"""Spectral thresholding denoising via filterbank analysis-modify-synthesis.

Applies soft, hard, or Wiener thresholding per filterbank channel.
Tight-frame analysis-synthesis guarantees energy preservation and quantifiable error.

Example:
    denoised, stats = denoise(noisy_signal, fs=16000, threshold_db=-30, method='wiener')
"""

from __future__ import annotations

import torch
from cool_frames.torch.filterbanks import filterbank, filterbankdual, ifilterbank
from cool_frames.torch.filters import audfilters


def denoise(
    f: torch.Tensor, fs: float, threshold_db: float = -30, method: str = "wiener"
) -> tuple[torch.Tensor, dict]:
    """Denoise audio via filterbank-domain spectral thresholding.

    Parameters
    ----------
    f : torch.Tensor
        Input audio signal (1D).
    fs : float
        Sample rate (Hz).
    threshold_db : float, optional
        Threshold level in dB (default -30). Used differently by each method:
        - 'hard': magnitude threshold in dB below peak per channel
        - 'soft': soft thresholding with magnitude threshold
        - 'wiener': noise level estimate for Wiener filtering
    method : {'wiener', 'hard', 'soft'}, optional
        Thresholding method (default 'wiener').

    Returns
    -------
    f_denoised : torch.Tensor
        Denoised signal.
    stats : dict
        Dictionary with keys:
        - 'energy_before': per-channel energy before thresholding
        - 'energy_after': per-channel energy after thresholding
        - 'channel_freqs': center frequencies of analysis filters
        - 'method': thresholding method used
    """
    device = f.device
    dtype = f.dtype

    # Design ERB-spaced auditory filterbank (tight frame)
    g, a, fc, L, _info = audfilters(fs, len(f))

    # Analyse
    c = filterbank(f, g, a)  # list of tensors, one per channel

    # Compute per-channel energy before thresholding
    energy_before = torch.tensor(
        [torch.sum(torch.abs(c_ch) ** 2).item() for c_ch in c], device=device, dtype=dtype
    )

    # Threshold per channel
    c_thresholded = _threshold_coefficients(c, threshold_db, method, device, dtype)

    # Compute per-channel energy after thresholding
    energy_after = torch.tensor(
        [torch.sum(torch.abs(c_ch) ** 2).item() for c_ch in c_thresholded],
        device=device,
        dtype=dtype,
    )

    # Synthesise with the canonical *dual* of the analysis frame.
    #
    # Until v0.1.1 this used `filterbanktight(g, a, L)`, which is the tight
    # frame associated with `g` — correct only if the analysis had also been
    # done with it.  Pairing `filterbank(f, g, a)` with a tight synthesis is
    # not a round trip: with thresholding effectively disabled the residual was
    # 28.7 (relative), where `filterbankdual` gives 4.2e-16.
    gd = filterbankdual(g, a, L, real=True)

    f_denoised = ifilterbank(c_thresholded, gd, a, len(f))

    stats = {
        "energy_before": energy_before,
        "energy_after": energy_after,
        "channel_freqs": fc,
        "method": method,
    }

    return f_denoised, stats


def _threshold_coefficients(
    c: list[torch.Tensor],
    threshold_db: float,
    method: str,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    """Apply thresholding rule to filterbank coefficients.

    Parameters
    ----------
    c : list of torch.Tensor
        Coefficient list (one tensor per channel).
    threshold_db : float
        Threshold level in dB.
    method : {'wiener', 'hard', 'soft'}
        Thresholding method.
    device : torch.device
        Target device.
    dtype : torch.dtype
        Target dtype.

    Returns
    -------
    c_thresh : list of torch.Tensor
        Thresholded coefficients.
    """
    c_out = []
    threshold_linear = 10 ** (threshold_db / 20)

    for ch_data in c:
        mag = torch.abs(ch_data)

        if method == "hard":
            # Hard thresholding: zero out magnitudes below threshold
            peak_mag = (
                torch.max(mag)
                if torch.max(mag) > 0
                else torch.tensor(1.0, device=device, dtype=dtype)
            )
            tau = threshold_linear * peak_mag
            mask = (mag > tau).float()
            c_out.append(ch_data * mask)

        elif method == "soft":
            # Soft thresholding: shrink magnitudes
            peak_mag = (
                torch.max(mag)
                if torch.max(mag) > 0
                else torch.tensor(1.0, device=device, dtype=dtype)
            )
            tau = threshold_linear * peak_mag
            shrunken_mag = torch.clamp(mag - tau, min=0)
            phase = torch.angle(ch_data)
            c_out.append(shrunken_mag * torch.exp(1j * phase))

        elif method == "wiener":
            # Wiener filtering: assume noise level, scale by SNR per frame
            frame_energy = mag**2
            noise_level_sq = (threshold_linear**2) * torch.mean(frame_energy)
            snr = frame_energy / (noise_level_sq + 1e-10)
            wiener_gain = snr / (1 + snr)
            c_out.append(ch_data * wiener_gain)

    return c_out
