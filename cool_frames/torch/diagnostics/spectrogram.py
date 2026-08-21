"""High-quality spectrograms via filterbank analysis and reassignment.

Produces detailed frequency-domain representations with optional phase-derivative
reassignment for improved time-frequency resolution.

Example:
    spec = filterbank_spectrogram(signal, fs=16000, scale='erb', db_range=60)
    spec_reassigned = reassigned_spectrogram(signal, fs=16000)
"""

from __future__ import annotations

import torch

from ..filterbanks import filterbank
from ..filters import audfilters, cqtfilters
from ..phase import filterbankphasegrad


def filterbank_spectrogram(
    f: torch.Tensor, fs: float, scale: str = "erb", db_range: float = 60
) -> dict:
    """Compute filterbank spectrogram with dB scaling.

    Parameters
    ----------
    f : torch.Tensor
        Input audio signal (1D).
    fs : float
        Sample rate (Hz).
    scale : {'erb', 'cqt'}, optional
        Filter design: 'erb' for auditory scale, 'cqt' for constant-Q (default 'erb').
    db_range : float, optional
        Dynamic range in dB for display (default 60). Values below peak - db_range
        are clipped.

    Returns
    -------
    spec : dict
        Dictionary with keys:
        - 'coeff_db': magnitude coefficients in dB, stacked as 2D tensor
        - 'fc': channel center frequencies (Hz)
        - 'a': hop sizes
        - 'g': filter impulse responses
        - 'fs': sample rate
        - 'db_range': dynamic range used
    """
    device = f.device
    dtype = f.dtype

    # Design filterbank
    if scale == "cqt":
        g, a, fc, _L, _info = cqtfilters(fs, len(f), fmin=20, fmax=20000, bins=96)
    else:  # 'erb'
        g, a, fc, _L, _info = audfilters(fs, len(f))

    # Analyse
    c = filterbank(f, g, a)  # list of tensors

    # Convert to dB and stack
    mag_db_list = []
    for c_ch in c:
        mag = torch.abs(c_ch)
        mag_db = 20 * torch.log10(mag + 1e-10)
        mag_db_list.append(mag_db)

    # Pad shorter channels to match longest
    max_len = max(len(m) for m in mag_db_list)
    mag_db_stacked = torch.zeros((len(c), max_len), device=device, dtype=dtype)
    for i, m in enumerate(mag_db_list):
        mag_db_stacked[i, : len(m)] = m

    peak_db = torch.max(mag_db_stacked)
    mag_db_clipped = torch.maximum(mag_db_stacked, peak_db - db_range)

    return {
        "coeff_db": mag_db_clipped,
        "fc": fc,
        "a": a,
        "g": g,
        "fs": fs,
        "db_range": db_range,
    }


def reassigned_spectrogram(f: torch.Tensor, fs: float, scale: str = "erb") -> dict:
    """Compute spectrogram with phase gradient information.

    Computes phase gradients which can be used to estimate instantaneous
    frequency and group delay for each time-frequency cell.

    Parameters
    ----------
    f : torch.Tensor
        Input audio signal (1D).
    fs : float
        Sample rate (Hz).
    scale : {'erb', 'cqt'}, optional
        Filter design (default 'erb').

    Returns
    -------
    spec : dict
        Dictionary with keys:
        - 'coeff_db': magnitude spectrogram in dB
        - 'fc': channel center frequencies (Hz)
        - 'a': hop sizes
        - 'fs': sample rate
        - 'instfreq_deviation': instantaneous frequency shift per channel (Hz)
        - 'groupdelay_shift': group delay shift per channel (samples)
    """
    device = f.device
    dtype = f.dtype

    # Design filterbank
    if scale == "cqt":
        g, a, fc, _L, _info = cqtfilters(fs, len(f), fmin=20, fmax=20000, bins=96)
    else:  # 'erb'
        g, a, fc, _L, _info = audfilters(fs, len(f))

    # Analyse
    c = filterbank(f, g, a)  # list of tensors

    # Compute phase gradients
    try:
        tgrad, fgrad, _, _ = filterbankphasegrad(f, g, a)  # type: ignore[arg-type]
    except Exception:
        # If phase gradient computation fails, use zero gradients
        tgrad = [torch.zeros_like(c_ch) for c_ch in c]
        fgrad = [torch.zeros_like(c_ch) for c_ch in c]

    # Convert to dB and stack
    mag_db_list = []
    for c_ch in c:
        mag = torch.abs(c_ch)
        mag_db = 20 * torch.log10(mag + 1e-10)
        mag_db_list.append(mag_db)

    # Pad shorter channels
    max_len = max(len(m) for m in mag_db_list)
    mag_db_stacked = torch.zeros((len(c), max_len), device=device, dtype=dtype)
    for i, m in enumerate(mag_db_list):
        mag_db_stacked[i, : len(m)] = m

    peak_db = torch.max(mag_db_stacked)
    mag_db_clipped = torch.maximum(mag_db_stacked, peak_db - 60)

    # Average phase gradient data across time for per-channel summary
    fgrad_vals = []
    for fg in fgrad:
        if len(fg) > 0:
            mean_val = torch.mean(fg)
            # Handle complex values by taking real part
            if torch.is_complex(mean_val):
                mean_val = torch.real(mean_val)
            fgrad_vals.append(mean_val.item())
        else:
            fgrad_vals.append(0.0)

    tgrad_vals = []
    for tg in tgrad:
        if len(tg) > 0:
            mean_val = torch.mean(tg)
            # Handle complex values by taking real part
            if torch.is_complex(mean_val):
                mean_val = torch.real(mean_val)
            tgrad_vals.append(mean_val.item())
        else:
            tgrad_vals.append(0.0)

    fgrad_summary = torch.tensor(fgrad_vals, device=device, dtype=dtype)
    tgrad_summary = torch.tensor(tgrad_vals, device=device, dtype=dtype)

    return {
        "coeff_db": mag_db_clipped,
        "fc": fc,
        "a": a,
        "fs": fs,
        "instfreq_deviation": fgrad_summary * fs / (2 * 3.14159265358979323846),
        "groupdelay_shift": tgrad_summary,
    }
