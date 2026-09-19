"""High-quality spectrograms via filterbank analysis and reassignment.

Produces detailed frequency-domain representations with optional phase-derivative
reassignment for improved time-frequency resolution.

Example:
    spec = filterbank_spectrogram(signal, fs=16000, scale='erb', db_range=60)
    spec_reassigned = reassigned_spectrogram(signal, fs=16000)
"""

from __future__ import annotations

import numpy as np
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
        - 'instfreq_deviation': mean instantaneous frequency minus centre frequency, per channel (Hz, energy-weighted)
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

    # Compute phase gradients.  No fallback: a failure here used to be
    # swallowed and replaced by zero gradients, the pattern that made the
    # numpy version return identically zero summaries for every input
    # (DEFECT_REGISTER #8).
    tgrad, fgrad, s_pow, _ = filterbankphasegrad(f, g, a)  # type: ignore[arg-type]

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

    # Per-channel summaries.  `tgrad` is the *absolute* instantaneous
    # frequency normalised so that 2 = fs (Hz = tgrad * fs / 2); `fgrad` is
    # the group delay in samples.  The two used to be swapped here (the
    # "instfreq_deviation" was built from `fgrad` and the "groupdelay_shift"
    # from `tgrad`), with a spurious 1 / (2*pi); both now match the numpy
    # backend: the energy-weighted mean instantaneous frequency minus the
    # channel's centre frequency in Hz, and the mean group delay in samples.
    fc_hz = np.asarray(fc, dtype=float)
    if_vals = []
    gd_vals = []
    for ch, (tg, fg, sm) in enumerate(zip(tgrad, fgrad, s_pow)):
        tg = torch.real(tg).reshape(-1).to(dtype)
        fg = torch.real(fg).reshape(-1).to(dtype)
        w = torch.real(sm).reshape(-1).to(dtype)
        if tg.numel() == 0:
            if_vals.append(0.0)
        elif float(w.sum()) > 0:
            if_vals.append(float((tg * w).sum() / w.sum()) * fs / 2.0 - float(fc_hz[ch]))
        else:
            if_vals.append(float(tg.mean()) * fs / 2.0 - float(fc_hz[ch]))
        gd_vals.append(float(fg.mean()) if fg.numel() > 0 else 0.0)

    ifd_summary = torch.tensor(if_vals, device=device, dtype=dtype)
    gd_summary = torch.tensor(gd_vals, device=device, dtype=dtype)

    return {
        "coeff_db": mag_db_clipped,
        "fc": fc,
        "a": a,
        "fs": fs,
        "instfreq_deviation": ifd_summary,
        "groupdelay_shift": gd_summary,
    }
