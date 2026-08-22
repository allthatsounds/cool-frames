"""High-quality spectrograms via filterbank analysis and reassignment.

Produces detailed frequency-domain representations with optional phase-derivative
reassignment for improved time-frequency resolution.

Example:
    spec = filterbank_spectrogram(signal, fs=16000, scale='erb', db_range=60)
    spec_reassigned = reassigned_spectrogram(signal, fs=16000)
"""
from __future__ import annotations

import numpy as np
from cool_frames.numpy.filterbanks import filterbank
from cool_frames.numpy.filters import audfilters, cqtfilters
from cool_frames.numpy.phase import filterbankphasegrad


def filterbank_spectrogram(
    f: np.ndarray,
    fs: float,
    scale: str = 'erb',
    db_range: float = 60
) -> dict:
    """Compute filterbank spectrogram with dB scaling.

    Parameters
    ----------
    f : np.ndarray
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
        - 'coeff_db': magnitude coefficients in dB, stacked as 2D array
        - 'fc': channel center frequencies (Hz)
        - 'a': hop sizes
        - 'g': filter impulse responses
        - 'fs': sample rate
        - 'db_range': dynamic range used
    """
    # Design filterbank
    if scale == 'cqt':
        g, a, fc, L, _info = cqtfilters(fs, Ls=len(f), fmin=20, fmax=20000, bins=96)
    else:  # 'erb'
        g, a, fc, L, _info = audfilters(fs, len(f))

    # Analyse
    c = filterbank(f, g, a)  # list of arrays

    # Convert to dB and stack
    mag_db_list = []
    for c_ch in c:
        mag = np.abs(c_ch)
        mag_db = 20 * np.log10(mag + 1e-10)
        mag_db_list.append(mag_db)

    # Pad shorter channels to match the longest.  The pad value is the display
    # floor, not 0.0 dB: channels are decimated by different hops, so most of
    # the image is padding, and padding at magnitude 1.0 let `peak_db` be set by
    # the padding rather than the signal for any quiet input.
    peak_db = float(max(np.max(m) for m in mag_db_list))
    floor_db = peak_db - db_range

    max_len = max(len(m) for m in mag_db_list)
    mag_db_stacked = np.full((len(c), max_len), floor_db, dtype=float)
    for i, m in enumerate(mag_db_list):
        mag_db_stacked[i, :len(m)] = m

    mag_db_clipped = np.maximum(mag_db_stacked, floor_db)

    return {
        'coeff_db': mag_db_clipped,
        'fc': fc,
        'a': a,
        'g': g,
        'fs': fs,
        'db_range': db_range,
    }


def reassigned_spectrogram(
    f: np.ndarray,
    fs: float,
    scale: str = 'erb',
    db_range: float = 60,
) -> dict:
    """Compute spectrogram with phase gradient information.

    Computes phase gradients which can be used to estimate instantaneous
    frequency and group delay for each time-frequency cell.

    Parameters
    ----------
    f : np.ndarray
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
    # Design filterbank
    if scale == 'cqt':
        g, a, fc, L, _info = cqtfilters(fs, Ls=len(f), fmin=20, fmax=20000, bins=96)
    else:  # 'erb'
        g, a, fc, L, _info = audfilters(fs, len(f))

    # Analyse
    c = filterbank(f, g, a, L=L)  # list of arrays

    # Compute phase gradients.
    #
    # Until v0.1.1 this called `filterbankphasegrad(c, a, fc)` — coefficients
    # as the signal, hop sizes as the filters, centre frequencies as the hops —
    # and unpacked the 4-tuple return into two names.  It could never succeed,
    # and a bare `except Exception` turned the failure into zeros, so both
    # documented outputs were identically zero for every input.
    #
    # The real signature is
    #     filterbankphasegrad(f, g, a, L) -> (tgrad, fgrad, s, c)
    # with `tgrad` the normalised instantaneous frequency and `fgrad` the group
    # delay in samples — the opposite of how the two were mapped below.
    tgrad, fgrad, _s, _c = filterbankphasegrad(f, g, a, L)

    # Convert to dB
    mag_db_list = [20 * np.log10(np.abs(c_ch) + 1e-10) for c_ch in c]

    # Stack, padding the shorter (more heavily decimated) channels.
    #
    # The pad value has to be the display floor, not 0.0: channels are
    # decimated by different hops, so on a typical ERB bank ~88 % of the image
    # is padding.  Padding at 0.0 dB (magnitude 1.0) meant `peak_db` could be
    # the *padding* rather than the signal — for a quiet input the whole 60 dB
    # window was anchored ~30 dB too high and most real coefficients fell below
    # the floor.
    peak_db = float(max(np.max(m) for m in mag_db_list))
    floor_db = peak_db - db_range

    max_len = max(len(m) for m in mag_db_list)
    mag_db_stacked = np.full((len(c), max_len), floor_db, dtype=float)
    for i, m in enumerate(mag_db_list):
        mag_db_stacked[i, :len(m)] = m

    mag_db_clipped = np.maximum(mag_db_stacked, floor_db)

    # Average phase-gradient data across time for a per-channel summary.
    # `tgrad` is instantaneous frequency (normalised), `fgrad` group delay.
    tgrad_summary = np.array([np.mean(tg) if len(tg) > 0 else 0.0 for tg in tgrad])
    fgrad_summary = np.array([np.mean(fg) if len(fg) > 0 else 0.0 for fg in fgrad])

    return {
        'coeff_db': mag_db_clipped,
        'fc': fc,
        'a': a,
        'fs': fs,
        'instfreq_deviation': tgrad_summary * fs / (2 * np.pi),
        'groupdelay_shift': fgrad_summary,
    }
