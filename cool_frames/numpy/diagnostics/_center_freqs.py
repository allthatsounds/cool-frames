"""numpy.diagnostics._center_freqs – estimate filterbank centre frequencies.

Moved here from cool_frames.filterbanks 2026-06-12: it is an *estimator* (circular
centre-of-gravity of |H|), a sibling of the other diagnostics, not a core
analysis/synthesis op.
"""
from __future__ import annotations

import numpy as np


def center_freqs(g: list[dict], L: int) -> np.ndarray:
    """Compute normalized center frequencies for a filterbank.

    Parameters
    ----------
    g : list of filter dicts
        Each filter must have 'H' (callable or array) and other properties.
    L : int
        System length / DFT length.

    Returns
    -------
    cfreq : ndarray, shape (M,)
        Normalized center frequencies in [-1, 1].
        Computed as circular center of gravity of filter magnitudes.

    Notes
    -----
    The center frequency is computed from the absolute value of the
    transfer function using the circular center of gravity method.
    """
    from ..filters._filters import filter_freqresp

    M = len(g)
    cfreq = np.zeros(M, dtype=float)

    # Circular indices: exp(2πi*n/L) for n = 0, 1, ..., L-1
    circ_ind = np.exp(2j * np.pi * np.arange(L) / float(L))

    for m, g_m in enumerate(g):
        H, _ = filter_freqresp(g_m, L)

        mag = np.abs(H)
        norm_val = np.sum(mag)
        if norm_val > 1e-30:
            mag = mag / norm_val

        center_of_gravity = np.sum(circ_ind * mag)

        if abs(center_of_gravity) > 1e-30:
            arg_z = np.angle(center_of_gravity)
            cfreq[m] = arg_z / np.pi
        else:
            cfreq[m] = 0.0

    return cfreq
