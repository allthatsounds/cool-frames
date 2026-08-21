"""
numpy/filters/_tfr.py
=====================
Compute per-channel time-frequency ratios (tfr) from filterbank descriptors.

For frequency-domain filters (audfilters, cqtfilters, hopfilters), the tfr
is ``L / gamma`` where ``gamma`` is computed from the filter's frequency
response shape via ``_comp_tfrfromwin``.

For time-domain windows (gabfilters), the tfr is ``gamma / L``.

The distinction arises because ``_comp_tfrfromwin`` measures the window's
support in whichever domain it is given.  A frequency-domain filter's
"width" in frequency maps to ``gamma`` via the same formula, but the
physical tfr requires the reciprocal.
"""
from __future__ import annotations

import numpy as np

from ._gabfilters import _comp_tfrfromwin


def compute_tfr_from_filters(
    g: list[dict],
    L: int,
    *,
    default_tfr: float = 1.0,
    min_tfr: float = 1e-8,
) -> np.ndarray:
    """Compute per-channel tfr from a filterbank descriptor list.

    Parameters
    ----------
    g : list of dict
        Filter descriptors as returned by ``audfilters``, ``cqtfilters``,
        ``hopfilters``, ``gabfilters``, etc.  Each dict must have an ``'H'``
        key whose value is either a callable ``H(L) -> ndarray`` or a
        numpy array.
    L : int
        Transform length.
    default_tfr : float
        Value to use for channels where ``_comp_tfrfromwin`` fails
        (e.g. complement lowpass/highpass filters with flat frequency
        responses).  Default 1.0.
    min_tfr : float
        Floor value to clamp tfr away from zero.  Default 1e-8.

    Returns
    -------
    tfr : ndarray, shape (M,)
        Per-channel time-frequency ratios.
    """
    M = len(g)
    tfr = np.full(M, default_tfr, dtype=float)

    for m in range(M):
        H = g[m]["H"]
        # Evaluate if callable (lazy frequency-domain filters)
        if callable(H):
            try:
                H_val = np.abs(H(L)).ravel()
            except Exception:
                continue
        else:
            H_val = np.abs(np.asarray(H)).ravel()

        if len(H_val) == 0 or np.max(H_val) < 1e-30:
            continue

        gamma = _comp_tfrfromwin(H_val)

        if gamma <= 0 or not np.isfinite(gamma):
            continue

        # Frequency-domain filters: tfr = L / gamma
        tfr_m = L / gamma
        if not np.isfinite(tfr_m) or tfr_m < min_tfr:
            tfr_m = default_tfr

        tfr[m] = tfr_m

    # Clamp
    tfr = np.maximum(tfr, min_tfr)
    return tfr  # type: ignore[no-any-return]
