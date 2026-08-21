"""
numpy.core._norm
=================
Window / signal normalisation helpers shared across all LTFAT packages.

Previously in the separate ``ltfat_core`` package; vendored here to
remove the external dependency.
"""

from __future__ import annotations

import numpy as np


def normalize_window(g: np.ndarray, norm: str) -> np.ndarray:
    """Apply normalisation to a window (simple internal helper).

    For the full public ``setnorm`` (multi-dim, returns norm value),
    use :func:`cool_frames.numpy.filterbanks.setnorm`.

    Parameters
    ----------
    g : ndarray
        Input array.
    norm : str
        ``"2"`` — unit L2 norm (default for pgauss);
        ``"inf"`` — peak = 1;
        ``"1"`` — unit L1 norm;
        ``"null"`` or ``"none"`` — no change.

    Returns
    -------
    g_out : ndarray
        Normalised copy.
    """
    norm = norm.lower().strip()
    if norm in ("2", "energy"):
        n = np.linalg.norm(g)
        return g / n if n > 0 else g
    elif norm in ("inf", "peak"):
        m = np.max(np.abs(g))
        return g / m if m > 0 else g
    elif norm in ("1",):
        s = np.sum(np.abs(g))
        return g / s if s > 0 else g
    elif norm in ("null", "none", ""):
        return g
    else:
        raise ValueError(f"Unknown norm type: {norm!r}")
