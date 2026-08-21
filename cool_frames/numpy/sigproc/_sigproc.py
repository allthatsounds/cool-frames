"""
numpy/sigproc/_sigproc.py
=========================
Coefficient-domain sparsity primitives: thresholding and sparsity selection.
These are the building blocks of the sparsity-based methods (denoising,
sparse solvers) that operate on filterbank coefficients.

Public home: cool_frames.sigproc. Mirrors cool_frames.torch.sigproc.

MATLAB originals: sigproc/{thresh,largestn,largestr}.m
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# thresh – coefficient thresholding (hard / soft / wiener)
# ---------------------------------------------------------------------------

def thresh(x, lam, mode: str = "hard"):
    """Coefficient thresholding.

    Parameters
    ----------
    x : array_like
        Input coefficients (real or complex).
    lam : float or array_like
        Threshold value.  Scalar → uniform threshold.  Array of same
        shape as *x* (or broadcastable) → element-wise threshold.
    mode : str
        ``'hard'`` (default), ``'soft'``, or ``'wiener'``.

    Returns
    -------
    xo : ndarray
        Thresholded coefficients.
    N : int
        Number of coefficients kept (non-zero after thresholding).
    """
    x = np.asarray(x)
    lam = np.asarray(lam)
    was_real = np.isrealobj(x)

    mode = mode.lower()
    if mode == "hard":
        mask = np.abs(x) >= lam
        xo = x * mask
    elif mode == "soft":
        ax = np.abs(x)
        shrunk = ax - lam
        shrunk = np.maximum(shrunk, 0.0)
        # Preserve phase for complex; sign for real
        xo = shrunk * np.exp(1j * np.angle(x))
        if was_real:
            xo = xo.real
    elif mode == "wiener":
        ax = np.abs(x)
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(ax > 0, lam / ax, 0.0)
        gain = np.maximum(1.0 - ratio ** 2, 0.0)
        xo = x * gain
    else:
        raise ValueError(f"Unknown thresholding mode: {mode!r}")

    N = int(np.count_nonzero(xo))
    return xo, N


# ---------------------------------------------------------------------------
# expand – dynamic range expansion
# ---------------------------------------------------------------------------

def largest(x, amount, mode: str = "hard"):
    """Keep the largest coefficients by magnitude (unifies largestn/largestr).

    Parameters
    ----------
    x : array_like
    amount : int or float
        ``amount >= 1`` (integer-valued) → absolute count N to keep;
        ``0 < amount < 1`` → fraction of coefficients to keep.
    mode : {'hard','soft','wiener'}

    Returns
    -------
    xo : ndarray  (same shape as x, thresholded)
    N_kept : int
    """
    x = np.asarray(x)
    ss = x.size
    if amount <= 0:
        return np.zeros_like(x), 0
    if amount >= 1 and amount == int(amount):
        N = int(amount)
    else:
        N = round(ss * amount)
    N = min(N, ss)
    if N <= 0:
        return np.zeros_like(x), 0
    flat = np.abs(x).ravel()
    lam = np.partition(flat, ss - N)[ss - N]
    return thresh(x, lam, mode=mode)
