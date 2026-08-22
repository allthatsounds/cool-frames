"""
numpy/phaseret/_gla.py
=======================
Griffin-Lim Algorithm adapted for filterbanks.

Port of ``phaseret/gabor/gla.m``.  The original operates in the Gabor
(DGT) domain; this version uses the filterbank analysis/synthesis cycle
from layer2.

The algorithm iterates:
    1. Synthesise signal from current coefficients
    2. Re-analyse signal
    3. Replace magnitudes with target, keep phase

Supports the fast Griffin-Lim (fGLA) acceleration from Perraudin et al.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..filterbanks._core import filterbank, ifilterbank
from ..filterbanks._frame import filterbankdual


def gla(
    s_list: list[np.ndarray],
    g: list[dict],
    a,
    *,
    L: int | None = None,
    Ls: int | None = None,
    real: bool = False,
    maxit: int = 100,
    tol: float = 1e-6,
    method: Literal["gla", "fgla"] = "gla",
    alpha: float = 0.99,
    startphase: Literal["input", "zero", "rand"] = "zero",
    seed: int | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]:
    """Griffin-Lim Algorithm for filterbanks.

    Reconstructs phase from magnitude spectra via iterative analysis-synthesis,
    optionally with fast acceleration [gla-griffin]_ and [gla-perraudin]_.

    Parameters
    ----------
    s_list : list of M arrays
        Target magnitudes per channel, each shape (N_m,).
    g : list of M filter dicts
        Analysis filterbank.
    a : hop sizes
    L : DFT length (inferred if None)
    Ls : output signal length for trimming
    real : use real (single-sided) synthesis
    maxit : maximum iterations
    tol : convergence tolerance on spectral convergence
    method : ``'gla'`` or ``'fgla'`` (fast Griffin-Lim)
    alpha : acceleration parameter for fGLA
    startphase : ``'input'``, ``'zero'``, or ``'rand'``
    seed : int, optional
        Seed for ``startphase='rand'``.  ``None`` (the default) draws from
        fresh entropy, so repeated calls differ; pass an integer to make a
        random start reproducible.

    Returns
    -------
    c : list of M complex arrays — coefficients with reconstructed phase
    f : reconstructed signal
    relres : (iter,) array of per-iteration residuals
    niter : number of iterations performed

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.phase import gla
    >>> # Generate some target magnitudes
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> from cool_frames.numpy.filterbanks import filterbank
    >>> x = np.random.default_rng(0).standard_normal(8000)
    >>> s_list = [np.abs(cm) for cm in filterbank(x, g, a, L=L)]
    >>> c, f, relres, niter = gla(s_list, g, a, L=L, maxit=10)
    >>> len(c) == len(g)  # same number of channels
    True
    >>> f.shape[0] > 0  # reconstructed signal
    True

    References
    ----------
    .. [gla-griffin] D. Griffin and J. Lim, "Signal estimation from modified short-time Fourier transform,"
           IEEE Trans. Acoustics, Speech, Signal Process., vol. 32, no. 2, pp. 236–243, 1984.
           doi:10.1109/TASSP.1984.1164317
    .. [gla-perraudin] N. Perraudin et al., "A fast Griffin-Lim algorithm," IEEE Workshop ASPAA, 2013.
    """
    M = len(g)
    s_abs = [np.abs(np.asarray(s)).ravel() for s in s_list]

    # Compute dual window for synthesis
    from ..filterbanks._utils import normalise_a

    a_norm = normalise_a(a, M)

    N = [len(s) for s in s_abs]
    if L is None:
        afrac = a_norm[:, 0] / a_norm[:, 1]
        L = int(round(N[0] * afrac[0]))

    if real:
        gd = filterbankdual(g, a_norm, L)
    else:
        gd = filterbankdual(g, a_norm, L, real=False)

    # Initialise coefficients
    if startphase == "zero":
        c = [s.copy().astype(complex) for s in s_abs]
    elif startphase == "rand":
        rng = np.random.default_rng(seed)
        c = [s * np.exp(2j * np.pi * rng.random(len(s))) for s in s_abs]
    else:  # 'input'
        c = [np.asarray(s, dtype=complex).ravel().copy() for s in s_list]

    # Normalization
    s_flat = np.concatenate(s_abs)
    norm_s = np.linalg.norm(s_flat)
    if norm_s == 0:
        norm_s = 1.0  # type: ignore[assignment]

    relres_list = []

    if method == "gla":
        for _it in range(maxit):
            # Synthesis
            f = ifilterbank(c, gd, a_norm, Ls=L, real=real)

            # Analysis
            c_new = filterbank(np.real(f) if real else f, g, a_norm, L=L)

            # Residual
            c_new_abs = np.concatenate([np.abs(np.asarray(cn).ravel()) for cn in c_new])
            res = float(np.linalg.norm(c_new_abs - s_flat) / norm_s)
            relres_list.append(res)

            # Phase replacement
            for m in range(M):
                cn = np.asarray(c_new[m]).ravel()
                phase = np.angle(cn)
                c[m] = s_abs[m] * np.exp(1j * phase)

            if res < tol:
                break

    elif method == "fgla":
        told = [ci.copy() for ci in c]
        for _it in range(maxit):
            f = ifilterbank(c, gd, a_norm, Ls=L, real=real)
            c_new = filterbank(np.real(f) if real else f, g, a_norm, L=L)

            c_new_abs = np.concatenate([np.abs(np.asarray(cn).ravel()) for cn in c_new])
            res = float(np.linalg.norm(c_new_abs - s_flat) / norm_s)
            relres_list.append(res)

            # Phase update
            tnew = []
            for m in range(M):
                cn = np.asarray(c_new[m]).ravel()
                phase = np.angle(cn)
                tnew.append(s_abs[m] * np.exp(1j * phase))

            # Acceleration
            c = [tnew[m] + alpha * (tnew[m] - told[m]) for m in range(M)]
            told = tnew

            if res < tol:
                break

        # The momentum step leaves the constraint set, so the extrapolated
        # point has |c| != s.  Project it back before returning: every other
        # member of the family guarantees |c_out| == s, and callers rely on it.
        # Re-projecting the extrapolate (rather than returning the last
        # projected iterate `told`) keeps the phase progress the momentum
        # bought, and measures at least as good on consistency at every
        # iteration count.
        c = [s_abs[m] * np.exp(1j * np.angle(c[m])) for m in range(M)]

    else:
        raise ValueError(f"Unknown method '{method}', expected 'gla' or 'fgla'")

    # Final synthesis
    f = ifilterbank(c, gd, a_norm, Ls=Ls or L, real=real)
    if real:
        f = np.real(f)

    niter = len(relres_list)
    relres = np.array(relres_list)

    return c, f, relres, niter
