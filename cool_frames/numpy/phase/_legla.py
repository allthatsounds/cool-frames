"""
numpy/phase/_legla.py
=====================
Le Roux's Griffin-Lim Algorithm (LEGLA) for filterbanks.

Where GLA projects onto the range of the analysis operator by synthesising and
re-analysing, LEGLA applies the same projection as a *convolution* with a
precomputed kernel, truncated at a relative threshold.  See
``_leglakernel.py`` for the kernel construction and its validation; the short
version is::

    (P c)_m[n] = sum_{m'} sum_{n'} k_{m,m'}[(n a_m - n' a_m') mod L] c_{m'}[n']
    k_{m,m'}   = ifft( conj(G_m) * Gd_{m'} )

Discarding kernel entries below ``relthr`` times the peak drops whole channel
pairs — filters in different parts of the spectrum barely interact — and gives
a cheaper, genuinely different projection operator.

Supported:
  - ``'trunc'``    : truncate the kernel at ``relthr``
  - ``'modtrunc'`` : additionally zero the self-term ``k_{m,m}[0]``, so a
                      coefficient does not contribute to its own update
  - ``'legla'``    : plain iteration
  - ``'flegla'``   : fast variant with momentum

Setting ``relthr=0`` keeps the entire kernel, which is mathematically identical
to GLA's full projection — the property the kernel is tested against.

Before v0.1.1 this module ignored ``relthr`` and ``variant`` entirely and ran a
full projection, making ``legla`` a bit-identical alias for ``gla``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..filterbanks._core import filterbank, ifilterbank
from ..filterbanks._frame import filterbankdual


def legla(
    s_list: list[np.ndarray],
    g: list[dict],
    a,
    *,
    L: int | None = None,
    Ls: int | None = None,
    real: bool = False,
    maxit: int = 100,
    tol: float = 1e-6,
    method: Literal["legla", "flegla"] = "legla",
    alpha: float = 0.99,
    startphase: Literal["input", "zero", "rand"] = "zero",
    seed: int | None = None,
    variant: Literal["trunc", "modtrunc"] = "trunc",
    relthr: float = 1e-3,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]:
    """Le Roux's Griffin-Lim for filterbanks.

    Efficient phase reconstruction with audio declipping [legla-siedenburg]_
    and frame operator theory [legla-perraudin]_.

    The LEGLA update replaces the full analysis-synthesis cycle with a
    truncated convolution against a precomputed projection kernel.

    Parameters
    ----------
    s_list : list of M arrays — target magnitudes
    g : list of M filter dicts
    a : hop sizes
    L, Ls, real, maxit, tol, method, alpha, startphase, seed : same as :func:`gla`
    variant : ``'trunc'`` (truncate at ``relthr``) or ``'modtrunc'``
        (additionally zero the self-term ``k_{m,m}[0]``, so a coefficient makes
        no contribution to its own update).
    relthr : float
        Relative threshold for kernel truncation: entries below
        ``relthr * max|k|`` are discarded.  ``0.0`` keeps the whole kernel and
        reproduces GLA's exact projection; the default ``1e-3`` typically
        retains a fifth of the channel pairs.  Larger values are cheaper per
        iteration and change the operator more.

    Notes
    -----
    LEGLA trades a one-off cost for a cheaper iteration: the kernel takes
    O(M^2) FFTs to build, after which each projection is a sparse matrix
    product instead of two full FFT passes per channel.  It therefore wins only
    once ``maxit`` is large enough to amortise the setup.  Measured on a
    23-channel ERB bank at ``Ls=512``::

        maxit=10    gla 0.07 s    legla(relthr=1e-2) 0.22 s
        maxit=100   gla 0.66 s    legla(relthr=1e-2) 0.29 s

    For a handful of iterations, use :func:`gla`.

    With fractional hop sizes there is no integer lag grid, so the convolution
    form does not apply and this falls back to the exact projection.

    Returns
    -------
    c, f, relres, niter : same as :func:`gla`

    References
    ----------
    .. [legla-siedenburg] K. Siedenburg, I. Kowalski, and M. Dörfler, "Audio declipping with social sparsity,"
           IEEE ICASSP, 2014.
    .. [legla-perraudin] N. Perraudin and P. Balazs, "Generalisation of the frame operator and spectrum
           for frames in Hilbert C*-modules," 2016.
    """
    M = len(g)
    s_abs = [np.abs(np.asarray(s)).ravel() for s in s_list]

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
    else:
        c = [np.asarray(s, dtype=complex).ravel().copy() for s in s_list]

    s_flat = np.concatenate(s_abs)
    norm_s = np.linalg.norm(s_flat)
    if norm_s == 0:
        norm_s = 1.0  # type: ignore[assignment]

    # The LEGLA update replaces the full synthesise-and-reanalyse projection
    # with a *truncated* one: the projection operator is a convolution whose
    # kernel is built once (see _leglakernel.py) and thresholded at `relthr`.
    #
    # `relthr=0` keeps the whole kernel and is mathematically identical to the
    # full projection; larger values discard channel pairs and lags that
    # contribute little, giving a cheaper — and genuinely different — operator.
    hops = np.round(a_norm[:, 0] / a_norm[:, 1]).astype(int)
    use_kernel = relthr >= 0 and np.allclose(a_norm[:, 0] / a_norm[:, 1], hops)

    if use_kernel:
        from ._leglakernel import LeglaKernel

        kernel = LeglaKernel(
            g,
            gd,
            hops,
            N,
            L,
            real=real,
            relthr=relthr,
            zero_self_term=(variant == "modtrunc"),
        )

        def project(c_in):
            return kernel.project(c_in)

    else:
        # Fractional hop sizes have no integer lag grid, so the convolution
        # form does not apply; fall back to the exact projection.
        def project(c_in):
            f = ifilterbank(c_in, gd, a_norm, Ls=L, real=real)
            return filterbank(np.real(f) if real else f, g, a_norm, L=L)

    relres_list = []

    if method == "legla":
        for _it in range(maxit):
            # Project
            c_proj = project(c)

            # Residual
            c_proj_abs = np.concatenate([np.abs(np.asarray(cp).ravel()) for cp in c_proj])
            res = float(np.linalg.norm(c_proj_abs - s_flat) / norm_s)
            relres_list.append(res)

            # Phase update with magnitude constraint.  `variant` is applied
            # when the kernel is built, not here — the old branch computed
            # angle(c + (proj - c)) == angle(proj), i.e. nothing.
            for m in range(M):
                c[m] = s_abs[m] * np.exp(1j * np.angle(np.asarray(c_proj[m]).ravel()))

            if res < tol:
                break

    elif method == "flegla":
        told = [ci.copy() for ci in c]
        for _it in range(maxit):
            c_proj = project(c)

            c_proj_abs = np.concatenate([np.abs(np.asarray(cp).ravel()) for cp in c_proj])
            res = float(np.linalg.norm(c_proj_abs - s_flat) / norm_s)
            relres_list.append(res)

            tnew = [
                s_abs[m] * np.exp(1j * np.angle(np.asarray(c_proj[m]).ravel())) for m in range(M)
            ]

            c = [tnew[m] + alpha * (tnew[m] - told[m]) for m in range(M)]
            told = tnew

            if res < tol:
                break

        # See the note in _gla.py: the momentum step leaves the constraint set,
        # so project the extrapolated point back before returning.
        c = [s_abs[m] * np.exp(1j * np.angle(c[m])) for m in range(M)]

    f = ifilterbank(c, gd, a_norm, Ls=Ls or L, real=real)
    if real:
        f = np.real(f)

    return c, f, np.array(relres_list), len(relres_list)
