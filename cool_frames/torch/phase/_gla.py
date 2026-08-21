"""
torch/phase/_gla.py
====================
Griffin-Lim Algorithm for filterbanks (PyTorch).

Native torch port of the numpy ``_gla.py``.  Uses ``torch.fft`` through the
torch filterbank analysis/synthesis kernels, making the forward pass fully
differentiable.

Both standard GLA and fast GLA (fGLA) are supported.
"""

from __future__ import annotations

from typing import Literal

import torch

from ..filterbanks._core import filterbank, ifilterbank


def gla(
    s_list: list[torch.Tensor],
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
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, int]:
    """Griffin-Lim Algorithm for filterbanks (PyTorch).

    Differentiable phase reconstruction via iterative analysis-synthesis
    [torch_gla-griffin]_ with fast acceleration [torch_gla-perraudin]_.

    Parameters
    ----------
    s_list : list of M tensors
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

    Returns
    -------
    c : list of M complex tensors — coefficients with reconstructed phase
    f : reconstructed signal tensor
    relres : (iter,) tensor of per-iteration residuals
    niter : number of iterations performed

    Examples
    --------
    >>> import torch
    >>> from cool_frames.torch.phase import gla
    >>> from cool_frames.torch.filters import audfilters
    >>> # Create target magnitudes (e.g., from a spectrogram)
    >>> s = [torch.abs(torch.randn(64, dtype=torch.complex64)) for _ in range(28)]
    >>> g = audfilters(16000)
    >>> c, f, relres, niter = gla(s, g, a=64, maxit=50)
    >>> f.shape[0] > 0
    True
    >>> niter <= 50
    True

    References
    ----------
    .. [torch_gla-griffin] D. Griffin and J. Lim, "Signal estimation from modified short-time Fourier transform,"
           IEEE Trans. Acoustics, Speech, Signal Process., vol. 32, no. 2, pp. 236–243, 1984.
           doi:10.1109/TASSP.1984.1164317
    .. [torch_gla-perraudin] N. Perraudin et al., "A fast Griffin-Lim algorithm," IEEE Workshop ASPAA, 2013.
    """
    M = len(g)

    # Determine device and dtype from inputs
    device = s_list[0].device if isinstance(s_list[0], torch.Tensor) else torch.device("cpu")
    dtype = torch.float64

    s_abs = [torch.abs(s.to(dtype=dtype, device=device)).flatten() for s in s_list]

    # Compute dual window (setup-time, numpy)
    from ...numpy.filterbanks._frame import filterbankdual
    from ...numpy.filterbanks._utils import normalise_a

    a_norm = normalise_a(a, M)
    N = [len(s) for s in s_abs]

    if L is None:
        afrac = a_norm[:, 0] / a_norm[:, 1]
        L = int(round(N[0] * afrac[0]))

    gd = filterbankdual(g, a_norm, L, real=real)

    # Initialise coefficients
    if startphase == "zero":
        c = [s.clone().to(dtype=torch.complex128) for s in s_abs]
    elif startphase == "rand":
        c = [
            s * torch.exp(2j * torch.pi * torch.rand(len(s), device=device, dtype=dtype))
            for s in s_abs
        ]
    else:  # 'input'
        c = [s.to(dtype=torch.complex128, device=device).flatten().clone() for s in s_list]

    # Normalisation
    s_flat = torch.cat(s_abs)
    norm_s = torch.linalg.norm(s_flat).item()
    if norm_s == 0:
        norm_s = 1.0

    relres_list = []

    if method == "gla":
        for _it in range(maxit):
            # Synthesis
            f = ifilterbank(c, gd, a_norm, Ls=L, real=real)

            # Analysis
            f_input = f.real if (real and f.is_complex()) else f
            c_new = filterbank(f_input, g, a_norm, L=L)

            # Residual
            c_new_abs = torch.cat([torch.abs(cn.flatten()) for cn in c_new])
            res = float(torch.linalg.norm(c_new_abs - s_flat).item() / norm_s)
            relres_list.append(res)

            # Phase replacement
            for m in range(M):
                cn = c_new[m].flatten()
                phase = torch.angle(cn)
                c[m] = s_abs[m] * torch.exp(1j * phase)

            if res < tol:
                break

    elif method == "fgla":
        told = [ci.clone() for ci in c]
        for _it in range(maxit):
            f = ifilterbank(c, gd, a_norm, Ls=L, real=real)

            f_input = f.real if (real and f.is_complex()) else f
            c_new = filterbank(f_input, g, a_norm, L=L)

            c_new_abs = torch.cat([torch.abs(cn.flatten()) for cn in c_new])
            res = float(torch.linalg.norm(c_new_abs - s_flat).item() / norm_s)
            relres_list.append(res)

            # Phase update
            tnew = []
            for m in range(M):
                cn = c_new[m].flatten()
                phase = torch.angle(cn)
                tnew.append(s_abs[m] * torch.exp(1j * phase))

            # Acceleration
            c = [tnew[m] + alpha * (tnew[m] - told[m]) for m in range(M)]
            told = tnew

            if res < tol:
                break
    else:
        raise ValueError(f"Unknown method '{method}', expected 'gla' or 'fgla'")

    # Final synthesis
    f = ifilterbank(c, gd, a_norm, Ls=Ls or L, real=real)
    if real and f.is_complex():
        f = f.real

    niter = len(relres_list)
    relres = torch.tensor(relres_list, dtype=dtype, device=device)

    return c, f, relres, niter
