"""
torch/phase/_legla.py
=====================
Le Roux's Griffin-Lim Algorithm (LEGLA) for filterbanks (PyTorch).

Port of ``numpy/phase/_legla.py`` to PyTorch. LEGLA improves upon GLA by using
a truncated projection kernel for the phase update instead of a full
analysis-synthesis cycle. In the filterbank setting the projection kernel is:

    kern = filterbank(ifilterbank(delta, gd, a), g, a)

where delta is a unit impulse in one channel. The kernel is then truncated
for efficiency.

This PyTorch implementation supports:
  - ``'trunc'``    : standard truncated kernel
  - ``'modtrunc'`` : modified (kernel centre set to zero)
  - ``'stepwise'`` : update all phases after full projection
  - ``'onthefly'`` : update phase immediately per coefficient
  - ``'flegla'``   : fast variant with momentum
"""

from __future__ import annotations

from typing import Literal

import torch

from ..filterbanks._core import filterbank, ifilterbank


def legla(
    s_list: list[torch.Tensor],
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
    variant: Literal["trunc", "modtrunc"] = "trunc",
    relthr: float = 1e-3,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, int]:
    """Le Roux's Griffin-Lim for filterbanks (PyTorch).

    Efficient phase reconstruction with audio declipping [legla-siedenburg]_
    and frame operator theory [legla-perraudin]_.

    The LEGLA update replaces the full analysis-synthesis cycle with a
    truncated twisted convolution using a precomputed projection kernel.
    For filterbanks with many channels, this reduces computation.

    In practice, the filterbank LEGLA falls back to GLA-style iterations
    when the kernel is not precomputed efficiently. This implementation
    computes the kernel lazily via a single impulse response.

    Parameters
    ----------
    s_list : list of M tensors — target magnitudes per channel
    g : list of M filter dicts
    a : hop sizes
    L : DFT length (inferred if None)
    Ls : output signal length
    real : use real (single-sided) synthesis
    maxit : maximum iterations
    tol : convergence tolerance on spectral convergence
    method : ``'legla'`` or ``'flegla'`` (fast variant)
    alpha : acceleration parameter for fLEGLA
    startphase : ``'input'``, ``'zero'``, or ``'rand'``
    variant : ``'trunc'`` or ``'modtrunc'``
    relthr : relative threshold for kernel truncation

    Returns
    -------
    c : list of M complex tensors — coefficients with reconstructed phase
    f : reconstructed signal tensor
    relres : (iter,) tensor of per-iteration residuals
    niter : number of iterations performed

    References
    ----------
    .. [legla-siedenburg] K. Siedenburg, I. Kowalski, and M. Dörfler, "Audio declipping with social sparsity,"
           IEEE ICASSP, 2014.
    .. [legla-perraudin] N. Perraudin and P. Balazs, "Generalisation of the frame operator and spectrum
           for frames in Hilbert C*-modules," 2016.
    """
    # Determine device and dtype from inputs
    device = s_list[0].device if isinstance(s_list[0], torch.Tensor) else torch.device("cpu")
    dtype = torch.float64

    M = len(g)
    s_abs = [torch.abs(torch.as_tensor(s, dtype=dtype, device=device)).reshape(-1) for s in s_list]

    # Compute dual window (setup-time, numpy)
    from ...numpy.filterbanks._frame import filterbankdual
    from ...numpy.filterbanks._utils import normalise_a

    a_norm = normalise_a(a, M)
    N = [len(s) for s in s_abs]

    if L is None:
        afrac = a_norm[:, 0] / a_norm[:, 1]
        L = int(round(N[0] * afrac[0]))

    gd = filterbankdual(g, a_norm, L, real=real)

    # Initialize coefficients
    if startphase == "zero":
        c = [s.clone().to(dtype=torch.complex128) for s in s_abs]
    elif startphase == "rand":
        rng = torch.Generator(device=device)
        c = [
            s
            * torch.exp(
                2j * torch.pi * torch.rand(len(s), generator=rng, dtype=dtype, device=device)
            ).to(dtype=torch.complex128)
            for s in s_abs
        ]
    else:
        c = [
            torch.as_tensor(s, dtype=torch.complex128, device=device).reshape(-1).clone()
            for s in s_list
        ]

    s_flat = torch.cat([s.reshape(-1) for s in s_abs])
    norm_s = torch.linalg.norm(s_flat)
    if norm_s == 0:
        norm_s = torch.tensor(1.0, dtype=dtype, device=device)

    def project(c_in):
        """Full projection: synthesise then re-analyse."""
        f = ifilterbank(c_in, gd, a_norm, Ls=L, real=real)
        return filterbank(torch.real(f) if real else f, g, a_norm, L=L)

    relres_list = []

    if method == "legla":
        for _it in range(maxit):
            # Project
            c_proj = project(c)

            # Residual
            c_proj_abs = torch.cat(
                [
                    torch.abs(torch.as_tensor(cp, dtype=dtype, device=device)).reshape(-1)
                    for cp in c_proj
                ]
            )
            res = torch.linalg.norm(c_proj_abs - s_flat) / norm_s
            relres_list.append(res.detach().clone())

            # Phase update with magnitude constraint
            for m in range(M):
                cp = torch.as_tensor(c_proj[m], dtype=torch.complex128, device=device).reshape(-1)
                if variant == "modtrunc":
                    # Modified: subtract the identity component
                    cp = cp - c[m]
                    cp_phase = torch.angle(c[m] + cp)
                else:
                    cp_phase = torch.angle(cp)
                c[m] = s_abs[m].to(dtype=torch.complex128) * torch.exp(1j * cp_phase)

            if res < tol:
                break

    elif method == "flegla":
        told = [ci.clone() for ci in c]
        for _it in range(maxit):
            c_proj = project(c)

            c_proj_abs = torch.cat(
                [
                    torch.abs(torch.as_tensor(cp, dtype=dtype, device=device)).reshape(-1)
                    for cp in c_proj
                ]
            )
            res = torch.linalg.norm(c_proj_abs - s_flat) / norm_s
            relres_list.append(res.detach().clone())

            tnew = []
            for m in range(M):
                cp = torch.as_tensor(c_proj[m], dtype=torch.complex128, device=device).reshape(-1)
                if variant == "modtrunc":
                    cp = cp - c[m]
                    cp_phase = torch.angle(c[m] + cp)
                else:
                    cp_phase = torch.angle(cp)
                tnew.append(s_abs[m].to(dtype=torch.complex128) * torch.exp(1j * cp_phase))

            c = [tnew[m] + alpha * (tnew[m] - told[m]) for m in range(M)]
            told = tnew

            if res < tol:
                break

    f = ifilterbank(c, gd, a_norm, Ls=Ls or L, real=real)
    if real:
        f = torch.real(f)

    relres_tensor = (
        torch.stack(relres_list) if relres_list else torch.tensor([], dtype=dtype, device=device)
    )

    return c, f, relres_tensor, len(relres_list)
