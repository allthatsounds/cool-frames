"""
torch/phase/_decolbfgs.py
==========================
Décorsière's L-BFGS phase retrieval adapted for filterbanks (PyTorch).

Port of ``phaseret/gabor/decolbfgs.m`` and numpy implementation.

The algorithm formulates phase retrieval as an unconstrained smooth
optimization problem.  Given target magnitudes ``s``, it minimizes:

    J(x) = sum_m sum_n | |c_m[n]|^p - s_m[n]^p |^2

where c = filterbank(x, g, a) and p is a compression parameter
(default 2/3).  The gradient is computed via the chain rule using
the filterbank analysis/synthesis operators.

The optimization uses L-BFGS from ``torch.optim``.

References: Décorsière et al., 2015.
"""

from __future__ import annotations

from typing import Literal

import torch

from ..filterbanks._core import filterbank, ifilterbank
from ..filterbanks._frame import filterbankdual


def decolbfgs(
    s_list: list[torch.Tensor],
    g: list[dict],
    a,
    *,
    L: int | None = None,
    Ls: int | None = None,
    real: bool = False,
    maxit: int = 100,
    tol: float = 1e-6,
    p: float = 2.0 / 3.0,
    startphase: Literal["input", "zero", "rand"] = "zero",
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, int]:
    """Décorsière's L-BFGS phase retrieval for filterbanks (PyTorch).

    Minimizes a smooth objective based on compressed magnitude
    differences using L-BFGS.

    Parameters
    ----------
    s_list : list of M tensors — target magnitudes
    g : list of M filter dicts
    a : hop sizes
    L : DFT length
    Ls : output signal length
    real : use real (single-sided) synthesis
    maxit : maximum L-BFGS iterations
    tol : convergence tolerance
    p : compression parameter for the objective (default: 2/3)
    startphase : ``'input'``, ``'zero'``, or ``'rand'``

    Returns
    -------
    c : list of M complex tensors — coefficients with reconstructed phase
    f : reconstructed signal tensor
    relres : tensor of residuals at each iteration
    niter : number of iterations performed
    """
    # Determine device and dtype from inputs
    device = s_list[0].device if isinstance(s_list[0], torch.Tensor) else torch.device("cpu")
    dtype = torch.float64

    M = len(g)
    s_abs = [torch.abs(s.to(dtype=dtype, device=device)).flatten() for s in s_list]

    # Setup-time: compute dual window (numpy)
    from ...numpy.filterbanks._utils import normalise_a

    a_norm = normalise_a(a, M)

    N = [len(s) for s in s_abs]
    if L is None:
        afrac = a_norm[:, 0] / a_norm[:, 1]
        L = int(round(N[0] * afrac[0]))

    gd = filterbankdual(g, a_norm, L, real=real)

    # Precompute target magnitudes raised to power p
    s_p = [torch.pow(s + torch.finfo(dtype).tiny, p) for s in s_abs]

    s_flat = torch.cat(s_abs)
    norm_s = torch.linalg.norm(s_flat).item()
    if norm_s == 0:
        norm_s = 1.0

    # Initial signal estimate
    if startphase == "zero":
        c0 = [s.clone().to(dtype=torch.complex128) for s in s_abs]
    elif startphase == "rand":
        c0 = [
            s * torch.exp(2j * torch.pi * torch.rand(len(s), device=device, dtype=dtype))
            for s in s_abs
        ]
    else:  # 'input'
        c0 = [s.to(dtype=torch.complex128, device=device).flatten().clone() for s in s_list]

    # Get initial signal via synthesis
    x0 = ifilterbank(c0, gd, a_norm, Ls=L, real=real)
    if real and x0.is_complex():
        x0 = x0.real
    if real:
        x0_flat = x0.flatten()
    else:
        x0_flat = torch.cat([x0.real.flatten(), x0.imag.flatten()])

    relres_list = []

    # Flatten x as optimization variable (requires_grad=True)
    x_var = x0_flat.clone().to(dtype=dtype, device=device).requires_grad_(True)

    # L-BFGS optimizer
    optimizer = torch.optim.LBFGS(
        [x_var],
        lr=1.0,
        max_iter=maxit,
        max_eval=maxit * 20,
        tolerance_grad=tol,
        tolerance_change=tol * tol,
        line_search_fn="strong_wolfe",
    )

    def _closure():
        """Closure function for L-BFGS optimizer."""
        optimizer.zero_grad()

        # Reconstruct x from flattened variable
        if real:
            x = x_var
        else:
            n = len(x_var) // 2
            x = x_var[:n] + 1j * x_var[n:]

        # Analysis
        c = filterbank(x, g, a_norm, L=L)

        # Objective: sum |  |c_m[n]|^p - s_m[n]^p |^2
        obj = torch.tensor(0.0, dtype=dtype, device=device, requires_grad=True)
        for m in range(M):
            cm = c[m].flatten()
            cm_abs = torch.abs(cm)
            cm_p = torch.pow(cm_abs + torch.finfo(dtype).tiny, p)
            diff = cm_p - s_p[m]
            obj = obj + torch.sum(diff**2)

        # Residual for tracking
        with torch.no_grad():
            c_abs_flat = torch.cat([torch.abs(cm.flatten()) for cm in c])
            res = float(torch.linalg.norm(c_abs_flat - s_flat).item() / norm_s)
            relres_list.append(res)

        # Compute gradient
        obj.backward()

        return obj

    # Run L-BFGS
    try:  # noqa: SIM105
        optimizer.step(_closure)
    except Exception:
        # If L-BFGS fails, continue with last state
        pass

    niter = len(relres_list)

    # Recover signal and coefficients
    with torch.no_grad():
        if real:
            f = x_var.clone()
        else:
            n = len(x_var) // 2
            f = x_var[:n] + 1j * x_var[n:]

        c = filterbank(f, g, a_norm, L=L)

        # Enforce magnitude constraint on final output
        for m in range(M):
            cm = c[m].flatten()
            phase = torch.angle(cm)
            c[m] = s_abs[m] * torch.exp(1j * phase)

        # Final synthesis
        f = ifilterbank(c, gd, a_norm, Ls=Ls or L, real=real)
        if real and f.is_complex():
            f = f.real

    relres = (
        torch.tensor(relres_list, dtype=dtype, device=device)
        if relres_list
        else torch.tensor([0.0], dtype=dtype, device=device)
    )

    return c, f, relres, niter
