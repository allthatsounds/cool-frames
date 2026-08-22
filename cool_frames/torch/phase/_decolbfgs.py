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

from .._dtypes import resolve
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
    seed: int | None = None,
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
    seed : int, optional
        Seed for ``startphase='rand'``.  ``None`` (the default) draws from
        fresh entropy, so repeated calls differ; pass an integer to make a
        random start reproducible.

    Returns
    -------
    c : list of M complex tensors — coefficients with reconstructed phase
    f : reconstructed signal tensor
    relres : tensor of residuals at each iteration
    niter : number of iterations performed
    """
    # Determine device and dtype from inputs
    device = s_list[0].device if isinstance(s_list[0], torch.Tensor) else torch.device("cpu")
    # The caller's dtype wins; see cool_frames/torch/_dtypes.py.
    dtype, cdtype = resolve(*(s_list if isinstance(s_list, list) else [s_list]))

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

    # Precompute target magnitudes raised to power p.
    #
    # Detached on purpose: these feed the *inner* L-BFGS objective, whose
    # closure calls `.backward()` once per line-search probe.  If they stayed
    # attached to the caller's graph, the second probe would fail with "Trying
    # to backward through the graph a second time".  The caller's gradient path
    # runs through `s_abs` in the final magnitude projection instead, which is
    # both correct and cheap.
    s_det = [s.detach() for s in s_abs]
    s_p = [torch.pow(s + torch.finfo(dtype).tiny, p) for s in s_det]

    s_flat = torch.cat(s_det)
    norm_s = torch.linalg.norm(s_flat).item()
    if norm_s == 0:
        norm_s = 1.0

    # Generator for `startphase='rand'`.  `None` means torch's global RNG,
    # i.e. unseeded; an explicit `seed` makes a random start reproducible.
    _rng = None
    if seed is not None:
        _rng = torch.Generator(device=device)
        _rng.manual_seed(int(seed))

    # Initial signal estimate.  Detached for the same reason as `s_p` above:
    # this only seeds the inner optimisation.
    if startphase == "zero":
        c0 = [s.clone().to(dtype=cdtype) for s in s_det]
    elif startphase == "rand":
        c0 = [
            s
            * torch.exp(
                2j * torch.pi * torch.rand(len(s), generator=_rng, device=device, dtype=dtype)
            )
            for s in s_det
        ]
    else:  # 'input'
        c0 = [s.detach().to(dtype=cdtype, device=device).flatten().clone() for s in s_list]

    # Get initial signal via synthesis
    x0 = ifilterbank(c0, gd, a_norm, Ls=L, real=real)
    if real and x0.is_complex():
        x0 = x0.real
    if real:
        x0_flat = x0.flatten()
    else:
        x0_flat = torch.cat([x0.real.flatten(), x0.imag.flatten()])

    relres_list = []

    # Flatten x as optimization variable (requires_grad=True).
    #
    # `.detach()` before `.clone()` is required, not cosmetic: when `s_list`
    # itself requires grad, `x0_flat` is a non-leaf node of the caller's graph,
    # and `torch.optim.LBFGS` rejects a non-leaf tensor outright ("can't
    # optimize a non-leaf Tensor").  Detaching starts a fresh graph for the
    # inner optimisation, which is what an inner solve should do anyway — the
    # returned coefficients are then a function of `s` through the magnitude
    # projection below, not through the L-BFGS trajectory.
    x_var = x0_flat.detach().clone().to(dtype=dtype, device=device).requires_grad_(True)

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

    # `relres_list` has one entry per closure evaluation, and L-BFGS calls the
    # closure several times per iteration during its line search.  Report true
    # iterations so that `niter` means the same thing as in the NumPy backend.
    niter = int(optimizer.state.get(x_var, {}).get("n_iter", len(relres_list)))

    # Recover the optimised signal and read off its phase.  This part is
    # deliberately outside the autograd graph: backpropagating through an
    # L-BFGS trajectory is neither meaningful nor affordable.
    with torch.no_grad():
        if real:
            f_opt = x_var.detach().clone()
        else:
            n = len(x_var) // 2
            f_opt = x_var[:n] + 1j * x_var[n:]

        phases = [torch.angle(cm.flatten()) for cm in filterbank(f_opt, g, a_norm, L=L)]

    # The magnitude constraint *is* differentiable, and it is where the caller's
    # `s` enters the result — so build the output here, outside `no_grad`.
    # Without this the returned coefficients have no grad_fn at all and any
    # loss built on them fails with "element 0 of tensors does not require
    # grad", which defeats the point of a torch backend.
    c = [s_abs[m] * torch.exp(1j * phases[m]) for m in range(M)]

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
