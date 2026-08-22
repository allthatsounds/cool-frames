"""
torch/phase/_legla.py
=====================
Le Roux's Griffin-Lim Algorithm (LEGLA) for filterbanks (PyTorch).

Port of ``numpy/phase/_legla.py``.  The truncated projection kernel is built
once with NumPy (see ``cool_frames/numpy/phase/_leglakernel.py``, which also
documents and validates the construction) and applied here as a torch sparse
matrix product, so the two backends agree numerically and the iteration stays
differentiable.

Supported:
  - ``'trunc'``    : truncate the kernel at ``relthr``
  - ``'modtrunc'`` : additionally zero the self-term ``k_{m,m}[0]``
  - ``'legla'``    : plain iteration
  - ``'flegla'``   : fast variant with momentum

``relthr=0`` keeps the whole kernel and reproduces GLA's exact projection.

Before v0.1.1 this module ignored ``relthr`` and ``variant`` and ran a full
projection, making ``legla`` a bit-identical alias for ``gla``.
"""

from __future__ import annotations

from typing import Literal

import torch

from .._dtypes import resolve
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
    seed: int | None = None,
    variant: Literal["trunc", "modtrunc"] = "trunc",
    relthr: float = 1e-3,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor, int]:
    """Le Roux's Griffin-Lim for filterbanks (PyTorch).

    Efficient phase reconstruction with audio declipping [legla-siedenburg]_
    and frame operator theory [legla-perraudin]_.

    The LEGLA update replaces the full analysis-synthesis cycle with a
    truncated convolution against a precomputed projection kernel.

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
    seed : int, optional
        Seed for ``startphase='rand'``.  ``None`` (the default) draws from
        fresh entropy, so repeated calls differ; pass an integer to make a
        random start reproducible.
    variant : ``'trunc'`` (truncate at ``relthr``) or ``'modtrunc'``
        (additionally zero the self-term ``k_{m,m}[0]``).
    relthr : float
        Relative threshold for kernel truncation; entries below
        ``relthr * max|k|`` are discarded.  ``0.0`` reproduces GLA's exact
        projection.  See the NumPy implementation for the cost trade-off.

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
    # The caller's dtype wins; see cool_frames/torch/_dtypes.py.
    dtype, cdtype = resolve(*(s_list if isinstance(s_list, list) else [s_list]))

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
    # Generator for `startphase='rand'`.  `None` means torch's global RNG,
    # i.e. unseeded; an explicit `seed` makes a random start reproducible.
    _rng = None
    if seed is not None:
        _rng = torch.Generator(device=device)
        _rng.manual_seed(int(seed))

    if startphase == "zero":
        c = [s.clone().to(dtype=cdtype) for s in s_abs]
    elif startphase == "rand":
        c = [
            s
            * torch.exp(
                2j * torch.pi * torch.rand(len(s), generator=_rng, dtype=dtype, device=device)
            ).to(dtype=cdtype)
            for s in s_abs
        ]
    else:
        c = [torch.as_tensor(s, dtype=cdtype, device=device).reshape(-1).clone() for s in s_list]

    s_flat = torch.cat([s.reshape(-1) for s in s_abs])
    norm_s = torch.linalg.norm(s_flat)
    if norm_s == 0:
        norm_s = torch.tensor(1.0, dtype=dtype, device=device)

    # Truncated projection kernel — see cool_frames/numpy/phase/_leglakernel.py
    # for the construction and its validation.  The kernel is a setup-time
    # object built with NumPy (as the dual windows already are); only its
    # application runs in torch, as a sparse matrix product, which keeps both
    # numerical parity with the NumPy backend and differentiability.
    import numpy as np

    hops = np.round(a_norm[:, 0] / a_norm[:, 1]).astype(int)
    use_kernel = relthr >= 0 and np.allclose(a_norm[:, 0] / a_norm[:, 1], hops)

    if use_kernel:
        from ...numpy.phase._leglakernel import LeglaKernel

        _kernel = LeglaKernel(
            g,
            gd,
            hops,
            N,
            L,
            real=real,
            relthr=relthr,
            zero_self_term=(variant == "modtrunc"),
        )
        offs = _kernel.offs

        def _to_sparse(mat):
            coo = mat.tocoo()
            idx = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)).to(device)
            val = torch.from_numpy(coo.data.astype(np.complex128)).to(device)
            # check_invariants=False: the indices come from scipy's COO format,
            # which already guarantees them, and the check is not free.
            return torch.sparse_coo_tensor(
                idx,
                val,
                size=mat.shape,
                dtype=cdtype,
                device=device,
                check_invariants=False,
            ).coalesce()

        P_t = _to_sparse(_kernel._P)
        Q_t = _to_sparse(_kernel._Q) if _kernel._Q is not None else None

        def project(c_in):
            x = torch.cat([cm.reshape(-1) for cm in c_in]).to(cdtype)
            y = torch.sparse.mm(P_t, x.unsqueeze(1)).squeeze(1)
            if Q_t is not None:
                y = y + torch.sparse.mm(Q_t, torch.conj(x).unsqueeze(1)).squeeze(1)
            return [y[offs[m] : offs[m + 1]] for m in range(M)]

    else:
        # Fractional hop sizes have no integer lag grid; fall back to the exact
        # synthesise-then-reanalyse projection.
        def project(c_in):
            f = ifilterbank(c_in, gd, a_norm, Ls=L, real=real)
            return filterbank(torch.real(f) if real else f, g, a_norm, L=L)

    relres_list = []

    if method == "legla":
        for _it in range(maxit):
            # Project
            c_proj = project(c)

            # Residual
            # `abs` first, *then* cast to the real dtype.  Casting first
            # discards the imaginary part (torch warns about it) and made the
            # reported residual |Re(c)| instead of |c| — 0.185 where the NumPy
            # backend reported 0.070, which also broke the `tol` comparison.
            c_proj_abs = torch.cat(
                [
                    torch.abs(torch.as_tensor(cp, device=device)).to(dtype).reshape(-1)
                    for cp in c_proj
                ]
            )
            res = torch.linalg.norm(c_proj_abs - s_flat) / norm_s
            relres_list.append(res.detach().clone())

            # Phase update with magnitude constraint
            for m in range(M):
                # `variant` is applied when the kernel is built, not here — the
                # old branch computed angle(c + (proj - c)) == angle(proj).
                cp = torch.as_tensor(c_proj[m], dtype=cdtype, device=device).reshape(-1)
                c[m] = s_abs[m].to(dtype=cdtype) * torch.exp(1j * torch.angle(cp))

            if res < tol:
                break

    elif method == "flegla":
        told = [ci.clone() for ci in c]
        for _it in range(maxit):
            c_proj = project(c)

            # `abs` first, *then* cast to the real dtype.  Casting first
            # discards the imaginary part (torch warns about it) and made the
            # reported residual |Re(c)| instead of |c| — 0.185 where the NumPy
            # backend reported 0.070, which also broke the `tol` comparison.
            c_proj_abs = torch.cat(
                [
                    torch.abs(torch.as_tensor(cp, device=device)).to(dtype).reshape(-1)
                    for cp in c_proj
                ]
            )
            res = torch.linalg.norm(c_proj_abs - s_flat) / norm_s
            relres_list.append(res.detach().clone())

            tnew = []
            for m in range(M):
                cp = torch.as_tensor(c_proj[m], dtype=cdtype, device=device).reshape(-1)
                tnew.append(s_abs[m].to(dtype=cdtype) * torch.exp(1j * torch.angle(cp)))

            c = [tnew[m] + alpha * (tnew[m] - told[m]) for m in range(M)]
            told = tnew

            if res < tol:
                break

        # See _gla.py: project the extrapolated point back onto |c| == s.
        c = [s_abs[m].to(c[m].dtype) * torch.exp(1j * torch.angle(c[m])) for m in range(M)]

    f = ifilterbank(c, gd, a_norm, Ls=Ls or L, real=real)
    if real:
        f = torch.real(f)

    relres_tensor = (
        torch.stack(relres_list) if relres_list else torch.tensor([], dtype=dtype, device=device)
    )

    return c, f, relres_tensor, len(relres_list)
