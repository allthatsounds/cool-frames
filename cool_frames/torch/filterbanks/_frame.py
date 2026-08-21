"""
torch/filterbanks/_frame.py
============================
Frame theory wrappers — delegates to NumPy for the matrix solves
(setup-time), returns torch-compatible filter dicts.

These are all setup-time operations (called once before the main loop),
so the numpy delegation has negligible cost.  GPU tensors are moved to
CPU automatically and the results are placed back on the original device.
"""

from __future__ import annotations

import numpy as np
import torch

from ...numpy.filterbanks._core import ifilterbankiter as _np_ifilterbankiter
from ...numpy.filterbanks._frame import (
    filterbankbounds as _np_filterbankbounds,
)
from ...numpy.filterbanks._frame import (
    filterbankdual as _np_filterbankdual,
)
from ...numpy.filterbanks._frame import (
    filterbankfreqz as _np_filterbankfreqz,
)
from ...numpy.filterbanks._frame import (
    filterbankresponse as _np_filterbankresponse,
)
from ...numpy.filterbanks._frame import (
    filterbankscale as _np_filterbankscale,
)
from ...numpy.filterbanks._frame import (
    filterbanktight as _np_filterbanktight,
)
from ..filters._wrappers import numpy_filters_to_torch

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _torch_filters_to_numpy(g_torch: list[dict], L: int) -> list[dict]:
    """Convert torch filter dicts back to numpy for the frame solvers."""
    g_np = []
    for gm in g_torch:
        d = dict(gm)
        H = d.get("H")
        if isinstance(H, torch.Tensor):
            d["H"] = H.detach().cpu().numpy()
        g_np.append(d)
    return g_np


def _infer_device(g: list[dict]) -> torch.device:
    """Infer the torch device from a list of filter dicts."""
    for gm in g:
        H = gm.get("H")
        if isinstance(H, torch.Tensor):
            return H.device
    return torch.device("cpu")


def _frame_solver(
    np_fn,
    g: list[dict],
    a,
    L: int,
    *,
    real: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex128,
) -> list[dict]:
    """Generic wrapper: numpy frame solver → torch filter dicts."""
    if device is None:
        device = _infer_device(g)
    g_np = _torch_filters_to_numpy(g, L)
    result_np = np_fn(g_np, a, L, real=real)
    return numpy_filters_to_torch(result_np, L, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# frame response & bounds
# ---------------------------------------------------------------------------


def filterbankresponse(
    g: list[dict],
    a,
    L: int,
    real: bool = False,
) -> torch.Tensor:
    """Compute the frame response (diagonal of the frame operator).

    Returns a real-valued tensor of shape ``(L,)``.

    Examples
    --------
    >>> from cool_frames.torch.filters import audfilters
    >>> g, a, _, L, _ = audfilters(16000, 8000)
    >>> resp = filterbankresponse(g, a, L)
    >>> resp.shape
    torch.Size([10368])
    """
    g_np = _torch_filters_to_numpy(g, L)
    resp_np = _np_filterbankresponse(g_np, a, L, real=real)
    return torch.tensor(resp_np, dtype=torch.float64, device=_infer_device(g))


def filterbankbounds(
    g: list[dict],
    a,
    L: int,
    real: bool = True,
) -> tuple[float, float]:
    """Compute the frame bounds *(A, B)* of a filterbank.

    Delegates to the NumPy implementation.  ``real`` defaults to ``True``
    (single-sided real-audio frames), matching the numpy backend.

    Examples
    --------
    >>> from cool_frames.torch.filters import audfilters
    >>> g, a, _, L, _ = audfilters(16000, 8000)
    >>> A, B = filterbankbounds(g, a, L)
    >>> A > 0 and B > A  # A ≤ I ≤ B for frame property
    True
    """
    g_np = _torch_filters_to_numpy(g, L)
    A, B = _np_filterbankbounds(g_np, a, L, real=real)
    return float(A), float(B)


def filterbankfreqz(
    g: list[dict],
    a,
    L: int,
) -> torch.Tensor:
    """Compute the frequency responses of all filters, stacked as rows.

    Returns a complex tensor of shape ``(M, L)``.

    Examples
    --------
    >>> from cool_frames.torch.filters import audfilters
    >>> g, a, _, L, _ = audfilters(16000, 8000)
    >>> H = filterbankfreqz(g, a, L)
    >>> H.shape
    torch.Size([10368, 35])
    """
    g_np = _torch_filters_to_numpy(g, L)
    H_np = _np_filterbankfreqz(g_np, a, L)
    return torch.tensor(H_np, dtype=torch.complex128, device=_infer_device(g))


# ---------------------------------------------------------------------------
# dual / tight frames
# ---------------------------------------------------------------------------


def filterbankdual(
    g: list[dict],
    a,
    L: int,
    *,
    real: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex128,
) -> list[dict]:
    """Compute the canonical dual filterbank.

    ``real`` defaults to ``True`` (single-sided real-audio frames),
    matching the numpy backend.  Pass ``real=False`` for complex /
    two-sided frames.

    Examples
    --------
    >>> from cool_frames.torch.filters import audfilters
    >>> g, a, _, L, _ = audfilters(16000, 8000)
    >>> g_dual = filterbankdual(g, a, L)
    >>> len(g_dual)
    35
    """
    return _frame_solver(_np_filterbankdual, g, a, L, real=real, device=device, dtype=dtype)


def filterbanktight(
    g: list[dict],
    a,
    L: int,
    *,
    real: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex128,
) -> list[dict]:
    """Compute the canonical tight filterbank.

    ``real`` defaults to ``True`` (single-sided real-audio frames),
    matching the numpy backend.  Pass ``real=False`` for complex /
    two-sided frames.

    Examples
    --------
    >>> from cool_frames.torch.filters import audfilters
    >>> g, a, _, L, _ = audfilters(16000, 8000)
    >>> g_tight = filterbanktight(g, a, L)
    >>> len(g_tight)
    35
    """
    return _frame_solver(_np_filterbanktight, g, a, L, real=real, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# scaling
# ---------------------------------------------------------------------------


def filterbankscale(
    g: list[dict],
    s,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.complex128,
    L: int | None = None,
) -> list[dict]:
    """Scale filterbank channels by per-channel factors *s*.

    Parameters
    ----------
    g : list of filter dicts
    s : array-like of length M, or scalar
    device, dtype : optional overrides for the output tensors
    L : signal length (needed to materialise filters for conversion)

    Examples
    --------
    >>> from cool_frames.torch.filters import audfilters
    >>> g, a, _, L, _ = audfilters(16000, 8000)
    >>> scales = [0.5] * len(g)
    >>> g_scaled = filterbankscale(g, scales, L=L)
    >>> len(g_scaled)
    35
    """
    if device is None:
        device = _infer_device(g)
    # Infer L from filter H lengths if not provided
    if L is None:
        for gm in g:
            H = gm.get("H")
            if isinstance(H, (np.ndarray, torch.Tensor)):
                L = len(H)
                break
        if L is None:
            raise ValueError("Cannot infer L — pass it explicitly")

    g_np = _torch_filters_to_numpy(g, L)
    gs_np = _np_filterbankscale(g_np, s)
    return numpy_filters_to_torch(gs_np, L, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# iterative synthesis
# ---------------------------------------------------------------------------


def ifilterbankiter(
    c: list[torch.Tensor],
    g: list[dict],
    a,
    Ls: int | None = None,
    tol: float = 1e-6,
    maxit: int = 100,
    alg: str = "cg",
    real: bool | None = None,
) -> tuple[torch.Tensor, float, int]:
    """Iterative filterbank synthesis via conjugate gradients.

    Delegates to the NumPy implementation and converts back to torch.

    Parameters
    ----------
    real : if True, use real-filterbank synthesis.  Defaults to ``None``,
        meaning derive it from the filters — see the NumPy ``ifilterbankiter``.
        The old ``False`` default reconstructed the flagship ``audfilters`` bank
        with 23 % error where the correct mode reaches 4.4e-16.  This is the
        same defect as ``filterbankiter``'s and the NumPy twin's, and it
        outlived both of those fixes by being in a fourth place nobody looked.

    Returns
    -------
    xr     : reconstructed signal tensor, in the coefficients' own real dtype
    relres : relative residual
    niter  : number of iterations used

    Examples
    --------
    >>> from cool_frames.torch.filters import audfilters
    >>> g, a, _, L, _ = audfilters(16000, 8000)
    >>> from cool_frames.torch.filterbanks import filterbank
    >>> c = filterbank(torch.randn(8000), g, a, L=L)
    >>> xr, relres, niter = ifilterbankiter(c, g, a, Ls=8000)
    >>> xr.shape
    torch.Size([8000])
    """
    c_np = [cm.detach().cpu().numpy() if isinstance(cm, torch.Tensor) else cm for cm in c]

    device = torch.device("cpu")
    for cm in c:
        if isinstance(cm, torch.Tensor):
            device = cm.device
            break

    g_np = _torch_filters_to_numpy(g, Ls or 0)

    result = _np_ifilterbankiter(c_np, g_np, a, Ls=Ls, tol=tol, maxit=maxit, alg=alg, real=real)
    xr_np, relres, niter = result[0], result[1], result[2]

    # Follow the coefficients' precision rather than forcing float64: the torch
    # backend is dtype-polymorphic everywhere else, and silently widening a
    # float32 pipeline here is the thing that polymorphism exists to avoid.
    out_dtype = torch.float64
    for cm in c:
        if isinstance(cm, torch.Tensor):
            out_dtype = (
                torch.float32 if cm.dtype in (torch.complex64, torch.float32) else torch.float64
            )
            break

    xr = torch.as_tensor(xr_np, dtype=out_dtype, device=device)
    return xr, relres, niter


# ---------------------------------------------------------------------------
# iterative analysis (CG on FF*)
# ---------------------------------------------------------------------------


def filterbankiter(
    f: torch.Tensor,
    g: list[dict],
    a,
    L: int | None = None,
    tol: float = 1e-6,
    maxit: int = 100,
    alg: str = "cg",
    real: bool | None = None,
) -> tuple[list[torch.Tensor], float, int]:
    """Iterative filterbank analysis via Conjugate Gradient.

    Solves ``FF* x = f`` then returns ``c = F x``, producing coefficients
    that give perfect reconstruction through the synthesis operator without
    needing the dual frame.

    All internal operations use torch tensors and are differentiable with
    respect to *f*.

    Parameters
    ----------
    f     : (Ls,) or (Ls, W) tensor
    g     : list of M filter dicts
    a     : hop sizes
    L     : DFT length (computed from Ls if omitted)
    tol   : relative residual tolerance
    maxit : maximum CG iterations
    alg   : ``'cg'`` or ``'pcg'``
    real  : if True, use real-filterbank synthesis.  Defaults to ``None``,
            meaning derive it from the filters — see the NumPy
            ``filterbankiter`` for why: the old ``False`` default diverged on
            the package's flagship single-sided bank (100 iterations to a
            relative residual of 58) and disagreed with every sibling in the
            family.  Kept in parity with the NumPy backend deliberately; a
            default that differs between backends is its own bug.

    Returns
    -------
    (c, relres, niter) : coefficients, relative residual, iteration count
    """
    from ..core._core import _normalise_a
    from ..core._core import filterbanklength as _filterbanklength
    from ._core import filterbank as _filterbank
    from ._core import ifilterbank as _ifilterbank

    mono = f.dim() == 1
    if mono:
        f = f.unsqueeze(1)
    Ls, _W = f.shape
    device = f.device

    M = len(g)
    a_norm = _normalise_a(a, M)

    if L is None:
        L = _filterbanklength(Ls, a_norm)

    if real is None:
        from ...numpy.filterbanks._core import filterbank_is_real as _is_real

        real = bool(_is_real(g, a_norm, int(L)))

    # Zero-pad
    if Ls < L:
        f_pad = torch.nn.functional.pad(f, (0, 0, 0, L - Ls))
    else:
        f_pad = f[:L]

    b = f_pad.to(torch.complex128).reshape(-1)
    norm_b = float(torch.linalg.norm(b))
    if norm_b == 0.0:
        c_out = _filterbank(torch.zeros(Ls, device=device), g, a, L)
        return c_out, 0.0, 0

    # Frame operator: A x = ifft( synth( ana( ifft(X) ) ) )
    #
    # The operator acts on the full length-L vector.  Until v0.1.1 this sliced
    # `x_vec[:Ls]` and let `filterbank` re-pad, which makes the map a
    # projection rather than F*F: with Ls < L the iteration diverged (relres
    # 399.7 after 60 iterations, against NumPy's 8.5e-11 in 17).  With Ls == L
    # the slice was a no-op, which is why it went unnoticed.
    def _apply_frame_op(x_vec: torch.Tensor) -> torch.Tensor:
        x_sig = x_vec
        if mono:
            c_tmp = _filterbank(x_sig.real, g, a, L)
        else:
            c_tmp = _filterbank(x_sig.real.unsqueeze(1), g, a, L)
        if real:
            out = 2.0 * _ifilterbank(c_tmp, g, a, Ls=L, real=False).real
        else:
            out = _ifilterbank(c_tmp, g, a, Ls=L, real=False)
        return out.reshape(-1).to(torch.complex128)

    # Preconditioner
    precond = None
    if alg == "pcg":
        try:
            resp = filterbankresponse(g, a, L, real=real)
            resp_safe = torch.where(resp.abs() < 1e-14, torch.ones_like(resp), resp)
            precond = (1.0 / resp_safe).to(torch.complex128)
        except Exception:
            precond = None

    # CG iteration
    x = torch.zeros(L, dtype=torch.complex128, device=device)
    r = b.clone()
    z = (precond * r) if precond is not None else r.clone()
    p = z.clone()
    rz = torch.vdot(r, z).real

    niter = 0
    relres_final = 1.0

    for k in range(maxit):
        Ap = _apply_frame_op(p)
        pAp = torch.vdot(p, Ap).real.item()
        if abs(pAp) < 1e-30:
            break
        alpha = rz.item() / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        relres_k = float(torch.linalg.norm(r)) / norm_b
        relres_final = relres_k
        niter = k + 1

        if relres_k < tol:
            break

        z_new = (precond * r) if precond is not None else r.clone()
        rz_new = torch.vdot(r, z_new).real
        if abs(rz.item()) < 1e-30:
            break
        beta = rz_new.item() / rz.item()
        p = z_new + beta * p
        rz = rz_new

    # Compute c = F x
    x_real = x.real[:Ls] if Ls <= L else x.real
    if mono:
        c_out = _filterbank(x_real, g, a, L)
    else:
        c_out = _filterbank(x_real.unsqueeze(1), g, a, L)
    return c_out, relres_final, niter


# ---------------------------------------------------------------------------
# filterbanklengthcoef – infer L from coefficient arrays
# ---------------------------------------------------------------------------


def filterbanklengthcoef(coef, a) -> int:
    """Determine the filterbank length *L* from coefficient tensors.

    Parameters
    ----------
    coef : list of M tensors (N_m, ...) or a single (N, M) tensor
    a    : hop sizes — scalar, (M,), or (M, 2) fractional

    Returns
    -------
    L : int

    Raises
    ------
    ValueError
        If the inferred *L* is inconsistent across channels.
    """
    import numpy as np

    if isinstance(coef, (list, tuple)):
        cl = np.array([c.shape[0] if isinstance(c, torch.Tensor) else len(c) for c in coef])
    else:
        if isinstance(coef, torch.Tensor):
            M_coef = coef.shape[1] if coef.dim() > 1 else 1
            cl = np.full(M_coef, coef.shape[0])
        else:
            coef = np.asarray(coef)
            M_coef = coef.shape[1] if coef.ndim > 1 else 1
            cl = np.full(M_coef, coef.shape[0])

    a = np.asarray(a)
    if a.ndim == 0:
        a = np.full(len(cl), int(a))
    if a.ndim == 1:
        a = a.ravel()
        if len(a) == 1:
            a = np.repeat(a, len(cl))
        L_arr = a * cl
    else:
        L_arr = a[:, 0] * cl / a[:, 1]

    L_arr = L_arr.astype(float)
    if np.ptp(L_arr) > 0.5:
        raise ValueError(
            "Invalid set of coefficients. The product of the number of "
            "coefficients and the channel time shift must be the same for "
            "all channels."
        )

    return int(round(L_arr[0]))
