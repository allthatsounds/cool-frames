"""
torch/phase/_constphase.py
===========================
Heap-based phase reconstruction (PGHI) — wraps the NumPy implementation.

This is the non-differentiable "reference" version.  For a differentiable
pipeline, use :func:`constphase_nonuniform` from ``_diff_constphase.py``.
"""

from __future__ import annotations

import numpy as np
import torch

from ...numpy.phase._constphase import filterbankconstphase as _np_filterbankconstphase
from .._dtypes import resolve
from ..filterbanks._frame import _torch_filters_to_numpy


def filterbankconstphase(
    f: torch.Tensor | np.ndarray,
    g: list[dict],
    a=None,
    L: int | None = None,
    fc: np.ndarray | None = None,
    tol: float = 1e-6,
    tgrad: list | None = None,
    fgrad: list | None = None,
    sqtfr: np.ndarray | None = None,
    fs: float | None = None,
    rng: object | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Phase reconstruction using heap-based PGHI (wraps NumPy).

    This delegates to the NumPy heap-based implementation, so it is
    **not** differentiable.  Use :func:`constphase_nonuniform` for a
    differentiable alternative.

    Parameters
    ----------
    f : signal tensor or array
    g : list of filter dicts
    a : hop sizes
    L : DFT length
    fc : centre frequencies (optional)
    tol : magnitude threshold
    tgrad, fgrad : pre-computed phase gradients (optional)

    Returns
    -------
    c : list of M complex tensors (with reconstructed phase)
    mask : tensor of booleans (which coefficients were above threshold)

    Examples
    --------
    >>> import torch
    >>> from cool_frames.torch.phase import filterbankconstphase
    >>> from cool_frames.torch.filters import audfilters
    >>> x = torch.randn(8000)
    >>> g, a, fc, L, _ = audfilters(16000, 8000)
    >>> c, mask = filterbankconstphase(x, g, a, L=L)
    >>> len(c) == len(g)
    True
    >>> c[0].dtype
    torch.complex64
    """
    # Convert signal to numpy.  The caller's dtype decides the output width;
    # the numpy core always computes in double.
    if isinstance(f, torch.Tensor):
        device = f.device
        _dtype, cdtype = resolve(f)
        f_np = f.detach().cpu().numpy()
    else:
        device = torch.device("cpu")
        cdtype = torch.complex128
        f_np = np.asarray(f)

    # Convert filters
    g_np = _torch_filters_to_numpy(g, L or 0)

    # Convert gradients if provided
    tgrad_np = None
    fgrad_np = None
    if tgrad is not None:
        tgrad_np = [
            t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
            for t in tgrad
        ]
    if fgrad is not None:
        fgrad_np = [
            f_.detach().cpu().numpy() if isinstance(f_, torch.Tensor) else np.asarray(f_)
            for f_ in fgrad
        ]

    result = _np_filterbankconstphase(
        f_np,
        g_np,
        a=a,
        L=L,
        fc=fc,
        tol=tol,
        tgrad=tgrad_np,
        fgrad=fgrad_np,
        sqtfr=sqtfr,
        fs=fs,
        rng=rng,
    )

    c_np, mask_np = result

    c_torch = [torch.as_tensor(cm, dtype=cdtype, device=device) for cm in c_np]
    # Per-channel boolean masks, matching the coefficient structure exactly —
    # the same shape the NumPy backend returns.  This used to collapse to a
    # single tensor (and to `torch.tensor([1])` on the branch where NumPy
    # returned a bare list), so the two backends disagreed about the *shape* of
    # the second return value as well as whether there was one.
    mask_torch = [torch.as_tensor(np.asarray(mm), device=device) for mm in mask_np]

    return c_torch, mask_torch
