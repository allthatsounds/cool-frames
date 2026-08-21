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
) -> tuple[list[torch.Tensor], torch.Tensor]:
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
    >>> g = audfilters(16000)
    >>> c, mask = filterbankconstphase(x, g, a=64)
    >>> len(c) == len(g)
    True
    >>> c[0].dtype
    torch.complex128
    """
    # Convert signal to numpy
    if isinstance(f, torch.Tensor):
        device = f.device
        f_np = f.detach().cpu().numpy()
    else:
        device = torch.device("cpu")
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
    )

    # The numpy function returns (c_list, mask) or just c_list
    if isinstance(result, tuple):
        c_np, mask_np = result
    else:
        c_np = result
        mask_np = np.ones(1)

    c_torch = [torch.tensor(cm, dtype=torch.complex128, device=device) for cm in c_np]
    mask_torch = (
        torch.tensor(mask_np, device=device)
        if isinstance(mask_np, np.ndarray)
        else torch.tensor([1])
    )

    return c_torch, mask_torch
