"""The RTISI-LA family for the torch backend: runs the NumPy implementation.

RTISI-LA is a sequential, frame-by-frame algorithm with a nonlinear step per
frame; it carries no gradient (neither did the torch port it replaces, whose
phase updates were in-place assignments) and gains nothing from a GPU.  The
torch functions therefore convert their arguments, run
:mod:`cool_frames.numpy.phase`, and return tensors on the caller's device in
the caller's precision (see ``cool_frames/torch/_dtypes.py``).  The two
backends give the same result by construction.
"""

from __future__ import annotations

import numpy as np
import torch

from .._dtypes import resolve


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _filters_to_numpy(g):
    if isinstance(g, (list, tuple)) and g and isinstance(g[0], dict):
        return [{k: _to_numpy(v) for k, v in gm.items()} for gm in g]
    return _to_numpy(g)


def _device(*xs) -> torch.device:
    for x in xs:
        if isinstance(x, torch.Tensor):
            return x.device
        if isinstance(x, (list, tuple)):
            for y in x:
                if isinstance(y, torch.Tensor):
                    return y.device
                if isinstance(y, dict):
                    for v in y.values():
                        if isinstance(v, torch.Tensor):
                            return v.device
    return torch.device("cpu")


def delegate(np_fn, s_list, g, a, M, kwargs: dict):
    """Run ``np_fn(s, g, a, M, **kwargs)`` on NumPy copies of the arguments
    and return ``(c, f, relres, niter)`` with ``c`` and ``f`` as tensors."""
    tensors = (
        [s_list]
        if isinstance(s_list, torch.Tensor)
        else [s for s in s_list if isinstance(s, torch.Tensor)]
    )
    dtype, cdtype = resolve(*tensors) if tensors else (torch.float64, torch.complex128)
    device = _device(s_list, g)
    if isinstance(s_list, torch.Tensor):
        s_np = _to_numpy(s_list)
    elif isinstance(s_list, np.ndarray):
        s_np = s_list
    else:
        s_np = [np.asarray(_to_numpy(s)) for s in s_list]
    a_np = _to_numpy(a)
    c, f, relres, niter = np_fn(s_np, _filters_to_numpy(g), a_np, M, **kwargs)

    def to_t(x, complex_):
        arr = np.asarray(x)
        dt = cdtype if (complex_ or np.iscomplexobj(arr)) else dtype
        return torch.as_tensor(arr).to(device=device, dtype=dt)

    c_t = [to_t(cm, True) for cm in c] if isinstance(c, list) else to_t(c, True)
    return c_t, to_t(f, False), float(relres), int(niter)
