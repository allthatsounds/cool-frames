"""cool_frames.torch – PyTorch backend for GPU-accelerated, differentiable
filterbank analysis/synthesis.

Each function mirrors the NumPy API but:
  - Accepts and returns ``torch.Tensor`` instead of ``np.ndarray``
  - All operations are differentiable w.r.t. the input signal
  - GPU execution via standard PyTorch device placement
  - Filter design delegates to the NumPy backend (setup-time only)

Quick start::

    import torch
    from cool_frames.torch.filters import audfilters
    from cool_frames.torch.filterbanks import filterbank, filterbankdual, ifilterbank

    g, a, fc, L, info = audfilters(16000, 4096)  # numpy-based design → tensors
    gd = filterbankdual(g, a, L)                  # dual frame for synthesis
    x = torch.randn(L)
    c = filterbank(x, g, a)                       # differentiable analysis
    # ... manipulate c ...
    x_hat = ifilterbank(c, gd, a)                 # differentiable synthesis (real=True default)
"""

import importlib as _importlib

# Re-export everything from cool_frames (the LTFAT-equivalent numpy package) so users
# can `from cool_frames.torch import X` and get either the torch version (if it exists
# in this package) or the cool_frames version. Was `cool_frames.torch` before 2026-06-04.
# The numpy re-export comes FIRST so the torch submodules imported below take
# precedence over their numpy namesakes (core, filterbanks, ...).
try:
    from cool_frames import *  # noqa: F403
    from cool_frames import __all__ as _cool_frames_all
except ImportError:
    # cool_frames not installed — cool_frames.torch can still expose its own surface
    _cool_frames_all = []

# Torch backends must WIN over the numpy namesakes bound by ``from cool_frames import *``.
# A plain ``from . import filterbanks`` does NOT override them: Python sees the
# attribute already exists (bound by the star import) and keeps the numpy module,
# so ``cool_frames.torch.filterbanks`` would silently resolve to numpy. Force each torch
# submodule via importlib so attribute access (``cool_frames.torch.filterbanks``,
# ``.phase``, …) returns the torch implementation.
_torch_surface = [
    "core",
    "diagnostics",
    "filterbanks",
    "filters",
    "operators",
    "phase",
    "sigproc",
]
for _name in _torch_surface:
    globals()[_name] = _importlib.import_module(f".{_name}", __name__)

__all__ = sorted(set(_cool_frames_all) | set(_torch_surface))
