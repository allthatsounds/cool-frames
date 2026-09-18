"""cool_frames.numpy – NumPy reference backend (LTFAT-equivalent).

The invertible filterbank core: filter design (``filters``), analysis/
synthesis and frame theory (``filterbanks``), the discrete Gabor transform
(``gabor``), frame multipliers (``operators``), phase gradients and retrieval (``phase``), plus
``diagnostics`` (filterbank inspection) and ``sigproc`` (coefficient-domain
sparsity primitives).
"""
from . import (
    core,
    diagnostics,
    filterbanks,
    filters,
    gabor,
    operators,
    phase,
    sigproc,
)

__all__ = [
    "core",
    "diagnostics",
    "filters",
    "filterbanks",
    "gabor",
    "operators",
    "phase",
    "sigproc",
]
