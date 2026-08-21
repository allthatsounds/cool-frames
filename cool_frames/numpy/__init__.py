"""cool_frames.numpy – NumPy reference backend (LTFAT-equivalent).

The invertible filterbank core: filter design (``filters``), analysis/
synthesis and frame theory (``filterbanks``), frame multipliers
(``operators``), phase gradients and retrieval (``phase``), plus
``diagnostics`` (filterbank inspection) and ``sigproc`` (coefficient-domain
sparsity primitives).
"""
from . import (
    core,
    diagnostics,
    filterbanks,
    filters,
    operators,
    phase,
    sigproc,
)

__all__ = [
    "core",
    "diagnostics",
    "filters",
    "filterbanks",
    "operators",
    "phase",
    "sigproc",
]
