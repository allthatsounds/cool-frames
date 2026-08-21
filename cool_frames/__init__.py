"""
cool_frames — Computational Listening Algorithms
==========================================

Public API
----------
The recommended import path is::

    from cool_frames.filters import audfilters, cqtfilters
    from cool_frames.filterbanks import filterbank, ifilterbank
    from cool_frames.operators import framemul, framemulinv
    from cool_frames.phase import filterbankphasegrad, gla, rtisila

Module overview
---------------
``core``        – Low-level FFT kernels and math utilities.
``filters``     – Filter design: auditory scales, windows, wavelets.
``filterbanks`` – Analysis, synthesis, frame theory, and signal processing.
``operators``   – Frame multipliers and TF-domain operators.
``phase``       – Phase gradients, reconstruction, and retrieval.
``diagnostics`` – Filterbank inspection and recommendation utilities.
``sigproc``     – Coefficient-domain sparsity primitives.

Backends
--------
``cool_frames.numpy``  – NumPy reference implementation.
``cool_frames.torch``  – PyTorch backend (GPU/autodiff).
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
from . import numpy as _numpy_backend

__version__ = "0.1.0"
__all__ = [
    "core", "diagnostics", "filters", "filterbanks",
    "operators", "phase", "sigproc",
    "__version__",
]
