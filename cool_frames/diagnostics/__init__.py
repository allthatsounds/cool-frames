"""cool_frames.diagnostics – re-exports from the NumPy backend.

Utilities for inspecting filterbanks: signal-driven designer recommendation,
centre-frequency estimation, and (reassigned) spectrograms.
"""
from ..numpy.diagnostics import *  # noqa: F401,F403
from ..numpy.diagnostics import __all__  # noqa: F401

# Make `cool_frames.diagnostics.admissibility` resolve as a real submodule
# path, not just an attribute, so `from ... import x` works.
import sys as _sys  # noqa: E402

from ..numpy.diagnostics import admissibility  # noqa: F401,E402

_sys.modules[__name__ + ".admissibility"] = admissibility
