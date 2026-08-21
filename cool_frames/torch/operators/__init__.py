"""torch.operators – frame multipliers and TF-domain operators (PyTorch).

Operators act on time-frequency representations built by the filterbanks
and filters modules.  The central object is the frame multiplier:
analyse, multiply by a TF symbol, synthesise.

Higher-level ridge-guided multipliers (``ridges_to_symbol``,
``fit_ridge_multiplier``, ``denoise_by_ridges``) were parked in
``torch_additions.operators`` on 2026-06-21 — mirroring the numpy ridge move out
of ``cool_frames.numpy.operators`` — to keep the shipped core focused. Re-add by moving
``_ridge_mul.py`` (and ``phase/_ridge.py``) back.

See MATH_REFERENCE.md §15a for the mathematical background.
"""

from ._framemul import (
    framemul,
    framemuladj,
    framemulappr,
    framemuleigs,
    framemulinv,
)

__all__ = [
    "framemul",
    "framemuladj",
    "framemulappr",
    "framemuleigs",
    "framemulinv",
]
