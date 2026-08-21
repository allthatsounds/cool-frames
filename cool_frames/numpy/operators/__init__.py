"""numpy.operators – frame multipliers and TF-domain operators (LTFAT-equivalent).

The central object is the frame multiplier: analyse, multiply by a TF symbol,
synthesise. See MATH_REFERENCE.md §15a for the mathematical background.

A multiplier is the natural way to express any linear time-frequency edit —
masking, equalisation, denoising — with an explicit, inspectable symbol, and
``framemulinv`` / ``framemulappr`` let you invert or best-approximate one.
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
