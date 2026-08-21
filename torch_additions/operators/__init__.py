"""torch_additions.operators — ridge-guided frame multipliers (PyTorch).

Parked here out of ``cool_frames.torch.operators``: ridge-guided multipliers are a
research extension rather than part of the core transform surface.
The ``Ridge``/``extract_ridges``/``segment_ridges`` extraction primitives the
multiplier depends on travel alongside in ``torch_additions.phase._ridge``.
"""

from ._ridge_mul import (
    denoise_by_ridges,
    fit_ridge_multiplier,
    ridges_to_symbol,
)

__all__ = ["denoise_by_ridges", "fit_ridge_multiplier", "ridges_to_symbol"]
