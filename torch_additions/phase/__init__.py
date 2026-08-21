"""torch_additions.phase — ridge-extraction primitives backing the ridge multipliers."""

from ._ridge import Ridge, extract_ridges, segment_ridges

__all__ = ["Ridge", "extract_ridges", "segment_ridges"]
