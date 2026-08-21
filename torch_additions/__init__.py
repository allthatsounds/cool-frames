"""torch_additions — parked torch-backend extras beyond the cool_frames core.

Holds torch functionality that is
intentionally kept out of the shipped ``cool_frames`` package so the core stays focused
on the LTFAT-equivalent transform surface, but preserved in-repo (and out of the
``cool_frames`` wheel) so it can be re-added easily.

Currently parked here (2026-06-21):
  - ``torch_additions.operators`` — ridge-guided frame multipliers
    (``ridges_to_symbol``, ``fit_ridge_multiplier``, ``denoise_by_ridges``).
  - ``torch_additions.recipes`` — ``audio_denoising`` (``denoise``) and
    ``magnitude_to_audio`` (``reconstruct``), application recipes that sit
    above the core transform surface.

Depends on ``cool_frames`` (specifically ``cool_frames.torch``). To re-add any piece to the
core, move the module back under ``cool_frames/torch/`` and restore the export.
"""

__all__ = ["operators", "phase", "recipes"]
