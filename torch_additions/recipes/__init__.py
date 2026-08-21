"""torch_additions.recipes — application recipes with no cool_frames-core home (PyTorch).

``audio_denoising`` (``denoise``) and ``magnitude_to_audio`` (``reconstruct``)
were promoted out of ``cool_frames.torch.recipes`` on 2026-06-21. Unlike the
spectrogram helpers (which moved to ``cool_frames.torch.diagnostics``), these
have no numpy-core counterpart, so they are parked here.
"""

from .audio_denoising import denoise
from .magnitude_to_audio import reconstruct

__all__ = ["denoise", "reconstruct"]
