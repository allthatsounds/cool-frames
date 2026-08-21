"""torch.diagnostics – filterbank inspection utilities (PyTorch).

Differentiable, torch-tensor spectrogram helpers, mirroring ``cool_frames.diagnostics``.
Promoted out of ``cool_frames.torch.recipes`` on 2026-06-21.

The signal-driven ``recommend_filterbank`` and ``center_freqs`` helpers are
design-time (NumPy) utilities and remain available through ``cool_frames.torch``'s
re-export of ``cool_frames.diagnostics``.
"""

from .spectrogram import filterbank_spectrogram, reassigned_spectrogram

__all__ = ["filterbank_spectrogram", "reassigned_spectrogram"]
