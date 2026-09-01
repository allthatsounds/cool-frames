"""numpy.diagnostics – filterbank inspection utilities.

Signal-driven filterbank recommendation, centre-frequency estimation, and
spectrogram helpers (including phase-gradient reassignment). Frame-health
numbers A/B/κ come from ``filterbankbounds(..., return_kappa=True)`` directly.
"""
from ._center_freqs import center_freqs
from .recommend_filterbank import FilterbankRecommendation, recommend_filterbank
from .spectrogram import filterbank_spectrogram, reassigned_spectrogram

__all__ = [
    "center_freqs",
    "recommend_filterbank",
    "FilterbankRecommendation",
    "filterbank_spectrogram",
    "reassigned_spectrogram",
]

from .admissibility import (  # noqa: E402
    NotAFrameWarning,
    check_admissible,
    max_overlap_for_kappa,
    min_bins,
    min_channels,
    predict_admissible,
    ripple_curve,
)

__all__ = list(__all__) + [
    "NotAFrameWarning",
    "check_admissible",
    "predict_admissible",
    "ripple_curve",
    "max_overlap_for_kappa",
    "min_channels",
    "min_bins",
]
