"""torch.filters – Filter design wrappers (delegates to NumPy, returns tensors).

Filter design is a setup-time operation that doesn't need gradients.
These thin wrappers call the NumPy implementations and convert the
resulting filter descriptors so that the ``H`` arrays are
``torch.Tensor`` objects suitable for the torch analysis/synthesis
kernels.
"""

from ._biquad import biquad_response, biquadfilter, comp_biquad
from ._wrappers import (
    audfilters,
    cqtfilters,
    filter_freqresp,
    filterbanklength,
    firwin,
    gabfilters,
    numpy_filters_to_torch,
    warpedfilters,
    waveletfilters,
)

__all__ = [
    "audfilters",
    "biquad_response",
    "biquadfilter",
    "comp_biquad",
    "cqtfilters",
    "filter_freqresp",
    "filterbanklength",
    "firwin",
    "gabfilters",
    "numpy_filters_to_torch",
    "warpedfilters",
    "waveletfilters",
]
