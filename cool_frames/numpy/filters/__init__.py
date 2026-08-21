"""numpy.filters – filter design: auditory scales, window functions, blfilter, audfilters, cqtfilters, wavelets, warped filters."""
from ._audscale import (
    GREENWOOD_DEFAULTS,
    audfiltbw,
    audspace,
    audtofreq,
    erbtofreq,
    freqtoaud,
    freqtoerb,
    gammatonefir,
)
from ._cqtfilters import cqtfilters
from . import lowlevel  # single-filter constructors: blfilter/firfilter/freqfilter/warpedblfilter
from ._design import audfilters, filterbanklength, partial_tighten
from ._filters import (
    biquadfilter,
    filter_freqresp,
)
from ._firwin import firwin, pgauss
from ._freqwin import freqwin
from ._freqwavelet import freqwavelet
from ._gabfilters import gabfilters
from ._greenwoodfilters import greenwoodfilters
from ._tfr import compute_tfr_from_filters
from ._warpedfilters_design import warpedfilters
from ._waveletfilters import waveletfilters


__all__ = [
    "audfiltbw",
    "audfilters",
    "audspace",
    "audtofreq",
    "biquadfilter",
    "cqtfilters",
    "erbtofreq",
    "filter_freqresp",
    "filterbanklength",
    "firwin",
    "lowlevel",
    "pgauss",
    "freqwin",
    "GREENWOOD_DEFAULTS",
    "greenwoodfilters",
    "freqtoaud",
    "freqtoerb",
    "freqwavelet",
    "compute_tfr_from_filters",
    "gabfilters",
    "partial_tighten",
    "gammatonefir",
    "warpedfilters",
    "waveletfilters",
]
