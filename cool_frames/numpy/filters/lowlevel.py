"""cool_frames.filters.lowlevel -- single-filter constructors.

These build ONE filter at a time and sit a tier *below* the high-level
designers (:func:`audfilters`, :func:`cqtfilters`, ...). Use them to assemble a
custom bank by hand or to make a one-off band-limited / FIR / frequency-domain /
warped filter:

* :func:`blfilter`      -- band-limited (frequency-domain) filter
* :func:`firfilter`     -- time-domain FIR filter
* :func:`freqfilter`    -- general frequency-domain filter
* :func:`warpedblfilter`-- band-limited filter on a warped frequency axis

They are grouped here (rather than in the flat ``cool_frames.filters`` namespace) so the
high-level designers a student reaches for first are not mixed with these
lower-level building blocks. Import as e.g. ``from cool_frames.filters.lowlevel import
blfilter``.
"""
from ._filters import blfilter, firfilter, freqfilter
from ._warpedfilters import warpedblfilter

__all__ = ["blfilter", "firfilter", "freqfilter", "warpedblfilter"]
