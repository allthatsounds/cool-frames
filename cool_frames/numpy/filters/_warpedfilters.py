"""
numpy/layer1/_warpedfilters.py
==============================
Warped filter constructors and helpers.

Ports of:
  - comp_warpedfoff.m     → comp_warpedfoff
  - comp_warpedfreqresponse.m → comp_warpedfreqresponse
  - warpedblfilter.m      → warpedblfilter

MATLAB originals:
  layer1/filter_prep/comp_warpedfoff.m
  layer1/filter_prep/comp_warpedfreqresponse.m
  layer1/filter_constructors/warpedblfilter.m
"""
from __future__ import annotations

import math

import numpy as np

from ..core._core import modcent
from ._firwin import firwin_eval

# ---------------------------------------------------------------------------
# Normalisation helper (matches MATLAB setnorm)
# ---------------------------------------------------------------------------

def _setnorm(g: np.ndarray, norm: str) -> np.ndarray:
    """Apply normalisation to a 1-D array."""
    norm = norm.lower() if norm else "null"
    if norm in ("null", "none", ""):
        return g
    elif norm in ("inf", "peak"):
        mx = np.max(np.abs(g))
        return g / mx if mx > 0 else g
    elif norm in ("1", "area"):
        s = np.sum(np.abs(g))
        return g / s if s > 0 else g
    elif norm in ("2", "energy"):
        s = math.sqrt(float(np.sum(np.abs(g)**2)))
        return g / s if s > 0 else g
    else:
        return g


# ---------------------------------------------------------------------------
# comp_warpedfoff – frequency offset for warped filters
# ---------------------------------------------------------------------------

def comp_warpedfoff(fc: float, bw: float, fs: float, L: int,
                    freqtoscale, scaletofreq,
                    do_symmetric: bool = False) -> int:
    """Compute the frequency offset (foff) for a warped filter.

    Parameters
    ----------
    fc : float – centre frequency in Hz
    bw : float – bandwidth in scale units
    fs : float – sampling rate
    L  : int   – transform length
    freqtoscale : callable(float) → float
    scaletofreq : callable(float) → float
    do_symmetric : bool

    Returns
    -------
    foff : int (0-based Python index)
    """
    fc_was_negative = fc < 0

    if fc_was_negative and do_symmetric:
        fc = -fc
        fcscale = freqtoscale(fc)
        # MATLAB: foff = -floor(scaletofreq(fcscale+.5*bw)/fs*L)+1
        #
        # The ``+1`` here is NOT a 1-based indexing artifact, and stripping it
        # as one (which is what this line used to do) put every mirrored channel
        # exactly one bin low.  It is part of the mirror arithmetic: the
        # response is reversed with ``H[::-1]``, which maps element i to
        # element (n-1-i), and that n-1 is what the +1 compensates for.
        #
        # Measured on a log-warped complex bank: with the +1 dropped, a
        # negative-fc channel and its positive twin differ by 0.18 (median,
        # relative L2, after mirroring) and a shift search finds a uniform
        # -1 bin offset on all 32 pairs; with it restored the two agree to
        # 0.0 exactly.  The positive branch's ``+1`` genuinely is 1-based and
        # is correctly dropped below.
        foff = -math.floor(scaletofreq(fcscale + 0.5 * bw) / fs * L) + 1
    else:
        fcscale = freqtoscale(fc)
        # MATLAB: foff = floor(scaletofreq(fcscale-.5*bw)/fs*L)+1
        # Convert to 0-based: the +1 in MATLAB makes it 1-based → 0-based is floor(...)
        foff = math.floor(scaletofreq(fcscale - 0.5 * bw) / fs * L)

    return foff  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# comp_warpedfreqresponse – transfer function of warped filter
# ---------------------------------------------------------------------------

def comp_warpedfreqresponse(winname: str, fc: float, bw: float,
                            fs: float, L: int,
                            freqtoscale, scaletofreq,
                            norm: str = "null",
                            do_symmetric: bool = False) -> np.ndarray:
    """Compute the transfer function of a warped filter.

    Parameters
    ----------
    winname : str – window type (passed to firwin_eval)
    fc : float – centre frequency in Hz
    bw : float – bandwidth in scale units
    fs : float – sampling rate
    L  : int   – transform length
    freqtoscale : callable(float or ndarray) → float or ndarray
    scaletofreq : callable(float or ndarray) → float or ndarray
    norm : str – normalisation flag
    do_symmetric : bool

    Returns
    -------
    H : 1-D ndarray – truncated transfer function
    """
    fc_was_negative = fc < 0

    if fc_was_negative and do_symmetric:
        fc = -fc

    fcscale = freqtoscale(fc)

    # Compute scale-domain bin positions for all DFT bins
    freqs = fs * np.arange(L) / L
    if not do_symmetric:
        freqs = modcent(freqs, fs)  # type: ignore[assignment]
    bins_lo = freqtoscale(freqs)  # type: ignore[assignment]

    # Mirror bins for high-frequency overlap
    nyquest2 = 2 * freqtoscale(fs / 2)
    bins_hi = nyquest2 + bins_lo

    # Rescale bins: firwin expects positions in [-0.5, 0.5] where
    # the window has width 1 centred at 0
    bins_lo = (bins_lo - fcscale) / bw
    bins_hi = (bins_hi - fcscale) / bw

    # Frequency offset (0-based)
    pos_lo = comp_warpedfoff(fc if not fc_was_negative else -fc, bw, fs, L,
                             freqtoscale, scaletofreq, do_symmetric)

    # Upper extent.
    #
    # Note for anyone tempted to "fix" the mirrored branch to use -A (the
    # mirror of the lower edge) so the width comes out as B - A: don't.  On that
    # branch the underlying response was built at +|fc|, so its energy sits at
    # bins [A, B]; ``pos_lo`` is -B, and the subsequent ``np.roll(H, -pos_lo)``
    # shifts that energy to [A + B, 2B].  The deliberately wide 2B window is
    # what captures it before ``H[::-1]`` brings it back to [0, B - A] for
    # placement at foff = -B.  Narrowing the window to B - A truncates the
    # energy away entirely (measured: mirror-symmetry error goes from 0.10 to
    # 1.00, i.e. the filter becomes empty).
    pos_hi = math.floor(scaletofreq(fcscale + 0.5 * bw) / fs * L)
    if pos_hi > L / 2:
        pos_hi = math.floor(scaletofreq(fcscale + 0.5 * bw - nyquest2) / fs * L)

    # Evaluate windows at warped positions
    win_lo = firwin_eval(winname, bins_lo)
    win_hi = firwin_eval(winname, bins_hi)

    H = win_lo + win_hi
    H = np.nan_to_num(H, nan=0.0)

    # Apply normalisation
    H = _setnorm(H, norm)

    # Circshift by -pos_lo
    H = np.roll(H, -pos_lo)

    # Truncate: compute the upper index in the shifted domain
    upidx = int(modcent(pos_hi - pos_lo, L))
    H = H[:upidx]

    if fc_was_negative and do_symmetric:
        H = H[::-1].copy()
        # The window taken above is deliberately wide (about 2B rather than the
        # filter's own B - A), because that width is what the roll-and-reverse
        # arithmetic needs in order to land the response on the right bins.  It
        # is not the filter's support: everything past the first B - A bins is
        # the aliased ``win_hi`` term, which the positive twin's narrower window
        # discards and which this branch was keeping.
        #
        # Measured on a log-warped complex bank, channel fc = -2321.6 came out
        # with 3334 nonzero bins and 4.5x the energy of its +2321.6 twin — the
        # twin's 1291 bins were all present and correct, with a spurious
        # 2043-bin tail attached.  Trim to the twin's width.
        lo_edge = math.floor(scaletofreq(fcscale - 0.5 * bw) / fs * L)
        width = int(modcent(pos_hi - lo_edge, L))
        if 0 < width < H.size:
            H = H[:width].copy()

    return H


# ---------------------------------------------------------------------------
# warpedblfilter – warped band-limited filter descriptor
# ---------------------------------------------------------------------------

def warpedblfilter(winname: str, fsupp: float, fc: float, *,
                   fs: float = 2.0,
                   freqtoscale=None,
                   scaletofreq=None,
                   norm: str = "energy",
                   delay: int = 0,
                   scal: float = 1.0,
                   do_symmetric: bool = False) -> dict:
    """Construct a warped band-limited filter descriptor.

    Parameters
    ----------
    winname : str
        Window type passed to firwin_eval.
    fsupp : float
        Frequency support in scale units (bandwidth).
    fc : float
        Centre frequency in Hz.
    fs : float
        Sampling rate.
    freqtoscale : callable
        Convert Hz → scale units.
    scaletofreq : callable
        Convert scale units → Hz.
    norm : str
        Normalisation (``'energy'``, ``'1'``, ``'inf'``).
    delay : int
        Delay in samples.
    scal : float
        Additional scaling factor.
    do_symmetric : bool
        Whether the warping is symmetric about DC, i.e. whether ``freqtoscale``
        diverges at 0 (as a log-like scale does) so that negative centre
        frequencies must be obtained by mirroring the positive ones rather than
        by evaluating the warp below zero.

        ``comp_warpedfreqresponse`` and ``comp_warpedfoff`` have always taken
        this flag; ``warpedblfilter`` did not accept it and so could not pass it
        on, which left every negative-frequency channel of a
        ``freqrange='complex'`` bank evaluating a warp outside its domain.

    Returns
    -------
    g : dict with keys ``'H'``, ``'foff'``, ``'realonly'``, ``'delay'``, ``'fs'``.
    """
    if freqtoscale is None or scaletofreq is None:
        raise ValueError("freqtoscale and scaletofreq must be provided")

    def H(L: int) -> np.ndarray:
        h = comp_warpedfreqresponse(
            winname, fc, fsupp, fs, L,
            freqtoscale, scaletofreq,
            norm=norm,
            do_symmetric=do_symmetric,
        )
        return h * scal

    def foff(L: int) -> int:
        return comp_warpedfoff(fc, fsupp, fs, L,
                               freqtoscale, scaletofreq,
                               do_symmetric)

    return {
        "H": H,
        "foff": foff,
        "realonly": 1 if fc != 0 else 0,
        "delay": int(delay),
        "fs": fs,
    }
