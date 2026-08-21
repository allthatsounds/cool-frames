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
        # Convert to 0-based: subtract 1
        foff = -math.floor(scaletofreq(fcscale + 0.5 * bw) / fs * L)
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

    # Upper extent
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
                   scal: float = 1.0) -> dict:
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
            norm=norm
        )
        return h * scal

    def foff(L: int) -> int:
        return comp_warpedfoff(fc, fsupp, fs, L,
                               freqtoscale, scaletofreq)

    return {
        "H": H,
        "foff": foff,
        "realonly": 1 if fc != 0 else 0,
        "delay": int(delay),
        "fs": fs,
    }
