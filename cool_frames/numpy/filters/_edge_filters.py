"""
_edge_filters.py
=================
Shared complement-based edge-filter constructor for DC and Nyquist.

All filterbank design functions (audfilters, cqtfilters, hopfilters,
waveletfilters) use this module to build their DC (lowpass) and Nyquist
(highpass) edge filters.  The complement construction guarantees that
the edge filter exactly fills the spectral gap left by the inner
channels, minimising the frame condition number κ = B/A.

Theory
------
Given inner filters g_inner with hop sizes a_inner, the frame response
is::

    S_inner(k) = Σ_m |H_m(k)|² / a_m

The complement filter fills the gap::

    H_edge(k) = P(k) · √( max(S_inner) − S_inner(k) ) · scal

where P(k) is a Hann-taper prototype centred at the target frequency
(0 Hz for DC, fs/2 for Nyquist).  The prototype's bandwidth and taper
ratio control where the edge filter tapers off.  The ``√(S_max − S)``
factor ensures the combined frame response is exactly flat within the
prototype's support.

For design functions that lack auditory-scale information to derive the
prototype bandwidth from spacing, a heuristic based on the nearest
inner channel is used:

    fsupp_dc  = 2 · fc[first_inner]
    fsupp_nyq = 2 · (fs/2 − fc[last_inner])
    taper_ratio = fsupp_neighbour / fsupp_edge  (clamped to [0, 1])
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_complement_lowpass(
    g_inner: list[dict],
    a_inner,
    fc_first_inner: float,
    fs: float,
    scal: float,
    fsupp_lp: float,
    taper_ratio: float,
    min_win: int = 4,
) -> dict:
    """Build a complement lowpass edge filter centred at DC.

    Parameters
    ----------
    g_inner : list[dict]
        Inner channel filter descriptors (no edge filters).
    a_inner : ndarray
        Hop sizes for the inner channels (1-D or (M, 2)).
    fc_first_inner : float
        Centre frequency of the first inner channel (Hz).
    fs : float
        Sampling rate (Hz).
    scal : float
        Amplitude scaling factor (typically ``sqrt(a_dc) / sqrt(2)``).
    fsupp_lp : float
        Bandwidth (Hz) of the DC prototype filter.  Typically
        ``2 · fc_first_inner`` or derived from auditory-scale spacing.
    taper_ratio : float
        Plateau-to-support ratio of the Hann-taper prototype, in [0, 1].
        0 → pure Hann (no plateau), 1 → rectangular.
    min_win : int
        Minimum window length in DFT bins.

    Returns
    -------
    dict
        Filter descriptor with callable H(L) and foff(L).
    """
    from ..filterbanks._frame import filterbankresponse
    from ._cqtfilters import _make_direct_filter
    from ._filters import filter_freqresp as _ffr

    taper_ratio = float(max(0.0, min(1.0, taper_ratio)))

    # Build a Hann-taper prototype at DC (realonly=0)
    P0 = _make_direct_filter(
        0.0, fsupp_lp, fs, scal=1.0, norm="inf",
        taper_ratio=taper_ratio if taper_ratio > 0 else None,
        min_win=min_win,
    )

    def H(L: int) -> np.ndarray:
        P0_full, _ = _ffr(P0, L)
        S = filterbankresponse(g_inner, a_inner, L, real=False)
        S_pos = S[:L // 2 + 1]
        S_max = float(np.max(S_pos))
        Hinv = np.sqrt(np.maximum(S_max - S, 0.0))
        C_full = P0_full * Hinv
        # Extract at prototype support bins
        Lw = len(P0["H"](L))
        foff_v = int(P0["foff"](L))
        idx = np.mod(np.arange(foff_v, foff_v + Lw), L)
        return C_full[idx] * scal  # type: ignore[no-any-return]

    def foff(L: int) -> int:
        return int(P0["foff"](L))

    return {"H": H, "foff": foff, "realonly": 0, "delay": 0, "fs": fs}


def build_complement_highpass(
    g_inner: list[dict],
    a_inner,
    fc_last_inner: float,
    fs: float,
    scal: float,
    fsupp_hp: float,
    taper_ratio: float,
    min_win: int = 4,
) -> dict:
    """Build a complement highpass edge filter centred at Nyquist.

    Parameters
    ----------
    g_inner : list[dict]
        Inner channel filter descriptors (no edge filters).
    a_inner : ndarray
        Hop sizes for the inner channels (1-D or (M, 2)).
    fc_last_inner : float
        Centre frequency of the last inner channel (Hz).
    fs : float
        Sampling rate (Hz).
    scal : float
        Amplitude scaling factor (typically ``sqrt(a_nyq) / sqrt(2)``).
    fsupp_hp : float
        Bandwidth (Hz) of the Nyquist prototype filter.  Typically
        ``2 · (fs/2 − fc_last_inner)`` or derived from auditory-scale
        spacing.
    taper_ratio : float
        Plateau-to-support ratio of the Hann-taper prototype, in [0, 1].
    min_win : int
        Minimum window length in DFT bins.

    Returns
    -------
    dict
        Filter descriptor with callable H(L) and foff(L).
    """
    from ..filterbanks._frame import filterbankresponse
    from ._cqtfilters import _make_direct_filter
    from ._filters import filter_freqresp as _ffr

    taper_ratio = float(max(0.0, min(1.0, taper_ratio)))

    # Build prototype at DC, then roll to Nyquist inside H(L).
    # This avoids the realonly halving that would occur if we placed
    # the prototype directly at fs/2 via _make_direct_filter.
    PK_dc = _make_direct_filter(
        0.0, fsupp_hp, fs, scal=1.0, norm="inf",
        taper_ratio=taper_ratio if taper_ratio > 0 else None,
        min_win=min_win,
    )

    def H(L: int) -> np.ndarray:
        PK_dc_full, _ = _ffr(PK_dc, L)
        PK_nyq_full = np.roll(PK_dc_full, L // 2)
        S = filterbankresponse(g_inner, a_inner, L, real=False)
        S_pos = S[:L // 2 + 1]
        S_max = float(np.max(S_pos))
        Hinv = np.sqrt(np.maximum(S_max - S, 0.0))
        C_full = PK_nyq_full * Hinv
        # Extract at prototype support bins shifted to Nyquist
        Lw = len(PK_dc["H"](L))
        foff_dc = int(PK_dc["foff"](L))
        foff_hp = L // 2 + foff_dc
        idx = np.mod(np.arange(foff_hp, foff_hp + Lw), L)
        return C_full[idx] * scal  # type: ignore[no-any-return]

    def foff(L: int) -> int:
        Lw = len(PK_dc["H"](L))
        foff_dc = int(PK_dc["foff"](L))
        return L // 2 + foff_dc

    return {"H": H, "foff": foff, "realonly": 0, "delay": 0, "fs": fs}


# ---------------------------------------------------------------------------
# Convenience: derive prototype parameters from inner-channel geometry
# ---------------------------------------------------------------------------

def edge_params_from_geometry(
    fc_inner_edge: float,
    fsupp_inner_edge: float,
    fs: float,
    target: str = "dc",
) -> tuple[float, float]:
    """Derive edge-filter bandwidth and taper ratio from the nearest
    inner channel.

    This is the generic heuristic used by ``cqtfilters`` and
    ``hopfilters`` (which don't have auditory-scale spacing to derive
    the prototype parameters).

    Parameters
    ----------
    fc_inner_edge : float
        Centre frequency of the nearest inner channel (Hz).
        First inner channel for DC, last inner channel for Nyquist.
    fsupp_inner_edge : float
        Bandwidth of that inner channel (Hz).
    fs : float
        Sampling rate (Hz).
    target : str
        ``'dc'`` or ``'nyquist'``.

    Returns
    -------
    fsupp_edge : float
        Prototype bandwidth (Hz).
    taper_ratio : float
        Taper ratio in [0, 1].
    """
    nf = fs / 2.0
    if target == "dc":
        fsupp_edge = 2.0 * fc_inner_edge
    else:
        fsupp_edge = 2.0 * (nf - fc_inner_edge)

    # Taper: ratio of the neighbouring inner channel's bandwidth to the
    # edge filter's bandwidth.  Clamped to [0, 1].
    if fsupp_edge > 0 and fsupp_inner_edge > 0:
        taper_ratio = min(1.0, fsupp_inner_edge / fsupp_edge)
    else:
        taper_ratio = 0.0

    return fsupp_edge, taper_ratio
