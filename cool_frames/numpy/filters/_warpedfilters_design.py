"""
numpy/layer1/_warpedfilters_design.py
=====================================
Top-level warped filterbank design.

Port of: layer1/filter_design/warpedfilters.m

Authors (original MATLAB): Nicki Holighaus, Zdenek Prusa
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..core._core import filterbanklength, floor23
from ._edge_filters import (
    build_complement_highpass,
    build_complement_lowpass,
)
from ._firwin import firwin_eval
from ._warpedfilters import warpedblfilter

# ---------------------------------------------------------------------------
# Helper: comp_filterbank_a
# ---------------------------------------------------------------------------

def _comp_filterbank_a(a, M: int) -> np.ndarray:
    """Expand *a* to (M, 2) fractional form ``[num, den]``."""
    a = np.asarray(a, dtype=float)
    if a.ndim == 0:
        return np.column_stack([np.full(M, float(a)), np.ones(M)])  # type: ignore[no-any-return]
    if a.ndim == 1:
        return np.column_stack([a[:M], np.ones(M)])  # type: ignore[no-any-return]
    if a.shape[1] == 1:
        return np.column_stack([a[:M, 0], np.ones(M)])  # type: ignore[no-any-return]
    return a[:M, :]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Nyquist and zero-frequency filters
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Main: warpedfilters
# ---------------------------------------------------------------------------

def warpedfilters(
    freqtoscale,
    scaletofreq,
    fs: float,
    fmin: float,
    fmax: float,
    bins: int,
    Ls: int,
    *,
    window: str = "hann",
    bwmul: float = 1.0,
    redmul: float = 1.0,
    min_win: int = 1,
    norm: str = "inf",
    sampling: str = "regsampling",
    freqrange: str = "real",
) -> tuple[list[dict], np.ndarray, np.ndarray, int, dict]:
    """Construct a frequency-warped filterbank.

    Parameters
    ----------
    freqtoscale : callable
        Function converting Hz → scale units.
    scaletofreq : callable
        Function converting scale units → Hz.
    fs : float
        Sampling rate (Hz).
    fmin : float
        Minimum frequency (Hz).
    fmax : float
        Maximum frequency (Hz).
    bins : int
        Number of filters per scale unit.
    Ls : int
        Signal length.
    window : str
        Window type from firwin.
    bwmul : float
        Bandwidth multiplier.
    redmul : float
        Redundancy multiplier.
    min_win : int
        Minimum window length.
    norm : str
        Filter normalisation: ``'inf'`` (peak), ``'energy'`` (unit energy),
        ``'1'`` (unit ℓ¹ norm).  Default ``'inf'``.
    sampling : str
        ``'regsampling'``, ``'uniform'``, ``'fractional'``,
        ``'fractionaluniform'``.
    freqrange : str
        ``'real'`` or ``'complex'``.

    Returns
    -------
    g : list of filter dicts
    a : ndarray – downsampling rates
    fc : ndarray – centre frequencies (Hz)
    L : int – admissible transform length
    info : dict – design metadata (fc, a, L, designer)

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import warpedfilters
    >>> # Warped filterbank with logarithmic frequency warping
    >>> freqtoscale = lambda f: np.log2(np.maximum(f, 1))
    >>> scaletofreq = lambda s: 2.0 ** s
    >>> g, a, fc, L, _info = warpedfilters(
    ...     freqtoscale, scaletofreq, 8000, 100, 4000, 12, 1024
    ... )
    >>> len(g) > 0  # at least DC + inner + Nyquist
    True
    >>> a.shape[0] == len(g)
    True
    """
    nf = fs / 2  # Nyquist

    if fmax > nf:
        fmax = nf

    # Handle fmin = 0 with scale that goes to -inf
    if fmin <= 0 and np.isinf(freqtoscale(0)):
        fmin = scaletofreq(freqtoscale(1))

    # Determine range of scale channels
    chan_min = math.floor(bins * freqtoscale(fmin)) / bins
    if chan_min >= fmax:
        raise ValueError("Invalid frequency scale, try lowering fmin")

    chan_max = chan_min
    while scaletofreq(chan_max) <= fmax:
        chan_max += 1 / bins
    while scaletofreq(chan_max + bwmul) >= nf:
        chan_max -= 1 / bins

    # Prepare scale vector and centre frequencies
    scalevec = np.arange(chan_min, chan_max + 0.5 / bins, 1 / bins)
    fc_list = list(scaletofreq(scalevec))
    fc_list.append(nf)
    if fmin != 0:
        fc_list.insert(0, 0.0)

    fc_arr = np.array(fc_list)
    M = len(fc_arr)

    # Set bandwidths
    fsupp = np.zeros(M)
    fsupp_idx = 0
    if fmin != 0:
        fsupp[0] = math.ceil(2 * scaletofreq(chan_min - 1 / bins + bwmul)) + 2
        fsupp_idx = 1
    fsupp[fsupp_idx:M-1] = np.ceil(
        scaletofreq(scalevec + bwmul) - scaletofreq(scalevec - bwmul)
    ) + 2
    fsupp[M-1] = math.ceil(2 * (nf - scaletofreq(chan_max + 1 / bins - bwmul))) + 2

    # Capture the single-sided edge bandwidths for the admissibility check
    # before ``freqrange='complex'`` mirrors ``fsupp`` below.
    _fsupp_dc, _fsupp_nyq = float(fsupp[0]), float(fsupp[M - 1])

    # Subsampling rates
    aprecise = fs / fsupp
    aprecise[1:-1] = aprecise[1:-1] / redmul

    if np.any(aprecise < 1):
        raise ValueError(f"Maximum redundancy mult. for this setting is "
                         f"{float(np.min(fs / fsupp)):.2f}")

    # Compute downsampling
    if sampling == "regsampling":
        a = floor23(aprecise)
        if np.isscalar(a):
            a = np.array([a])
        a = np.atleast_1d(a).astype(int)
        L = filterbanklength(Ls, a)

        iters = 0
        while 2 * Ls < L and not np.all(a == a[0]) and iters < 100:
            maxa = np.max(a)
            remaining = a[a != maxa]
            if len(remaining) > 0:
                a[a == maxa] = np.max(remaining)
            L = filterbanklength(Ls, a)
            iters += 1

    elif sampling == "fractional":
        L = Ls
        N = np.ceil(Ls / aprecise).astype(int)
        a = np.column_stack([np.full(M, Ls, dtype=int), N])

    elif sampling == "fractionaluniform":
        L = Ls
        N_val = int(np.ceil(Ls / np.min(aprecise)))
        a = np.tile([Ls, N_val], (M, 1))

    elif sampling == "uniform":
        a_val = int(math.floor(np.min(aprecise)))
        L = filterbanklength(Ls, a_val)
        a = np.full(M, a_val, dtype=int)

    else:
        raise ValueError(f"Unknown sampling mode {sampling!r}")

    # Expand a
    afull = _comp_filterbank_a(a, M)

    # Scaling factors
    scal = np.sqrt(afull[:, 0] / afull[:, 1])

    if freqrange == "real":
        scal[0] = scal[0] / math.sqrt(2)
        scal[M-1] = scal[M-1] / math.sqrt(2)
    elif freqrange == "complex":
        # `a` is 1-D for regsampling/uniform and 2-D only for the fractional
        # modes, so an unconditional vstack raised
        # "all the input array dimensions except for the concatenation axis
        # must match" on the default sampling.  `waveletfilters` has the same
        # construct written correctly; mirror it.
        _stack = np.vstack if np.asarray(a).ndim == 2 else np.concatenate
        a = _stack([a, np.flipud(a[1:M-1])])  # type: ignore[assignment]
        scal = np.concatenate([scal, np.flipud(scal[1:M-1])])  # type: ignore[assignment]
        fc_arr = np.concatenate([fc_arr, -np.flipud(fc_arr[1:M-1])])  # type: ignore[assignment]
        fsupp = np.concatenate([fsupp, np.flipud(fsupp[1:M-1])])  # type: ignore[assignment]

    # Determine symmetry flag.
    #
    # A log-like warp diverges at DC, so its negative-frequency channels cannot
    # be built by evaluating the warp below zero — they have to be mirrored from
    # the positive side.  That is what this flag records, and until now it was
    # computed and then dropped on the floor: `warpedblfilter` took no such
    # argument, so every negative-fc channel of a `freqrange='complex'` bank was
    # built by evaluating the warp outside its domain.
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            symmetry = bool(freqtoscale(0) < -1e10)
    except (ValueError, RuntimeWarning, FloatingPointError):
        symmetry = True

    # Build inner filters via warpedblfilter
    g: list[dict[str, Any] | None] = [None] * len(fc_arr)

    g_idx_start = 0
    if fmin != 0:
        g_idx_start = 1

    inner_indices = list(range(g_idx_start, M - 1))
    if freqrange == "complex":
        inner_indices += list(range(M, len(fc_arr)))

    for idx in inner_indices:
        g[idx] = warpedblfilter(
            window,
            bwmul * 2,
            float(fc_arr[idx]),
            fs=fs,
            freqtoscale=freqtoscale,
            scaletofreq=scaletofreq,
            scal=float(scal[idx]),
            norm=norm,
            do_symmetric=symmetry,
        )

    # Edge filters via shared complement construction
    g_inner = [g[i] for i in inner_indices if i < M]
    a_inner = a[g_idx_start:M-1] if a.ndim == 1 else a[g_idx_start:M-1, :]

    if fmin != 0:
        # DC lowpass
        fc_first = float(fc_arr[g_idx_start])
        fsupp_dc = float(fsupp[0])
        # Derive taper from the ratio of first inner channel bandwidth
        # to the DC filter bandwidth
        fsupp_first_inner = float(fsupp[g_idx_start])
        ratio_dc = min(1.0, fsupp_first_inner / fsupp_dc) if fsupp_dc > 0 else 0.0

        g[0] = build_complement_lowpass(
            g_inner, a_inner, fc_first, fs,  # type: ignore[arg-type]
            scal=float(scal[0]),
            fsupp_lp=fsupp_dc,
            taper_ratio=ratio_dc,
            min_win=min_win,
        )

    # Nyquist highpass
    fc_last = float(fc_arr[M - 2]) if M >= 2 else 0.0
    fsupp_nyq = float(fsupp[M - 1])
    fsupp_last_inner = float(fsupp[M - 2]) if M >= 2 else 0.0
    ratio_nyq = min(1.0, fsupp_last_inner / fsupp_nyq) if fsupp_nyq > 0 else 0.0

    g[M - 1] = build_complement_highpass(
        g_inner, a_inner, fc_last, fs,  # type: ignore[arg-type]
        scal=float(scal[M - 1]),
        fsupp_hp=fsupp_nyq,
        taper_ratio=ratio_nyq,
        min_win=min_win,
    )

    from ..diagnostics.admissibility import check_admissible

    # The warped rule: channels sit uniformly in the scale coordinate and each
    # filter is +/- bwmul wide *in that coordinate*, so the interval is derived
    # from ``scalevec`` rather than from Hz supports.
    admissible = check_admissible(
        None, None, fs=fs, L=int(L),
        fsupp_dc=_fsupp_dc, fsupp_nyq=_fsupp_nyq,
        warped=(scalevec, scaletofreq, bwmul), min_win=1,
        designer="warpedfilters")

    # LTFAT publishes no info struct at all for warpedfilters -- no fc, no
    # tfr -- so `filterbankconstphase`'s magnitude path had no sqtfr to use on
    # this designer and no reference to copy one from. The rule LTFAT applies
    # to the designers that DO expose one was recovered from its own exports
    # (see `tfr_from_bandwidth`) and depends only on the designed bandwidth
    # and the window, both of which warpedfilters has. So it is derived here
    # rather than left blank. It is NOT validated against LTFAT and cannot be.
    from ._tfr import tfr_from_bandwidth

    # As for the other designers: the two edge channels' bandwidths are the
    # ones captured before the `freqrange='complex'` mirror, not whatever
    # `fsupp` holds at those positions now.
    _bw = np.asarray(fsupp, dtype=float).copy()
    _bw[0] = float(_fsupp_dc)
    _bw[min(M - 1, len(_bw) - 1)] = float(_fsupp_nyq)

    info = {"fc": fc_arr, "a": a, "L": int(L), "designer": "warpedfilters",
            "fsupp": fsupp, "scalevec": scalevec, "bwmul": float(bwmul),
            "fsupp_dc": _fsupp_dc, "fsupp_nyq": _fsupp_nyq,
            "tfr": tfr_from_bandwidth(_bw, fs, int(L)),
            "tfr_source": "derived (no LTFAT reference exists)",
            "admissible": admissible}
    return g, a, fc_arr, int(L), info  # type: ignore[return-value]
