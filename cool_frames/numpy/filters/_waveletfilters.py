"""
numpy/layer1/_waveletfilters.py
===============================
Top-level wavelet filterbank design.

Port of: layer1/filter_design/waveletfilters.m

Authors (original MATLAB): Nicki Holighaus, Zdenek Prusa,
                            Guenther Koliander, Clara Hollomey
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from ..core._core import filterbanklength, floor23
from ._edge_filters import (
    build_complement_highpass,
    build_complement_lowpass,
    edge_params_from_geometry,
)
from ._freqwavelet import freqwavelet


# ---------------------------------------------------------------------------
# Helper: painless hop caps
# ---------------------------------------------------------------------------

def _painless_caps(winCell, L: int, scales, trunc_at: float, basefc: float,
                   aprecise, lp_num: int, M2: int,
                   quantise: bool = True) -> np.ndarray:
    r"""Per-channel painless hop caps :math:`a_m \le \lfloor L / W_m \rfloor`.

    A filterbank's frame operator is diagonal in frequency -- the *painless*
    case, and the only case in which :func:`filterbankdual` /
    :func:`filterbanktight` are exact -- when every channel's frequency
    support fits inside one alias-free period, i.e. ``W_m <= L / a_m``.

    ``W_m`` is measured, not modelled: the channel is realised at this ``L``
    and its live bins counted.  A modelled width would have to reproduce
    ``freqwavelet``'s rounding, and it is exactly that kind of near-miss that
    leaves a bank one bin short of painless with no symptom other than a bad
    dual.

    The lowpass channels have no ``freqwavelet`` realisation at this point, so
    their bandwidth is bounded by ``aprecise`` (which is what sets it).  That
    bound is loose; the realised bank is re-checked at the end of
    :func:`waveletfilters` and a violation there is reported.

    ``quantise`` rounds each cap DOWN to a ``floor23`` (:math:`2^i 3^j`) value
    so integer hops share factors and ``lcm(a) -> L`` stays small.  Rounding
    down can only strengthen the inequality.  Rational (``fractional``) hops
    need no such quantisation and pass ``quantise=False``.
    """
    gl_tmp, _ = freqwavelet(
        winCell, L, scales,
        output_format="asfreqfilter",
        efsuppthr=trunc_at,
        basefc=basefc,
    )
    if not isinstance(gl_tmp, list):
        gl_tmp = [gl_tmp]
    widths = np.empty(len(gl_tmp), dtype=int)
    for j, gm in enumerate(gl_tmp):
        Hj = np.asarray(gm["H"](L)).ravel()
        nz = np.flatnonzero(np.abs(Hj) > 1e-10)
        widths[j] = (nz[-1] - nz[0] + 1) if nz.size else 1

    caps = np.empty(M2, dtype=float)
    for k in range(lp_num):
        caps[k] = max(1.0, math.floor(L / max(1.0, float(aprecise[k]))))
    caps[lp_num:M2] = np.maximum(1, (L // np.maximum(widths, 1)))
    if quantise:
        caps = np.array([max(1, floor23(int(c))) for c in caps], dtype=float)
    return caps


def _painless_ratio(g: list[dict], a, L: int) -> tuple[float, int]:
    """Worst realised ``a_m * W_m / L`` over the finished bank, and the count
    of channels above 1.

    This is the same inequality :func:`_painless_caps` enforces, but measured
    on the bank that is actually returned -- after the lowpass and Nyquist
    complements have been appended and after ``redtar`` has rewritten the
    hops.  Those later steps are exactly where a cap applied mid-design stops
    being a guarantee.
    """
    a_arr = np.asarray(a, dtype=float)
    if a_arr.ndim == 2:
        a_rat = a_arr[:, 0] / a_arr[:, 1]
    else:
        a_rat = a_arr.ravel()
    worst = 0.0
    nbad = 0
    for m, gm in enumerate(g):
        H = gm.get("H")
        if H is None:
            continue
        Hm = np.asarray(H(L) if callable(H) else H).ravel()
        nz = np.flatnonzero(np.abs(Hm) > 1e-10)
        if not nz.size:
            continue
        W = int(nz[-1] - nz[0] + 1)
        r = float(a_rat[m % len(a_rat)]) * W / L
        worst = max(worst, r)
        # One bin of slack: several designers emit L/a + 1 bins by
        # construction and are painless in practice.
        if W > L / float(a_rat[m % len(a_rat)]) + 1:
            nbad += 1
    return worst, nbad


def _repair_complement_hops(g: list[dict], a, L: int) -> int:
    r"""Lower any channel's hop until it meets its own painless limit.

    :func:`_painless_caps` runs while the hops are being chosen, but the DC
    lowpass and the Nyquist highpass complements are *appended afterwards* and
    simply inherit the smallest wavelet hop.  A complement is wide by
    construction -- it spans everything the wavelets left uncovered -- so on a
    sparse scale set it can arrive several times over its limit while every
    wavelet channel is comfortably inside.  At ``fs = 8000``, ``Ls = 1024``,
    24 scales at 6/octave the Nyquist complement occupied 1113 of 1728 bins at
    ``a = 6``: ``aW/L = 3.87``, and the exact oracle put the lower frame bound
    at 0 while the diagonal estimator reported a healthy ``kappa = 4.4``.

    Lowering a hop never invalidates the painless inequality for any other
    channel, so this is a safe local repair.  Two constraints shape it:

    * ``L`` must stay a whole number of hops, so an integer hop is lowered to
      the largest **divisor of L** that is within the limit rather than to the
      limit itself.  (Rational ``[L, N]`` hops are divisor-free: raising ``N``
      is enough.)
    * ``g[m]["H"]`` was scaled by ``sqrt(a_m)``, the package's per-channel
      energy convention, so changing the hop without rescaling would leave the
      channel with the gain of a hop it no longer has.  The response is
      rescaled by ``sqrt(a_new / a_old)`` to keep the convention intact.

    Returns the number of channels repaired.
    """
    a_arr = np.asarray(a)
    fixed = 0
    for m, gm in enumerate(g):
        H = gm.get("H")
        if H is None:
            continue
        Hm = np.asarray(H(L) if callable(H) else H).ravel()
        nz = np.flatnonzero(np.abs(Hm) > 1e-10)
        if not nz.size:
            continue
        W = int(nz[-1] - nz[0] + 1)
        if a_arr.ndim == 2:
            a_old = float(a_arr[m, 0]) / float(a_arr[m, 1])
            if W <= L / a_old + 1:
                continue
            a_arr[m, 1] = max(int(a_arr[m, 1]), W)
            a_new_m = float(a_arr[m, 0]) / float(a_arr[m, 1])
        else:
            a_old = float(a_arr[m])
            if W <= L / a_old + 1:
                continue
            cap = max(1, int(L // W))
            d = cap
            while d > 1 and L % d:
                d -= 1
            if d >= a_old:
                continue
            a_arr[m] = d
            a_new_m = float(d)
        s = math.sqrt(a_new_m / a_old)
        if callable(H):
            # Keep it lazy: some channels build their response from L, and
            # freezing it here would pin the filter to this one length.
            gm["H"] = (lambda fn, sc: lambda Lq: np.asarray(fn(Lq)) * sc)(H, s)
        else:
            gm["H"] = Hm * s
        fixed += 1
    return fixed


# ---------------------------------------------------------------------------
# Helper: comp_filterbank_a
# ---------------------------------------------------------------------------

def _comp_filterbank_a(a, M: int) -> np.ndarray:
    """Expand *a* to (M, 2) fractional form ``[num, den]``.

    Accepts a scalar, a 1-D ``(M,)`` array of integer hops (the regsampling /
    uniform convention, matching audfilters/cqtfilters/greenwoodfilters), a
    column ``(M,1)``, or an already-fractional ``(M,2)`` array.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:                       # scalar or (M,) integer hops
        if a.size == 1:
            return np.column_stack([np.full(M, a.flat[0]), np.ones(M)])  # type: ignore[no-any-return]
        return np.column_stack([a[:M], np.ones(M)])  # type: ignore[no-any-return]
    if a.shape[1] == 1:                   # (M,1) or (1,1) column
        col = a[:, 0]
        if col.size == 1:
            return np.column_stack([np.full(M, col[0]), np.ones(M)])  # type: ignore[no-any-return]
        return np.column_stack([col[:M], np.ones(M)])  # type: ignore[no-any-return]
    return a[:M, :]                        # already (M, 2)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Lowpass helpers
# ---------------------------------------------------------------------------

def _wavelet_lowpass(gout, a, L, lowpass_bandwidth, taper_ratio, scal_, flags_real, min_win=4):
    """Construct a single lowpass filter from the filterbank response.

    Port of the nested ``wavelet_lowpass`` function in waveletfilters.m,
    now delegating to the shared complement builder.
    """
    # The wavelet lowpass uses normalised frequency (fs=2.0) and
    # lowpass_bandwidth in normalised units.  Convert to Hz-like parameters
    # that the shared builder expects.
    fs_norm = 2.0
    fsupp_lp = lowpass_bandwidth * fs_norm   # bandwidth in normalised Hz

    # Derive a nominal fc for the first inner channel from the wavelet filters.
    # The complement builder needs fc_first_inner, but for wavelet lowpass
    # the prototype is always centred at DC.  We pass a small nominal value.
    fc_first_inner = fsupp_lp / 2.0  # half the support

    glow = build_complement_lowpass(
        gout, a, fc_first_inner, fs_norm,
        scal=scal_,
        fsupp_lp=fsupp_lp,
        taper_ratio=taper_ratio,
        min_win=min_win,
    )

    Lw = lambda L_: min(math.ceil(lowpass_bandwidth * L_), L_)
    foff_func = lambda L_: -int(Lw(L_) // 2)

    infolow = {
        "fc": np.array([0.0]),
        "foff": np.array([foff_func(L)]),
        "fsupp": np.array([Lw(L)]),
        "basefc": np.array([0.0]),
        "scale": np.array([0.0]),
        "dilation": np.array([0.0]),
        "bw": np.array([lowpass_bandwidth]),
        "tfr": np.array([1.0]),
        "aprecise": np.array([lowpass_bandwidth * L]),
        "a_natural": np.array([[L, math.ceil(lowpass_bandwidth * L)]]),
        "cauchyAlpha": None,
    }

    return glow, infolow


def _wavelet_highpass(gout, a, L, fc_last, fsupp_last, scal_, min_win=4):
    """Construct a Nyquist highpass complement filter.

    cool_frames extension (no LTFAT equivalent): mirrors :func:`_wavelet_lowpass` at the
    upper edge so a real wavelet bank covers ``[0, Nyquist]`` and is invertible.
    ``fc_last`` / ``fsupp_last`` are the centre frequency and support (in the
    ``fs=2`` normalised convention) of the highest-frequency wavelet.
    """
    fs_norm = 2.0
    fsupp_hp, taper_ratio = edge_params_from_geometry(
        float(fc_last), float(fsupp_last), fs_norm, target="nyquist")

    ghigh = build_complement_highpass(
        gout, a, float(fc_last), fs_norm,
        scal=float(scal_),
        fsupp_hp=fsupp_hp,
        taper_ratio=taper_ratio,
        min_win=min_win,
    )

    bw = fsupp_hp / fs_norm
    Lw = lambda L_: min(math.ceil(bw * L_), L_)
    infohigh = {
        "fc": np.array([1.0]),                    # normalised Nyquist
        "foff": np.array([L // 2 - int(Lw(L) // 2)]),
        "fsupp": np.array([Lw(L)]),
        "basefc": np.array([1.0]),
        "scale": np.array([0.0]),
        "dilation": np.array([0.0]),
        "bw": np.array([bw]),
        "tfr": np.array([1.0]),
        "aprecise": np.array([bw * L]),
        "a_natural": np.array([[L, math.ceil(bw * L)]]),
        "cauchyAlpha": None,
    }
    return ghigh, infohigh


def _wavelet_lowpass_repeat(winCell, scales_sorted_2, L, lowpass_number,
                            lowpass_at_zero, scal_vec, trunc_at, norm):
    """Construct repeated lowpass filters.

    Port of the nested ``wavelet_lowpass_repeat`` function in waveletfilters.m.
    """
    LP_range = 0.1 / scales_sorted_2[0]
    LP_step = abs(0.1 / scales_sorted_2[1] - LP_range)

    if scales_sorted_2[0] > 0:
        glow_list, infolow = freqwavelet(
            winCell, L,
            np.full(lowpass_number, scales_sorted_2[0]),
            output_format="asfreqfilter",
            efsuppthr=trunc_at,
            basefc=0.1,
            scal=scal_vec,
            norm=norm,
        )

        if not isinstance(glow_list, list):
            glow_list = [glow_list]

        # Correct foff for each lowpass filter
        for kk in range(lowpass_number):
            idx = lowpass_number - kk - 1
            orig_foff = glow_list[idx]["foff"]

            def _make_shifted_foff(orig_f, shift_k=kk+1):
                def new_foff(L_):
                    return orig_f(L_) - round(L_ * shift_k * LP_step / 2)
                return new_foff

            glow_list[idx]["foff"] = _make_shifted_foff(orig_foff)
            infolow["foff"][idx] = glow_list[idx]["foff"](L)

        infolow["fc"] = LP_range - np.arange(lowpass_number, 0, -1) * LP_step

    elif scales_sorted_2[0] < 0:
        if lowpass_at_zero:
            lowpass_number = lowpass_number - 1

        glow_list, infolow = freqwavelet(
            winCell, L,
            np.full(lowpass_number, scales_sorted_2[0]),
            output_format="asfreqfilter",
            efsuppthr=trunc_at,
            basefc=0.1,
            scal=scal_vec[:lowpass_number],
            norm=norm,
            freqrange="negative",
        )

        if not isinstance(glow_list, list):
            glow_list = [glow_list]

        for kk in range(lowpass_number):
            orig_foff = glow_list[kk]["foff"]

            def _make_shifted_foff(orig_f, shift_k=kk+1):
                def new_foff(L_):
                    return orig_f(L_) + math.floor(L_ * shift_k * LP_step / 2)
                return new_foff

            glow_list[kk]["foff"] = _make_shifted_foff(orig_foff)
            infolow["foff"][kk] = glow_list[kk]["foff"](L)

        infolow["fc"] = LP_range + np.arange(1, lowpass_number + 1) * LP_step
    else:
        raise ValueError("Scales must be nonzero")

    return glow_list, infolow


# ---------------------------------------------------------------------------
# Helper: exact bin intervals for the admissibility predictor
# ---------------------------------------------------------------------------

def _bins_to_linear(A: int, W: int, fs: float, L: int) -> list[tuple[float, float]]:
    """``(fc, fsupp)`` pairs in Hz covering exactly the ``W`` bins from ``A``.

    ``predict_admissible``'s linear rule puts an interval of
    ``odd(round(L*fsupp/fs))`` bins centred on bin ``round(L*fc/fs)``, so a
    single pair can only express an *odd* number of bins.  An even-width
    channel is therefore split into two odd intervals offset by one bin,
    whose union is exactly ``[A, A+W-1]``.
    """
    if W <= 0:
        return []
    if W % 2 == 1:
        return [((A + W // 2) * fs / L, W * fs / L)]
    if W == 2:
        return [(A * fs / L, fs / L), ((A + 1) * fs / L, fs / L)]
    Wp = W - 1
    return [((A + Wp // 2) * fs / L, Wp * fs / L),
            ((A + 1 + Wp // 2) * fs / L, Wp * fs / L)]


# ---------------------------------------------------------------------------
# Helpers: merge info dicts
# ---------------------------------------------------------------------------

def _merge_info(info1, info2, prepend=True):
    """Merge two info dicts, concatenating array fields."""
    merged = {}
    for key in info1:
        v1 = info1[key]
        v2 = info2.get(key)
        if v2 is None:
            merged[key] = v1
            continue
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            if key == "a_natural":
                if prepend:
                    merged[key] = np.vstack([v2, v1])
                else:
                    merged[key] = np.vstack([v1, v2])
            else:
                if prepend:
                    merged[key] = np.concatenate([v2, v1])
                else:
                    merged[key] = np.concatenate([v1, v2])
        else:
            merged[key] = v1  # Keep first for non-array fields
    return merged


# ---------------------------------------------------------------------------
# Main: waveletfilters
# ---------------------------------------------------------------------------

def waveletfilters(
    fs: float,
    Ls: int,
    *,
    fmin: float | None = None,
    fmax: float | None = None,
    bins=12,
    scales=None,
    wavelet=None,
    sampling: str = "regsampling",
    painless: bool = True,
    lowpass: str = "single",
    highpass: str = "auto",
    freqrange: str = "real",
    norm: str = "null",
    redmul: float = 1.0,
    redtar: float | None = None,
    delay=0,
    trunc_at: float = 1e-5,
    min_win: int = 4,
    hop_ms: float | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray, int, dict]:
    """Construct a wavelet filterbank.

    .. note:: **Parameter-order convention.**
       ``waveletfilters`` takes ``fs`` (sampling rate) first and ``Ls``
       (signal length) second, matching the other frequency-scale designers
       (``audfilters``, ``cqtfilters``, ``greenwoodfilters``, ``gabfilters``).
       This diverges from MATLAB LTFAT (which puts ``Ls`` first because its
       scales are dimensionless).

    .. note:: ``hop_ms`` requires a real ``fs`` (use ``fs=2.0`` for the
       dimensionless wavelet convention; ``hop_ms`` then refers to that
       normalised rate).

    Parameters
    ----------
    fs : float
        Sampling rate in Hz. Pass ``fs=2.0`` for the dimensionless LTFAT
        convention (normalised so Nyquist = 1).
    Ls : int
        Signal length.
    fmin : float, optional
        Lowest wavelet centre frequency in Hz (default 50). Used only on the
        default ``scales=None`` path.
    fmax : float, optional
        Highest wavelet centre frequency in Hz; defaults to Nyquist (``fs/2``).
        Used only on the default ``scales=None`` path.
    bins : int
        Voices (wavelets) per octave for the cqt-style geometric spacing
        (default 12, i.e. semitones). Used only on the default
        ``scales=None`` path.
    scales : array_like, optional
        Power-user escape hatch: explicit vector of wavelet scales (positive,
        dimensionless dilations). When given it OVERRIDES ``fmin``/``fmax``/
        ``bins``. When ``None`` (default), scales are derived from
        ``fmin``/``fmax``/``bins``.
    wavelet : str or list, optional
        Wavelet family with optional parameters (e.g. ``'cauchy'`` or
        ``['cauchy', 300]``). Default ``None`` -> ``['cauchy', 300]``.
        Replaces the former ``name`` argument.
    sampling : str
        ``'regsampling'``, ``'uniform'``, ``'fractional'``,
        ``'fractionaluniform'``.
    painless : bool
        Hop strategy.  ``True`` (default) caps each channel's hop at its
        painless limit (``a_m <= floor(L / W_m)``, ``W_m`` = that channel's
        measured DFT-bin support), which makes the frame operator diagonal in
        frequency and therefore makes :func:`filterbankdual` /
        :func:`filterbanktight` exact.

        ``False`` keeps the aggressive ``floor23`` + lcm-reduction heuristic:
        roughly 6x cheaper in coefficients, and fine for analysis-only work
        (scalograms, feature extraction), but the resulting bank is **not**
        painless and its diagonal dual does not reconstruct.  At
        ``fs = 8000``, ``Ls = 4096``, 64 geometric scales the two settings
        measure:

        =============  ===========  =============  ==================
        ``painless``   redundancy   ``max aW/L``   round-trip error
        =============  ===========  =============  ==================
        ``True``       8.98         0.98           4.9e-16
        ``False``      1.39         22.8           7.5e-01
        =============  ===========  =============  ==================

        With ``painless=False`` use :func:`ifilterbankiter` (iterative,
        exact) rather than :func:`filterbankdual` if you need to invert.

        .. versionchanged:: 0.1.1
           The default was ``False``.  It produced a bank whose canonical dual
           lost 75 % of the signal with no warning, which is not a defensible
           default for a designer whose siblings (``audfilters``,
           ``cqtfilters``, ``greenwoodfilters``) all reconstruct to 5e-16 out
           of the box.  ``painless`` is also now honoured by ``'uniform'``,
           ``'fractional'`` and ``'fractionaluniform'``; it used to be
           silently ignored by all three.
    lowpass : str
        ``'single'``, ``'repeat'``, ``'none'``.
    highpass : str
        **cool_frames extension (no LTFAT equivalent).** ``'auto'`` (default) appends a
        Nyquist highpass complement for a real bank (``freqrange='real'``,
        ``lowpass != 'none'``) so the filterbank covers ``[0, Nyquist]`` and is
        invertible; ``'none'`` reproduces LTFAT's behaviour (which does not cover
        the band above the highest wavelet and is documented as not guaranteed
        invertible). Invertibility still requires the *scales* to span toward
        Nyquist; the complement closes the final gap but cannot bridge an
        arbitrarily wide one without poor conditioning.
    freqrange : str
        ``'real'``, ``'complex'``, ``'analytic'``.
    norm : str
        Normalisation flag.
    redmul : float
        Redundancy multiplier (default 1).
    redtar : float or None
        Target redundancy.
    delay : int or callable or array
        Channel delays.
    trunc_at : float
        Truncation threshold for effective support.
    min_win : int
        Minimum window length in DFT bins for the lowpass edge filter
        (default 4).

    Returns
    -------
    gout : list of filter dicts
    a : ndarray – downsampling rates
    fc : ndarray – centre frequencies (Hz)
    L : int – admissible transform length
    info : dict – metadata

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import waveletfilters
    >>> scales = np.array([1.0, 2.0, 4.0, 8.0])
    >>> g, a, fc, L, info = waveletfilters(16000, 1024, scales=scales, wavelet='cauchy')
    >>> len(g)  # one DC lowpass + four wavelets + one Nyquist highpass
    6
    >>> a.shape[0] == len(g)
    True
    """
    if wavelet is None:
        wavelet = ["cauchy", 300]
    if isinstance(wavelet, str):
        wavelet = [wavelet]
    winCell = list(wavelet)

    # ------------------------------------------------------------------
    # Scale generation: cqt-style fmin/fmax/bins default, scales= override.
    # ------------------------------------------------------------------
    basefc = 0.1  # hardcoded mother-wavelet centre frequency (passed to freqwavelet)
    _fmin_given = fmin is not None
    _fmax_given = fmax is not None
    _bins_given = bool(np.any(np.asarray(bins) != 12))
    if scales is not None:
        if _fmin_given or _fmax_given or _bins_given:
            warnings.warn(
                "waveletfilters: explicit scales= overrides fmin/fmax/bins.",
                stacklevel=2)
    else:
        nyq = fs / 2.0
        if fmax is None:
            fmax = nyq          # interface consistency with cqtfilters
        if fmin is None:
            fmin = 50.0
        if fmin <= 0 or fmax <= fmin:
            raise ValueError("require 0 < fmin < fmax for the default scale grid")
        n = max(1, int(round(bins * np.log2(fmax / fmin))))  # bins = voices/octave
        k = np.arange(n + 1)
        fc_grid = fmin * (fmax / fmin) ** (k / n)            # ascending Hz, fc[-1]==fmax
        scales = 0.05 * fs / fc_grid
        scales = np.maximum(scales, basefc)                  # scale >= basefc (freqwavelet)

    # hop_ms implies a single common hop -> uniform sampling.
    if hop_ms is not None and sampling != "uniform":
        if sampling != "regsampling":
            warnings.warn(
                f"waveletfilters: hop_ms overrides sampling={sampling!r} -> 'uniform'.",
                stacklevel=2)
        sampling = "uniform"

    scales = np.atleast_1d(np.asarray(scales, dtype=float)).ravel()
    if np.any(scales <= 0):
        raise ValueError("scales must be positive")

    # Sort descending
    scales_sorted = np.sort(scales)[::-1]
    M = len(scales)

    # Generate mother wavelet to determine natural subsampling
    _, info_mother = freqwavelet(
        winCell, Ls, 1.0,
        output_format="asfreqfilter",
        efsuppthr=trunc_at,
        basefc=0.1,
    )
    basea = float(np.asarray(info_mother["aprecise"]).item())

    lowpass_at_zero = False

    # Determine lowpass filter count and aprecise for lowpass channels
    if lowpass == "repeat":
        lp_num = scales_sorted[1] / (scales_sorted[0] - scales_sorted[1])
        if abs(lp_num - round(lp_num)) < 1e3 * np.finfo(float).eps:
            lp_num = round(lp_num)
            lowpass_at_zero = True
        lp_num = max(1, int(math.floor(lp_num)))
        M2 = M + lp_num
        aprecise_lp = np.full(lp_num, basea * scales_sorted[0])

    elif lowpass == "single":
        lp_num = 1
        lowpass_at_zero = True
        M2 = M + 1
        aprecise_lp = np.array([0.2 * scales_sorted[3] * Ls]) if len(scales_sorted) >= 4 else np.array([basea * scales_sorted[0]])  # type: ignore[assignment]

    else:  # 'none'
        lp_num = 0
        M2 = M
        aprecise_lp = np.array([])  # type: ignore[assignment]

    aprecise = np.concatenate([aprecise_lp, basea * scales])  # type: ignore[assignment]

    if np.any(aprecise < 1):
        raise ValueError("Bandwidth of one of the filters exceeds fs")

    aprecise = aprecise / redmul
    if np.any(aprecise < 1):
        raise ValueError(f"Maximum redundancy mult. for this setting is {float(np.min(basea / scales).item()):.2f}")

    # Compute downsampling rates
    if sampling == "regsampling":
        a = np.ones(M2, dtype=int)

        lower_scale = math.floor(math.log2(1 / np.max(scales)))
        upper_scale = math.floor(math.log2(1 / np.min(scales)))

        for kk in range(lower_scale, upper_scale + 1):
            tempidx = np.where(np.floor(np.log2(1.0 / scales)) == kk)[0]
            if len(tempidx) == 0:
                continue
            # Index into aprecise (offset by lp_num)
            min_idx = tempidx[np.argmin(1.0 / scales[tempidx])]
            a_val = floor23(aprecise[min_idx + lp_num])
            a[tempidx + lp_num] = a_val

        # Lowpass channels get a=1 initially (or floor23 of their aprecise)
        for k in range(lp_num):
            a[k] = max(1, floor23(aprecise[k]))

        L = filterbanklength(Ls, a)

        # Heuristic to reduce lcm(a)
        iters = 0
        while 2 * Ls < L and not np.all(a == a[0]) and iters < 100:
            maxa = np.max(a)
            a[a == maxa] = np.max(a[a != maxa]) if np.any(a != maxa) else maxa  # type: ignore[assignment]
            L = filterbanklength(Ls, a)
            iters += 1

        if painless:
            # Cap each channel's hop at its painless limit a_m <= floor(L / W_m)
            # so the bank is a guaranteed frame (A>0). W_m is the channel's
            # DFT-bin support; for the wavelet channels this is the effective
            # support of freqwavelet at the current L, for the (wide, DC-side)
            # lowpass it is ceil(aprecise) which already bounds its bandwidth.
            # The support widths scale ~linearly with L, so one re-solve of L
            # after capping converges; we iterate a couple of times for safety.
            for _ in range(3):
                caps = _painless_caps(winCell, L, scales, trunc_at, basefc,
                                      aprecise, lp_num, M2).astype(int)
                a_capped = np.minimum(a, caps)
                if np.array_equal(a_capped, a) and filterbanklength(Ls, a_capped) == L:
                    a = a_capped
                    break
                a = a_capped
                L = filterbanklength(Ls, a)

            # Keep the admissible length manageable: reducing the largest hops
            # only lowers them (painless inequality stays satisfied) and shrinks
            # lcm(a) -> L. Snap distinct hop values down toward the smallest one.
            lcm_iters = 0
            while 2 * Ls < L and not np.all(a == np.min(a)) and lcm_iters < 200:
                maxa = int(np.max(a))
                nxt = int(np.max(a[a != maxa]))
                a[a == maxa] = nxt
                L = filterbanklength(Ls, a)
                lcm_iters += 1

        # Integer (regular) hops -> 1-D (M,) array, matching audfilters /
        # cqtfilters / greenwoodfilters. The (M,2) rational form is reserved for
        # the genuinely fractional sampling modes below.

    elif sampling == "fractional":
        L = Ls
        a_rat = np.asarray(aprecise, dtype=float).copy()
        if painless:
            # Rational hops are exact, so the cap needs no floor23 rounding.
            a_rat = np.minimum(a_rat, _painless_caps(
                winCell, L, scales, trunc_at, basefc, aprecise, lp_num, M2,
                quantise=False))
        N = np.ceil(Ls / a_rat).astype(int)
        a = np.column_stack([np.full(M2, Ls, dtype=int), N])  # type: ignore[assignment]

    elif sampling == "fractionaluniform":
        L = Ls
        if lowpass_at_zero:
            aprecise[1:] = np.min(aprecise[1:])  # type: ignore[assignment]
        else:
            aprecise[:] = np.min(aprecise)  # type: ignore[assignment]
        a_rat = np.asarray(aprecise, dtype=float).copy()
        if painless:
            caps = _painless_caps(winCell, L, scales, trunc_at, basefc,
                                  aprecise, lp_num, M2, quantise=False)
            # "uniform" is the point of this mode: one hop for every wavelet
            # channel, so the cap has to be the tightest one, not per-channel.
            if lowpass_at_zero:
                a_rat[0] = min(a_rat[0], caps[0])
                a_rat[1:] = min(float(np.min(a_rat[1:])), float(np.min(caps[1:])))
            else:
                a_rat[:] = min(float(np.min(a_rat)), float(np.min(caps)))
        N = np.ceil(Ls / a_rat).astype(int)
        a = np.column_stack([np.full(M2, Ls, dtype=int), N])  # type: ignore[assignment]

    elif sampling == "uniform":
        a_painless = max(1, int(math.floor(np.min(aprecise))))
        if painless:
            # `aprecise` is the mother wavelet's natural *time*-domain
            # subsampling; the painless condition is a statement about
            # *frequency* support, and for a heavy-tailed wavelet at
            # trunc_at = 1e-5 the two differ by an order of magnitude.  Solve
            # for L once with the uncapped hop, cap, then re-solve.
            for _ in range(3):
                L_try = filterbanklength(Ls, a_painless)
                caps = _painless_caps(winCell, L_try, scales, trunc_at, basefc,
                                      aprecise, lp_num, M2)
                a_next = max(1, min(a_painless, int(np.min(caps))))
                if a_next == a_painless:
                    break
                a_painless = a_next
        if hop_ms is not None:
            a_val = max(1, int(round(hop_ms / 1000.0 * fs)))
            if a_val > a_painless:
                warnings.warn(
                    f"waveletfilters: uniform hop {a_val} (hop_ms={hop_ms}) exceeds the "
                    f"painless limit ({a_painless}); the wide high-frequency filters are "
                    f"undersampled. The bank stays invertible via ifilterbankiter but is "
                    f"ill-conditioned (expect many CG iterations) -- add scales for better "
                    f"conditioning, or use gabfilters for a well-conditioned uniform-hop "
                    f"(STFT) analysis.",
                    stacklevel=2)
        else:
            a_val = a_painless
        L = filterbanklength(Ls, a_val)
        a = np.full(M2, a_val, dtype=int)  # 1-D (M,): integer hops

    else:
        raise ValueError(f"Unknown sampling mode {sampling!r}")

    # Expand a to fractional form
    afull = _comp_filterbank_a(a, M2)

    # Delay vector
    if callable(delay):
        delayvec = np.array([delay(kk, afull[kk, 0] / afull[kk, 1])
                             for kk in range(M2)])
    elif np.ndim(delay) == 0:
        delayvec = np.full(M2, int(delay))
    else:
        delayvec = np.atleast_1d(np.asarray(delay)).ravel()
        if len(delayvec) < M2:
            delayvec = np.pad(delayvec, (0, M2 - len(delayvec)))

    # Scaling factors
    scal = np.sqrt(afull[:, 0] / afull[:, 1])

    if freqrange == "real" and lowpass_at_zero:
        scal[0] = scal[0] / math.sqrt(2)
    elif freqrange == "complex":
        # Mirror onto the negative frequencies. Stack along the channel axis
        # whether ``a`` is 1-D (integer hops) or 2-D (fractional [num,den]).
        _stack = np.vstack if np.asarray(a).ndim == 2 else np.concatenate
        if lowpass_at_zero:
            a = _stack([a, np.flipud(a[1:])])  # type: ignore[assignment]
            scal = np.concatenate([scal, np.flipud(scal[1:])])  # type: ignore[assignment]
            delayvec = np.concatenate([delayvec, np.flipud(delayvec[1:])])  # type: ignore[assignment]
        else:
            a = _stack([a, np.flipud(a)])  # type: ignore[assignment]
            scal = np.concatenate([scal, np.flipud(scal)])  # type: ignore[assignment]
            delayvec = np.concatenate([delayvec, np.flipud(delayvec)])  # type: ignore[assignment]

    a_new = a.copy()  # type: ignore[misc]

    # Adjust for redtar
    if redtar is not None:
        if a.ndim == 2 and a.shape[1] == 2:  # type: ignore[misc]
            a_old = a[:, 0].astype(float) / a[:, 1].astype(float)
        else:
            a_old = a.ravel().astype(float)

        if freqrange != "real":
            org_red = np.sum(1.0 / a_old)
        elif lowpass_at_zero:
            org_red = 1.0 / a_old[0] + np.sum(2.0 / a_old[1:])
        else:
            org_red = np.sum(2.0 / a_old)

        # Clamp at 1: `np.floor` of an aggressive ratio produced a = 0, and
        # `filterbank` then raised "negative dimensions are not allowed".
        a_new = np.maximum(1, np.floor(a.astype(float) * org_red / redtar).astype(int))

        if callable(delay):
            delayvec = np.array([delay(kk, float(np.asarray(a_new[kk]).ravel()[0]))
                                 for kk in range(M2)])
            if freqrange == "complex":
                if lowpass_at_zero:
                    delayvec = np.concatenate([delayvec, np.flipud(delayvec[1:])])
                else:
                    delayvec = np.concatenate([delayvec, np.flipud(delayvec)])

        if sampling != "uniform":
            # `N_new`, not `N_old`.  Until v0.1.1 this re-encoded the *original*
            # hops in fractional form, so `redtar` had no effect at all on any
            # non-uniform sampling mode: redtar=0.5 and redtar=50 produced
            # bit-identical filters and identical redundancy.
            N_new = np.ceil(L / a_new.ravel().astype(float)).astype(int)
            a_new = np.column_stack([np.full(len(N_new), L, dtype=int), N_new])  # type: ignore[assignment]
        else:
            L = filterbanklength(L, a_new)

    # Retrieve the wavelets
    _frange = "positive"
    if freqrange == "analytic":
        _frange = "analytic"

    gout, info = freqwavelet(
        winCell, L, scales,
        output_format="asfreqfilter",
        efsuppthr=trunc_at,
        basefc=0.1,
        scal=scal[lp_num:M2],
        delay=delayvec[lp_num:M2].astype(int),
        norm=norm,
        freqrange=_frange,
    )

    if not isinstance(gout, list):
        gout = [gout]

    # Bin geometry of the wavelet channels at this L, captured before the
    # lowpass/highpass merges renumber ``info``.  ``freqwavelet`` stores
    # ``foff = ceil(L/fs * flo)`` and a response of ``floor(L/fs * fhi) - foff``
    # bins, so ``info["fsupp"]`` (= that count + 1) is one more than the number
    # of bins the channel actually occupies.  See the admissibility block below.
    _wav_A = np.asarray(info["foff"], dtype=int).ravel().copy()
    _wav_nbins = np.asarray(info["fsupp"], dtype=int).ravel().copy() - 1
    _wav_fsupp_raw = np.asarray(info["fsupp"], dtype=float).ravel().copy()
    _wav_fc_norm = np.asarray(info["fc"], dtype=float).ravel().copy()

    # cool_frames extension (no LTFAT equivalent): build a Nyquist highpass complement
    # from the pristine wavelet channels so a real bank covers [0, Nyquist] and
    # is invertible (A>0). LTFAT's waveletfilters leaves this band uncovered and
    # is documented as not guaranteed invertible. Built here, appended below.
    _ghigh = None
    _infohigh = None
    _a_hp = None
    want_highpass = (highpass != "none" and freqrange == "real"
                     and lowpass != "none" and len(gout) >= 1)
    if want_highpass:
        wfc = np.asarray(info["fc"], dtype=float).ravel()
        wfsupp = np.asarray(info["fsupp"], dtype=float).ravel()
        jmax = int(np.argmax(wfc))
        a_wav = a_new[lp_num:, :] if a_new.ndim == 2 else a_new[lp_num:].reshape(-1, 1)
        scal_hp = float(scal[lp_num + jmax]) / math.sqrt(2.0)
        try:
            _ghigh, _infohigh = _wavelet_highpass(
                list(gout), a_wav, L, wfc[jmax], wfsupp[jmax], scal_hp,
                min_win=min_win,
            )
            wrows = a_new[lp_num:, :] if a_new.ndim == 2 else a_new[lp_num:].reshape(-1, 1)
            if wrows.shape[1] == 2:
                jhop = int(np.argmin(wrows[:, 0] / wrows[:, 1]))
            else:
                jhop = int(np.argmin(wrows[:, 0]))
            _a_hp = wrows[jhop:jhop + 1, :]
        except Exception:
            _ghigh = None        # never let the complement break construction

    if freqrange == "complex":
        gout_neg, info_neg = freqwavelet(
            winCell, L, -np.flipud(scales),
            output_format="asfreqfilter",
            efsuppthr=trunc_at,
            basefc=0.1,
            freqrange="negative",
            scal=scal[M2:M2 + M] if len(scal) > M2 else scal[lp_num:M2],
            delay=delayvec[M2:M2 + M].astype(int) if len(delayvec) > M2 else delayvec[lp_num:M2].astype(int),
            norm=norm,
        )
        if not isinstance(gout_neg, list):
            gout_neg = [gout_neg]
        gout = gout + gout_neg
        info = _merge_info(info_neg, info, prepend=False)

    # Generate lowpass filters
    if lowpass == "single":
        if len(scales_sorted) < 4:
            raise ValueError("Lowpass generation requires at least 4 scales")
        lp_bw = 0.2 / scales_sorted[3]
        taper_ratio = 1 - scales_sorted[3] / scales_sorted[1]
        # Pass only the wavelet filters and their subsampling rates (skip lowpass row)
        a_wavelet = a_new[1:] if a_new.ndim == 1 else a_new[1:, :]
        glow, infolow = _wavelet_lowpass(
            gout, a_wavelet, L, lp_bw, taper_ratio,
            scal[0], freqrange == "real", min_win=min_win,
        )
        gout = [glow] + gout
        info = _merge_info(info, infolow, prepend=True)

    elif lowpass == "repeat":
        if len(scales_sorted) < 2:
            raise ValueError("Lowpass generation requires at least 2 scales")
        glow_list, infolow = _wavelet_lowpass_repeat(
            winCell, scales_sorted[:2], L, lp_num,
            lowpass_at_zero, scal[:lp_num], trunc_at, norm
        )
        gout = glow_list + gout
        info = _merge_info(info, infolow, prepend=True)

        if freqrange == "complex":
            ghigh_list, infohigh = _wavelet_lowpass_repeat(
                winCell, -scales_sorted[:2], L, lp_num,
                lowpass_at_zero, scal[-lp_num:][::-1], trunc_at, norm
            )
            gout = gout + ghigh_list
            info = _merge_info(info, infohigh, prepend=False)

    # Append the Nyquist highpass complement (cool_frames extension; real banks only).
    if _ghigh is not None:
        gout = gout + [_ghigh]
        info = _merge_info(info, _infohigh, prepend=False)
        if a_new.ndim == 2:
            a_new = np.vstack([a_new, _a_hp])
        else:
            a_new = np.concatenate([a_new, _a_hp.ravel()])

    # Apply delays.  The loop used to stop at `lp_num`, which excludes the
    # complement highpass appended above — so `delay=5` produced descriptor
    # delays [5, 5, 5, 5, 5, 5, 5, 0] and that last channel was left
    # un-delayed relative to the rest of the bank.
    for kk in range(min(len(gout), len(delayvec))):
        gout[kk]["delay"] = int(delayvec[kk])
    for kk in range(len(delayvec), len(gout)):
        # Channels appended after `delayvec` was built (the complement
        # highpass) take the same delay as the rest of the bank.
        gout[kk]["delay"] = int(delayvec[-1]) if len(delayvec) else 0

    # ── Admissibility ────────────────────────────────────────────────────
    # Announce a non-frame geometry here, where the parameters were chosen,
    # rather than letting it surface later as an all-zero dual.
    #
    # Geometry.  A wavelet of scale s is a dilation of the mother wavelet, so
    # its effective support edges are a *fixed multiple* of its centre
    # frequency -- the dilation cancels:
    #
    #     fc_m  = (fs/2) * basefc / s_m
    #     f_lo  = max(0,  r0 * fc_m),   r0 = fsupp_[0] / peakpos
    #     f_hi  = min(fs, r4 * fc_m),   r4 = fsupp_[4] / peakpos
    #
    # with r0, r4 the support ratios of the mother wavelet at ``trunc_at``.
    # freqwavelet then occupies the bins
    #
    #     A_m = ceil(L/fs * f_lo)  ...  B_m = floor(L/fs * f_hi) - 1
    #
    # which is what ``info["foff"]`` and ``info["fsupp"] - 1`` record, so we
    # read them off directly rather than recomputing the rounding.  Endpoint
    # bins are alive (|H| = trunc_at * peak), hence ``window="rect"``.
    #
    # ``_interval_linear`` can only build odd-width intervals, so an
    # even-width channel is handed over as two overlapping odd ones whose
    # union is exactly [A, B]; rounding it up to one odd interval instead
    # would over-cover the bin above B and mispredict a bank whose only gap
    # is that single bin.
    #
    # The two complements are Hann-taper prototypes of
    # ``odd(max(min_win, round(L*bw)))`` bins centred on 0 and on fs/2.  The
    # taper does not vanish at its endpoints (it only reaches half height
    # there), so all of those bins are alive too -- except when the taper
    # ratio degenerates to 0 and ``_make_direct_filter`` falls back to a plain
    # Hann, which does have a dead bin at each end.  The ``sqrt(S_max - S)``
    # factor can null further bins, but only where the inner channels already
    # attain the maximum, i.e. only where they cover the bin themselves.
    _admissible = None
    _fsupp_inner_hz = _wav_nbins.astype(float) * fs / L
    _fsupp_dc_hz = 0.0
    _fsupp_nyq_hz = 0.0
    if freqrange == "real" and lowpass == "single" and _ghigh is not None:
        from ..diagnostics.admissibility import check_admissible

        _fc_pred: list[float] = []
        _fsupp_pred: list[float] = []
        for _A, _W in zip(_wav_A.tolist(), _wav_nbins.tolist()):
            for _f, _s in _bins_to_linear(int(_A), int(_W), fs, L):
                _fc_pred.append(_f)
                _fsupp_pred.append(_s)

        # DC complement: bandwidth lp_bw = 0.2 / scales_sorted[3] in the
        # fs = 2 normalised convention, i.e. bw/fs = lp_bw.
        _lp_bw = 0.2 / scales_sorted[3]
        _taper_dc = 1 - scales_sorted[3] / scales_sorted[1]
        _W_dc = max(min_win, round(L * _lp_bw))
        _W_dc = _W_dc if _W_dc % 2 else _W_dc + 1
        if _taper_dc <= 0:                      # plain Hann -> 2 dead bins
            _W_dc -= 2
        _fsupp_dc_hz = _W_dc * fs / L

        # Nyquist complement: same prototype, bandwidth 2*(1 - fc_last) in the
        # fs = 2 convention, centred on fs/2.
        _jmax = int(np.argmax(_wav_fc_norm))
        _fsupp_hp, _taper_hp = edge_params_from_geometry(
            float(_wav_fc_norm[_jmax]), float(_wav_fsupp_raw[_jmax]), 2.0,
            target="nyquist")
        _W_hp = max(min_win, round(L * _fsupp_hp / 2.0))
        _W_hp = _W_hp if _W_hp % 2 else _W_hp + 1
        if _taper_hp <= 0:                      # plain Hann -> 2 dead bins
            _W_hp -= 2
        _fsupp_nyq_hz = _W_hp * fs / L

        _admissible = check_admissible(
            np.asarray(_fc_pred, dtype=float),
            np.asarray(_fsupp_pred, dtype=float),
            fs=fs, L=int(L),
            fsupp_dc=_fsupp_dc_hz, fsupp_nyq=_fsupp_nyq_hz,
            min_win=1, window="rect", designer="waveletfilters")
    # Any other channel layout -- lowpass='none' or 'repeat', a two-sided
    # (complex) bank, or a bank built without the Nyquist complement -- has
    # DC/Nyquist edges the closed-form predictor cannot express (it always
    # assumes one complement centred on each).  Rather than return no verdict
    # at all -- `lowpass='none'` leaves 79 DFT bins uncovered at the default
    # settings and used to report `admissible=None`, which reads as "fine" --
    # fall back to measuring the realised diagonal response.  That is O(L*M)
    # and exact for the question being asked (is any bin covered by nothing),
    # it just cannot say *which parameter* to change, which is what the
    # closed-form predictor is for.
    if _admissible is None:
        from ..filterbanks._frame import filterbankresponse

        _resp = filterbankresponse(gout, _comp_filterbank_a(a_new, len(gout)),
                                   int(L), real=(freqrange == "real"))
        _dead = np.flatnonzero(_resp <= 1e-12 * max(float(np.max(_resp)), 1e-300))
        _admissible = {
            "is_frame": bool(_dead.size == 0),
            "first_hole_bin": int(_dead[0]) if _dead.size else None,
            "n_hole_bins": int(_dead.size),
            "source": "measured",
        }
        if _dead.size:
            warnings.warn(
                f"waveletfilters: this geometry is not a frame. {_dead.size} "
                f"DFT bins are covered by no filter, the first at bin "
                f"{int(_dead[0])} (~{_dead[0] * fs / L:.1f} Hz), so the lower "
                f"frame bound is 0 and the bank is not invertible. "
                f"lowpass='single' (with highpass='auto') covers both edges.",
                stacklevel=2,
            )

    info["startindex"] = lp_num
    info["designer"] = "waveletfilters"
    # Hz, matching the other designers.  NOTE: the pre-existing
    # ``info["fsupp"]`` stays in DFT bins (freqwavelet's own convention).
    info["fsupp_inner"] = _fsupp_inner_hz
    info["fsupp_dc"] = float(_fsupp_dc_hz)
    info["fsupp_nyq"] = float(_fsupp_nyq_hz)
    info["admissible"] = _admissible

    # ── Painless condition, measured on the bank that is actually returned ──
    # `_painless_caps` enforces the inequality mid-design, but the lowpass and
    # Nyquist complements are appended afterwards and `redtar` rewrites every
    # hop, so the guarantee has to be re-checked here rather than assumed.
    #
    # This is reported, not enforced: `painless=False` is a legitimate choice
    # for analysis-only work.  What is not legitimate is finding out about it
    # from a reconstruction that silently loses three quarters of the signal.
    if painless and redtar is None:
        # Not under `redtar`.  There the caller has named the redundancy they
        # want and `a_new` has already been rewritten to hit it; repairing the
        # hops afterwards would quietly overrule that.  It overruled it badly,
        # too: on `sampling='uniform'` the repair pulled the wide channels back
        # down and made redundancy non-monotone in `redtar`
        # (0.5 -> 18.0, 2.0 -> 8.9, 20.0 -> 13.6).  A redundancy target and the
        # painless condition are competing requests; the explicit one wins, and
        # the check below reports what it cost.
        _repair_complement_hops(gout, a_new, int(L))
    _pl_ratio, _pl_bad = _painless_ratio(gout, a_new, int(L))
    info["painless"] = bool(_pl_bad == 0)
    info["painless_ratio"] = float(_pl_ratio)
    if _pl_bad:
        warnings.warn(
            f"waveletfilters: {_pl_bad} of {len(gout)} channels violate the "
            f"painless condition (worst a*W/L = {_pl_ratio:.3g}, needs <= 1). "
            f"The frame operator is not diagonal, so filterbankdual and "
            f"filterbanktight are approximate here and reconstruction will "
            f"lose energy. Pass painless=True for an exactly invertible bank, "
            f"or invert with ifilterbankiter. info['painless_ratio'] carries "
            f"the measured value.",
            stacklevel=2,
        )

    fc = (fs / 2) * info["fc"]

    return gout, a_new, fc, L, info
