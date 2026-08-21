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
    build_complement_lowpass,
    build_complement_highpass,
    edge_params_from_geometry,
)
from ._freqwavelet import freqwavelet

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
    painless: bool = False,
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
        Hop strategy for ``sampling='regsampling'`` only (no-op otherwise).
        ``False`` (default) keeps the aggressive ``floor23`` + lcm-reduction
        heuristic, which is fast but may NOT form a frame (``A`` can be ~0) for
        some scale sets -- acceptable when a tight frame is not required.
        ``True`` caps each channel's hop at its painless limit
        (``a_m <= floor(L / W_m)``, ``W_m`` = that channel's DFT-bin support)
        so the bank is a guaranteed frame (``A > 0``).
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
                gl_tmp, _ = freqwavelet(
                    winCell, L, scales,
                    output_format="asfreqfilter",
                    efsuppthr=trunc_at,
                    basefc=basefc,
                )
                if not isinstance(gl_tmp, list):
                    gl_tmp = [gl_tmp]
                widths = np.empty(M, dtype=int)
                for j, gm in enumerate(gl_tmp):
                    Hj = np.asarray(gm["H"](L)).ravel()
                    nz = np.flatnonzero(np.abs(Hj) > 1e-10)
                    widths[j] = (nz[-1] - nz[0] + 1) if nz.size else 1
                caps = np.empty(M2, dtype=int)
                for k in range(lp_num):
                    caps[k] = max(1, int(math.floor(L / max(1.0, aprecise[k]))))
                caps[lp_num:M2] = np.maximum(1, (L // np.maximum(widths, 1)))
                # Round each cap DOWN to a floor23 (2^i*3^j) value so the hops
                # share factors and lcm(a) -> L stays small; floor23 of cap is
                # always <= cap, preserving the painless inequality a_m <= cap.
                caps = np.array([max(1, floor23(int(c))) for c in caps], dtype=int)
                a_capped = np.minimum(a, caps)
                if np.array_equal(a_capped, a) and L == filterbanklength(Ls, a_capped):
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
        N = np.ceil(Ls / aprecise).astype(int)
        a = np.column_stack([np.full(M2, Ls, dtype=int), N])  # type: ignore[assignment]

    elif sampling == "fractionaluniform":
        L = Ls
        if lowpass_at_zero:
            aprecise[1:] = np.min(aprecise[1:])  # type: ignore[assignment]
        else:
            aprecise[:] = np.min(aprecise)  # type: ignore[assignment]
        N = np.ceil(Ls / aprecise).astype(int)
        a = np.column_stack([np.full(M2, Ls, dtype=int), N])  # type: ignore[assignment]

    elif sampling == "uniform":
        a_painless = max(1, int(math.floor(np.min(aprecise))))
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

        a_new = np.floor(a.astype(float) * org_red / redtar).astype(int)
        scal_arr = np.full(M2, org_red / redtar)

        if callable(delay):
            delayvec = np.array([delay(kk, float(np.asarray(a_new[kk]).ravel()[0]))
                                 for kk in range(M2)])
            if freqrange == "complex":
                if lowpass_at_zero:
                    delayvec = np.concatenate([delayvec, np.flipud(delayvec[1:])])
                else:
                    delayvec = np.concatenate([delayvec, np.flipud(delayvec)])

        if sampling != "uniform":
            N_old = np.ceil(L / a_old).astype(int)
            N_new = np.ceil(L / a_new.ravel().astype(float)).astype(int)
            a_new = np.column_stack([np.full(len(N_new), L, dtype=int), N_old])  # type: ignore[assignment]
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

    # Apply delays to lowpass filters
    for kk in range(lp_num):
        gout[kk]["delay"] = int(delayvec[kk])

    info["startindex"] = lp_num
    fc = (fs / 2) * info["fc"]

    return gout, a_new, fc, L, info
