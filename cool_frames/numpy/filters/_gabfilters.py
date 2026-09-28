"""
numpy/layer1/_gabfilters.py
===========================
Linearly-spaced Gabor filterbank construction.

Constructs M (or M2 = floor(M/2)+1 for 'real' mode) modulated copies of a
prototype window, equivalent to the DGT/DGTREAL with the time-invariant
phase convention.

MATLAB original
---------------
  layer1/filter_design/gabfilters.m
  utils/legacy/gabor/gabwin.m
  utils/legacy/gabor/dgtlength.m
  layer1/filter_prep/comp_tfrfromwin.m
"""
from __future__ import annotations

import math
import warnings

import numpy as np

from ..core._core import involute

# ---------------------------------------------------------------------------
# dgtlength – smallest admissible DGT length
# ---------------------------------------------------------------------------

def _dgtlength(Ls: int, a: int, M: int) -> int:
    """Compute next admissible DGT length: ``ceil(Ls / lcm(a, M)) * lcm(a, M)``.

    Port of ``dgtlength.m``.
    """
    b = math.lcm(a, M)
    return int(math.ceil(Ls / b) * b)


# ---------------------------------------------------------------------------
# _gabwin – resolve window specification to a numeric vector
# ---------------------------------------------------------------------------

def _gabwin(g, M: int, norm: str = "energy") -> np.ndarray:
    """Resolve a window specification to a numeric vector of length M.

    Handles:
      * string  – passed to ``firwin(name, M, norm=norm)``
      * ndarray – used directly (must be length M or L)
    """
    if isinstance(g, str):
        from ._firwin import firwin as _firwin
        return _firwin(g, M, norm=norm)
    else:
        return np.asarray(g, dtype=float).ravel()


# ---------------------------------------------------------------------------
# _fir2long – zero-pad FIR window to length L  (centred, periodic)
# ---------------------------------------------------------------------------

def _fir2long(g: np.ndarray, L: int) -> np.ndarray:
    """Zero-pad a centred FIR window *g* to length *L*.

    Port of ``fir2long.m``:  the window is assumed to be in DFT ordering
    (DC at index 0).  We keep the first ceil(len(g)/2) and last floor(len(g)/2)
    samples and pad zeros in between.
    """
    Lg = len(g)
    if Lg == L:
        return g.copy()
    if Lg > L:
        raise ValueError(f"fir2long: window length {Lg} > target {L}")

    out = np.zeros(L, dtype=g.dtype)
    # First half (including DC)
    n1 = int(math.ceil(Lg / 2))
    out[:n1] = g[:n1]
    # Second half (negative frequencies)
    n2 = Lg - n1
    if n2 > 0:
        out[L - n2:] = g[n1:]
    return out


# ---------------------------------------------------------------------------
# _winwidthatheight – window width at a given relative height
# ---------------------------------------------------------------------------

def _winwidthatheight(g: np.ndarray, atheight: float) -> float:
    """Width of a symmetric window at a relative height.

    Port of ``winwidthatheight.m`` (nested in ``comp_tfrfromwin.m``).

    The original assumed DFT ordering — peak at index 0 — and read
    ``g[0 : gl//2 + 1]``.  That holds for the time-domain prototype
    ``gabfilters`` passes in, but *not* for the frequency responses the filter
    designers store, which are peak-centred within their compact support.  For
    such an array the first sample is already below threshold, so the crossing
    indices were pinned at 0 and 1 regardless of the window: ``w/gl`` collapsed
    to 1 and ``compute_tfr_from_filters`` returned the same ``gamma`` for a
    Hann, a rectangle, a triangle and a three-bin needle (11597.8 for all four,
    where a Hann should give ~2891).

    Rolling the peak to index 0 first makes the routine correct for both
    layouts and leaves the DFT-ordered case untouched (its argmax is already 0).
    """
    g = np.asarray(g)
    gl = len(g)
    if gl == 0:
        return 0.0

    peak = int(np.argmax(g))
    if peak != 0:
        g = np.roll(g, -peak)

    gmax = float(np.max(g))
    fracofmax = gmax * atheight  # threshold value

    # First half of window (peak up to the far side)
    half = g[: gl // 2 + 1]

    # Find where the window crosses the threshold
    exact = np.where(half == fracofmax)[0]
    if len(exact) > 0:
        return 2.0 * float(exact[0])

    # Interpolate between last-above and first-below
    above = np.where(half > fracofmax)[0]
    below = np.where(half < fracofmax)[0]

    if len(below) == 0:
        return float(gl)

    ind1 = int(above[-1]) if len(above) > 0 else 0
    ind2 = int(below[0])
    denom = half[ind1] - half[ind2]
    if abs(denom) < 1e-30:
        return 2.0 * float(ind1)
    rest = 1.0 - (fracofmax - half[ind2]) / denom
    return 2.0 * (ind1 + rest)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# _comp_tfrfromwin – time-frequency ratio from a window
# ---------------------------------------------------------------------------

def _comp_tfrfromwin(g: np.ndarray, atheight: float | None = None) -> float:
    """Compute the time-frequency ratio ``gamma / L`` of a window.

    Port of ``comp_tfrfromwin.m``:
        gl = winwidthatheight(g, 1e-10)
        w  = winwidthatheight(g, atheight)
        Cg = -pi/4 * (w/gl)^2 / log(atheight)
        gamma = Cg * gl^2
        tfr(L) = gamma / L

    Returns ``gamma`` (multiply by 1/L to get the tfr for a specific L).
    """
    if atheight is None:
        atheight = 10 ** (-3.0 / 10.0)   # ~0.5012, half-power height

    gl = _winwidthatheight(g, 1e-10)
    w = _winwidthatheight(g, atheight)

    if gl == 0:
        return 0.0

    Cg = -math.pi / 4.0 * (w / gl) ** 2 / math.log(atheight)
    gamma = Cg * gl ** 2
    return gamma


# ---------------------------------------------------------------------------
# gabfilters – public API
# ---------------------------------------------------------------------------

def _gab_bank(g0: np.ndarray, a: int, M: int, L: int, fs, real: bool,
              windowaxis: str = "time") -> list[dict]:
    """The filter descriptors of a Gabor bank with numeric window ``g0``.

    ``g0`` is used as given (no normalisation), so the same construction
    builds a bank from a canonical dual or tight window.  Every channel
    stores the whole ``L``-point transform of the window (``windowaxis='time'``)
    or the window itself (``'freq'``).  The interior channels share one
    read-only response and the edge channels another, so a bank costs
    ``2 L`` complex values rather than ``M2 L``; nothing in the package
    writes to a filter's ``H`` in place.

    In ``'time'`` mode each descriptor also carries ``"gabor"``: the window,
    ``a``, ``M``, ``real`` and the channel index.  The frame algebra reads it
    to use the Gabor closed forms (``gabdual``, ``gabtight``,
    ``gabframebounds``), which are exact and need no ``L``-length blocks.
    """
    g0 = np.asarray(g0)
    if windowaxis == "time":
        gnum = np.fft.fftshift(np.fft.fft(involute(_fir2long(g0, L))))
    else:
        gnum = np.conj(np.fft.fftshift(g0))
    Lg = len(gnum)
    half_lo = Lg // 2
    M2 = M // 2 + 1 if real else M

    # In real (single-sided) mode the DC and Nyquist channels have no
    # conjugate partner, so `ifilterbank(..., real=True)`'s 2*real(ifft) fold
    # double-counts them.  Every other designer in the package compensates with
    # a 1/sqrt(2) on the two edge channels (see `_design.py`, `_cqtfilters.py`,
    # `_greenwoodfilters.py`, `_waveletfilters.py`, `_warpedfilters_design.py`);
    # gabfilters did not, which left a 4x-overlap Hann DGT — an exactly tight
    # frame — reading kappa = 1.667 with a 67 % response spike at DC and
    # Nyquist.
    edge_scal = np.ones(M2, dtype=float)
    if real and M2 > 1:
        edge_scal[0] /= math.sqrt(2.0)
        # The top channel is the Nyquist bin only when M is even; for odd M the
        # single-sided range stops just short of it and needs no correction.
        if M % 2 == 0:
            edge_scal[-1] /= math.sqrt(2.0)

    shared: dict[float, np.ndarray] = {}

    def _resp(scale: float) -> np.ndarray:
        if scale not in shared:
            h = gnum * scale if scale != 1.0 else gnum.copy()
            h.flags.writeable = False
            shared[scale] = h
        return shared[scale]

    win = None
    if windowaxis == "time":
        win = np.array(g0, copy=True)
        win.flags.writeable = False
    gout = []
    for kk in range(M2):
        filt = {
            "H": _resp(float(edge_scal[kk])),
            "foff": int(kk * L / M - half_lo),
            "realonly": 0,
            "delay": 0,
            "fs": fs,
        }
        if win is not None:
            filt["gabor"] = {"window": win, "a": int(a), "M": int(M),
                             "real": bool(real), "k": kk}
        gout.append(filt)
    return gout


def gabfilters(fs: float, Ls: int, *,
               window="hann",
               window_ms: float | None = None,
               hop_ms: float | None = None,
               M: int | None = None,
               a: int | None = None,
               real: bool = True,
               norm: str = "energy",
               windowaxis: str = "time") -> tuple[list[dict], np.ndarray, np.ndarray, int, dict]:
    """Construct a uniform (linearly-spaced) Gabor / STFT filterbank.

    Builds ``M2 = floor(M/2)+1`` (real mode, default) or ``M`` (complex mode)
    frequency-shifted copies of a prototype window, equivalent to the DGT with
    the time-invariant phase convention.

    .. note:: **Consistent designer interface.**
       Like ``audfilters`` / ``cqtfilters`` / ``greenwoodfilters``, this takes
       ``fs`` (sampling rate) first and returns ``(g, a, fc, L, info)`` with
       ``fc`` in **Hz**.  Time parameters may be given in **milliseconds**
       (``window_ms`` / ``hop_ms``) — the friendly path — or directly in samples
       (``M`` = window length / channels, ``a`` = hop); supply at most one of
       each.  A bare ``gabfilters(fs, Ls)`` uses a sensible default lattice.

    Parameters
    ----------
    fs : float
        Sampling rate (Hz).  Used for the ms↔samples conversion and to report
        ``fc`` in Hz.
    Ls : int
        Signal length (samples).
    window : str or array_like
        Prototype window.  A string is a ``firwin`` name (e.g. ``'hann'``); a
        numeric array is used directly.  (LTFAT called this ``g``.)
    window_ms : float, optional
        Window length / channel spacing in milliseconds (``M = round(window_ms/1000*fs)``).
        Mutually exclusive with ``M``.
    hop_ms : float, optional
        Hop (time step) in milliseconds (``a = round(hop_ms/1000*fs)``).
        Mutually exclusive with ``a``.
    M : int, optional
        Number of frequency channels = window length (samples). If neither
        ``M`` nor ``window_ms`` is given, defaults to a 32 ms window.
    a : int, optional
        Hop size (samples). If neither ``a`` nor ``hop_ms`` is given, defaults
        to ``M//4`` (≈4× redundancy).
    real : bool
        If ``True`` (default), only the non-negative-frequency channels are
        returned (``M2 = floor(M/2)+1`` filters).  Mimics ``dgtreal``.
    norm : str
        Window normalisation: ``'energy'`` (default), ``'1'``, ``'inf'``.
    windowaxis : str
        ``'time'`` (default) or ``'freq'``.

    Notes
    -----
    As in LTFAT, every channel stores the whole ``L``-point transform of the
    window (``M2 * L`` complex values in all), because a time-limited window
    is not band-limited; this is what makes the bank equal to ``dgtreal``
    with the time-invariant phase convention at every ``L``.  For long
    signals use :func:`cool_frames.gabor.dgtreal` directly, which is exact
    and needs no ``L``-length filters.

    Returns
    -------
    gout : list[dict]
        Filter descriptors (keys ``'H'``, ``'foff'``, ``'realonly'``,
        ``'delay'``, ``'fs'``).
    aout : ndarray, shape (Mout,)
        Hop sizes (uniform, ``a`` per channel) — 1-D integer array, consistent
        with the other designers.
    fc : ndarray, shape (Mout,)
        Centre frequencies in **Hz**.
    L : int
        Admissible transform length (``dgtlength(Ls, a, M)``).
    info : dict
        Extra information (``'fc'``, ``'tfr'``).

    Examples
    --------
    >>> from cool_frames.numpy.filters import gabfilters
    >>> g, a, fc, L, info = gabfilters(16000, 16000)              # defaults
    >>> g, a, fc, L, info = gabfilters(16000, 16000, window_ms=32, hop_ms=8)
    """
    fs = float(fs)
    Ls = int(Ls)
    windowaxis = windowaxis.lower()
    if windowaxis not in ("time", "freq"):
        raise ValueError(f"gabfilters: windowaxis must be 'time' or 'freq', got {windowaxis!r}")
    if isinstance(window, (int, float, np.integer, np.floating)):
        raise ValueError(
            "gabfilters: 'window' is a window name (e.g. 'hann') or an array, "
            "not a length; set the window length via window_ms or M.")

    # Resolve M (channels = window length) and a (hop) from ms / explicit / defaults.
    if window_ms is not None:
        if M is not None:
            raise ValueError("gabfilters: pass either window_ms or M, not both.")
        M = int(round(window_ms / 1000.0 * fs))
    if hop_ms is not None:
        if a is not None:
            raise ValueError("gabfilters: pass either hop_ms or a, not both.")
        a = int(round(hop_ms / 1000.0 * fs))
    if M is None:
        M = int(round(0.032 * fs))           # default 32 ms window
    if a is None:
        a = max(1, int(M) // 4)              # default ~4x redundancy
    M = int(M)
    a = max(1, int(a))
    if M < 2:
        raise ValueError(f"gabfilters: need M>=2 channels (got {M}); increase window_ms/M.")
    if a > M:
        warnings.warn(
            f"gabfilters: hop a={a} exceeds M={M} channels (redundancy<1); the bank "
            f"is undersampled and not an invertible frame. Reduce hop_ms/a or "
            f"increase window_ms/M.",
            stacklevel=2)

    g = window   # internal alias; the construction below uses `g`
    L = _dgtlength(Ls, a, M)

    # Resolve window to a numeric vector (length M, energy-normalised)
    g0 = _gabwin(g, M, norm=norm)

    # Centre frequencies: 2*k/M for k = 0, …, M-1  (normalised to [0,2))
    fc_full = 2.0 * np.arange(M) / M
    Mfull = M

    # Truncate for real mode
    if real:
        M2 = M // 2 + 1
        fc_out = fc_full[:M2].copy()
    else:
        M2 = M
        fc_out = fc_full.copy()

    # Return centre frequencies in Hz, consistent with the other designers
    # (audfilters/cqtfilters/greenwoodfilters/waveletfilters). ``fc_full`` is
    # normalised to Nyquist = 1, so scale by fs/2 when fs is known; if fs is not
    # given there is no Hz mapping and the normalised values are returned.
    if fs is not None:
        fc_out = fc_out * (float(fs) / 2.0)

    # ── The stored response is the whole transformed window ──────────────
    # LTFAT's gabfilters.m stores ``gtmp.H = gnum`` -- all ``Lg`` bins -- with
    # ``foff = kk*L/M - floor(Lg/2)``.  A window of M samples is time-limited,
    # so its L-point spectrum is not band-limited: its main lobe alone spans
    # ``~4*L/M`` bins for a Hann window, and its sidelobes reach every bin.
    #
    # Until 2026-09-28 this port kept only ``M`` bins around the peak, on the
    # reasoning that "M nonzero samples concentrate within ~M bins".  That is
    # backwards: the support in bins scales with ``L/M``, not with ``M``.  The
    # kept band was ``M**2/L`` channel spacings wide, so the bank departed
    # from the DGT it documents as ``L`` grew past ``M**2`` and stopped being
    # a frame near ``L = M**2``: at ``M = 480, a = 120, L = 144000`` the
    # magnitudes differed from ``dgtreal`` by 28 % and the bounds read
    # ``(2.53, 3.84)`` against ``(4, 4)``; ``M = 16, a = 4, L = 1024`` -- a
    # tight Hann frame -- was reported as not a frame at all.
    #
    # Storing all ``Lg`` bins restores LTFAT's behaviour: the bank now equals
    # ``dgtreal`` with the time-invariant phase convention at every ``L``.
    # The channels share their response (``_gab_bank``), so the bank costs
    # ``2 L`` complex values, and the frame algebra takes the Gabor closed
    # forms, so its dual and bounds cost no more than ``gabdual``'s.
    # Analysis and synthesis still touch all ``L`` bins per channel; for long
    # signals ``cool_frames.gabor.dgtreal`` is the fast path, as in LTFAT.
    # In ``windowaxis='freq'`` mode ``gnum`` already has ``M`` bins, as in
    # LTFAT, so nothing changes there.
    gout = _gab_bank(g0, a, Mfull, L, fs, real, windowaxis)
    Lg = L if windowaxis == "time" else len(g0)
    Lg_compact = Lg

    # Hop sizes: uniform, a for every channel (1-D integer array)
    aout = np.full(M2, a, dtype=int)

    # No painless warning.  Every other designer warns when its lattice
    # exceeds the painless limit, because a non-uniform bank's dual is then
    # approximate; until the uniform branch of `filterbankdual` existed this
    # one did too (round trip 4.9e-4 at the defaults).  A uniform bank gets
    # its exact canonical dual whatever its lattice.

    # Time-frequency ratio
    gamma = _comp_tfrfromwin(g0)
    tfr = gamma / L if L > 0 else 0.0
    if windowaxis == "freq":
        tfr = 1.0 / tfr if tfr != 0 else 0.0

    # ── Admissibility ────────────────────────────────────────────────────
    # Announce a non-frame geometry here, where the parameters were chosen,
    # rather than letting it surface later as an all-zero dual.
    #
    # windowaxis='time': every channel now stores the full transformed window
    # (``gl = L`` bins), so nothing is uncovered in frequency and the only way
    # to fail is in time.  A window of at most M samples makes the frame
    # operator diagonal in time (the "painless in time" case of Daubechies,
    # Grossmann and Meyer), proportional to the a-periodisation
    #
    #     D(r) = sum_k |g0[r + k*a]|^2 ,   r = 0 .. a-1 ,
    #
    # so the bank is a frame iff ``min D > 0`` and its condition number is
    # ``max D / min D``.  That is exact, not a prediction.
    #
    # windowaxis='freq' stores the window itself as the frequency response,
    # whose live width depends on how many endpoint bins that window zeroes
    # out; we report no verdict there rather than an unvalidated one.
    gl = Lg_compact
    fsupp_hz = float(gl) * fs / L
    fsupp_all = np.full(M2, fsupp_hz, dtype=float)
    fsupp_dc = fsupp_hz
    fsupp_nyq = fsupp_hz

    if windowaxis == "time" and len(g0) <= Mfull:
        # Periodise the window as it actually sits in the length-L signal
        # (``_fir2long`` wraps its second half to the end), not by its index
        # in ``g0``: the positions modulo ``a`` differ unless ``a`` divides M.
        w2 = np.abs(_fir2long(g0, L)) ** 2
        D = w2.reshape(L // a, a).sum(axis=0)
        peak = float(D.max()) if D.size else 0.0
        holes = np.flatnonzero(D <= peak * 1e-12) if peak > 0 else np.arange(a)
        is_frame = bool(peak > 0 and holes.size == 0)
        kappa = float(peak / D.min()) if is_frame else math.inf
        admissible = {
            "is_frame": is_frame,
            "first_hole_bin": None,          # no frequency hole is possible
            "n_hole_bins": 0,
            "first_hole_sample": None if is_frame else int(holes[0]),
            "n_hole_samples": 0 if is_frame else int(holes.size * (L // a)),
            "rho": float(a) / float(len(w2)),
            "kappa_pred": kappa,
            "usable": is_frame,
        }
        if not is_frame:
            from ..diagnostics.admissibility import NotAFrameWarning
            warnings.warn(
                f"gabfilters: this geometry is not a frame. The hop a={a} leaves "
                f"{holes.size} of every {a} samples outside the window's support "
                f"(the first at offset {int(holes[0])}), so the lower frame bound "
                f"is zero. Use a <= the window's nonzero length -- see "
                f"cool_frames.diagnostics.admissibility.",
                NotAFrameWarning,
                stacklevel=2,
            )
    else:
        admissible = None

    info = {
        "fc": fc_out,
        "tfr": tfr,
        "designer": "gabfilters",
        "fsupp": fsupp_all,
        "fsupp_inner": fsupp_all[1:-1],
        "fsupp_dc": float(fsupp_dc),
        "fsupp_nyq": float(fsupp_nyq),
        "admissible": admissible,
    }

    return gout, aout, fc_out, L, info
