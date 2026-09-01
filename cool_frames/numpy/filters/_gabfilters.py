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

def _identity_scaletofreq(u):
    """Gabor channels are uniform in frequency itself: ``g(f) = f``."""
    return u



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

    # Build the prototype frequency response
    if windowaxis == "time":
        # gnum = fftshift(fft(involute(fir2long(g0, L))))
        g_long = _fir2long(g0, L)
        g_inv = involute(g_long)
        gnum = np.fft.fftshift(np.fft.fft(g_inv))
    else:
        # freq mode: gnum = conj(fftshift(g0))
        gnum = np.conj(np.fft.fftshift(g0))

    Lg = len(gnum)

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

    # ── Extract compact prototype of length Lg_compact ────────────────
    # The time-domain prototype has M nonzero samples, so its frequency
    # support is concentrated within ~M bins.  Storing only the compact
    # support (rather than the full L-length FFT) keeps the filter descriptors
    # consistent with blfilter / waveletfilters output.
    #
    # NOTE: this does *not* establish the painless condition, contrary to what
    # this comment claimed before v0.1.1.  Painlessness needs support <= N =
    # L/a, i.e. M <= L/a, which the default lattice (a = M//4) never satisfies:
    # it needs M**2 <= 4L.  A warning is emitted below when it is violated, in
    # line with the other designers.
    #
    # We keep Lg_compact = M bins centred on the peak of gnum (which sits
    # at index Lg//2 after the fftshift above).
    Lg_compact = len(g0)            # = M  (the window length)
    center = Lg // 2
    half_lo = Lg_compact // 2
    half_hi = Lg_compact - half_lo
    gnum_compact = gnum[center - half_lo : center + half_hi].copy()

    # Build filter descriptors.
    #
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
        if Mfull % 2 == 0:
            edge_scal[-1] /= math.sqrt(2.0)

    gout = []
    for kk in range(M2):
        filt = {
            "H": gnum_compact * edge_scal[kk],
            "foff": int(kk * L / Mfull - half_lo),
            "realonly": 0,
            "delay": 0,
            "fs": fs,
        }
        gout.append(filt)

    # Hop sizes: uniform, a for every channel (1-D integer array)
    aout = np.full(M2, a, dtype=int)

    # Painless check.  Every other designer warns when its lattice exceeds the
    # painless limit; gabfilters used to claim (in a comment) that it always
    # satisfied it, and warned only when redundancy dropped below 1.
    _support = len(gnum_compact)
    _N = L / float(a)
    if _support > _N:
        warnings.warn(
            f"gabfilters: filter support ({_support} bins) exceeds the painless "
            f"limit N = L/a = {_N:.0f}, so `filterbankdual`/`filterbanktight` "
            f"return an approximate dual (relative reconstruction error ~1e-4 "
            f"at the defaults, larger for wider windows). The bank is still a "
            f"well-conditioned frame — use `ifilterbankiter(c, g, a, L)` for "
            f"exact reconstruction, or choose a <= L/M for a painless lattice.",
            stacklevel=2,
        )

    # Time-frequency ratio
    gamma = _comp_tfrfromwin(g0)
    tfr = gamma / L if L > 0 else 0.0
    if windowaxis == "freq":
        tfr = 1.0 / tfr if tfr != 0 else 0.0

    # ── Admissibility ────────────────────────────────────────────────────
    # Announce a non-frame geometry here, where the parameters were chosen,
    # rather than letting it surface later as an all-zero dual.
    #
    # Unlike the painless designers, gabfilters builds its prototype in the
    # TIME domain, so its realised frequency support is not a designed
    # bandwidth but the width of the compact block it stores: every channel
    # occupies exactly ``gl = len(gnum_compact)`` DFT bins, whatever the
    # window shape, laid out at
    #
    #     A_k = k*(L/M) - gl//2,   B_k = A_k + gl - 1,   k = 0 .. Mout-1
    #
    # (``L = dgtlength(Ls, a, M)`` is a multiple of M, so ``k*L/M`` is exact).
    # There are no dead endpoint bins: the truncated tails of the transformed
    # window are generically nonzero, and where they do vanish exactly (the
    # Dirichlet zeros at multiples of L/M) the bin carries another channel's
    # peak.  So the bank is a frame iff L/M <= gl.
    #
    # gabfilters' warping coordinate is frequency itself (``g(f) = f``), so
    # the interval is handed to the predictor through the warped hook with
    # ``scaletofreq = identity``; that is the one route that can express an
    # even-width interval exactly (``_interval_linear`` always builds an odd
    # one).  The quarter-bin inset in ``bwmul`` is what makes floor()/ceil()
    # land on A and B for both parities of gl.
    # ``gl == M`` (a named window, or an array of length M) is what makes the
    # "no dead bins" claim above true for *any* window shape: the transformed
    # window vanishes exactly at L-bin offsets that are multiples of L/gl, and
    # with gl == M that lattice is the channel lattice L/M, so every such bin
    # carries some other channel's peak.  A window array of a different length
    # puts the two lattices out of step -- with gl = L, for instance, a
    # periodic Hann has a three-bin spectrum, nothing like its gl bins -- and
    # the dead bins are then window-dependent, so we report no verdict.
    gl = Lg_compact
    fsupp_hz = float(gl) * fs / L
    fsupp_all = np.full(M2, fsupp_hz, dtype=float)
    q = L // Mfull
    nyq_bin = L // 2
    A_top = (M2 - 1) * q - gl // 2
    B_top = A_top + gl - 1
    representable = (gl == Mfull)
    if not real:
        # two-sided bank: no channel is the Nyquist complement.  A gap in a
        # uniform layout repeats every L/M bins, so bin L/2 can never be the
        # *only* hole -- a one-bin stub there is therefore exact.
        last = M2
        W_nyq = 1
    elif B_top >= nyq_bin:
        # the top channel reaches Nyquist: folded about it, it covers
        # [min(A_top, L - B_top), L/2] -- exactly an interval centred on
        # Nyquist, which is how the predictor models the edge.
        last = M2 - 1
        W_nyq = 2 * (nyq_bin - min(A_top, L - B_top)) + 1
    elif B_top <= nyq_bin - 2:
        # the top channel stops short of Nyquist: it is an ordinary inner
        # channel and [B_top+1, L/2-1] is uncovered, which the one-bin stub
        # still reports as a hole.
        last = M2
        W_nyq = 1
    else:
        # B_top == L/2 - 1: bin L/2 is uncovered but the predictor always
        # covers it (it assumes a Nyquist complement), so this hole is
        # invisible to it.  With gl == M that only happens for odd M at
        # L/M = M+1 (L even) or M+2 (L odd) -- both of which have L/M > gl,
        # so the gaps *between* the channels give the right verdict anyway.
        last = M2
        W_nyq = 1
    A_inner = np.arange(1, last, dtype=float) * q - gl // 2
    u_inner = (A_inner + gl / 2.0) * fs / L
    bwmul = (gl / 2.0 - 0.25) * fs / L
    fsupp_dc = fsupp_hz               # folds to [0, gl//2], the exact coverage
    fsupp_nyq = W_nyq * fs / L

    if windowaxis == "time" and representable:
        from ..diagnostics.admissibility import check_admissible

        admissible = check_admissible(
            None, None, fs=fs, L=int(L),
            fsupp_dc=fsupp_dc, fsupp_nyq=fsupp_nyq,
            warped=(u_inner, _identity_scaletofreq, bwmul),
            min_win=1, window="rect", designer="gabfilters")
    else:
        # Two layouts we cannot express, so we report no verdict rather than
        # an unvalidated one: windowaxis='freq', which stores the window
        # itself as the frequency response so the live width is gl minus
        # however many endpoint bins that particular window zeroes out
        # (window- and parity-dependent), and a window array whose length is
        # not M (see above).
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
