"""
numpy/layer1/_firwin.py
=======================
Window functions for use in FIR filter design (port of firwin.m).

All windows are evaluated using the WPE (Whole-Point Even) convention:
  n = 0, 1, …, M-1  →  x = n / M  ∈ [0, 1)
The peak is at index 0 (zero-frequency sample at the start).
This matches LTFAT's zero-phase window convention.

MATLAB original: layer1/window_functions/firwin.m
"""
from __future__ import annotations

import math

import numpy as np

from ..core._norm import normalize_window

# ---------------------------------------------------------------------------
# Helper: normalised sample positions
# ---------------------------------------------------------------------------

def _x(M: int) -> np.ndarray:
    """Sample positions for a window of length *M* in WPE convention:
    ``n / M`` where n ∈ [0, M). Peak is at n=0."""
    n = np.arange(M, dtype=float)
    return n / M


def _xc(x: np.ndarray) -> np.ndarray:
    """WPE positions folded to [-0.5, 0.5): the signed distance from the peak
    (index 0), so that a shape defined around 0 is sampled whole-point even."""
    x = np.asarray(x, dtype=float)
    return np.where(x < 0.5, x, x - 1.0)


def _truncgauss_percent(name: str) -> float:
    """The height (in %) at which 'truncgauss' is cut: 1 by default, or the
    number appended to the name ('truncgauss20'), as in LTFAT."""
    tail = name[len("truncgauss"):]
    if not tail:
        return 1.0
    try:
        pct = float(tail)
    except ValueError:
        raise ValueError(f"firwin: cannot read a percentage from {name!r}") from None
    if not 0.0 < pct < 100.0:
        raise ValueError(f"firwin: the truncgauss percentage must lie in (0, 100); got {pct}")
    return pct


# Shapes that are frequency responses or impulse responses with their own
# bandwidth, not zero-phase windows: firwin has none of them (neither has
# LTFAT's firwin).  Until 2026-09-26 it returned a Gaussian under the name
# 'roex' and a causal envelope with a spurious unit first sample under
# 'gammatone', and audfilters(window='gammatone') built filters 3750 times too
# wide from it.
_NOT_WINDOWS = {
    "gammatone": "freqwin('gammatone', L, bw) for its frequency response, or "
                 "gammatonefir(fc, fs) for its impulse response",
    "roex": "freqwin('roex', L, bw) for the rounded-exponential frequency response",
}


def _not_a_window(name: str) -> None:
    if name in _NOT_WINDOWS:
        raise ValueError(f"firwin: {name!r} is not a window shape; use {_NOT_WINDOWS[name]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def firwin(name: str, M: int, norm: str = "inf", *,
           beta: float | None = None, shift: float = 0.0) -> np.ndarray:
    """Evaluate a named symmetric FIR window of length *M*.

    The returned vector uses the WPE (Whole-Point Even) convention where
    the peak is at index 0. This is LTFAT's zero-phase window ordering.

    Parameters
    ----------
    name : str
        Window type.  Supported: ``'hann'``, ``'hanning'``, ``'sine'``,
        ``'sqrthann'``, ``'cosine'``, ``'hamming'``, ``'blackman'``,
        ``'blackman2'``, ``'rect'``, ``'square'``, ``'tria'``,
        ``'bartlett'``, ``'sqrttria'``, ``'itersine'``, ``'ogg'``,
        ``'nuttall'``, ``'nuttall10'``, ``'nuttall01'``, ``'nuttall11'``,
        ``'nuttall20'``, ``'gauss'``, ``'butterworth'``, ``'truncgauss'``
        (cut at 1 % of its height, or at the percentage appended to the name,
        e.g. ``'truncgauss20'``, as in LTFAT) and ``'kaiser'``.  Every window
        peaks at index 0 and is whole-point even.  ``'gammatone'`` and
        ``'roex'`` are not window shapes and raise ``ValueError`` naming
        :func:`freqwin` / :func:`gammatonefir`.
    M : int
        Window length (number of samples).
    norm : str
        Normalisation: ``'inf'`` / ``'peak'`` (default, unit-peak, no scaling),
        ``'energy'`` (``sqrt(L)`` scaling), ``'1'`` / ``'area'``
        (``L`` scaling).
    shift : float
        Evaluate the window shifted by this many samples (fractional), as
        LTFAT's ``firwin(..., 'shift', s)``: sample *n* takes the shape's value
        at ``(n + shift) / M`` from the peak; samples pushed past half a
        window from the peak are zero.  ``blfilter(..., pedantic=True)`` uses
        it for the sub-bin centre frequency.  Default 0.
    beta : float, optional
        Kaiser-Bessel shape parameter; **required** when ``name='kaiser'``
        (0 -> rectangular, larger -> narrower mainlobe).  Ignored otherwise.
        Folds in the former ``firkaiser``.

    Returns
    -------
    g : (M,) float64 array
        Window vector.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import firwin
    >>> w = firwin('hann', 512)
    >>> w.shape
    (512,)
    >>> float(w[0])
    1.0
    >>> w = firwin('blackman', 256, norm='energy')
    >>> round(float(np.sqrt(np.sum(w**2))), 12)  # 'energy' means unit L2
    1.0
    """
    name = name.lower().strip()
    _not_a_window(name)
    if shift:
        if name == "kaiser":
            raise NotImplementedError("firwin: shift is not implemented for 'kaiser'")
        xs = _xc(_x(M)) + shift / M
        g = firwin_eval(name, np.mod(xs, 1.0)) * (np.abs(xs) <= 0.5)
        return _apply_norm(np.maximum(g, 0.0), norm, M)
    x = _x(M)
    n = np.arange(M, dtype=float)

    # --- window shape ---
    if name in ("hann", "hanning", "nuttall10"):
        # WPE hann: 0.5 * (1 + cos(2*pi*n/M))
        g = 0.5 * (1.0 + np.cos(2.0 * np.pi * x))

    elif name in ("sine", "sqrthann", "cosine"):
        # Sine window is sqrt of hann window
        g = np.sqrt(np.maximum(0.0, 0.5 * (1.0 + np.cos(2.0 * np.pi * x))))

    elif name in ("hamming",):
        g = 0.54 + 0.46 * np.cos(2 * np.pi * x)

    elif name == "blackman":
        g = 0.42 + 0.5 * np.cos(2 * np.pi * x) + 0.08 * np.cos(4 * np.pi * x)

    elif name == "blackman2":
        g = (7938 / 18608
             + 9240 / 18608 * np.cos(2 * np.pi * x)
             + 1430 / 18608 * np.cos(4 * np.pi * x))

    elif name in ("rect", "square"):
        g = np.ones(M)
        # For even M, set Nyquist bin to 0
        if M % 2 == 0:
            g[M // 2] = 0.0

    elif name in ("tria", "bartlett"):
        # Triangular window: 1 - 2*min(n, M-n) / M
        g = 1.0 - 2.0 * np.minimum(n, M - n) / M

    elif name == "sqrttria":
        g = np.sqrt(np.maximum(1.0 - 2.0 * np.minimum(n, M - n) / M, 0.0))

    elif name in ("itersine", "ogg"):
        # itersine: sin(pi/2 * cos(pi * x)^2) where x = n/M
        g = np.sin(np.pi / 2 * np.cos(np.pi * x) ** 2)

    elif name == "nuttall":
        a = [0.355768, 0.487396, 0.144232, 0.012604]
        g = sum(a[k] * np.cos(2 * np.pi * k * x) for k in range(4))  # type: ignore[assignment]

    elif name == "nuttall01":
        g = 0.53836 + 0.46164 * np.cos(2 * np.pi * x)

    elif name == "nuttall11":
        g = 0.40897 + 0.5 * np.cos(2 * np.pi * x) + 0.09103 * np.cos(4 * np.pi * x)

    elif name == "nuttall20":
        g = (3 / 8) + 0.5 * np.cos(2 * np.pi * x) + (1 / 8) * np.cos(4 * np.pi * x)

    elif name == "gauss":
        # Gaussian with sigma = 0.2 of the window length (0.4 of its half),
        # centred on index 0.  (Until 2026-09-26 it peaked at the middle
        # sample, against this module's convention.)
        g = np.exp(-0.5 * (_xc(x) / 0.2) ** 2)

    elif name.startswith("truncgauss"):
        # LTFAT: exp(4 log(p/100) x^2), height p % at x = +-1/2 (p = 1 by
        # default); the Cg that pghi_findgamma tabulates for 'truncgauss'.
        # (Until 2026-09-26: a Gaussian at the middle sample, zeroed below
        # exp(-2) -- pghi_findgamma of it was 2.2 times the tabulated value.)
        g = np.exp(4.0 * math.log(_truncgauss_percent(name) / 100.0) * _xc(x) ** 2)

    elif name == "butterworth":
        # Butterworth-like plateau of order 4 and half-width 0.3, centred on
        # index 0 (until 2026-09-26 centred on the middle sample).
        g = 1.0 / (1.0 + (np.abs(_xc(x)) / 0.3) ** 8)

    elif name == "kaiser":
        if beta is None:
            raise ValueError("firwin: 'kaiser' window requires beta=")
        if M == 1:
            g = np.array([1.0])
        else:
            mm = M - 1
            kk = np.arange(M, dtype=float) + (M % 2) / 2.0 - 0.5
            kk = 2.0 * beta / mm * np.sqrt(np.maximum(kk * (mm - kk), 0.0))
            g = np.i0(kk) / np.i0(beta)
        g = np.fft.ifftshift(g)          # peak to index 0 (DFT ordering)
        if M % 2 == 0:
            g[M // 2] = 0.0              # zero Nyquist (whole-point even)
        g = np.real(g)

    elif name == "taper":
        # Special: name passed as list ['hann', 'taper', ratio] — handled in blfilter
        g = np.ones(M)

    else:
        raise ValueError(f"firwin: unknown window type {name!r}")

    # Clip tiny negatives from cosine evaluation
    g = np.maximum(g, 0.0)

    # --- normalisation ---
    g = _apply_norm(g, norm, M)
    return g


def firwin_taper(name1: str, name2: str, ratio: float,
                 M: int, norm: str = "inf") -> np.ndarray:
    """Tapering window: flat plateau fading into *name1* shape.

    Corresponds to ``blfilter({'hann', 'taper', ratio}, …)``.

    Parameters
    ----------
    name1 : str
        Window name (used for taper shape description).
    name2 : str
        Secondary name (typically 'taper').
    ratio : float
        Taper ratio (width of transition region relative to window length).
    M : int
        Window length.
    norm : str
        Normalization type.

    Returns
    -------
    np.ndarray
        Tapered window vector.

    Examples
    --------
    >>> from cool_frames.numpy.filters._firwin import firwin_taper
    >>> w = firwin_taper('hann', 'taper', 0.2, 512)
    >>> w.shape
    (512,)
    """
    # Construct a taper: central plateau of width (1-ratio)*M,
    # cosine roll-off of width ratio*M/2 on each side
    x = _x(M)
    plateau_half = (1.0 - ratio) / 2.0
    g = np.ones(M)
    transition = np.abs(x) > plateau_half
    xt = (np.abs(x[transition]) - plateau_half) / ratio
    g[transition] = 0.5 + 0.5 * np.cos(np.pi * xt)
    g = np.maximum(g, 0.0)  # type: ignore[assignment]
    return _apply_norm(g, norm, M)


def _apply_norm(g: np.ndarray, norm: str, M: int) -> np.ndarray:
    """Apply normalization to a window vector.

    Parameters
    ----------
    g : np.ndarray
        Window vector.
    norm : str
        ``'inf'``/``'peak'`` — unit peak; ``'energy'``/``'2'`` — unit L2 norm;
        ``'1'``/``'area'`` — unit L1 norm; ``'null'``/``'none'`` — unchanged.
    M : int
        Window length.  Accepted for signature stability and no longer read —
        the norms are computed from ``g`` itself.

    Returns
    -------
    np.ndarray
        Normalized window.

    Raises
    ------
    ValueError
        If norm is not recognized.

    Notes
    -----
    ``'energy'``/``'2'`` used to *multiply by* ``sqrt(M)`` and ``'1'``/``'area'``
    to multiply by ``M`` — neither of which normalises anything.  A "unit
    energy" Hann window of length 512 came back with an L2 norm of 313.5.

    That contradicted every other definition in the package —
    ``core._norm.normalize_window``, the public ``setnorm``, and
    ``_warpedfilters._setnorm`` all divide by the norm — and LTFAT, where
    ``'2'``/``'energy'`` means unit L2.  ``firwin``'s own default is ``'inf'``,
    which was always correct, so the damage was confined to callers who asked
    for energy or area explicitly: ``gabfilters`` is the one in-tree caller
    that does, via ``_gabwin``.
    """
    norm = norm.lower().strip()
    if norm in ("null", "none", ""):
        return g
    if norm in ("energy", "2"):
        n = float(np.linalg.norm(g))
        return g / n if n > 0 else g
    elif norm in ("1", "area"):
        s = float(np.sum(np.abs(g)))
        return g / s if s > 0 else g
    elif norm in ("inf", "peak"):
        mx = float(np.max(np.abs(g)))
        return g / mx if mx > 0 else g
    else:
        raise ValueError(f"firwin: unknown norm {norm!r}")


# ---------------------------------------------------------------------------
# firwin_eval – evaluate window at arbitrary sample positions
# (needed by comp_warpedfreqresponse; MATLAB firwin accepts float vectors)
# ---------------------------------------------------------------------------

def firwin_eval(name: str, x: np.ndarray) -> np.ndarray:
    """Evaluate a named window at arbitrary sample positions *x*.

    Evaluates the window shape at arbitrary positions in WPE (Whole-Point Even)
    convention where x = n/M for n in [0, M).

    Parameters
    ----------
    name : str
        Window type (same names as :func:`firwin`).
    x : ndarray
        Sample positions, typically in ``[0, 1)``.

    Returns
    -------
    g : ndarray, same shape as *x*
        Window values at the given positions.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters._firwin import firwin_eval
    >>> x = np.linspace(0, 1, 100)
    >>> g = firwin_eval('hann', x)
    >>> g.shape
    (100,)
    >>> float(g[0])
    1.0
    """
    name = name.lower().strip()
    _not_a_window(name)
    x = np.asarray(x, dtype=float)

    if name in ("hann", "hanning", "nuttall10"):
        g = 0.5 * (1.0 + np.cos(2.0 * np.pi * x))

    elif name in ("sine", "sqrthann", "cosine"):
        # Sine window is sqrt of hann window
        g = np.sqrt(np.maximum(0.0, 0.5 * (1.0 + np.cos(2.0 * np.pi * x))))

    elif name in ("hamming",):
        g = 0.54 + 0.46 * np.cos(2 * np.pi * x)

    elif name == "blackman":
        g = 0.42 + 0.5 * np.cos(2 * np.pi * x) + 0.08 * np.cos(4 * np.pi * x)

    elif name == "blackman2":
        g = (7938 / 18608
             + 9240 / 18608 * np.cos(2 * np.pi * x)
             + 1430 / 18608 * np.cos(4 * np.pi * x))

    elif name in ("rect", "square"):
        g = np.ones_like(x)
        # For even-length signals, zero the Nyquist bin at x=0.5
        g = np.where(np.abs(x - 0.5) < 1e-10, 0.0, g)

    elif name in ("tria", "bartlett"):
        # Triangular window: g = 1 - 2*min(x, 1-x) for x in [0, 1]
        g = 1.0 - 2.0 * np.minimum(x, 1.0 - x)

    elif name == "sqrttria":
        # Square root of triangular
        g = np.sqrt(np.maximum(1.0 - 2.0 * np.minimum(x, 1.0 - x), 0.0))

    elif name in ("itersine", "ogg"):
        g = np.sin(np.pi / 2 * np.cos(np.pi * x) ** 2)

    elif name == "nuttall":
        a = [0.355768, 0.487396, 0.144232, 0.012604]
        g = sum(a[k] * np.cos(2 * np.pi * k * x) for k in range(4))  # type: ignore[assignment]

    elif name == "nuttall01":
        g = 0.53836 + 0.46164 * np.cos(2 * np.pi * x)

    elif name == "nuttall11":
        g = 0.40897 + 0.5 * np.cos(2 * np.pi * x) + 0.09103 * np.cos(4 * np.pi * x)

    elif name == "nuttall20":
        g = (3 / 8) + 0.5 * np.cos(2 * np.pi * x) + (1 / 8) * np.cos(4 * np.pi * x)

    elif name == "gauss":
        g = np.exp(-0.5 * (_xc(x) / 0.2) ** 2)

    elif name.startswith("truncgauss"):
        g = np.exp(4.0 * math.log(_truncgauss_percent(name) / 100.0) * _xc(x) ** 2)

    elif name == "butterworth":
        g = 1.0 / (1.0 + (np.abs(_xc(x)) / 0.3) ** 8)

    else:
        raise ValueError(f"firwin_eval: unknown window type {name!r}")

    # The cosine-based formulas are periodic with period 1, so x
    # values in [0,1) already represent the full window in WPE order.
    # Only zero values that lie genuinely outside a single period,
    # i.e. x < 0 or x >= 1.
    g = g * ((x >= 0.0) & (x < 1.0)).astype(float)
    g = np.maximum(g, 0.0)
    return g  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# ERB-bandwidth constant for 'hann' window
# (used by audfilters via helper_filtergeneratorfunc)
# ---------------------------------------------------------------------------

def hann_winbw(probelen: int = 10_000) -> float:
    """Return the ERB-type bandwidth constant for a Hann window.

    Computes the relative energy bandwidth of a Hann window as:
    ``winbw = norm(firwin('hann', probelen, 'inf'))^2 / probelen``

    Parameters
    ----------
    probelen : int
        Probe length for window evaluation. Default: 10000.

    Returns
    -------
    float
        Bandwidth constant (relative energy per Hz).

    Examples
    --------
    >>> from cool_frames.numpy.filters._firwin import hann_winbw
    >>> bw = hann_winbw()
    >>> 0.0 < bw < 1.0
    True
    """
    h = firwin("hann", probelen, norm="inf")
    return float(np.dot(h, h) / probelen)


def window_winbw(name: str, probelen: int = 10_000, **kw) -> float:
    """Equivalent noise bandwidth of any named window.

    The generalisation of :func:`hann_winbw`, matching LTFAT's
    ``audfilters.m``, which computes the constant from whichever window it was
    given rather than from a fixed value::

        winbw = norm(firwin(win, probelen, 'inf'))^2 / probelen

    Returns 3/8 for ``'hann'`` and 1/2 for ``'sine'``, as LTFAT does.

    Parameters
    ----------
    name : str
        Window name, as accepted by :func:`firwin`.
    probelen : int
        Probe length. Default 10000.
    **kw
        Extra arguments forwarded to :func:`firwin` (``beta`` for Kaiser).

    Returns
    -------
    float
        Bandwidth constant (relative energy per Hz).

    Examples
    --------
    >>> from cool_frames.numpy.filters._firwin import window_winbw
    >>> round(window_winbw("hann"), 6)
    0.375
    >>> round(window_winbw("sine"), 6)
    0.5
    """
    h = firwin(name, probelen, norm="inf", **kw)
    return float(np.dot(h, h) / probelen)


# ---------------------------------------------------------------------------
# pgauss — periodic (sampled) Gaussian Gabor window (relocated from core)
# ---------------------------------------------------------------------------

def pgauss(
    L: int,
    tfr: float = 1.0,
    *,
    width: float | None = None,
    bw: float | None = None,
    cf: float = 0.0,
    delay: float = 0.0,
    centering: str = "wp",
    atheight: float = 0.5,
    norm: str = "2",
) -> np.ndarray:
    """Sampled, periodized Gaussian.

    Parameters
    ----------
    L : int
        Signal length.
    tfr : float
        Time-to-frequency support ratio.  ``tfr > 1`` means wider time
        support than frequency support.  Default is 1 (self-dual under
        DFT).  Ignored when *width* or *bw* is given.
    width : float or None
        Set the effective support (in samples) at *atheight* fraction of
        the peak.  Overrides *tfr*.
    bw : float or None
        Set the bandwidth in normalised frequency.  Overrides *tfr*.
    cf : float
        Centre frequency in bins (modulates the Gaussian).
    delay : float
        Shift the Gaussian by *delay* samples.
    centering : ``"wp"`` or ``"hp"``
        Whole-point even (default) or half-point even centering.
    atheight : float
        Height fraction used with *width* (default 0.5 = half-height).
    norm : str
        Normalisation: ``"2"`` (unit L2 norm, default), ``"inf"`` (peak 1),
        ``"1"`` (unit L1 norm), ``"null"`` (no normalisation).

    Returns
    -------
    g : ndarray, shape (L,)
        Periodic Gaussian, L2-normalised by default.

    Notes
    -----
    The output is whole-point even (centred at index 0), so
    ``np.fft.fft(pgauss(L, tfr))`` is real-valued.  The DFT of
    ``pgauss(L, tfr)`` equals ``pgauss(L, 1/tfr)`` (up to
    normalisation).

    For generating optimal Gabor windows, use ``pgauss(L, a * M / L)``
    where *a* is the hop size and *M* the number of channels.
    """
    if L < 1:
        raise ValueError("L must be >= 1")

    # Resolve tfr from width or bw
    if width is not None:
        tfr = math.pi / (4 * math.log(1.0 / atheight)) * width**2 / L
    elif bw is not None:
        tfr = L / (bw * L / 2) ** 2

    # Centering offset
    cent = 0.0 if centering == "wp" else 0.5

    # Compute the periodized Gaussian (matches comp_pgauss.m)
    g = _comp_pgauss(L, tfr, cent - delay, cf)

    # Normalise
    g = normalize_window(g, norm)

    return g


def _comp_pgauss(
    L: int,
    w: float,
    c_t: float,
    c_f: float,
) -> np.ndarray:
    """Core computation of the periodized Gaussian.

    Direct port of LTFAT ``comp_pgauss.m``.

    Parameters
    ----------
    L : int
        Length.
    w : float
        Time-frequency ratio.
    c_t : float
        Time centering (0 = whole-point, 0.5 = half-point).
    c_f : float
        Frequency centering (in bins).
    """
    g = np.zeros(L, dtype=complex)
    if L == 0:
        return g.real

    sqrtl = math.sqrt(L)
    safe = 4.0

    # Keep delay in a sane interval
    c_t = c_t % L

    # Number of periods to sum (beyond [-safe, safe] the Gaussian is
    # numerically zero)
    sqw = math.sqrt(L / math.sqrt(w)) if w > 0 else 1.0
    nk = math.ceil(safe / sqw) if sqw > 0 else 0

    lr = np.arange(L, dtype=np.float64) + c_t

    for k in range(-nk, nk + 1):
        g += np.exp(
            -math.pi * (lr / sqrtl - k * sqrtl) ** 2 / w + 2j * math.pi * c_f * (lr / L - k)
        )

    # Normalise to unit L2 norm
    g = g / np.linalg.norm(g)

    return g.real if c_f == 0 and (c_t % 1) == 0 else g  # type: ignore[no-any-return]
