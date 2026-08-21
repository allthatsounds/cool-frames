"""
numpy/layer1/_wavelet.py
========================
Wavelet generator functions and helpers.

Ports of:
  - helper_waveletgeneratorfunc.m  → wavelet_generator_func
  - octave_lambertw (Lambert W)    → lambertw
  - determine_freqatheight          → _determine_freqatheight

MATLAB original: utils/helpers/helper_waveletgeneratorfunc.m
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Lambert W function  (port of octave_lambertw from helper_waveletgeneratorfunc.m)
# ---------------------------------------------------------------------------

def lambertw(z, b: int = 0):
    """Lambert W function (branch *b*).

    Solves ``W(z) * exp(W(z)) = z``.

    Parameters
    ----------
    z : float or complex or ndarray
    b : int
        Branch index.  Only 0 and -1 give real values for real *z*.

    Returns
    -------
    w : same type as *z*
    """
    z = np.asarray(z, dtype=complex)
    scalar = z.ndim == 0
    z = np.atleast_1d(z)

    # --- initial guess: series expansion about -1/e ---
    w = (1 - 2 * abs(b)) * np.sqrt(2 * np.e * z + 2) - 1

    # --- asymptotic expansion at 0 and Inf ---
    v = np.log(z + (~(z.astype(bool) | bool(b))).astype(complex)) + 2j * np.pi * b
    v = v - np.log(v + (v == 0).astype(complex))

    # --- choose strategy ---
    c = np.abs(z + 1 / np.e)
    c = c > (1.45 - 1.1 * abs(b))
    c = c | (b * np.imag(z) > 0) | ((np.imag(z) == 0) & (b == 1))
    w = (1 - c) * w + c * v

    # --- Halley iteration ---
    for _ in range(10):
        p = np.exp(w)
        t = w * p - z
        f = (w != -1).astype(complex)
        t = f * t / (p * (w + f) - 0.5 * (w + 2.0) * t / (w + f))
        w = w - t
        if (np.abs(np.real(t)) < 2.48e-16 * (1.0 + np.abs(np.real(w)))).all() and \
           (np.abs(np.imag(t)) < 2.48e-16 * (1.0 + np.abs(np.imag(w)))).all():
            break

    result = w[0] if scalar else w
    # Return real if input was real and branch supports it
    if np.isrealobj(z) and b in (0, -1):
        result = np.real(result)
    return result


# ---------------------------------------------------------------------------
# Helper: find frequency at a given height (bisection)
# ---------------------------------------------------------------------------

def _determine_freqatheight(fun, peakpos: float, thr: float,
                            descending: bool) -> float:
    """Find frequency where ``|fun(f)| == thr`` via bisection.

    Parameters
    ----------
    fun : callable  (float array → float array)
    peakpos : float – peak position of the wavelet
    thr : float – target height (e.g., 0.5 for -3 dB)
    descending : bool
        If True, search to the right of peakpos (descending flank).
        If False, search to the left (ascending flank).
    """
    tol = 1e-12
    if descending:
        lo, hi = peakpos, peakpos * 100
        # Expand hi until fun(hi) < thr
        for _ in range(100):
            if abs(float(np.real(fun(np.array([hi]))).item())) < thr:
                break
            hi *= 2
        for _ in range(200):
            mid = (lo + hi) / 2
            val = abs(float(np.real(fun(np.array([mid]))).item()))
            if abs(val - thr) < tol:
                break
            if val > thr:
                lo = mid
            else:
                hi = mid
        return mid
    else:
        lo, hi = 0.0, peakpos
        # Make sure lo is below threshold
        for _ in range(200):
            mid = (lo + hi) / 2
            val = abs(float(np.real(fun(np.array([mid]))).item()))
            if abs(val - thr) < tol:
                break
            if val < thr:
                lo = mid
            else:
                hi = mid
        return mid


# ---------------------------------------------------------------------------
# B-spline basis functions (orders 1–5) for fbsp wavelet
# ---------------------------------------------------------------------------

def _bspline(order: int, x: np.ndarray) -> np.ndarray:
    """Cardinal B-spline of given order evaluated at *x*."""
    x = np.asarray(x, dtype=float)
    if order == 1:
        return ((x >= 0) & (x < 1)).astype(float)
    elif order == 2:
        return (  # type: ignore[no-any-return]
            ((x >= 0) & (x < 1)) * x + ((x >= 1) & (x < 2)) * (2 - x)
        )
    elif order == 3:
        return (  # type: ignore[no-any-return]
            ((x >= 0) & (x < 1)) * (0.5 * x**2) +
            ((x >= 1) & (x < 2)) * (-x**2 + 3*x - 1.5) +
            ((x >= 2) & (x < 3)) * (0.5 * x**2 - 3*x + 4.5)
        )
    elif order == 4:
        return (  # type: ignore[no-any-return]
            ((x >= 0) & (x < 1)) * (x**3 / 6) +
            ((x >= 1) & (x < 2)) * (-x**3/2 + 2*x**2 - 2*x + 2/3) +
            ((x >= 2) & (x < 3)) * (x**3/2 - 4*x**2 + 10*x - 22/3) +
            ((x >= 3) & (x < 4)) * (-x**3/6 + 2*x**2 - 8*x + 32/3)
        )
    elif order == 5:
        return ((x >= 0) & (x < 1)) * (x**4 / 24) + \
               ((x >= 1) & (x < 2)) * (-x**4/6 + 5*x**3/6 - 5*x**2/4 + 5*x/6 - 5/24) + \
               ((x >= 2) & (x < 3)) * (x**4/4 - 5*x**3/2 + 35*x**2/4 - 25*x/2 + 155/24) + \
               ((x >= 3) & (x < 4)) * (-x**4/6 + 5*x**3/2 - 55*x**2/4 + 65*x/2 - 655/24) + \
               ((x >= 4) & (x < 5)) * (x**4/24 - 5*x**3/6 + 25*x**2/4 - 125*x/6 + 625/24)
    else:
        raise ValueError(f"B-spline order must be 1–5, got {order}")


# ---------------------------------------------------------------------------
# Main: wavelet generator function
# ---------------------------------------------------------------------------

# Supported wavelet types
WAVELET_TYPES = ("cauchy", "morse", "morlet", "fbsp", "analyticsp", "cplxsp")


def wavelet_generator_func(
    name: str | list | tuple,
    *,
    negative: bool = False,
    efsuppthr: float = 1e-5,
    bwthr: float = 10**(-3/10),
) -> tuple:
    """Return a wavelet function handle, support, peak position, and cauchyAlpha.

    Parameters
    ----------
    name : str or list/tuple
        Wavelet type, optionally with parameters.
        Examples: ``'cauchy'``, ``['cauchy', 300]``, ``['morse', 300, 0, 3]``,
        ``['morlet', 4]``, ``['fbsp', 4, 2]``, ``['analyticsp', 4, 2]``,
        ``['cplxsp', 4, 2]``.
    negative : bool
        If True, construct wavelet at negative frequencies.
    efsuppthr : float
        Effective support threshold.
    bwthr : float
        Bandwidth threshold (height for bandwidth measurement).

    Returns
    -------
    fun : callable(y: ndarray) → ndarray
        Wavelet function in normalised frequency domain.
    fsupp : (5,) float array
        Support vector [efsuppthr_asc, bwthr_asc, peak, bwthr_desc, efsuppthr_desc].
    peakpos : float
        Peak position.
    cauchy_alpha : float or None
        Equivalent Cauchy alpha parameter.
    """
    if isinstance(name, str):
        name = [name]
    else:
        name = list(name)

    wtype = name[0].lower().strip()
    if wtype not in WAVELET_TYPES:
        raise ValueError(f"Unknown wavelet type {wtype!r}. "
                         f"Supported: {WAVELET_TYPES}")

    args = name[1:]

    if wtype in ("cauchy", "morse"):
        alpha = float(args[0]) if len(args) >= 1 else 300.0
        beta = float(args[1]) if len(args) >= 2 else 0.0
        gamma = float(args[2]) if len(args) >= 3 else 3.0

        if alpha <= 1:
            raise ValueError(f"Alpha must be > 1 (got {alpha})")
        if gamma <= 0:
            raise ValueError(f"Gamma must be > 0 (got {gamma})")

        order = (alpha - 1) / (2 * gamma)
        peakpos = (order / (2 * np.pi * gamma)) ** (1 / gamma)

        # Log-normalisation constant
        log_norm = order / gamma - order / gamma * np.log(order / (2 * np.pi * gamma))

        if not negative:
            def fun(y):
                y = np.asarray(y, dtype=complex)
                return (np.real(y) > 0) * np.exp(
                    -2 * np.pi * y**gamma
                    + (order - 1j * beta) * np.log(y + 0j)
                    + log_norm
                )
        else:
            def fun(y):
                y = np.asarray(y, dtype=complex)
                ay = np.abs(y)
                return (np.real(y) < 0) * np.exp(
                    -2 * np.pi * ay**gamma
                    + (order - 1j * beta) * np.log(ay + 0j)
                    + log_norm
                )

        # Support via Lambert W
        def _freqatheight_asc(thr):
            val = lambertw(
                -thr**(gamma / order) / np.e, b=0
            )
            return float(np.real((-order / (2 * np.pi * gamma) * val) ** (1 / gamma)).item())

        def _freqatheight_desc(thr):
            val = lambertw(
                -thr**(gamma / order) / np.e, b=-1
            )
            return float(np.real((-order / (2 * np.pi * gamma) * val) ** (1 / gamma)).item())

        cauchy_alpha = alpha  # For cauchy; for morse, this is approximate

    elif wtype == "morlet":
        sigma = float(args[0]) if len(args) >= 1 else 4.0
        if sigma <= 1:
            raise ValueError(f"Sigma must be > 1 (got {sigma})")

        # Fixed-point iteration for peak position
        peakpos = float(sigma)
        for _ in range(100):
            peakpos_old = peakpos
            peakpos = sigma / (1 - np.exp(-sigma * peakpos))
            if abs(peakpos - peakpos_old) < 1e-6:
                break

        # Normalisation denominator
        denom = (np.exp(-0.5 * (sigma - peakpos)**2)
                 - np.exp(-0.5 * (sigma**2 + peakpos**2)))

        def fun(y):
            y = np.asarray(y, dtype=float)
            return (np.exp(-0.5 * (sigma - np.abs(y))**2)
                    - np.exp(-0.5 * (sigma**2 + np.abs(y)**2))) / denom

        _freqatheight_asc = lambda thr: _determine_freqatheight(fun, peakpos, thr, False)
        _freqatheight_desc = lambda thr: _determine_freqatheight(fun, peakpos, thr, True)
        cauchy_alpha = None  # Would need wpghi_findalpha

    elif wtype == "fbsp":
        order = int(args[0]) if len(args) >= 1 else 4
        fb = float(args[1]) if len(args) >= 2 else 2.0

        if order < 1 or order > 5:
            raise ValueError(f"fbsp order must be 1–5 (got {order})")
        if fb < 2:
            raise ValueError(f"fb must be >= 2 (got {fb})")

        peakpos = 1.0
        peak_val = float(np.asarray(_bspline(order, np.array([order / 2.0]))).item())

        def fun(y):
            y = np.asarray(y, dtype=float)
            return _bspline(order, (np.abs(y) - 1) * fb * order / 2 + order / 2) / peak_val

        _freqatheight_asc = lambda thr: _determine_freqatheight(fun, peakpos, thr, False)
        _freqatheight_desc = lambda thr: _determine_freqatheight(fun, peakpos, thr, True)
        cauchy_alpha = None

    elif wtype == "analyticsp":
        order = int(args[0]) if len(args) >= 1 else 4
        fb = int(args[1]) if len(args) >= 2 else 2

        if order < 1 or order > 5:
            raise ValueError(f"analyticsp order must be 1–5 (got {order})")
        if fb < 1:
            raise ValueError(f"fb must be >= 1 (got {fb})")

        peakpos = 1.0

        if not negative:
            def fun(y):
                y = np.asarray(y, dtype=float)
                return (y > 0) * (np.sinc(fb * (y - 1))**order
                                  + np.sinc(fb * (y + 1))**order)
        else:
            def fun(y):
                y = np.asarray(y, dtype=float)
                return (y < 0) * (np.sinc(fb * (np.abs(y) - 1))**order
                                  + np.sinc(fb * (np.abs(y) + 1))**order)

        # Height function for support computation (avoids sinc division issues)
        def _heightfun(y):
            y = np.asarray(y, dtype=float)
            eps_ = np.finfo(float).eps
            v = np.minimum(1.0, (y > 0) * (
                1.0 / np.abs(fb * (np.pi * y - np.pi) + eps_)**order
                + 1.0 / np.abs(fb * (np.pi * y + np.pi))**order
            ))
            return v

        _freqatheight_asc = lambda thr: _determine_freqatheight(_heightfun, peakpos, thr, False)
        _freqatheight_desc = lambda thr: _determine_freqatheight(_heightfun, peakpos, thr, True)
        cauchy_alpha = None

    elif wtype == "cplxsp":
        order = int(args[0]) if len(args) >= 1 else 4
        fb = int(args[1]) if len(args) >= 2 else 2

        if order < 1 or order > 5:
            raise ValueError(f"cplxsp order must be 1–5 (got {order})")
        if fb < 1:
            raise ValueError(f"fb must be >= 1 (got {fb})")

        peakpos = 1.0

        if not negative:
            def fun(y):
                y = np.asarray(y, dtype=float)
                return np.sinc(fb * (y - 1))**order
        else:
            def fun(y):
                y = np.asarray(y, dtype=float)
                return np.sinc(fb * (np.abs(y) - 1))**order

        def _heightfun(y):
            y = np.asarray(y, dtype=float)
            eps_ = np.finfo(float).eps
            return np.minimum(1.0, 1.0 / np.abs(fb * (np.pi * y - np.pi) + eps_)**order)

        _freqatheight_asc = lambda thr: _determine_freqatheight(_heightfun, peakpos, thr, False)
        _freqatheight_desc = lambda thr: _determine_freqatheight(_heightfun, peakpos, thr, True)
        cauchy_alpha = None

    # Build support vector
    fsupp = np.array([-np.inf, -np.inf, peakpos, np.inf, np.inf])
    if efsuppthr > 0:
        fsupp[0] = _freqatheight_asc(efsuppthr)
        fsupp[4] = _freqatheight_desc(efsuppthr)
    fsupp[1] = _freqatheight_asc(bwthr)
    fsupp[3] = _freqatheight_desc(bwthr)

    return fun, fsupp, peakpos, cauchy_alpha
