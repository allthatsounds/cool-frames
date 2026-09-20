"""
numpy/phase/_findgamma.py
=========================
Find the window constant gamma for PGHI / RTPGHI.

Port of PHASERET's ``gabor/pghi_findgamma.m``, including its local function
``findbestgauss``: the Gaussian ``exp(-pi l^2 / gamma)`` closest to a
window, with ``gamma = Cg * gl^2``.  ``wpghi_findgamma`` converts a filter
bank's time-frequency ratio to the same constant.
"""

from __future__ import annotations

import math

import numpy as np

# ======================================================================
# Precomputed window constants (from pghi_findgamma.m)
# ======================================================================

_PRECOMPUTED_CG = {
    "hann": 0.25645,
    "hanning": 0.25645,
    "nuttall10": 0.25645,
    "sqrthann": 0.41532,
    "cosine": 0.41532,
    "sine": 0.41532,
    "hamming": 0.29794,
    "nuttall01": 0.29610,
    "tria": 0.27561,
    "triangular": 0.27561,
    "bartlett": 0.27561,
    "sqrttria": 0.48068,
    "blackman": 0.17954,
    "blackman2": 0.18465,
    "nuttall": 0.12807,
    "nuttall12": 0.12807,
    "ogg": 0.35744,
    "itersine": 0.35744,
    "nuttall20": 0.14315,
    "nuttall11": 0.17001,
    "nuttall02": 0.18284,
    "nuttall30": 0.09895,
    "nuttall21": 0.11636,
    "nuttall03": 0.13369,
    "truncgauss": 0.17054704423023,
}


def _winwidthatheight(g: np.ndarray, atheight: float) -> float:
    """Width of window *g* at relative height *atheight*.

    Port of the nested ``winwidthatheight`` in pghi_findgamma.m.

    The MATLAB original reads ``gnum(1:floor(gl/2)+1)`` and looks for the
    threshold crossing as the index *increases*, which is only meaningful for
    DFT ordering — peak at index 0, decaying outwards.  Handed the centred
    ordering that ``scipy.signal.get_window`` and every other Python window
    source produces, the scan runs *up* the rising flank instead of down the
    falling one, the crossing indices pin near the peak, and the measured width
    comes out at roughly ``gl`` for any window shape.  Through v0.1.1 that made
    ``pghi_findgamma(hann_vector)`` return ``Cg = 2.195`` against the
    precomputed ``Cg = 0.25645`` for the same window — 8.5x too large, silently
    — and 6.1x to 10.3x for the other tabulated shapes.  Since ``gamma``
    scales the phase gradients PGHI integrates, an error of that size does not
    degrade the phase estimate, it replaces it.

    This is the same defect, in a second copy of the same helper, that was
    fixed in ``cool_frames/numpy/filters/_gabfilters.py`` — see
    ``test_window_width_measurement_depends_on_window_shape``.  The two copies
    are private, near-identical, and were never compared, so the fix landed in
    one of them.  ``test_findgamma.py`` now asserts they agree.

    Rolling the peak to index 0 first makes the routine correct for both
    layouts and leaves the DFT-ordered case bit-identical (its argmax is
    already 0).
    """
    g = np.asarray(g, dtype=float).ravel()
    gl = len(g)
    if gl == 0:
        return 0.0

    peak = int(np.argmax(g))
    if peak != 0:
        g = np.roll(g, -peak)

    gmax = float(np.max(g))
    fracofmax = gmax / (1.0 / atheight)  # = gmax * atheight

    half = gl // 2 + 1
    ghalf = g[:half]

    exact = np.where(ghalf == fracofmax)[0]
    if len(exact) > 0:
        return 2.0 * float(exact[0])

    above = np.where(ghalf > fracofmax)[0]
    below = np.where(ghalf < fracofmax)[0]

    if len(below) == 0:
        return float(gl)

    ind1 = above[-1] if len(above) > 0 else 0
    ind2 = below[0]

    rest = 1.0 - (fracofmax - g[ind2]) / (g[ind1] - g[ind2])
    return 2.0 * (ind1 + rest)  # type: ignore[no-any-return]


def _fir2long(g: np.ndarray, L: int) -> np.ndarray:
    """LTFAT 2.6 ``fir2long``: the first ``ceil(gl/2)`` samples at the start,
    the rest at the end, zeros between (no split middle sample)."""
    out = np.zeros(L)
    h = -(-g.size // 2)
    out[:h] = g[:h]
    out[L - (g.size - h) :] = g[h:]
    return out


def _long2fir(g: np.ndarray, gl: int) -> np.ndarray:
    """LTFAT 2.6 ``long2fir`` (``'unsymmetric'``): the first ``ceil(gl/2)``
    and the last ``floor(gl/2)`` samples."""
    h = -(-gl // 2)
    return np.concatenate([g[:h], g[g.size - (gl - h) :]])


def _findbestgauss(gnum: np.ndarray, atheightrange: np.ndarray | None = None) -> float:
    """PHASERET's ``findbestgauss`` (local to ``pghi_findgamma.m``): the
    height ``ah`` at which a Gaussian best matches the window.

    For every ``ah`` in the range, the window's width ``w`` at that height is
    measured and the Gaussian with the same width at the same height --
    ``pgauss(L, 'inf', 'width', w, 'atheight', ah)``, which is ``ah`` at
    ``+-w/2`` -- is compared with the peak-normalised window on ``L = 10 gl``
    samples; the closest one wins.

    The port this replaces built ``exp(-pi l^2 / ((w/2)^2 / -ln ah))``, which
    is ``ah**pi`` at ``+-w/2`` rather than ``ah``: every candidate was too
    narrow, the error decreased towards the widest one, and the search
    returned the top of its range (0.8) for four of the five tabulated
    windows.  Like :func:`_winwidthatheight` it expects the peak at index 0
    (LTFAT's FIR layout); a window with its peak elsewhere is rolled there.
    """
    from ..filters import pgauss

    if atheightrange is None:
        atheightrange = np.arange(0.01, 0.8005, 0.001)
    gnum = np.asarray(gnum, dtype=float).ravel()
    peak = int(np.argmax(gnum))
    if peak != 0:
        gnum = np.roll(gnum, -peak)
    L = 10 * gnum.size
    glong = _fir2long(gnum / np.max(np.abs(gnum)), L)
    norms = np.empty(len(atheightrange))
    for ii, ah in enumerate(atheightrange):
        w = _winwidthatheight(gnum, float(ah))
        gauss = pgauss(L, width=w, atheight=float(ah), norm="inf")
        norms[ii] = np.linalg.norm(glong - gauss)
    return float(atheightrange[int(np.argmin(norms))])


def pghi_findgamma(
    g, gl: int | None = None, a: int | None = None, M: int | None = None
) -> tuple[float, float]:
    """Find the gamma constant for PGHI / RTPGHI.

    A window name gives PHASERET's tabulated constant.  A numeric window
    (peak at index 0, as LTFAT stores FIR windows, or anywhere -- it is
    rolled there) gives PHASERET's search: the window is cut to its width at
    1e-10 of its peak, which is the ``gl`` of the result, and ``Cg`` comes
    from the Gaussian that best matches it (``findbestgauss``).  The two
    agree to 0.2 % at 1024 taps and to about 1 % at 256.

    Parameters
    ----------
    g : str or ndarray
        Window name (e.g. ``'hann'``) or numeric window vector.
    gl : int, optional
        Window length for a named window (required unless ``M`` is given).
        A numeric window has its own length; passing a different ``gl`` is
        an error.
    a : int, optional
        Hop size (required only for ``'gauss'`` window).
    M : int, optional
        Number of channels (required only for ``'gauss'`` window or
        as fallback gl).

    Returns
    -------
    gamma : float
        Window constant.  gamma = Cg * gl²
    Cg : float
        Normalised window constant.
    """
    # Named window — try precomputed
    if isinstance(g, str):
        name = g.lower()

        if name == "gauss":
            if a is None or M is None:
                raise ValueError("'gauss' window requires a and M")
            return float(a * M), float("nan")

        if name in _PRECOMPUTED_CG:
            Cg = _PRECOMPUTED_CG[name]
            if gl is None:
                if M is not None:
                    gl = M
                else:
                    raise ValueError("gl (window length) is required")
            return Cg * gl**2, Cg

        # Unknown name — fall through to numeric search
        raise ValueError(
            f"Unknown window name '{g}'. Pass a numeric window vector for search-based gamma."
        )

    # Numeric window: PHASERET's search.  The window is first cut to its
    # width at 1e-10 of its peak, which is also the ``gl`` of the result.
    g = np.asarray(g, dtype=float).ravel()
    if gl is not None and gl != g.size:
        raise ValueError(
            f"gl={gl} was given for a numeric window of {g.size} samples; a numeric "
            "window's length is its own (pass the window at the length you mean)"
        )
    peak = int(np.argmax(g))
    if peak != 0:
        g = np.roll(g, -peak)
    gl = int(math.floor(_winwidthatheight(g, 1e-10) + 0.5))  # MATLAB round
    g = _long2fir(g, gl)

    atheight = _findbestgauss(g)
    w = _winwidthatheight(g, atheight)

    Cg = -math.pi / 4.0 * (w / (gl - 1)) ** 2 / math.log(atheight)
    gamma = Cg * gl**2

    return gamma, Cg


def wpghi_findgamma(
    g=None, tfr=None, *, L: int | None = None, **kwargs
) -> tuple[float | np.ndarray, float]:
    """PGHI's window constant from a window, or from a time-frequency ratio.

    ``wpghi_findgamma(g, **kwargs)`` is :func:`pghi_findgamma`.

    ``wpghi_findgamma(tfr=tfr, L=L)`` converts the time-frequency ratio a
    filter bank's phase functions take (``filterbankconstphase``'s ``tfr``,
    the designers' ``info['tfr']``) to the Gabor constant: the Gaussian with
    ratio ``tfr`` at length ``L``, ``pgauss(L, tfr) ~ exp(-pi l^2 / (tfr L))``,
    has ``gamma = tfr * L`` -- PHASERET's own conversion for a window
    ``{'gauss', tfr}``.  ``tfr`` may be a scalar, one value per channel, or a
    callable of ``L``; ``gamma`` has the same shape, and ``Cg`` is NaN, as
    for ``pghi_findgamma('gauss', ...)``.

    Until this was fixed, ``tfr`` was accepted and ignored.
    """
    if tfr is None:
        if g is None:
            raise ValueError("wpghi_findgamma needs a window g or a time-frequency ratio tfr")
        return pghi_findgamma(g, **kwargs)
    if g is not None:
        raise ValueError(
            "wpghi_findgamma takes a window g or a time-frequency ratio tfr, not both"
        )
    if kwargs:
        raise TypeError(f"wpghi_findgamma(tfr=...) takes only L; got {sorted(kwargs)}")
    if L is None:
        raise ValueError("wpghi_findgamma(tfr=...) needs the transform length L")
    t = tfr(L) if callable(tfr) else tfr
    t = np.asarray(t, dtype=float)
    if np.any(~np.isfinite(t)) or np.any(t <= 0):
        raise ValueError("tfr must be positive and finite")
    gamma = t * float(L)
    return (float(gamma) if gamma.ndim == 0 else gamma), float("nan")
