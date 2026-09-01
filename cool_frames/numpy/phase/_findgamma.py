"""
numpy/phaseret/_findgamma.py
==============================
Find the window constant gamma for PGHI / RTPGHI.

Port of ``phaseret/gabor/pghi_findgamma.m`` (Gabor domain) and
``filterbank/utils/legacy/gabor/wpghi_findalpha.m`` (filterbank domain).

The gamma parameter relates a given analysis window to the closest
Gaussian:  g(l) ≈ exp(-π l² / γ),  with  γ = Cg · gl².
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


def _findbestgauss(gnum: np.ndarray, atheightrange: np.ndarray | None = None) -> float:
    """Find the relative height at which a Gaussian best matches *gnum*.

    Simplified port — uses a brute-force search over atheight values.

    Like :func:`_winwidthatheight`, this works in DFT ordering: the zero-pad
    below splits *gnum* at its midpoint and wraps the second half to the end of
    ``glong``, and the reference Gaussian is built on the wrap-around distance
    ``min(l, L-l)``.  Both only line up if the window peaks at index 0, so the
    peak is rolled there first.

    Known limitation: the returned ``atheight`` regularly pins at the top of
    ``atheightrange`` (0.8) for the standard windows, which means the minimum is
    at the boundary rather than interior and the search has not actually found
    the best Gaussian.  With the ordering fixed, the resulting ``Cg`` is 7-45 %
    above the tabulated constant for the same window (Hann 0.309 vs 0.256;
    cosine 0.600 vs 0.415).  Pass the window *name* when you have it —
    :func:`pghi_findgamma` then returns the exact tabulated value and never
    reaches this search.  ``test_findgamma.py`` pins the gap so that narrowing
    it is a visible change.
    """
    if atheightrange is None:
        atheightrange = np.arange(0.01, 0.801, 0.001)

    gl = len(gnum)
    L = 10 * gl

    # Peak-normalise, then roll the peak to index 0 (see docstring).
    gnum = np.asarray(gnum, dtype=float).ravel()
    gnum = gnum / np.max(np.abs(gnum))
    peak = int(np.argmax(gnum))
    if peak != 0:
        gnum = np.roll(gnum, -peak)
    glong = np.zeros(L)
    half = (gl + 1) // 2
    glong[:half] = gnum[:half]
    glong[L - (gl - half) :] = gnum[half:]

    norms = np.zeros(len(atheightrange))
    for ii, ah in enumerate(atheightrange):
        w = _winwidthatheight(gnum, ah)
        # Build matching Gaussian: g(l) = exp(-π (l/σ)²) where σ = w/(2√(-ln(ah)))
        sigma_sq = (w / 2.0) ** 2 / (-math.log(ah)) if ah > 0 else 1.0
        l = np.arange(L)
        l = np.minimum(l, L - l)  # wrap-around distance
        gauss = np.exp(-math.pi * l**2 / sigma_sq) if sigma_sq > 0 else np.zeros(L)
        gauss /= np.max(gauss) if np.max(gauss) > 0 else 1.0
        norms[ii] = np.linalg.norm(glong - gauss)

    best_idx = int(np.argmin(norms))
    return float(atheightrange[best_idx])


def pghi_findgamma(
    g, gl: int | None = None, a: int | None = None, M: int | None = None
) -> tuple[float, float]:
    """Find the gamma constant for PGHI / RTPGHI.

    Parameters
    ----------
    g : str or ndarray
        Window name (e.g. ``'hann'``) or numeric window vector.
    gl : int, optional
        Window support length. Required for named windows without
        precomputed constants.
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

    # Numeric window — search
    g = np.asarray(g, dtype=float).ravel()
    if gl is None:
        gl = len(g)

    atheight = _findbestgauss(g)
    w = _winwidthatheight(g, atheight)

    Cg = -math.pi / 4.0 * (w / (gl - 1)) ** 2 / math.log(atheight)
    gamma = Cg * gl**2

    return gamma, Cg


def wpghi_findgamma(g, tfr: np.ndarray | None = None, **kwargs) -> tuple[float, float]:
    """Alias for ``pghi_findgamma`` — for filterbank WPGHI compatibility.

    For filterbank-based PGHI, gamma is typically computed per-channel
    and stored in the filter info dict.  This function is a convenience
    wrapper.
    """
    return pghi_findgamma(g, **kwargs)
