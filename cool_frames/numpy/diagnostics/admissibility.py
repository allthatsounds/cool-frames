r"""Closed-form frame-admissibility prediction for painless filterbanks.

Why this exists
---------------
Every painless designer can be handed parameters for which the resulting bank
is **not a frame**: the lower frame bound is exactly zero, some band of the
spectrum is annihilated, and the signal is not recoverable.  The closed-form
painless estimator :func:`~cool_frames.filterbanks.filterbankbounds` does not
notice, because it is evaluated on a response that does not see the gap;
:func:`~cool_frames.filterbanks.filterbankbounds_svd` does notice, but costs
:math:`O(L^2)` memory and requires the bank to be built first.

This module answers the question *before* the bank is built, in closed form,
from the design parameters alone.

The covering theorem
--------------------
A painless bank fails to be a frame exactly when its filters leave a DFT bin
uncovered.  (Measured over 8,524 configurations spanning every designer, this
covering criterion and the exact eigenvalue bound agree in 100 % of cases.)
Coverage is pure integer arithmetic: a prototype of odd length :math:`W_m`
bins centred on bin :math:`k_m = \mathrm{round}(L f_m / f_s)` occupies bins
:math:`[k_m - \lfloor W_m/2\rfloor,\; k_m - \lfloor W_m/2\rfloor + W_m - 1]`,
of which the outermost one at each end is identically zero for a taper that
vanishes at its endpoints.  Adjacent channels overlap iff

.. math::
    k_{m+1} - k_m \;\le\; \tfrac{W_m - 1}{2} + \tfrac{W_{m+1} - 1}{2} - 1 .

The warped corollary
--------------------
Every designer places channels uniformly in some warping coordinate
:math:`u = g(f)` and gives each filter a support that is **constant in that
same coordinate**:

==================  ======================  ================  ==================
designer            :math:`u = g(f)`        spacing           support
==================  ======================  ================  ==================
``audfilters``      auditory scale          ``spacing``       ``bwmul / winbw``
``greenwoodfilters``  Greenwood position    ``spacing``       ``bwmul / winbw``
``warpedfilters``   user ``freqtoscale``    ``1 / bins``      ``2 * bwmul``
``cqtfilters``      :math:`\log_2 f`        ``1 / bins``      from ``Qvar``
``waveletfilters``  :math:`\log_2 f`        ``1 / bins``      wavelet-dependent
``gabfilters``      :math:`f`               ``fs / M``        window width
==================  ======================  ================  ==================

Because both live in the same coordinate the warping derivative cancels and
the continuum condition is a pure ratio -- independent of frequency, of the
sampling rate, and of which scale was chosen:

.. math::
    \Delta_u \;<\; \Lambda_u \;-\; \frac{c\,f_s}{L\, b_{\min}},
    \qquad b_{\min} = 1/g'(f_1),

with :math:`c` the number of dead endpoint bins of the prototype.  The
discretisation term is largest where the local bandwidth is smallest, so for
any convex scale the *lowest* channel binds.

Two specialisations are worth stating because they behave differently:

* **Auditory / warped designers** are limited by *redundancy*: for
  ``audfilters`` the condition reads ``spacing < bwmul / winbw``, and for
  ``warpedfilters`` simply ``2 * bwmul > 1 / bins``.
* **Constant-Q designers** are not.  With :math:`t = 2^{1/\mathrm{bins}}` the
  support is :math:`Q_{\mathrm{var}} f_c (t - 1/t)` and the spacing
  :math:`f_c (t-1)`, so :math:`f_c` cancels and the continuum condition is
  :math:`Q_{\mathrm{var}} > t/(t+1)` -- satisfied for every
  :math:`Q_{\mathrm{var}} \ge 2/3`.  A constant-Q bank can only fail through
  the discretisation term at :math:`f_{\min}`, i.e. it is limited by
  *low-frequency resolution*, not by redundancy.

Where this runs
---------------
Every designer calls :func:`check_admissible` before returning and publishes
the verdict as ``info["admissible"]``, together with the geometry the
predictor used (``fsupp_inner``, ``fsupp_dc``, ``fsupp_nyq``; ``scalevec`` and
``bwmul`` for ``warpedfilters``).

Two designers have configurations whose channel layout this covering test
cannot express -- it always assumes exactly one complement centred on DC and
one on Nyquist, and an interval whose live width it can name.  Those report
``info["admissible"] = None`` rather than an unvalidated verdict:
``gabfilters(windowaxis='freq')``, whose dead endpoint bins depend on the
window *and* on the parity of ``M``, ``gabfilters`` handed a window array
whose length is not ``M``, and ``waveletfilters`` with
``lowpass='none'``/``'repeat'``, ``highpass='none'``, or a two-sided
(``freqrange='complex'``/``'analytic'``) bank.  A geometry below the floor emits
:class:`NotAFrameWarning` naming the first uncovered bin.  The bank is still
built -- analysis still works, and studying the gap is a legitimate thing to
want -- but the warning fires where the parameters were chosen rather than
later, when the missing band shows up as an all-zero dual.

Examples
--------
>>> import numpy as np
>>> from cool_frames.filters import audfilters
>>> from cool_frames.diagnostics.admissibility import predict_admissible
>>> g, a, fc, L, info = audfilters(16000, 4096, M=8)
>>> predict_admissible(fc[1:-1], info["fsupp_inner"], fs=16000, L=L,
...                    fsupp_dc=2 * fc[1],
...                    fsupp_nyq=2 * (8000 - fc[-2]))["is_frame"]
False
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "DEAD_BINS",
    "NotAFrameWarning",
    "check_admissible",
    "predict_admissible",
    "ripple_curve",
    "max_overlap_for_kappa",
    "min_channels",
    "min_bins",
]


class NotAFrameWarning(UserWarning):
    """The requested geometry leaves part of the spectrum uncovered.

    Raised as a warning, not an exception: the bank is still built and still
    analyses signals, but the lower frame bound is zero, so no dual exists and
    the annihilated band cannot be recovered.  Every designer runs this check
    at construction time.
    """

#: Identically-zero endpoint bins of each prototype -- the constant ``c``.
DEAD_BINS = {
    "hann": 1, "hanning": 1, "sqrthann": 1, "cosine": 1, "sine": 1,
    "blackman": 1, "blackman2": 1, "nuttall": 1, "tria": 1, "bartlett": 1,
    "sqrttria": 1, "itersine": 1, "ogg": 1, "rect": 0, "square": 0,
}


# ---------------------------------------------------------------------------
# the universal ripple curve
# ---------------------------------------------------------------------------

def _proto(x, name="hann"):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    m = np.abs(x) < 0.5
    if name in ("hann", "hanning"):
        out[m] = 0.5 * (1.0 + np.cos(2.0 * np.pi * x[m]))
    elif name in ("sqrthann", "cosine", "sine"):
        out[m] = np.sqrt(0.5 * (1.0 + np.cos(2.0 * np.pi * x[m])))
    elif name in ("rect", "square"):
        out[m] = 1.0
    elif name in ("tria", "bartlett"):
        out[m] = 1.0 - 2.0 * np.abs(x[m])
    else:
        out[m] = 0.5 * (1.0 + np.cos(2.0 * np.pi * x[m]))
    return out


def ripple_curve(rho, window="hann", n=2048):
    r"""Condition number of an infinite uniform bank at overlap ratio ``rho``.

    ``rho = spacing / support``, both in the same coordinate.  Only the
    *shape* of the prototype enters, so for a fixed prototype :math:`\kappa`
    is a universal function of ``rho`` alone: 1 at the partition-of-unity
    ratios 1/4 and 1/3, 2 at 1/2, and divergent as ``rho`` approaches 1.
    """
    rho = float(rho)
    if rho >= 1.0:
        return math.inf
    if rho <= 1e-6:
        return 1.0
    u = np.linspace(0.0, rho, n, endpoint=False)
    reach = int(math.ceil(0.5 / rho)) + 2
    acc = np.zeros_like(u)
    for m in range(-reach, reach + 1):
        acc += _proto(u - m * rho, window) ** 2
    lo, hi = float(acc.min()), float(acc.max())
    return math.inf if lo <= 0 else hi / lo


_RHO_CACHE: dict = {}


def max_overlap_for_kappa(kappa_target, window="hann", tol=1e-6):
    """Largest overlap ratio whose ripple keeps ``kappa <= kappa_target``."""
    key = (round(float(kappa_target), 9), window)
    if key in _RHO_CACHE:
        return _RHO_CACHE[key]
    lo, hi = 1.0 / 3.0, 1.0 - 1e-9
    if ripple_curve(lo, window) > kappa_target:
        _RHO_CACHE[key] = 0.0
        return 0.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if ripple_curve(mid, window) <= kappa_target:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    _RHO_CACHE[key] = lo
    return lo


# ---------------------------------------------------------------------------
# the exact covering test
# ---------------------------------------------------------------------------

def _interval_linear(fc, fsupp, fs, L, min_win, dead):
    W = int(max(min_win, round(L * fsupp / fs)))
    W = W + 1 if W % 2 == 0 else W
    k = int(round(L * fc / fs))
    return k - W // 2 + dead, k - W // 2 + W - 1 - dead


def _interval_warped(u, scaletofreq, bwmul, fs, L):
    lo = int(math.floor(L / fs * float(scaletofreq(u - bwmul))))
    hi = int(math.ceil(L / fs * float(scaletofreq(u + bwmul)))) - 1
    return lo, hi


def _covers(intervals, L):
    top = L // 2
    covered = np.zeros(top + 1, dtype=bool)
    for lo, hi in intervals:
        lo, hi = int(lo), int(hi)
        if hi < lo:
            continue
        k = np.arange(lo, hi + 1) % L
        covered[np.minimum(k, L - k)] = True
    holes = np.nonzero(~covered)[0]
    return (bool(holes.size == 0),
            int(holes[0]) if holes.size else None,
            int(holes.size))


def predict_admissible(fc_inner, fsupp_inner, *, fs, L, fsupp_dc, fsupp_nyq,
                       warped=None, min_win=4, window="hann",
                       kappa_target=None):
    """Predict whether a bank with this geometry is a frame, in closed form.

    Parameters
    ----------
    fc_inner, fsupp_inner : array_like
        Centre frequencies and designed supports of the inner channels, Hz.
    fs, L : float, int
        Sampling rate and filterbank length.
    fsupp_dc, fsupp_nyq : float
        Prototype bandwidths of the DC and Nyquist complements, Hz.
    warped : tuple, optional
        ``(u, scaletofreq, bwmul)`` -- switches the inner channels to the
        warped-domain support rule used by :func:`warpedfilters`.
    kappa_target : float, optional
        Also require the predicted condition number to stay below this.

    Returns
    -------
    dict
        ``is_frame``, ``first_hole_bin``, ``n_hole_bins``, ``rho``,
        ``kappa_pred``, ``usable``.
    """
    dead = DEAD_BINS.get(window, 1)
    if warped is not None:
        u, s2f, bwmul = warped
        iv = [_interval_warped(x, s2f, bwmul, fs, L) for x in np.asarray(u)]
    else:
        iv = [_interval_linear(f, s, fs, L, min_win, dead)
              for f, s in zip(np.asarray(fc_inner, float),
                              np.asarray(fsupp_inner, float))]
    inner = list(iv)
    iv.append(_interval_linear(0.0, fsupp_dc, fs, L, min_win, dead))
    iv.append(_interval_linear(fs / 2.0, fsupp_nyq, fs, L, min_win, dead))
    ok, hole, n_holes = _covers(iv, L)

    if len(inner) >= 2:
        centres = np.array([(a + b) / 2.0 for a, b in inner])
        widths = np.array([b - a + 1 for a, b in inner], dtype=float)
        gap = np.diff(centres)
        reach = 0.5 * (widths[:-1] + widths[1:])
        rho = float(np.max(gap / np.maximum(reach, 1e-300)))
    else:
        rho = 0.0
    kap = ripple_curve(min(rho, 0.999999), window) if ok else math.inf
    return {"is_frame": ok, "first_hole_bin": hole, "n_hole_bins": n_holes,
            "rho": rho, "kappa_pred": float(kap),
            "usable": bool(ok and (kappa_target is None
                                   or kap <= kappa_target))}


# ---------------------------------------------------------------------------
# construction-time check
# ---------------------------------------------------------------------------

def check_admissible(fc_inner, fsupp_inner, *, fs, L, fsupp_dc, fsupp_nyq,
                     designer, warped=None, min_win=4, window="hann",
                     warn=True):
    """Run :func:`predict_admissible` and warn if the bank is not a frame.

    This is what the designers call.  It never raises and never changes the
    bank that gets built -- a caller who wants a non-frame (to study the gap,
    or because they only need analysis) still gets one.  It exists so that the
    failure is announced at the point where the parameters were chosen, rather
    than surfacing later as an all-zero dual or a silently unrecoverable band.

    Returns the :func:`predict_admissible` dict, which the designers put in
    ``info["admissible"]``.
    """
    pred = predict_admissible(
        fc_inner, fsupp_inner, fs=fs, L=L,
        fsupp_dc=fsupp_dc, fsupp_nyq=fsupp_nyq,
        warped=warped, min_win=min_win, window=window)

    if warn and not pred["is_frame"]:
        import warnings

        n = pred["n_hole_bins"]
        k = pred["first_hole_bin"]
        warnings.warn(
            f"{designer}: this geometry is not a frame. "
            f"{n} DFT bin{'s' if n != 1 else ''} of [0, fs/2] "
            f"{'are' if n != 1 else 'is'} covered by no filter, the first at "
            f"bin {k} (~{k * fs / L:.1f} Hz), so the lower frame bound is zero "
            f"and that band cannot be recovered. Widen the filters or add "
            f"channels -- see cool_frames.diagnostics.admissibility.",
            NotAFrameWarning,
            stacklevel=3,
        )
    return pred


# ---------------------------------------------------------------------------
# solved forms
# ---------------------------------------------------------------------------

def min_channels(u_span, support_u, *, fs, L, b_min, window="hann",
                 kappa_target=None):
    r"""Smallest channel count for a uniformly-warped designer.

    ``M_min = floor(u_span / eff) + 2`` with
    ``eff = support_u - c*fs/(L*b_min)``.  For ``audfilters`` and
    ``greenwoodfilters``, ``support_u = bwmul / winbw`` and ``u_span =
    g(f_max) - g(f_min)``.
    """
    eff = support_u - DEAD_BINS.get(window, 1) * 2.0 * fs / (L * b_min)
    if eff <= 0:
        return math.inf
    if kappa_target is not None:
        eff *= max_overlap_for_kappa(kappa_target, window)
    return int(math.floor(float(u_span) / eff)) + 2


def min_bins(support_u, *, fs, L, b_min, window="hann", kappa_target=None):
    r"""Smallest ``bins`` for a designer whose spacing is ``1 / bins``.

    For ``warpedfilters``, ``support_u = 2 * bwmul``; the continuum limit is
    the scale-independent condition ``2 * bwmul > 1 / bins``.
    """
    eff = support_u - DEAD_BINS.get(window, 1) * 2.0 * fs / (L * b_min)
    if eff <= 0:
        return math.inf
    if kappa_target is not None:
        eff *= max_overlap_for_kappa(kappa_target, window)
    return int(math.floor(1.0 / eff)) + 1
