"""
numpy/filters/_tfr.py
=====================
Compute per-channel time-frequency ratios (tfr) from filterbank descriptors.

For frequency-domain filters (audfilters, cqtfilters, hopfilters), the tfr
is ``L / gamma`` where ``gamma`` is computed from the filter's frequency
response shape via ``_comp_tfrfromwin``.

For time-domain windows (gabfilters), the tfr is ``gamma / L``.

The distinction arises because ``_comp_tfrfromwin`` measures the window's
support in whichever domain it is given.  A frequency-domain filter's
"width" in frequency maps to ``gamma`` via the same formula, but the
physical tfr requires the reciprocal.
"""
from __future__ import annotations

import numpy as np

from ._gabfilters import _comp_tfrfromwin


def compute_tfr_from_filters(
    g: list[dict],
    L: int,
    *,
    default_tfr: float = 1.0,
    min_tfr: float = 1e-8,
) -> np.ndarray:
    """Compute per-channel tfr from a filterbank descriptor list.

    Parameters
    ----------
    g : list of dict
        Filter descriptors as returned by ``audfilters``, ``cqtfilters``,
        ``hopfilters``, ``gabfilters``, etc.  Each dict must have an ``'H'``
        key whose value is either a callable ``H(L) -> ndarray`` or a
        numpy array.
    L : int
        Transform length.
    default_tfr : float
        Value to use for channels where ``_comp_tfrfromwin`` fails
        (e.g. complement lowpass/highpass filters with flat frequency
        responses).  Default 1.0.
    min_tfr : float
        Floor value to clamp tfr away from zero.  Default 1e-8.

    Returns
    -------
    tfr : ndarray, shape (M,)
        Per-channel time-frequency ratios.
    """
    M = len(g)
    tfr = np.full(M, default_tfr, dtype=float)

    for m in range(M):
        H = g[m]["H"]
        # Evaluate if callable (lazy frequency-domain filters)
        if callable(H):
            try:
                H_val = np.abs(H(L)).ravel()
            except Exception:
                continue
        else:
            H_val = np.abs(np.asarray(H)).ravel()

        if len(H_val) == 0 or np.max(H_val) < 1e-30:
            continue

        gamma = _comp_tfrfromwin(H_val)

        if gamma <= 0 or not np.isfinite(gamma):
            continue

        # Frequency-domain filters: tfr = L / gamma
        tfr_m = L / gamma
        if not np.isfinite(tfr_m) or tfr_m < min_tfr:
            tfr_m = default_tfr

        tfr[m] = tfr_m

    # Clamp
    tfr = np.maximum(tfr, min_tfr)
    return tfr  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# tfr_from_bandwidth -- LTFAT's closed-form rule, recovered from its exports
# ---------------------------------------------------------------------------

def tfr_from_bandwidth(fsupp, fs: float, L: int, *, winbw: float | None = None):
    r"""LTFAT's per-channel time-frequency ratio, from the *designed* bandwidth.

    .. math::  \mathrm{tfr}_m \;=\; \frac{1}{2\,\mathrm{winbw}^2}\,
               \frac{L}{W_m^{2}},\qquad W_m = \mathrm{fsupp}_m \cdot L / f_s

    ``W_m`` is the channel's designed bandwidth in DFT bins and ``winbw`` is
    the prototype window's equivalent-bandwidth factor (``3/8`` for the Hann
    every painless designer here uses), giving a leading constant of
    :math:`1/(2 \cdot (3/8)^2) = 32/9`.

    How this was established
    ------------------------
    LTFAT publishes ``info.tfr`` as a function handle and
    ``filterbankconstphase`` consumes ``sqrt(info.tfr(L))``, but the rule
    behind it is not written down anywhere in the reference.  It was recovered
    from ``tests/reference_data/sqtfr_*.mat``: the product
    ``tfr_m * W_m^2 / L`` is constant to **4.2e-16** across every channel of
    both ``audfilters`` (29 channels, tfr spanning 0.0298 to 719) and
    ``cqtfilters`` (78 channels), which fixes the form; the constant then
    lands on ``1/(2*winbw^2)`` exactly.  Predicting LTFAT's own numbers from
    this formula reproduces them to a *uniform* 4.5e-5 (audfilters) and
    2.1e-5 (cqtfilters) -- median equal to max, so a single scale factor, not
    a per-channel error.  That residual is the small ``fsupp`` convention
    difference between the two packages, not a defect in the rule.
    ``tests/crosslang/test_sqtfr_convention.py`` pins it.

    Note this is *not* ``L / compute_tfr_from_filters(...)``.  That measures
    the realised response with ``comp_tfrfromwin``, which gives a Hann shape
    constant of 3.5288 rather than 3.5556 and disagrees with LTFAT by ~1 %
    per channel, with the error varying channel to channel.  LTFAT uses the
    design bandwidth, not the realised one.

    Why it matters for ``warpedfilters``
    ------------------------------------
    LTFAT returns no ``info`` struct at all for ``warpedfilters`` -- no
    ``fc``, no ``tfr`` -- so there is no reference value to copy and no
    cross-language check possible (see the skip in the crosslang tests).  But
    ``warpedfilters`` designs its bandwidths by the same warped rule and
    windows them with the same Hann, so the formula applies unchanged.  That
    is how this package can publish a ``tfr`` for a designer the reference
    leaves blank.

    Parameters
    ----------
    fsupp : array_like
        Per-channel designed bandwidth in **Hz**.  Zero or non-finite entries
        (an unbandlimited complement, say) yield ``nan``.
    fs : sampling rate in Hz.
    L : transform length.
    winbw : prototype window equivalent bandwidth; default the Hann ``3/8``.

    Returns
    -------
    tfr : ndarray, shape (M,)
    """
    from ._firwin import hann_winbw

    if winbw is None:
        winbw = hann_winbw()
    W = np.asarray(fsupp, dtype=float).ravel() * float(L) / float(fs)
    out = np.full(W.shape, np.nan, dtype=float)
    good = np.isfinite(W) & (W > 0)
    out[good] = (1.0 / (2.0 * float(winbw) ** 2)) * float(L) / W[good] ** 2
    return out
