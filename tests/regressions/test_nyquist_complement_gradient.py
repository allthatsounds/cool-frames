"""The Nyquist complement straddles the fold, and the phase gradient must cope.

A real filterbank's top channel has to be symmetric about ``L/2``: the band
just below Nyquist and its mirror just above are the same band. So its support
*crosses* ``L/2`` -- as do the upper tails of the topmost ordinary channels.

``comp_phasegradfilters`` weights each filter by its signed DFT index. Until
v0.1.1 it reduced each index mod ``L`` and centred it about ``L/2``
independently, which splits such a support: the lower half keeps indices near
``+L/2``, the upper half is thrown to near ``-L/2``, and the frequency-weighted
response very nearly cancels. The instantaneous frequency then came out near
zero instead of near Nyquist -- an error of about ``fs/2``.

Measured on cqtfilters at fs = 8000, Ls = 4096 (L = 5832, complement on bins
2774..3058 with L/2 = 2916), against MATLAB LTFAT's own ``filterbankphasegrad``
on the same bank: the complement's deviation read **-5106.8 Hz** where LTFAT
reports **0.0**, while every other channel already agreed to about 0.6 Hz.
With the support kept contiguous it reads 18.0 Hz and nothing else moves.

Note what these tests do *not* claim. On a symmetric real channel the phase
gradient collapses to the channel's own centre frequency, because such a
channel carries no net phase advance -- that is a property of the geometry, not
a defect, and it is why the tolerances below are stated against ``fc`` rather
than against the probe tone.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np
from cool_frames.filters import audfilters, cqtfilters
from cool_frames.numpy.phase import filterbankphasegrad

FS = 8000
LS = 4096


def _channel_median_if(f, g, a, L, m):
    """Median estimated instantaneous frequency in Hz on channel ``m``."""
    from cool_frames.filterbanks import filterbank

    tg, _fg, _s, _c = filterbankphasegrad(f, g, a, L)
    tg = np.concatenate([np.asarray(x).ravel() for x in tg])
    cc = filterbank(f, g, a, L=L)
    mag = np.concatenate([np.abs(np.asarray(x)).ravel() for x in cc])
    ar = np.asarray(a, float)
    ar = ar[:, 0] / ar[:, 1] if ar.ndim == 2 else ar
    N = np.ceil(L / ar).astype(int)
    chan = np.repeat(np.arange(len(g)), N)
    sel = chan == (m % len(g))
    mm = mag[sel]
    if mm.max() <= 0:
        pytest.skip("complement carries no energy for this probe")
    live = mm > 0.3 * mm.max()
    return float(np.median(tg[sel][live])) * FS / 2


@pytest.mark.parametrize(
    "designer,kw,tone",
    [
        ("audfilters", {}, 3996.0),
        ("cqtfilters", dict(fmin=50.0, fmax=3900.0, bins=12), 3950.0),
    ],
)
def test_nyquist_complement_reports_near_nyquist_not_near_zero(designer, kw, tone):
    """The regression itself: a tone inside the complement's band must not be
    reported as a near-DC instantaneous frequency."""
    warnings.simplefilter("ignore")
    build = {"audfilters": audfilters, "cqtfilters": cqtfilters}[designer]
    g, a, _fc, L, _info = build(FS, LS, **kw)
    t = np.arange(LS) / FS
    est = _channel_median_if(np.sin(2 * np.pi * tone * t), g, a, L, -1)

    # The complement is symmetric about the fold, so the estimate collapses to
    # its own centre frequency -- fs/2.  That is the geometry, not an error.
    # What must not happen is the old behaviour, which landed near 0.
    assert abs(est - FS / 2) < 0.05 * FS, (
        f"{designer}: Nyquist complement reports {est:.1f} Hz for a {tone} Hz "
        f"tone; expected about {FS / 2} Hz. A value near 0 means the signed "
        f"DFT index is splitting the support at L/2 again."
    )
    assert abs(est) > 0.25 * FS, (
        f"{designer}: Nyquist complement reports {est:.1f} Hz, which is the "
        f"near-zero signature of the pre-v0.1.1 index split."
    )


@pytest.mark.parametrize(
    "designer,kw",
    [
        ("audfilters", {}),
        ("cqtfilters", dict(fmin=50.0, fmax=3900.0, bins=12)),
    ],
)
def test_which_channels_straddle_the_fold(designer, kw):
    """Pins the premise, so a designer change that moves the crossing shows up
    here rather than as a mysterious gradient.

    It is *not* only the complement.  The topmost ordinary channels spill their
    upper tails past ``L/2`` as well -- three channels on audfilters, two on
    cqtfilters at these settings.  They were only mildly affected by the old
    index split because most of their energy sits below the fold, so the
    cancellation was partial; they agreed with LTFAT to about 1 Hz throughout.
    The complement, centred exactly on the fold, was the one that broke.
    """
    warnings.simplefilter("ignore")
    build = {"audfilters": audfilters, "cqtfilters": cqtfilters}[designer]
    g, _a, _fc, L, _info = build(FS, LS, **kw)
    straddling = []
    for m, gm in enumerate(g):
        H = gm["H"]
        H = np.asarray(H(L) if callable(H) else H).ravel()
        fo = gm["foff"]
        fo = int(fo(L)) if callable(fo) else int(fo)
        nz = np.flatnonzero(np.abs(H) > 1e-10)
        if not nz.size:
            continue
        lo, hi = fo + nz[0], fo + nz[-1]
        if lo < L // 2 < hi:
            straddling.append(m)
    M = len(g)
    assert straddling, f"{designer}: no channel crosses L/2 — the premise is gone"
    assert straddling[-1] == M - 1, (
        f"{designer}: the Nyquist complement (channel {M - 1}) must cross L/2; "
        f"crossing channels are {straddling}"
    )
    # A contiguous run at the top: anything else means the geometry moved.
    assert straddling == list(range(straddling[0], M)), (
        f"{designer}: expected the crossing channels to be a contiguous run "
        f"ending at the complement, got {straddling}"
    )
    assert M - straddling[0] <= 4, (
        f"{designer}: {M - straddling[0]} channels now cross L/2, which is more "
        f"than the top few — the fold handling deserves another look"
    )


@pytest.mark.parametrize(
    "designer,kw,tone",
    [
        ("audfilters", {}, 1000.0),
        ("cqtfilters", dict(fmin=50.0, fmax=3900.0, bins=12), 1000.0),
    ],
)
def test_inner_channels_are_untouched_by_the_fix(designer, kw, tone):
    """The interior is where the signal path is exact, and the index change
    must not have disturbed it."""
    warnings.simplefilter("ignore")
    build = {"audfilters": audfilters, "cqtfilters": cqtfilters}[designer]
    g, a, fc, L, _info = build(FS, LS, **kw)
    t = np.arange(LS) / FS
    m = int(np.argmin(np.abs(np.asarray(fc, float) - tone)))
    est = _channel_median_if(np.sin(2 * np.pi * tone * t), g, a, L, m)
    assert abs(est - tone) < 1.0, (
        f"{designer}: inner channel {m} reports {est:.3f} Hz for a {tone} Hz "
        f"tone; the signal path is exact here and must stay so"
    )
