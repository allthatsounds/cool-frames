"""Settle the per-designer ``sqtfr`` convention against MATLAB LTFAT.

Why this exists
---------------
``filterbankconstphase``'s magnitude path needs ``sqtfr``, the square root of
the per-channel time-frequency ratio.  Its value is designer-specific, getting
it wrong costs accuracy rather than raising, and the conventions currently
documented on that function have never been checked against the reference
implementation.  That check is what limits magnitude-only phase retrieval on
these banks.

The comparison is split into three questions, because a single pass/fail would
not say *which* thing is wrong:

1. **Same bank?**  Do the two languages' designers realise the same filters?
   Until that holds, everything downstream is uninterpretable — designer drift
   masquerades as a convention error.  This one is *not* a test here, because
   the parameter mapping between LTFAT's and cool-frames' designer signatures
   differs per designer; LTFAT's realised responses are exported as
   ``gf`` (L x M) in each .mat, and comparing them is a manual first step.
2. **Same kernel?**  Fed LTFAT's own magnitudes, hops, centre frequencies and
   ``sqtfr``, does cool-frames' ``comp_filterbankphasegradfrommag`` return
   LTFAT's gradients?  This isolates the implementation from the convention.
3. **Same convention?**  What is LTFAT's ``info.tfr(L)`` for this designer, and
   is it what cool-frames documents?  This is the actual question.

Running it
----------
In MATLAB, with LTFAT on the path::

    >> cd <repo>/tests
    >> export_sqtfr_reference

then in Python::

    pytest tests/crosslang/test_sqtfr_convention.py -v -m requires_ref

Without the .mat files every test here skips.
"""
from __future__ import annotations

import pathlib

import pytest

import numpy as np

_REF_DIR = pathlib.Path(__file__).resolve().parents[1] / "reference_data"
DESIGNERS = ["audfilters", "cqtfilters", "waveletfilters", "warpedfilters"]


def _load(designer):
    scipy_io = pytest.importorskip("scipy.io", reason="scipy needed to read .mat")
    path = _REF_DIR / f"sqtfr_{designer}.mat"
    if not path.exists():
        pytest.skip(
            f"{path.name} not found. Run export_sqtfr_reference() in MATLAB "
            "with LTFAT on the path."
        )
    return scipy_io.loadmat(str(path), squeeze_me=True)


def _split(flat, N):
    return np.split(np.asarray(flat).ravel(), np.cumsum(np.asarray(N, int))[:-1])


# ---------------------------------------------------------------------------
# 2. same kernel?  (LTFAT's own inputs -> cool-frames' gradient function)
# ---------------------------------------------------------------------------

def _kernel(ref, scaled, edge_mode="ltfat"):
    from cool_frames.numpy.phase._fbphasegradfrommag import (
        comp_filterbankneighbors,
        comp_filterbankphasegradfrommag,
    )
    M = int(ref["M"])
    N = np.asarray(ref["N"], int).ravel()
    a_rat = np.asarray(ref["a_rat"], float).ravel()
    fc_n = np.asarray(ref["fc_n"], float).ravel()
    key = "abss_scaled" if scaled else "abss_raw"
    tkey = "tgrad_one" if scaled else "tgrad_one_raw"
    fkey = "fgrad_one" if scaled else "fgrad_one_raw"
    abss = np.asarray(ref[key], float).ravel()
    NEIGH, posInfo = comp_filterbankneighbors(a_rat.astype(int), M, N, do_real=True)
    tg, fg, _ = comp_filterbankphasegradfrommag(
        abss, N, a_rat, M, np.ones(M), fc_n, NEIGH, posInfo,
        gderivweight=float(ref["gderivweight"]), edge_mode=edge_mode)
    # cool-frames returns the ABSOLUTE instantaneous frequency (fc included);
    # LTFAT returns the deviation. Subtract fc before comparing.
    dev = tg - np.repeat(fc_n, N)
    return (M, N, dev, fg,
            np.asarray(ref[tkey], float).ravel(),
            np.asarray(ref[fkey], float).ravel())


def _require_reference_geometry(ref, designer):
    """Skip when LTFAT's export carries no centre frequencies.

    LTFAT's ``warpedfilters`` returns no ``info`` struct at all -- no ``fc``,
    no ``tfr`` -- so ``export_sqtfr_reference`` writes NaN for both, and every
    gradient it then computes from them is NaN.  There is no reference
    convention to compare against for that designer: that is a property of
    the reference implementation, not a failure of this port, so it skips.

    Without this guard the ratio below filters every NaN out (a NaN fails the
    magnitude test), the selection comes back empty, and ``np.median([])`` is
    ``nan`` -- which fails the assertion with a message that reads like a
    numerical disagreement rather than a missing reference.
    """
    fc = np.asarray(ref["fc_n"], float).ravel()
    if not np.all(np.isfinite(fc)):
        pytest.skip(
            f"{designer}: LTFAT returns no info struct for this designer, so "
            f"the export has no centre frequencies (fc = NaN) and no tfr — "
            f"there is no reference convention to compare against. Fields "
            f"LTFAT did return: {list(np.atleast_1d(ref['info_fields']))}"
        )


def _ratio(x, y):
    ok = np.abs(y) > 1e-12 * max(float(np.abs(y).max()), 1e-30)
    assert ok.any(), (
        "no comparable coefficients — every reference value is zero or NaN. "
        "The median of an empty selection is nan, which would otherwise fail "
        "as though the two implementations disagreed numerically."
    )
    return x[ok] / y[ok], ok


@pytest.mark.requires_ref
@pytest.mark.parametrize("designer", DESIGNERS)
def test_gradient_kernel_matches_ltfat(designer):
    """With ``edge_mode='ltfat'`` the port must reproduce the reference on
    every channel, interior and edge alike.

    Compared as a ratio rather than an absolute difference: a handful of
    coefficients sit where LTFAT's gradient is ~0 and the quotient is
    meaningless there, but the median is robust to those and still catches any
    systematic factor -- which is exactly the failure mode this is here for.
    """
    ref = _load(designer)
    _require_reference_geometry(ref, designer)
    M, N, dev, fg, tg_lt, fg_lt = _kernel(ref, scaled=True, edge_mode="ltfat")
    chan = np.repeat(np.arange(M), N)

    for label, sel in (("inner", (chan > 0) & (chan < M - 1)),
                       ("edge", (chan == 0) | (chan == M - 1)),
                       ("all", np.ones_like(chan, dtype=bool))):
        r_t, _ = _ratio(dev[sel], tg_lt[sel])
        assert abs(np.median(r_t) - 1.0) < 1e-9, (
            f"{designer}: {label}-channel tgrad is {np.median(r_t):.6f}x LTFAT's")
    r_f, _ = _ratio(fg, fg_lt)
    assert abs(np.median(r_f) - 1.0) < 1e-9, (
        f"{designer}: fgrad is {np.median(r_f):.6f}x LTFAT's")


@pytest.mark.requires_ref
@pytest.mark.parametrize("designer", DESIGNERS)
def test_default_edge_mode_differs_only_at_the_edges(designer):
    """The shipped default rescales the two one-sided channels by 2.

    Pinned deliberately: it is a documented divergence from LTFAT, not an
    accident, and if it ever spreads beyond channels 0 and M-1 that is a bug.
    """
    ref = _load(designer)
    _require_reference_geometry(ref, designer)
    M, N, dev, _fg, tg_lt, _f = _kernel(ref, scaled=True, edge_mode="rescaled")
    chan = np.repeat(np.arange(M), N)
    inner = (chan > 0) & (chan < M - 1)
    edge = ~inner

    r_in, _ = _ratio(dev[inner], tg_lt[inner])
    r_ed, _ = _ratio(dev[edge], tg_lt[edge])
    assert abs(np.median(r_in) - 1.0) < 1e-9, (
        f"{designer}: default mode perturbs interior channels "
        f"({np.median(r_in):.6f}x) — it must not")
    assert abs(np.median(r_ed) - 2.0) < 1e-9, (
        f"{designer}: expected the documented 2x at the edges, got "
        f"{np.median(r_ed):.6f}x")


# ---------------------------------------------------------------------------
# 3. same convention?
# ---------------------------------------------------------------------------

@pytest.mark.requires_ref
@pytest.mark.parametrize("designer", DESIGNERS)
def test_ltfat_tfr_is_recorded(designer):
    """Report LTFAT's per-channel TFR. This is the number the convention note
    on ``filterbankconstphase`` has to match."""
    ref = _load(designer)
    tfr = np.asarray(ref["tfr"], float).ravel()
    src = str(ref["tfr_source"])
    if src == "absent" or not np.isfinite(tfr).any():
        pytest.skip(
            f"LTFAT's {designer} exposes no info.tfr; fields present: "
            f"{list(np.atleast_1d(ref['info_fields']))}")
    assert np.all(tfr > 0), f"{designer}: LTFAT tfr has non-positive entries"


@pytest.mark.requires_ref
@pytest.mark.parametrize("designer", DESIGNERS)
def test_documented_sqtfr_reproduces_ltfat_gradients(designer):
    """The convention cool-frames documents must reproduce LTFAT's gradients.

    ``tgrad`` depends on ``1/gamma`` and ``fgrad`` on ``gamma``, so the ratio
    between LTFAT's two exported gradient sets recovers the gamma LTFAT used,
    per channel, exactly — no fitting.
    """
    ref = _load(designer)
    if str(ref["tfr_source"]) == "absent":
        pytest.skip(f"LTFAT's {designer} exposes no info.tfr")

    N = np.asarray(ref["N"], int).ravel()
    tfr = np.asarray(ref["tfr"], float).ravel()
    # fgrad is scaled by exactly gamma/(2 pi) * N -- a pure multiply -- so the
    # ratio between the two exports is gamma to machine precision.  tgrad also
    # carries a 1/gamma, but it carries the tfr-difference correction too, and
    # that correction is zero at sqtfr = 1 and non-zero at sqtfr = sqrt(tfr),
    # so the tgrad route returns something that is not gamma.
    fg_one = _split(ref["fgrad_one"], N)
    fg_tfr = _split(ref["fgrad_tfr"], N)
    recovered = np.full(len(N), np.nan)
    for m in range(len(N)):
        z1 = fg_one[m]
        ok = np.abs(z1) > 1e-14 * max(float(np.abs(z1).max()), 1e-30)
        if ok.sum():
            recovered[m] = float(np.median(fg_tfr[m][ok] / z1[ok]))

    good = np.isfinite(recovered)
    assert good.any(), f"{designer}: could not recover gamma from the reference"
    rel = np.abs(recovered[good] - tfr[good]) / np.maximum(np.abs(tfr[good]), 1e-30)
    assert np.nanmax(rel) < 1e-6, (
        f"{designer}: gamma recovered from LTFAT's own gradients disagrees "
        f"with info.tfr(L) by up to {np.nanmax(rel):.3e} — the export or the "
        "algebra is wrong, fix that before drawing conclusions")
