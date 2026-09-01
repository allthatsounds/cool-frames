"""
test_findgamma.py
=================
``pghi_findgamma`` — the window constant PGHI scales its phase gradients by.

This module was covered at 11.8 %: the ``_PRECOMPUTED_CG`` table was imported
and nothing else ran.  The search path underneath it — ``_findbestgauss`` and
``_winwidthatheight``, 60 statements — had never been executed by a test.

What that hid
-------------
``_winwidthatheight`` is a *second copy* of a helper that lives in
``cool_frames/numpy/filters/_gabfilters.py``.  The audit found the copy over
there measuring every window as full-width — it scans ``g[:gl//2+1]`` for the
threshold crossing, which only tracks the falling flank if the window peaks at
index 0 — fixed it with a ``np.roll(g, -argmax)``, and wrote
``test_window_width_measurement_depends_on_window_shape`` to hold it.  The copy
in ``_findgamma.py`` kept the bug, because the two are private, near-identical,
and nothing compared them.

Measured consequence, for a 256-tap window:

===========  ==========  ======================  =============
window       table ``Cg``  numeric ``Cg`` (v0.1.1)  factor
===========  ==========  ======================  =============
hann            0.25645                  2.19460          8.6x
hamming         0.29794                  2.26669          7.6x
blackman        0.17954                  1.85417         10.3x
bartlett        0.27561                  2.02420          7.3x
cosine          0.41532                  2.54799          6.1x
===========  ==========  ======================  =============

The table row and the numeric row are the *same window*; the table is the
precomputed answer to the search.  Since ``gamma = Cg * gl**2`` multiplies the
phase gradients the heap integrator consumes, being 8.6x out does not blur the
phase estimate, it replaces it — the same failure mode as the missing
centre-frequency term the audit found in ``comp_filterbankphasegradfrommag``.

Anything reached through the window *name* was unaffected: named windows return
the tabulated constant and never enter the search.  Anyone who passed a numeric
window vector — which is what ``scipy.signal.get_window`` hands you, and the
only option for a window not in the table — got the wrong number with no
warning.

The tests below hold the fix, hold the two copies of the helper to each other,
and pin the residual gap that the fix does *not* close (see
``test_numeric_search_still_disagrees_with_the_table``).
"""

from __future__ import annotations

import pytest

import numpy as np
from cool_frames.numpy.phase import pghi_findgamma, wpghi_findgamma
from cool_frames.numpy.phase._findgamma import (
    _PRECOMPUTED_CG,
    _findbestgauss,
    _winwidthatheight,
)

GL = 256
TABULATED = ["hann", "hamming", "blackman", "bartlett", "cosine"]


def _window(name, gl=GL):
    from scipy.signal import get_window

    return np.asarray(get_window(name, gl, fftbins=True), dtype=float)


# ---------------------------------------------------------------------------
# The defect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TABULATED)
def test_window_width_is_independent_of_sample_ordering(name):
    """The same window in centred and DFT ordering must measure the same width.

    This is the assertion that was missing.  ``scipy.signal.get_window`` returns
    a *centred* window (peak at ``gl//2``); LTFAT's convention is DFT ordering
    (peak at index 0).  Both describe the same window and must yield the same
    constant.  Before the fix the centred form measured ``w = 256.6`` against
    the DFT form's ``w = 75.6`` for a 256-tap Hann — the scan ran up the rising
    flank instead of down the falling one and pinned at the peak.
    """
    centred = _window(name)
    wrapped = np.fft.ifftshift(centred)
    assert int(np.argmax(centred)) != 0, "fixture assumption: centred window peaks off-zero"
    assert int(np.argmax(wrapped)) == 0, "fixture assumption: wrapped window peaks at zero"

    for atheight in (0.1, 0.5, 0.8):
        w_c = _winwidthatheight(centred, atheight)
        w_w = _winwidthatheight(wrapped, atheight)
        assert abs(w_c - w_w) < 1e-9, (
            f"{name} at height {atheight}: centred ordering measures {w_c:.3f}, "
            f"DFT ordering measures {w_w:.3f} — the scan depends on sample order"
        )
        # And the width must be a fraction of the window, not the whole of it.
        assert 0.0 < w_c < GL, f"{name}: measured width {w_c:.1f} for a {GL}-tap window"


@pytest.mark.parametrize("name", TABULATED)
def test_gamma_is_independent_of_sample_ordering(name):
    """The public entry point inherits the property."""
    centred = _window(name)
    g_c, cg_c = pghi_findgamma(centred)
    g_w, cg_w = pghi_findgamma(np.fft.ifftshift(centred))
    assert abs(cg_c - cg_w) < 1e-12, f"{name}: Cg {cg_c:.6f} (centred) vs {cg_w:.6f} (DFT)"
    assert abs(g_c - g_w) < 1e-9


def test_the_two_copies_of_winwidthatheight_agree():
    """``_findgamma`` and ``_gabfilters`` each carry a private copy. Hold them together.

    The v0.1.1 audit fixed the ``_gabfilters`` one and left this one, which is
    how the 8.6x error survived the release.  Duplicated private helpers are
    fine; duplicated private helpers that nobody compares are how a fix lands in
    one of two places.  If the copies are ever unified, this test becomes
    trivially true rather than wrong, which is the right failure mode.
    """
    from cool_frames.numpy.filters._gabfilters import _winwidthatheight as gab_version

    rng = np.random.default_rng(0)
    cases = [_window(n) for n in TABULATED]
    cases.append(np.ones(64))  # rectangle
    cases.append(np.bartlett(101))
    # A deliberately lumpy window: the interpolation branches should still match.
    lumpy = np.hanning(128) * (1.0 + 0.1 * rng.standard_normal(128))
    cases.append(np.abs(lumpy))

    for i, w in enumerate(cases):
        for atheight in (0.05, 0.25, 0.5, 0.75):
            a = _winwidthatheight(w, atheight)
            b = gab_version(w, atheight)
            assert abs(a - b) < 1e-9, (
                f"case {i} at height {atheight}: _findgamma gives {a:.6f}, "
                f"_gabfilters gives {b:.6f} — the two copies have diverged"
            )


def test_measured_width_orders_by_window_concentration():
    """A narrower window must measure narrower. The broken version did not.

    With the scan pinned at the peak, a needle, a Hann and a rectangle all
    returned ~``gl``.  A width measurement that cannot tell a three-bin spike
    from a rectangle is not measuring width.
    """
    n = 129
    needle = np.zeros(n)
    needle[n // 2 - 1 : n // 2 + 2] = [0.5, 1.0, 0.5]
    hann = np.hanning(n)
    rect = np.ones(n)

    w_needle, w_hann, w_rect = (_winwidthatheight(w, 0.5) for w in (needle, hann, rect))
    assert w_needle < w_hann < w_rect, (
        f"widths do not order by concentration: needle {w_needle:.2f}, "
        f"hann {w_hann:.2f}, rect {w_rect:.2f}"
    )


# ---------------------------------------------------------------------------
# The tabulated path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(_PRECOMPUTED_CG))
def test_named_windows_return_the_tabulated_constant(name):
    """``gamma = Cg * gl**2`` exactly, for every name in the table."""
    gamma, Cg = pghi_findgamma(name, gl=GL)
    assert Cg == pytest.approx(_PRECOMPUTED_CG[name], rel=0, abs=0)
    assert gamma == pytest.approx(_PRECOMPUTED_CG[name] * GL**2, rel=1e-12)
    assert gamma > 0.0


def test_named_window_lookup_is_case_insensitive_and_accepts_aliases():
    assert pghi_findgamma("HANN", gl=GL) == pghi_findgamma("hann", gl=GL)
    # hanning/hann, tria/triangular/bartlett, cosine/sine/sqrthann are aliases.
    assert pghi_findgamma("hanning", gl=GL) == pghi_findgamma("hann", gl=GL)
    assert pghi_findgamma("triangular", gl=GL) == pghi_findgamma("tria", gl=GL)
    assert pghi_findgamma("sine", gl=GL) == pghi_findgamma("cosine", gl=GL)


def test_gauss_uses_the_lattice_rather_than_the_table():
    """For 'gauss', gamma is ``a*M`` exactly and ``Cg`` is undefined."""
    gamma, Cg = pghi_findgamma("gauss", a=32, M=256)
    assert gamma == pytest.approx(32 * 256)
    assert np.isnan(Cg), "Cg is not defined for the true Gaussian; it should be NaN"


def test_missing_arguments_raise_rather_than_guessing():
    with pytest.raises(ValueError, match="requires a and M"):
        pghi_findgamma("gauss")
    with pytest.raises(ValueError, match="requires a and M"):
        pghi_findgamma("gauss", a=32)
    with pytest.raises(ValueError, match="window length"):
        pghi_findgamma("hann")
    with pytest.raises(ValueError, match="Unknown window name"):
        pghi_findgamma("definitely-not-a-window", gl=GL)


def test_M_substitutes_for_gl_when_gl_is_absent():
    """Documented fallback: ``M`` stands in for the window length."""
    assert pghi_findgamma("hann", M=128) == pghi_findgamma("hann", gl=128)


def test_numeric_window_infers_its_own_length():
    w = _window("hann", 128)
    assert pghi_findgamma(w) == pghi_findgamma(w, gl=128)


# ---------------------------------------------------------------------------
# What the fix does *not* fix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", TABULATED)
def test_numeric_search_still_disagrees_with_the_table(name):
    """Pin the residual gap, which the ordering fix narrows but does not close.

    The table is the precomputed answer to this search, so in a correct port the
    two would agree.  They do not: after the ordering fix the search comes out
    7-45 % high (Hann 0.309 vs 0.256, cosine 0.600 vs 0.415).  The cause is
    visible in ``_findbestgauss`` — its ``atheight`` pins at 0.8, the top of the
    hardcoded ``atheightrange``, so the minimum is at the boundary and the
    search has not converged to anything.  The module already flags itself a
    "simplified port".

    This test asserts the *current* accuracy, both bounds.  The lower bound is
    the regression guard (the 6-10x error must not come back); the upper bound
    is the improvement guard — narrowing the search will fail this test, which
    is the intended prompt to update the numbers deliberately.  Do not relax it
    to make an unrelated change pass.
    """
    _, cg_numeric = pghi_findgamma(_window(name))
    ratio = cg_numeric / _PRECOMPUTED_CG[name]
    assert 1.0 <= ratio < 1.5, (
        f"{name}: numeric search gives Cg={cg_numeric:.5f} against a tabulated "
        f"{_PRECOMPUTED_CG[name]:.5f} (ratio {ratio:.3f}). Outside [1.0, 1.5) this is "
        "either the ordering defect returning or an improvement worth recording."
    )


def test_findbestgauss_pins_at_the_top_of_its_search_range():
    """The evidence for the paragraph above, asserted rather than asserted-about.

    ``_findbestgauss`` searches ``np.arange(0.01, 0.801, 0.001)`` and returns
    0.8 — the *last element* — for four of the five tabulated windows.  An
    interior minimum means the search converged; a boundary one means the range
    is too narrow and the returned height is an artefact of where the loop
    stopped.  Bartlett is the exception, minimising at 0.285, and it is no more
    accurate for it (1.17x the table against Blackman's 1.08x), which says the
    residual error is not only the pinning.

    When someone widens the range or ports ``findbestgauss`` properly, this
    fails and ``test_numeric_search_still_disagrees_with_the_table`` should be
    retightened at the same time.
    """
    top = 0.8  # last element of the hardcoded atheightrange
    pinned = {name: _findbestgauss(_window(name)) for name in TABULATED}

    assert pinned["bartlett"] < top - 0.01, (
        f"bartlett was the one interior minimum at 0.285; it now returns {pinned['bartlett']:.4f}"
    )
    for name in ("hann", "hamming", "blackman", "cosine"):
        assert pinned[name] == pytest.approx(top, abs=1e-9), (
            f"{name}: _findbestgauss returned {pinned[name]:.4f}, not the range boundary. "
            "If the search now finds an interior minimum, this limitation is fixed — "
            "update this test and test_numeric_search_still_disagrees_with_the_table."
        )


def test_wpghi_findgamma_ignores_its_tfr_argument():
    """``wpghi_findgamma(g, tfr)`` accepts ``tfr`` positionally and discards it.

    The signature promises a filterbank-domain variant; the body is
    ``return pghi_findgamma(g, **kwargs)`` — ``tfr`` is never read.  A caller
    computing a per-channel time-frequency ratio and passing it in gets the
    plain Gabor answer and no indication that their argument went nowhere.

    Pinned, not fixed: making ``tfr`` do something is a design decision about
    what per-channel gamma should mean, not a typo.  Until then this records
    that the parameter is inert.
    """
    a = wpghi_findgamma("hann", None, gl=GL)
    b = wpghi_findgamma("hann", np.linspace(1.0, 99.0, 17), gl=GL)
    c = pghi_findgamma("hann", gl=GL)
    assert a == b == c, "wpghi_findgamma's tfr argument now has an effect — document it"
