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

The tests below hold the fix and hold the two copies of the helper to each
other.

The second defect: ``findbestgauss``
------------------------------------
With the ordering fixed the search was still 7-45 % above the table, and its
best height pinned at 0.8, the top of the range, for four of five windows.
``findbestgauss`` exists -- a local function in PHASERET's
``pghi_findgamma.m``, not a file of its own -- and the "simplified port" had
built its candidate Gaussians as ``exp(-pi l^2 / ((w/2)^2 / -ln ah))``, which
is ``ah**pi`` at half the measured width instead of ``ah``, and skipped
PHASERET's first step (cut the window to its width at 1e-10 of the peak, the
``gl`` the constant is scaled by).  Ported line by line it reproduces
PHASERET's constants to 1e-15 given LTFAT's windows, finds interior heights,
and is within 0.2 % of the table at 1024 taps (the table was evidently
computed at some large length; the search converges to it as ``gl`` grows).
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
# The search is PHASERET's
# ---------------------------------------------------------------------------

# PHASERET's pghi_findgamma(firwin(name, gl)) in Octave (LTFAT 2.6.0,
# PHASERET 0.2.5): gamma.  Only windows cool-frames' firwin produces
# identically to LTFAT's; its even-length Hamming, Blackman2 and Nuttall01
# keep a non-zero sample at +-gl/2 that LTFAT zeroes (recorded in
# DEFECT_REGISTER.md), which changes the window and so the constant.
_PHASERET_GAMMA = {
    ("hann", 256): 16924.434359588988,
    ("hann", 255): 16793.275261037976,
    ("hann", 1024): 269200.23939710855,
    ("hann", 100): 2613.3068807951104,
    ("blackman", 256): 11846.872794250598,
    ("blackman", 1024): 188431.7814448479,
    ("tria", 256): 18187.524663545588,
    ("tria", 255): 18046.267198722799,
    ("sqrthann", 256): 27421.793551177281,
    ("nuttall", 1024): 134422.71282238577,
    ("itersine", 100): 3643.3914241262464,
    ("sqrttria", 255): 31474.519362240248,
    ("hamming", 255): 19506.118275918707,
}


@pytest.mark.parametrize("key", sorted(_PHASERET_GAMMA))
def test_numeric_search_is_phaserets(key):
    """``pghi_findgamma(numeric window)`` gives PHASERET's constant."""
    from cool_frames.filters import firwin

    name, gl = key
    gamma, _Cg = pghi_findgamma(firwin(name, gl))
    assert gamma == pytest.approx(_PHASERET_GAMMA[key], rel=1e-9)


@pytest.mark.parametrize("name", ["hann", "blackman", "tria", "sqrthann"])
def test_numeric_search_converges_to_the_table(name):
    """The table is the search's answer for a long window: within 0.2 % at
    1024 taps (was 7-45 % above it at 256 before the port, 0.5-1.6 % now)."""
    from cool_frames.filters import firwin

    _, cg = pghi_findgamma(firwin(name, 1024))
    ratio = cg / _PRECOMPUTED_CG[name]
    assert 1.0 <= ratio < 1.002, f"{name}: Cg {cg:.6f} vs table {_PRECOMPUTED_CG[name]}"


@pytest.mark.parametrize("name", TABULATED)
def test_findbestgauss_finds_an_interior_minimum(name):
    """The best height is inside the search range, not pinned at its top
    (0.8, where the broken Gaussian put four of these five windows)."""
    ah = _findbestgauss(np.fft.ifftshift(_window(name)))
    assert 0.02 < ah < 0.79, f"{name}: best height {ah}"


def test_findbestgauss_candidates_have_the_measured_width_at_the_height():
    """The candidate for height ``ah`` and width ``w`` is ``ah`` at ``+-w/2``
    (``pgauss(..., 'width', w, 'atheight', ah)``); the old one was ``ah**pi``."""
    from cool_frames.filters import pgauss

    L, w, ah = 2560, 100.0, 0.3
    g = pgauss(L, width=w, atheight=ah, norm="inf")
    x = np.arange(L, dtype=float)
    val = np.interp(w / 2, x[:200], g[:200])
    assert val == pytest.approx(ah, rel=1e-3)


def test_numeric_window_refuses_a_different_gl():
    with pytest.raises(ValueError, match="numeric window"):
        pghi_findgamma(_window("hann", 128), gl=256)


# ---------------------------------------------------------------------------
# wpghi_findgamma: the time-frequency ratio
# ---------------------------------------------------------------------------


def test_wpghi_findgamma_converts_a_time_frequency_ratio():
    """``tfr`` used to be accepted and ignored.  It is now converted:
    the Gaussian ``pgauss(L, tfr)`` is ``exp(-pi l^2 / (tfr L))``, so its
    constant is ``tfr * L`` -- PHASERET's conversion for ``{'gauss', tfr}``,
    and what ``pghi_findgamma`` finds for that Gaussian."""
    from cool_frames.filters import pgauss

    L = 4096
    gamma, Cg = wpghi_findgamma(tfr=2.5, L=L)
    assert gamma == pytest.approx(2.5 * L)
    assert np.isnan(Cg)
    per_channel, _ = wpghi_findgamma(tfr=np.array([1.0, 2.0, 4.0]), L=L)
    np.testing.assert_allclose(per_channel, [L, 2 * L, 4 * L])
    from_callable, _ = wpghi_findgamma(tfr=lambda n: 3.0 / n * 1000, L=L)
    assert from_callable == pytest.approx(3000.0)

    # The window path agrees with it for the Gaussian itself.
    g = pgauss(L, 2.5)
    g_search, _ = pghi_findgamma(g)
    assert g_search == pytest.approx(2.5 * L, rel=0.02)

    # The window path is unchanged, and the two cannot be mixed.
    assert wpghi_findgamma("hann", gl=GL) == pghi_findgamma("hann", gl=GL)
    with pytest.raises(ValueError, match="not both"):
        wpghi_findgamma("hann", 2.0, L=L)
    with pytest.raises(ValueError, match="length L"):
        wpghi_findgamma(tfr=2.0)
