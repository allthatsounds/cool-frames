"""
test_prop_biquad_filterbank.py
==============================
Python port of:
    layer2_filterbank/property/PropBiquadFilterbank.m

Verifies that a biquad-filter-based filterbank respects the same
frame-theoretic properties as blfilter-based ones:

(1) filterbankresponse is real and non-negative.
(2) filterbankbounds gives 0 < A <= B < inf.
(3) Analysis is linear.
(4) Frame inequality: A||x||² <= Σ(1/a_m)||c_m||² <= B||x||²
(5) filterbankscale multiplies the response by s².
(6) filterbankdual + ifilterbank gives perfect reconstruction.
(7) filterbanktight produces A_tight ≈ B_tight.
(8) Sum of individual responses equals total response.
"""

from __future__ import annotations

import pytest

import numpy as np


def _make_biquad_bank():
    """Return (g, a, L, M) for a small biquad filterbank."""
    from cool_frames.filters import (
        biquadfilter,  # type: ignore
        filterbanklength,  # type: ignore
    )
    Ls = 512
    M_biq = 6
    fcs = np.linspace(0.1, 0.9, M_biq)
    bws = 0.12 * np.ones(M_biq)
    g   = biquadfilter(fcs, bws)   # returns list of filter dicts
    a   = np.ones(M_biq, dtype=int)
    L   = filterbanklength(Ls, a)
    return g, a, L, M_biq, Ls


@pytest.mark.requires_impl
class TestBiquadResponseImpl:
    """PropBiquadFilterbank: filterbankresponse is real and non-negative."""

    def test_response_real_and_positive(self, needs_impl):
        from cool_frames.filterbanks import filterbankresponse  # type: ignore
        g, a, L, M, Ls = _make_biquad_bank()
        gf = np.real(np.asarray(filterbankresponse(g, a, L)))
        assert np.all(gf >= -1e-12), \
            f"biquad filterbankresponse must be non-negative; min={gf.min():.2e}"


@pytest.mark.requires_impl
class TestBiquadFrameBoundsImpl:
    """PropBiquadFilterbank: frame bounds are finite and ordered."""

    def test_bounds_ordered_finite(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds  # type: ignore
        g, a, L, M, Ls = _make_biquad_bank()
        A, B = filterbankbounds(g, a, L)
        assert A > 0,        f"biquad lower bound A={A} must be positive"
        assert B < np.inf,   "biquad upper bound B must be finite"
        assert A <= B + 1e-12, f"biquad bounds not ordered: A={A}, B={B}"


@pytest.mark.requires_impl
class TestBiquadLinearityImpl:
    """PropBiquadFilterbank: analysis is linear."""

    def test_linearity(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        g, a, L, M, Ls = _make_biquad_bank()
        rng   = np.random.default_rng(7)
        alpha = 1.5 + 0.7j
        beta  = -0.3 + 1.2j
        x = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
        y = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
        cx    = filterbank(x, g, a)
        cy    = filterbank(y, g, a)
        csum  = filterbank(alpha * x + beta * y, g, a)
        for m in range(M):
            expected = alpha * np.asarray(cx[m]) + beta * np.asarray(cy[m])
            rel_err  = np.linalg.norm(np.asarray(csum[m]) - expected) / \
                       (np.linalg.norm(expected) + 1e-15)
            assert rel_err < 1e-10, \
                f"biquad linearity band {m}: err={rel_err:.2e}"


@pytest.mark.requires_impl
class TestBiquadFrameInequalityImpl:
    """PropBiquadFilterbank: frame inequality A||x||² <= Σ … <= B||x||²."""

    def test_frame_inequality(self, needs_impl):
        from cool_frames.filterbanks import filterbank, filterbankbounds  # type: ignore
        g, a, L, M, Ls = _make_biquad_bank()
        # Complex test signals through a two-sided (realonly=0) bank -> use the
        # complex/two-sided bounds. The default real=True folds the response,
        # which doubles the bounds for realonly=0 filters and would not bracket
        # complex-signal energy ratios.
        A, B = filterbankbounds(g, a, L, real=False)
        rng  = np.random.default_rng(42)
        for trial in range(20):
            x   = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            c   = filterbank(x, g, a)
            ex  = np.linalg.norm(x) ** 2
            eTx = sum(np.linalg.norm(np.asarray(cm)) ** 2 / float(a_m)
                      for cm, a_m in zip(c, np.asarray(a).ravel()))
            assert eTx >= (A - 1e-6) * ex, \
                f"Trial {trial}: weighted energy {eTx:.4f} < A·||x||²={A*ex:.4f}"
            assert eTx <= (B + 1e-6) * ex, \
                f"Trial {trial}: weighted energy {eTx:.4f} > B·||x||²={B*ex:.4f}"


@pytest.mark.requires_impl
class TestBiquadScaleImpl:
    """PropBiquadFilterbank: filterbankscale multiplies response by s²."""

    def test_scale_multiplies_response(self, needs_impl):
        from cool_frames.filterbanks import (  # type: ignore
            filterbankresponse,
            filterbankscale,
        )
        g, a, L, M, Ls = _make_biquad_bank()
        gf_orig = np.real(np.asarray(filterbankresponse(g, a, L)))
        for s in [0.5, 2.0, 3.0]:
            gs        = filterbankscale(g, s)
            gf_scaled = np.real(np.asarray(filterbankresponse(gs, a, L)))
            err = np.linalg.norm(gf_scaled - s ** 2 * gf_orig) / \
                  (np.linalg.norm(gf_orig) + 1e-15)
            assert err < 1e-10, \
                f"biquad scale s={s}: response scaling error {err:.2e}"


@pytest.mark.requires_impl
class TestBiquadDualPRImpl:
    """PropBiquadFilterbank: filterbankdual → perfect reconstruction."""

    def test_dual_pr(self, needs_impl):
        from cool_frames.filterbanks import (  # type: ignore
            filterbank,
            filterbankdual,
            ifilterbank,
        )
        g, a, L, M, Ls = _make_biquad_bank()
        # Complex signals through a two-sided (realonly=0) bank: use the complex
        # dual and complex synthesis (real=False). (real=True is the single-sided
        # real-audio convention and would not invert a complex signal here.)
        gd  = filterbankdual(g, a, L, real=False)
        rng = np.random.default_rng(17)
        for trial in range(5):
            x  = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            c  = filterbank(x, g, a)
            xr = np.asarray(ifilterbank(c, gd, a, L, real=False))
            rel_err = np.linalg.norm(x - xr[:Ls]) / np.linalg.norm(x)
            assert rel_err < 1e-4, \
                f"biquad dual PR trial {trial}: err={rel_err:.2e}"


@pytest.mark.requires_impl
class TestBiquadTightImpl:
    """PropBiquadFilterbank: filterbanktight gives equal bounds."""

    def test_tight_equal_bounds(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds, filterbanktight  # type: ignore
        g, a, L, M, Ls = _make_biquad_bank()
        gt = filterbanktight(g, a, L)
        A_t, B_t = filterbankbounds(gt, a, L)
        rel_diff = abs(B_t - A_t) / (A_t + 1e-15)
        assert rel_diff < 1e-4, \
            f"biquad tight: A={A_t:.8f}, B={B_t:.8f}, rel_diff={rel_diff:.2e}"


@pytest.mark.requires_impl
class TestBiquadResponseIndividualImpl:
    """PropBiquadFilterbank: sum of individual responses equals total."""

    def test_individual_sum_equals_total(self, needs_impl):
        from cool_frames.filterbanks import (  # type: ignore
            filterbankfreqz,
            filterbankresponse,
        )
        from cool_frames.numpy.filterbanks._utils import normalise_a  # type: ignore
        g, a, L, M, Ls = _make_biquad_bank()
        gf_total = np.real(np.asarray(filterbankresponse(g, a, L)))
        # Compute individual responses manually: |H_m(k)|^2 / a_m
        a_norm = normalise_a(a, M)
        afrac = a_norm[:, 0] / a_norm[:, 1]
        H = filterbankfreqz(g, a_norm, L)
        gf_indiv = np.real(H * H.conj()) / afrac[np.newaxis, :]
        gf_sum = gf_indiv.sum(axis=1)
        err = np.linalg.norm(gf_total - gf_sum) / (np.linalg.norm(gf_total) + 1e-15)
        assert err < 1e-10, \
            f"biquad: sum of individual responses ≠ total (err={err:.2e})"
