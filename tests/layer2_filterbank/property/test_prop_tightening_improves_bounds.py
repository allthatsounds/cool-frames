"""
test_prop_tightening_improves_bounds.py
=========================================
Python port of:
    layer2_filterbank/property/PropTighteningImprovesBounds.m

Properties:
(1) filterbanktight: A_tight ≈ B_tight (unit condition number).
(2) Condition number never increases: B_tight/A_tight <= B_orig/A_orig.
(3) Number of filters preserved.
(4) filterbankdual satisfies reciprocal bounds: Ad = 1/B, Bd = 1/A
    (checked via positive-frequency half of filterbankresponse).
(5) Original frame is a valid frame: A >= 0, B < inf, A <= B.
"""

from __future__ import annotations

import pytest

import numpy as np


def _pos_freq_bounds(g, a, L):
    # Real-signal frame bounds via the folded response (kappa matches the SVD
    # ground truth). Formerly took min/max over the un-folded positive half to
    # dodge filterbankbounds returning A=0 on one-sided banks; the fold in
    # filterbankbounds(real=True) now handles that correctly.
    from cool_frames.filterbanks import filterbankbounds  # type: ignore
    return filterbankbounds(g, a, L, real=True)


@pytest.mark.requires_impl
class TestTightFrameUnitConditionImpl:
    """PropTighteningImprovesBounds: filterbanktight gives A_tight ≈ B_tight."""

    def test_tight_unit_condition(self, needs_impl):
        from cool_frames.filterbanks import filterbanktight  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gt = filterbanktight(g, a, L)
        At, Bt = _pos_freq_bounds(gt, a, L)
        assert At > 0, "Positive-freq tight lower bound must be positive"
        assert abs(At - Bt) / (At + 1e-15) < 0.01, \
            f"Tight frame pos-freq: A={At:.8f}, B={Bt:.8f} — not equal"


@pytest.mark.requires_impl
class TestTighteningDoesNotWorsenConditionImpl:
    """PropTighteningImprovesBounds: condition number does not increase."""

    def test_condition_not_worsened(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds, filterbanktight  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        A_orig, B_orig = filterbankbounds(g, a, L)
        gt       = filterbanktight(g, a, L)
        At, Bt   = filterbankbounds(gt, a, L)
        cond_orig  = B_orig / (A_orig + 1e-15)
        cond_tight = Bt     / (At     + 1e-15)
        assert cond_tight <= cond_orig + 1e-6, \
            f"Tight cond={cond_tight:.6f} exceeds original cond={cond_orig:.6f}"


@pytest.mark.requires_impl
class TestTighteningPreservesFilterCountImpl:
    """PropTighteningImprovesBounds: filterbanktight preserves M."""

    def test_filter_count_preserved(self, needs_impl):
        from cool_frames.filterbanks import filterbanktight  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        gt = filterbanktight(g, a, L)
        assert len(gt) == len(g), \
            f"filterbanktight changed filter count: {len(g)} → {len(gt)}"


@pytest.mark.requires_impl
class TestDualReciprocalBoundsImpl:
    """PropTighteningImprovesBounds: dual bounds are reciprocal of original."""

    def test_dual_reciprocal_bounds(self, needs_impl):
        from cool_frames.filterbanks import filterbankdual  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gd = filterbankdual(g, a, L)
        A_orig, B_orig = _pos_freq_bounds(g, a, L)
        Ad_eff, Bd_eff = _pos_freq_bounds(gd, a, L)
        assert abs(Ad_eff - 1.0 / B_orig) < 1e-3, \
            f"Dual Ad={Ad_eff:.8f}, expected 1/B={1/B_orig:.8f}"
        assert abs(Bd_eff - 1.0 / A_orig) < 1e-3, \
            f"Dual Bd={Bd_eff:.8f}, expected 1/A={1/A_orig:.8f}"


@pytest.mark.requires_impl
class TestOriginalFrameValidImpl:
    """PropTighteningImprovesBounds: original filterbank is a valid frame."""

    def test_original_is_frame(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        A, B = filterbankbounds(g, a, L)
        assert A >= 0,     f"Lower bound A={A:.8f} must be non-negative"
        assert B < np.inf, "Upper bound B must be finite"
        assert A <= B + 1e-10, f"Bounds not ordered: A={A}, B={B}"
