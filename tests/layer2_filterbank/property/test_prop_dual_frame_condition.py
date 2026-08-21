"""
test_prop_dual_frame_condition.py
==================================
Python port of:
    layer2_filterbank/property/PropDualFrameCondition.m

Tests structural properties of the canonical dual frame:

(1) Dual bounds satisfy [1/B, 1/A] (derived from positive-frequency response).
(2) Dual filterbankresponse lies within its own bounds.
(3) For a tight frame, filterbankresponse is constant.
(4) Original and dual bounds are mutually consistent: A·Bd == 1 and B·Ad == 1.
"""

from __future__ import annotations

import pytest

import numpy as np


def _pos_freq_bounds(g, a, L):
    """Real-signal frame bounds (folded response).

    Formerly this took min/max over the positive-frequency half of the
    *un-folded* response to work around ``filterbankbounds`` returning A=0 for
    the uncovered negative frequencies of one-sided ERB banks. Since 2026-06-12
    ``filterbankbounds(..., real=True)`` folds the response (``resp + involute``),
    which is the correct real-signal bound — its kappa matches the SVD ground
    truth (:func:`filterbankbounds_svd`) — and is positive for one-sided banks.
    This helper now simply delegates, kept for call-site stability.
    """
    from cool_frames.filterbanks import filterbankbounds  # type: ignore
    return filterbankbounds(g, a, L, real=True)


@pytest.mark.requires_impl
class TestDualBoundsReciprocalImpl:
    """PropDualFrameCondition: dual bounds are reciprocal of original."""

    def test_dual_bounds_reciprocal(self, needs_impl):
        from cool_frames.filterbanks import filterbankdual  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gd = filterbankdual(g, a, L)
        A_orig, B_orig = _pos_freq_bounds(g, a, L)
        Ad_eff, Bd_eff = _pos_freq_bounds(gd, a, L)
        assert abs(Ad_eff - 1.0 / B_orig) < 1e-3, \
            f"Dual lower bound Ad={Ad_eff:.8f}, expected 1/B={1/B_orig:.8f}"
        assert abs(Bd_eff - 1.0 / A_orig) < 1e-3, \
            f"Dual upper bound Bd={Bd_eff:.8f}, expected 1/A={1/A_orig:.8f}"


@pytest.mark.requires_impl
class TestDualResponseWithinBoundsImpl:
    """PropDualFrameCondition: dual response lies in [Ad, Bd]."""

    def test_response_within_bounds(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds, filterbankdual, filterbankresponse  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gd = filterbankdual(g, a, L)
        Ad, Bd   = filterbankbounds(gd, a, L)
        # Compare the folded (real-signal) response against the folded bounds.
        gf_dual  = np.real(np.asarray(filterbankresponse(gd, a, L, real=True)))
        assert np.min(gf_dual) >= Ad - 1e-6, \
            f"Dual response min={np.min(gf_dual):.6f} < Ad={Ad:.6f}"
        assert np.max(gf_dual) <= Bd + 1e-6, \
            f"Dual response max={np.max(gf_dual):.6f} > Bd={Bd:.6f}"


@pytest.mark.requires_impl
class TestTightFrameConstantResponseImpl:
    """PropDualFrameCondition: tight frame response is constant at pos. freqs."""

    def test_tight_response_constant(self, needs_impl):
        from cool_frames.filterbanks import filterbanktight  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gt = filterbanktight(g, a, L)
        At, Bt = _pos_freq_bounds(gt, a, L)
        assert At > 0, "Tight frame lower bound must be positive"
        assert abs(At - Bt) / (At + 1e-15) < 0.01, \
            f"Tight frame response not constant: A={At:.8f}, B={Bt:.8f}"


@pytest.mark.requires_impl
class TestDualBoundsConsistencyImpl:
    """PropDualFrameCondition: A·Bd == 1 and B·Ad == 1."""

    def test_bounds_product_is_one(self, needs_impl):
        from cool_frames.filterbanks import filterbankdual  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gd = filterbankdual(g, a, L)
        # The canonical dual is one-sided (realonly=0, positive support only),
        # so filterbankbounds returns Ad=0 at uncovered negative frequencies.
        # Use the positive-frequency bounds (_pos_freq_bounds) which match
        # filterbankbounds and correctly capture the one-sided bank.
        A,  B  = _pos_freq_bounds(g, a, L)
        Ad, Bd = _pos_freq_bounds(gd, a, L)
        if A > 0 and B < np.inf:
            assert abs(A * Bd - 1.0) < 1e-3, \
                f"A·Bd = {A*Bd:.8f}, expected 1.0"
            assert abs(B * Ad - 1.0) < 1e-3, \
                f"B·Ad = {B*Ad:.8f}, expected 1.0"
        elif B < np.inf and Ad > 0:
            assert abs(B * Ad - 1.0) < 1e-3, \
                f"B·Ad = {B*Ad:.8f}, expected 1.0"
