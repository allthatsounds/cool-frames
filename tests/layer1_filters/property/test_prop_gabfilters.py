"""
tests/layer1_filters/property/test_prop_gabfilters.py
=====================================================
Property-based tests for gabfilters.

Tests mathematical invariants across random parameter combinations.
"""
from __future__ import annotations

import math

import pytest

import numpy as np

try:
    from cool_frames.numpy.filters._gabfilters import gabfilters
except ImportError:
    gabfilters = None

pytestmark = pytest.mark.skipif(gabfilters is None,
                                reason="cool_frames not installed")

_N_TRIALS = 10
_TOL = 1e-9
_RNG = np.random.default_rng(42)


def _random_params(rng):
    """Generate random (Ls, a, M) triple with power-of-2 a and M."""
    M = 2 ** rng.integers(3, 9)
    a = 2 ** rng.integers(2, 7)
    Ls = rng.integers(100, 5001)
    return int(Ls), int(a), int(M)


def _dgtlength(Ls, a, M):
    b = math.lcm(a, M)
    return int(math.ceil(Ls / b) * b)


# ---------------------------------------------------------------------------
# 1. Filter count invariant
# ---------------------------------------------------------------------------

class TestFilterCountInvariant:

    def test_real_mode_filter_count(self):
        rng = np.random.default_rng(42)
        for _ in range(_N_TRIALS):
            Ls, a, M = _random_params(rng)
            gout, *_ = gabfilters(16000, Ls, window='hann', a=a, M=M, real=True)
            assert len(gout) == M // 2 + 1

    def test_complex_mode_filter_count(self):
        rng = np.random.default_rng(43)
        for _ in range(_N_TRIALS):
            Ls, a, M = _random_params(rng)
            gout, *_ = gabfilters(16000, Ls, window='hann', a=a, M=M, real=False)
            assert len(gout) == M


# ---------------------------------------------------------------------------
# 2. Centre frequency linearity
# ---------------------------------------------------------------------------

class TestFcLinearity:

    def test_fc_spacing_always_2_over_m(self):
        rng = np.random.default_rng(44)
        for _ in range(_N_TRIALS):
            Ls, a, M = _random_params(rng)
            _, _, fc, *_ = gabfilters(16000, Ls, window='hann', a=a, M=M)
            expected = 16000.0 / M          # fc in Hz: spacing = fs/M
            dfc = np.diff(fc)
            np.testing.assert_allclose(dfc, expected, rtol=1e-9)

    def test_fc_starts_at_zero(self):
        rng = np.random.default_rng(45)
        for _ in range(_N_TRIALS):
            Ls, a, M = _random_params(rng)
            _, _, fc, *_ = gabfilters(16000, Ls, window='hann', a=a, M=M)
            assert abs(fc[0]) < _TOL


# ---------------------------------------------------------------------------
# 3. DGT length invariant
# ---------------------------------------------------------------------------

class TestDgtLengthInvariant:

    def test_l_equals_dgtlength(self):
        rng = np.random.default_rng(46)
        for _ in range(_N_TRIALS):
            Ls, a, M = _random_params(rng)
            _, _, _, L, _ = gabfilters(16000, Ls, window='hann', a=a, M=M)
            assert L == _dgtlength(Ls, a, M)

    def test_l_ge_ls(self):
        rng = np.random.default_rng(47)
        for _ in range(_N_TRIALS):
            Ls, a, M = _random_params(rng)
            _, _, _, L, _ = gabfilters(16000, Ls, window='hann', a=a, M=M)
            assert L >= Ls


# ---------------------------------------------------------------------------
# 4. Prototype sharing
# ---------------------------------------------------------------------------

class TestPrototypeSharing:

    def test_all_filters_share_h(self):
        rng = np.random.default_rng(48)
        for _ in range(_N_TRIALS):
            Ls, a, M = _random_params(rng)
            gout, *_ = gabfilters(16000, Ls, window='hann', a=a, M=M)
            H0 = gout[0]['H']
            for g in gout[1:]:
                np.testing.assert_array_equal(g['H'], H0)


# ---------------------------------------------------------------------------
# 5. Foff spacing invariant
# ---------------------------------------------------------------------------

class TestFoffSpacing:

    def test_foff_spacing_is_l_over_m(self):
        rng = np.random.default_rng(49)
        for _ in range(_N_TRIALS):
            Ls, a, M = _random_params(rng)
            gout, _, _, L, _ = gabfilters(16000, Ls, window='hann', a=a, M=M)
            step = L / M
            for k in range(1, len(gout)):
                actual = gout[k]['foff'] - gout[k-1]['foff']
                assert abs(actual - step) < _TOL, \
                    f"foff step {actual} != L/M={step}"


# ---------------------------------------------------------------------------
# 6. TFR positivity
# ---------------------------------------------------------------------------

class TestTfrPositivity:

    def test_tfr_is_positive(self):
        rng = np.random.default_rng(50)
        for _ in range(_N_TRIALS):
            Ls, a, M = _random_params(rng)
            *_, info = gabfilters(16000, Ls, window='hann', a=a, M=M)
            assert info['tfr'] > 0, "TFR should be positive for Hann window"

    def test_tfr_varies_with_l(self):
        """TFR = gamma/L, so different L should give different tfr."""
        _, _, _, L1, info1 = gabfilters(16000, 100, window='hann', a=8, M=16)
        _, _, _, L2, info2 = gabfilters(16000, 1000, window='hann', a=8, M=16)
        if L1 != L2:
            assert info1['tfr'] != info2['tfr']
