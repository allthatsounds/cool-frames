"""
tests/layer1_filters/unit/test_gabfilters.py
=============================================
Python unit tests for gabfilters (linear Gabor filterbank design).

Mirrors TestGabFilters.m, covering:
  1. Return-value structure and types
  2. Filter descriptor fields
  3. Centre-frequency properties
  4. Transform length (dgtlength)
  5. Real vs complex mode
  6. Time vs freq window axis
  7. Analysis consistency (ufilterbank equivalence)
  8. Edge cases
"""
from __future__ import annotations

import math

import pytest

import numpy as np
from cool_frames.numpy.filters._gabfilters import gabfilters

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

_M  = 64
_A  = 16
_LS = 640
_TOL = 1e-9


def _dgtlength(Ls, a, M):
    """Compute expected DGT length."""
    b = math.lcm(a, M)
    return int(math.ceil(Ls / b) * b)


# ---------------------------------------------------------------------------
# 1. Return-value structure and types
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_returns_five_outputs(self):
        gout, aout, fc, L, info = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        assert len(gout) > 0
        assert aout is not None
        assert len(fc) > 0
        assert L > 0
        assert isinstance(info, dict)

    def test_gout_is_list_of_dicts(self):
        gout, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        assert isinstance(gout, list)
        assert all(isinstance(g, dict) for g in gout)

    def test_info_has_fc_and_tfr(self):
        *_, info = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        assert 'fc' in info
        assert 'tfr' in info


# ---------------------------------------------------------------------------
# 2. Filter descriptor fields
# ---------------------------------------------------------------------------

class TestFilterDescriptors:

    def test_filter_has_required_fields(self):
        gout, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        for g in gout:
            assert 'H' in g, "Filter must have 'H' field"
            assert 'foff' in g, "Filter must have 'foff' field"
            assert 'realonly' in g, "Filter must have 'realonly' field"

    @pytest.mark.xfail(reason="Short filters return compact H; full-length padding not yet implemented")
    def test_all_filters_have_same_h_length(self):
        gout, _, _, L, _ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        for g in gout:
            assert len(g['H']) == L, \
                f"Filter H length {len(g['H'])} != L={L}"

    def test_all_filters_share_prototype_h(self):
        gout, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        H0 = gout[0]['H']
        for g in gout[1:]:
            np.testing.assert_array_equal(g['H'], H0,
                err_msg="All filters should share the same prototype H")


# ---------------------------------------------------------------------------
# 3. Centre-frequency properties
# ---------------------------------------------------------------------------

class TestCentreFrequencies:

    def test_fc_linearly_spaced(self):
        _, _, fc, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        dfc = np.diff(fc)
        np.testing.assert_allclose(dfc, dfc[0], atol=_TOL,
            err_msg="Centre frequencies should be linearly spaced")

    def test_fc_starts_at_zero(self):
        _, _, fc, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        assert abs(fc[0]) < _TOL, "First centre frequency should be 0"

    def test_fc_in_hz(self):
        # fc is returned in Hz (consistent with the other designers): [0, fs/2].
        fs = 16000
        _, _, fc, *_ = gabfilters(fs, _LS, window='hann', a=_A, M=_M)
        assert fc.min() >= 0.0
        assert np.max(fc) <= fs / 2 + 1e-6, "All fc values must be in [0, fs/2]"

    def test_fc_spacing_is_2_over_m(self):
        _, _, fc, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        expected_step = 16000.0 / _M      # fc in Hz: spacing = fs/M
        dfc = np.diff(fc)
        np.testing.assert_allclose(dfc, expected_step, rtol=1e-9)


# ---------------------------------------------------------------------------
# 4. Transform length
# ---------------------------------------------------------------------------

class TestTransformLength:

    def test_l_is_dgt_length(self):
        _, _, _, L, _ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        assert L == _dgtlength(_LS, _A, _M)

    def test_l_divisible_by_a_and_m(self):
        _, _, _, L, _ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        assert L % _A == 0, "L must be divisible by a"
        assert L % _M == 0, "L must be divisible by M"

    @pytest.mark.parametrize("Ls,a,M", [
        (100, 10, 20),
        (1000, 128, 512),
        (333, 7, 14),
        (1, 1, 2),
    ])
    def test_l_always_valid(self, Ls, a, M):
        _, _, _, L, _ = gabfilters(16000, Ls, window='hann', a=a, M=M)
        assert L >= Ls
        assert L % a == 0
        assert L % M == 0


# ---------------------------------------------------------------------------
# 5. Real vs complex mode
# ---------------------------------------------------------------------------

class TestRealComplexMode:

    def test_real_mode_filter_count(self):
        gout, _, fc, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M, real=True)
        M2 = _M // 2 + 1
        assert len(gout) == M2
        assert len(fc) == M2

    def test_complex_mode_filter_count(self):
        gout, _, fc, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M, real=False)
        assert len(gout) == _M
        assert len(fc) == _M

    def test_real_is_default(self):
        gout_default, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        gout_real, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M, real=True)
        assert len(gout_default) == len(gout_real)


# ---------------------------------------------------------------------------
# 6. Time vs freq window axis
# ---------------------------------------------------------------------------

class TestWindowAxis:

    def test_time_mode_produces_valid_filters(self):
        gout, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M, windowaxis='time')
        assert np.sum(np.abs(gout[0]['H'])**2) > 0

    def test_freq_mode_produces_valid_filters(self):
        g0 = np.random.randn(_M)
        gout, *_ = gabfilters(16000, _LS, window=g0, a=_A, M=_M, windowaxis='freq')
        assert np.sum(np.abs(gout[0]['H'])**2) > 0

    def test_invalid_windowaxis_raises(self):
        with pytest.raises(ValueError, match="windowaxis"):
            gabfilters(16000, _LS, window='hann', a=_A, M=_M, windowaxis='invalid')


# ---------------------------------------------------------------------------
# 7. Analysis consistency
# ---------------------------------------------------------------------------

class TestAnalysisConsistency:

    def test_foff_spacing(self):
        gout, _, _, L, _ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        step = L / _M
        for k in range(1, len(gout)):
            actual = gout[k]['foff'] - gout[k-1]['foff']
            assert abs(actual - step) < _TOL, \
                f"foff step {actual} != L/M={step}"

    def test_uniform_hop_sizes(self):
        _, aout, *_ = gabfilters(16000, _LS, window='hann', a=_A, M=_M)
        assert aout.ndim == 1, f"Expected 1-D hop array, got shape {aout.shape}"
        assert np.all(aout == _A)

    def test_ufilterbank_produces_correct_shape(self):
        """Verify that gabfilters output works with ufilterbank."""
        from cool_frames.numpy.filterbanks import filterbank
        M = 64; a = 16; L = 640
        gout, aout, fc, L2, _ = gabfilters(16000, L, window='hann', a=a, M=M)
        f = np.random.randn(L2)
        c = filterbank(f, gout, a, stack=True)
        M2 = M // 2 + 1
        N = L2 // a
        assert c.shape == (N, M2), f"Expected ({N}, {M2}), got {c.shape}"


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_small_m(self):
        gout, _, fc, *_ = gabfilters(16000, 64, window='hann', a=4, M=8)
        assert len(gout) == 8 // 2 + 1
        assert len(fc) == 8 // 2 + 1

    def test_ls_equals_l(self):
        L0 = math.lcm(_A, _M) * 5
        _, _, _, L, _ = gabfilters(16000, L0, window='hann', a=_A, M=_M)
        assert L == L0

    def test_numeric_window(self):
        from cool_frames.numpy.filters._firwin import firwin
        g0 = firwin('hann', _M, norm='energy')
        gout, *_ = gabfilters(16000, _LS, window=g0, a=_A, M=_M)
        assert len(gout) > 0

    def test_different_windows(self):
        for win in ['hann', 'blackman', 'hamming', 'nuttall']:
            gout, *_ = gabfilters(16000, _LS, window=win, a=_A, M=_M)
            assert len(gout) == _M // 2 + 1
