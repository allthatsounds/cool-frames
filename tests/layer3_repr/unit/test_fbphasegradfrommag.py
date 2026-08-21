"""
tests/layer3_repr/unit/test_fbphasegradfrommag.py
=================================================
Unit tests for comp_filterbankphasegradfrommag and comp_filterbankneighbors.

Mirrors TestFbPhaseGradFromMag.m.
"""
from __future__ import annotations

import numpy as np
from cool_frames.numpy.phase._fbphasegradfrommag import (
    _pderiv,
    comp_filterbankneighbors,
    comp_filterbankphasegradfrommag,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_nonuniform_fb(M=3, seed=42):
    """Build a small non-uniform filterbank test case."""
    a = np.array([4, 8, 16])[:M]
    L = 64
    N = L // a
    fc = np.linspace(0, 1, M)
    sqtfr = np.linspace(0.5, 1.0, M)
    rng = np.random.default_rng(seed)
    Nsum = int(np.sum(N))
    abss = np.abs(rng.standard_normal(Nsum)) + 0.1
    NEIGH, posInfo = comp_filterbankneighbors(a, M, N, do_real=True)
    return abss, N, a, M, sqtfr, fc, NEIGH, posInfo


# ===========================================================================
# 1. comp_filterbankneighbors — structure
# ===========================================================================

class TestCompFilterbankNeighbors:

    def test_neigh_shape(self):
        a = np.array([4, 8, 16]); M = 3; N = np.array([16, 8, 4])
        NEIGH, posInfo = comp_filterbankneighbors(a, M, N, do_real=True)
        Nsum = int(np.sum(N))
        assert NEIGH.shape == (6, Nsum)
        assert posInfo.shape == (2, Nsum)

    def test_posinfo_channel_indices(self):
        a = np.array([4, 8, 16]); M = 3; N = np.array([16, 8, 4])
        _, posInfo = comp_filterbankneighbors(a, M, N, do_real=True)
        chanStart = np.concatenate([[0], np.cumsum(N)])
        for m in range(M):
            idx = slice(chanStart[m], chanStart[m + 1])
            assert np.all(posInfo[0, idx] == m), \
                f"Channel {m} posInfo[0] should all be {m}"

    def test_posinfo_time_positions(self):
        a = np.array([4, 8]); M = 2; N = np.array([8, 4])
        _, posInfo = comp_filterbankneighbors(a, M, N, do_real=True)
        # Channel 0: 0, 4, 8, ..., 28
        np.testing.assert_array_equal(
            posInfo[1, :N[0]], np.arange(N[0]) * a[0])
        # Channel 1: 0, 8, 16, 24
        np.testing.assert_array_equal(
            posInfo[1, N[0]:N[0] + N[1]], np.arange(N[1]) * a[1])

    def test_above_neighbors_exist(self):
        a = np.array([4, 8, 16]); M = 3; N = np.array([16, 8, 4])
        NEIGH, _ = comp_filterbankneighbors(a, M, N, do_real=True)
        # First channel should have above neighbours (rows 4,5)
        above = NEIGH[4, :N[0]]
        assert np.all(above >= 0), "First channel should have above neighbours"

    def test_last_channel_no_above_real_mode(self):
        a = np.array([4, 8, 16]); M = 3; N = np.array([16, 8, 4])
        NEIGH, _ = comp_filterbankneighbors(a, M, N, do_real=True)
        cs = int(np.sum(N[:2]))
        # Last channel should NOT have above neighbours in real mode
        above4 = NEIGH[4, cs:cs + N[2]]
        assert np.all(above4 == -1), \
            "Last channel should have no above neighbours in real mode"

    def test_complex_mode_wraps(self):
        a = np.array([4, 8, 16]); M = 3; N = np.array([16, 8, 4])
        NEIGH, _ = comp_filterbankneighbors(a, M, N, do_real=False)
        cs = int(np.sum(N[:2]))
        # Last channel SHOULD have above neighbours (wrapping to channel 0)
        above4 = NEIGH[4, cs:cs + N[2]]
        assert np.any(above4 >= 0), \
            "Last channel should wrap to first in complex mode"

    def test_uniform_case(self):
        """When all channels are equal, NEIGH should still be valid."""
        M = 4; a = np.array([8] * M); N = np.array([16] * M)
        NEIGH, posInfo = comp_filterbankneighbors(a, M, N, do_real=True)
        assert NEIGH.shape == (6, 64)
        assert posInfo.shape == (2, 64)


# ===========================================================================
# 2. _pderiv — periodic derivative
# ===========================================================================

class TestPderiv:

    def test_constant_gives_zero(self):
        f = np.ones(16) * 5.0
        fd = _pderiv(f, difforder=2)
        np.testing.assert_allclose(fd, 0, atol=1e-14)

    def test_sine_derivative(self):
        """pderiv of sin should approximate cos (scaled by 2π·L)."""
        N = 64
        x = np.arange(N) / N
        f = np.sin(2 * np.pi * x)
        fd = _pderiv(f, difforder=2)
        # pderiv returns L * central_diff / 2, which approximates
        # the derivative of a function on [0,1)
        # For sin(2πx), d/dx = 2π·cos(2πx), and pderiv returns N·(…)/2
        # so fd ≈ 2π·cos(2πx) scaled by N (then divided by N in the MATLAB caller)
        expected = 2 * np.pi * np.cos(2 * np.pi * x)
        # Normalize: caller usually divides by N
        np.testing.assert_allclose(fd / N, expected / N, atol=0.05)

    def test_order4(self):
        f = np.ones(16) * 3.0
        fd = _pderiv(f, difforder=4)
        np.testing.assert_allclose(fd, 0, atol=1e-14)


# ===========================================================================
# 3. comp_filterbankphasegradfrommag — output structure
# ===========================================================================

class TestOutputStructure:

    def test_output_shapes(self):
        abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _make_nonuniform_fb()
        tgrad, fgrad, logs = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo)
        Nsum = int(np.sum(N))
        assert tgrad.shape == (Nsum,)
        assert fgrad.shape == (Nsum,)
        assert logs.shape == (Nsum,)

    def test_outputs_are_finite(self):
        abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _make_nonuniform_fb()
        tgrad, fgrad, logs = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo)
        assert np.all(np.isfinite(tgrad))
        assert np.all(np.isfinite(fgrad))
        assert np.all(np.isfinite(logs))


# ===========================================================================
# 4. logs = log(abss + tiny)
# ===========================================================================

class TestLogs:

    def test_logs_is_log_magnitude(self):
        abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _make_nonuniform_fb()
        _, _, logs = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo)
        expected = np.log(abss + np.finfo(float).tiny)
        np.testing.assert_allclose(logs, expected, atol=1e-14)


# ===========================================================================
# 5. Constant magnitude → zero fgrad
# ===========================================================================

class TestConstantMagnitude:

    def test_const_mag_zero_fgrad(self):
        M = 3; a = np.array([4, 8, 16]); N = 64 // a
        Nsum = int(np.sum(N))
        abss = np.ones(Nsum)
        fc = np.array([0.0, 0.5, 1.0])
        sqtfr = np.array([0.5, 0.7, 1.0])
        NEIGH, posInfo = comp_filterbankneighbors(a, M, N, do_real=True)
        _, fgrad, _ = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo)
        np.testing.assert_allclose(fgrad, 0, atol=1e-10,
            err_msg="Constant magnitude should give zero fgrad")


# ===========================================================================
# 6. do_tfrdiff changes result
# ===========================================================================

class TestTfrDiff:

    def test_tfrdiff_changes_tgrad(self):
        abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _make_nonuniform_fb()
        tgrad1, _, _ = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo,
            gderivweight=0.5, do_tfrdiff=False)
        tgrad2, _, _ = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo,
            gderivweight=0.5, do_tfrdiff=True)
        assert not np.allclose(tgrad1, tgrad2), \
            "do_tfrdiff should change tgrad"

    def test_fgrad_unaffected_by_tfrdiff(self):
        abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _make_nonuniform_fb()
        _, fgrad1, _ = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo,
            gderivweight=0.5, do_tfrdiff=False)
        _, fgrad2, _ = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo,
            gderivweight=0.5, do_tfrdiff=True)
        np.testing.assert_array_equal(fgrad1, fgrad2,
            err_msg="do_tfrdiff should not affect fgrad")


# ===========================================================================
# 7. Scaling — gderivweight
# ===========================================================================

class TestGderivWeight:

    def test_zero_weight_equals_no_tfrdiff(self):
        """When gderivweight=0, do_tfrdiff should have no effect."""
        abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _make_nonuniform_fb()
        tgrad1, _, _ = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo,
            gderivweight=0.0, do_tfrdiff=False)
        tgrad2, _, _ = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo,
            gderivweight=0.0, do_tfrdiff=True)
        np.testing.assert_allclose(tgrad1, tgrad2, atol=1e-14)


# ===========================================================================
# 8. Uniform filterbank — sanity
# ===========================================================================

class TestUniformCase:

    def test_uniform_runs_and_finite(self):
        M = 4; a_val = 8; N_val = 16
        a = np.full(M, a_val); N = np.full(M, N_val)
        Nsum = int(np.sum(N))
        fc = np.linspace(0, 1, M)
        sqtfr = np.full(M, 0.8)
        rng = np.random.default_rng(99)
        abss = np.abs(rng.standard_normal(Nsum)) + 0.1
        NEIGH, posInfo = comp_filterbankneighbors(a, M, N, do_real=True)
        tgrad, fgrad, logs = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo)
        assert np.all(np.isfinite(tgrad))
        assert np.all(np.isfinite(fgrad))
