"""
tests/layer3_repr/property/test_prop_fbphasegradfrommag.py
==========================================================
Property-based tests for comp_filterbankphasegradfrommag.

Tests mathematical invariants across random non-uniform filterbank configs.
"""
from __future__ import annotations

import numpy as np
from cool_frames.numpy.phase._fbphasegradfrommag import (
    comp_filterbankneighbors,
    comp_filterbankphasegradfrommag,
)

_N_TRIALS = 8


def _random_nonuniform_fb(rng, M=None):
    """Generate a random non-uniform filterbank configuration."""
    if M is None:
        M = rng.integers(2, 7)
    # Random hop sizes (powers of 2)
    a = 2 ** rng.integers(2, 6, size=M)
    # Pick L divisible by all hop sizes
    L = int(np.lcm.reduce(a) * rng.integers(2, 8))
    N = L // a
    fc = np.sort(rng.uniform(0, 1, size=M))
    fc[0] = 0.0  # ensure DC
    sqtfr = np.abs(rng.standard_normal(M)) * 0.5 + 0.5
    Nsum = int(np.sum(N))
    abss = np.abs(rng.standard_normal(Nsum)) + 0.01
    NEIGH, posInfo = comp_filterbankneighbors(a, M, N, do_real=True)
    return abss, N, a, M, sqtfr, fc, NEIGH, posInfo


# ---------------------------------------------------------------------------
# 1. Output shape invariant
# ---------------------------------------------------------------------------

class TestOutputShapeInvariant:

    def test_shapes_match_nsum(self):
        rng = np.random.default_rng(100)
        for _ in range(_N_TRIALS):
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _random_nonuniform_fb(rng)
            tgrad, fgrad, logs = comp_filterbankphasegradfrommag(
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo)
            Nsum = int(np.sum(N))
            assert tgrad.shape == (Nsum,)
            assert fgrad.shape == (Nsum,)
            assert logs.shape == (Nsum,)


# ---------------------------------------------------------------------------
# 2. Finiteness
# ---------------------------------------------------------------------------

class TestFiniteness:

    def test_all_outputs_finite(self):
        rng = np.random.default_rng(101)
        for _ in range(_N_TRIALS):
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _random_nonuniform_fb(rng)
            tgrad, fgrad, logs = comp_filterbankphasegradfrommag(
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo)
            assert np.all(np.isfinite(tgrad)), "tgrad must be finite"
            assert np.all(np.isfinite(fgrad)), "fgrad must be finite"
            assert np.all(np.isfinite(logs)), "logs must be finite"


# ---------------------------------------------------------------------------
# 3. Constant magnitude → zero fgrad
# ---------------------------------------------------------------------------

class TestConstantMagnitudeProperty:

    def test_const_mag_zero_fgrad(self):
        rng = np.random.default_rng(102)
        for _ in range(_N_TRIALS):
            M = int(rng.integers(2, 6))
            a = 2 ** rng.integers(2, 5, size=M)
            L = int(np.lcm.reduce(a) * rng.integers(2, 5))
            N = L // a
            Nsum = int(np.sum(N))
            fc = np.sort(rng.uniform(0, 1, size=M))
            fc[0] = 0.0
            sqtfr = np.abs(rng.standard_normal(M)) * 0.5 + 0.5
            abss = np.ones(Nsum)  # constant
            NEIGH, posInfo = comp_filterbankneighbors(a, M, N, do_real=True)
            _, fgrad, _ = comp_filterbankphasegradfrommag(
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo)
            np.testing.assert_allclose(fgrad, 0, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Logs invariant
# ---------------------------------------------------------------------------

class TestLogsInvariant:

    def test_logs_equals_log_abss(self):
        rng = np.random.default_rng(103)
        for _ in range(_N_TRIALS):
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _random_nonuniform_fb(rng)
            _, _, logs = comp_filterbankphasegradfrommag(
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo)
            expected = np.log(abss + np.finfo(float).tiny)
            np.testing.assert_allclose(logs, expected, atol=1e-14)


# ---------------------------------------------------------------------------
# 5. Scaling by magnitude
# ---------------------------------------------------------------------------

class TestMagnitudeScaling:

    def test_scaling_abss_shifts_logs(self):
        """Multiplying abss by a constant shifts logs by log(constant)."""
        rng = np.random.default_rng(104)
        abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _random_nonuniform_fb(rng)
        scale = 3.7
        _, _, logs1 = comp_filterbankphasegradfrommag(
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo)
        _, _, logs2 = comp_filterbankphasegradfrommag(
            abss * scale, N, a, M, sqtfr, fc, NEIGH, posInfo)
        np.testing.assert_allclose(logs2 - logs1, np.log(scale), atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Gderivweight = 0 makes tfrdiff irrelevant
# ---------------------------------------------------------------------------

class TestGderivWeightZero:

    def test_zero_weight_makes_tfrdiff_irrelevant(self):
        rng = np.random.default_rng(105)
        for _ in range(_N_TRIALS):
            abss, N, a, M, sqtfr, fc, NEIGH, posInfo = _random_nonuniform_fb(rng)
            t1, _, _ = comp_filterbankphasegradfrommag(
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo,
                gderivweight=0.0, do_tfrdiff=False)
            t2, _, _ = comp_filterbankphasegradfrommag(
                abss, N, a, M, sqtfr, fc, NEIGH, posInfo,
                gderivweight=0.0, do_tfrdiff=True)
            np.testing.assert_allclose(t1, t2, atol=1e-14)


# ---------------------------------------------------------------------------
# 7. NEIGH structure consistency
# ---------------------------------------------------------------------------

class TestNeighConsistency:

    def test_neigh_indices_in_range(self):
        rng = np.random.default_rng(106)
        for _ in range(_N_TRIALS):
            M = int(rng.integers(2, 6))
            a = 2 ** rng.integers(2, 5, size=M)
            L = int(np.lcm.reduce(a) * rng.integers(2, 5))
            N = L // a
            Nsum = int(np.sum(N))
            NEIGH, _ = comp_filterbankneighbors(a, M, N, do_real=True)
            valid = NEIGH >= -1
            assert np.all(valid), "NEIGH values must be >= -1"
            active = NEIGH[NEIGH >= 0]
            assert np.all(active < Nsum), "Active NEIGH values must be < Nsum"
