"""
test_subsampling.py
===================
Python port of:
    layer0_ops/unit/TestSubsampling.m

Covers: comp_downs, comp_ups
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import comp_downs_ref, comp_ups_ref

# ---------------------------------------------------------------------------
# comp_downs – reference tests
# ---------------------------------------------------------------------------

class TestCompDownsReference:
    """
    MATLAB counterpart: TestSubsampling (comp_downs section).
    """

    def test_identity_a1(self):
        """comp_downs(x, 1) == x (a=1 is identity)."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            x = rng.standard_normal(128)
            np.testing.assert_array_equal(comp_downs_ref(x, 1), x)

    def test_output_length(self):
        """comp_downs(x, 2) has length ceil(L/2)."""
        rng = np.random.default_rng(1)
        for _ in range(10):
            L      = rng.integers(100, 501)
            x      = rng.standard_normal(int(L))
            result = comp_downs_ref(x, 2)
            assert len(result) == int(np.ceil(L / 2))

    def test_zero_input(self):
        """comp_downs of a zero vector is zero."""
        x = np.zeros(128)
        np.testing.assert_array_equal(comp_downs_ref(x, 2), np.zeros(64))

    def test_impulse(self):
        """comp_downs of an impulse at position 0 preserves that sample."""
        x    = np.zeros(128); x[0] = 1.0
        out  = comp_downs_ref(x, 2)
        assert out[0] == 1.0

    def test_noise_output_length(self):
        """comp_downs(noise, 2): output length = ceil(L/2)."""
        L   = 256
        x   = np.random.default_rng(2).standard_normal(L)
        out = comp_downs_ref(x, 2)
        assert len(out) == int(np.ceil(L / 2))

    def test_skip_parameter(self):
        """skip=3: first output is x[3]."""
        x    = np.arange(10, dtype=float)
        out  = comp_downs_ref(x, 2, skip=3)
        assert out[0] == x[3]


# ---------------------------------------------------------------------------
# comp_ups – reference tests
# ---------------------------------------------------------------------------

class TestCompUpsReference:
    """
    MATLAB counterpart: TestSubsampling (comp_ups section).
    """

    def test_identity_a1(self):
        """comp_ups(x, 1) == x."""
        rng = np.random.default_rng(3)
        for _ in range(10):
            x = rng.standard_normal(128)
            np.testing.assert_array_equal(comp_ups_ref(x, 1), x)

    def test_output_length(self):
        """comp_ups(x, a) has length a * len(x)."""
        rng = np.random.default_rng(4)
        for _ in range(10):
            x = rng.standard_normal(64)
            a = int(rng.integers(2, 9))
            assert len(comp_ups_ref(x, a)) == a * len(x)

    def test_energy_preserved(self):
        """Upsampling preserves energy: sum(|up|^2) == sum(|x|^2)."""
        rng = np.random.default_rng(5)
        for _ in range(10):
            x   = rng.standard_normal(64)
            a   = int(rng.integers(2, 9))
            up  = comp_ups_ref(x, a)
            np.testing.assert_allclose(
                np.sum(np.abs(up) ** 2), np.sum(np.abs(x) ** 2), rtol=1e-12
            )

    def test_cyclic_impulse_structure(self):
        """After downsample + upsample the impulse is at position 0."""
        x = np.zeros(128); x[0] = 1.0
        down = comp_downs_ref(x, 2)
        up   = comp_ups_ref(down, 2)
        assert up[0] > 0

