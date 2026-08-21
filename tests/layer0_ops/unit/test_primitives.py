"""
test_primitives.py
==================
Python port of:
    layer0_ops/unit/TestPrimitives.m

Covers: involute, modcent, fftindex, floor23

Test categories
---------------
  [Reference]  Always runs – uses numpy reference implementations from conftest.
  [Impl]       @pytest.mark.requires_impl – calls cool_frames.layer0.
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import fftindex_ref, floor23_ref, involute_ref, modcent_ref

# ---------------------------------------------------------------------------
# involute – reference tests
# ---------------------------------------------------------------------------

class TestInvoluteReference:
    """
    Tests for the numpy reference involute_ref().

    MATLAB counterpart: TestPrimitives (involute section).
    """

    def test_double_application_is_identity(self):
        """involute(involute(x)) == x for random complex x."""
        rng = np.random.default_rng(0)
        for _ in range(10):
            x      = rng.standard_normal(100) + 1j * rng.standard_normal(100)
            result = involute_ref(involute_ref(x))
            np.testing.assert_allclose(result, x, atol=1e-12,
                err_msg="involute_ref(involute_ref(x)) ≠ x")

    def test_fft_relation(self):
        """fft(involute(x)) == conj(fft(x)) to within 1e-12."""
        rng = np.random.default_rng(1)
        for _ in range(10):
            x   = rng.standard_normal(128) + 1j * rng.standard_normal(128)
            lhs = np.fft.fft(involute_ref(x))
            rhs = np.conj(np.fft.fft(x))
            np.testing.assert_allclose(lhs, rhs, atol=1e-12,
                err_msg="fft(involute_ref(x)) ≠ conj(fft(x))")

    def test_dc_element_is_conjugated(self):
        """involute(x)[0] == conj(x[0])."""
        rng = np.random.default_rng(2)
        for _ in range(10):
            x = rng.standard_normal(100) + 1j * rng.standard_normal(100)
            assert abs(involute_ref(x)[0] - np.conj(x[0])) < 1e-14


# ---------------------------------------------------------------------------
# involute – implementation accuracy tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestInvoluteImpl:
    """Verify cool_frames.layer0.involute matches involute_ref."""

    def test_matches_reference(self, needs_impl):
        from cool_frames.core import involute  # type: ignore

        rng = np.random.default_rng(42)
        x   = rng.standard_normal(128) + 1j * rng.standard_normal(128)
        np.testing.assert_allclose(involute(x), involute_ref(x), atol=1e-14)


# ---------------------------------------------------------------------------
# modcent – reference tests
# ---------------------------------------------------------------------------

class TestModcentReference:
    """
    Tests for modcent_ref().

    MATLAB counterpart: TestPrimitives (modcent section).
    """

    @pytest.mark.parametrize("r", [2.0, 4.0, 8.0, 16.0, 32.0])
    def test_output_in_range(self, r):
        """modcent(x, r) must lie in [-r/2, r/2) for any x."""
        x   = np.random.default_rng(0).standard_normal(10)
        res = modcent_ref(x, r)
        assert np.all(res >= -r / 2), f"r={r}: value below -r/2"
        assert np.all(res < r / 2),   f"r={r}: value ≥ r/2"

    def test_periodicity(self):
        """modcent(x + r, r) == modcent(x, r)."""
        rng = np.random.default_rng(1)
        r   = 16.0
        x   = rng.standard_normal(10)
        np.testing.assert_allclose(modcent_ref(x, r), modcent_ref(x + r, r), atol=1e-14)


@pytest.mark.requires_impl
class TestModcentImpl:
    def test_matches_reference(self, needs_impl):
        from cool_frames.core import modcent  # type: ignore

        x = np.random.default_rng(42).standard_normal(50)
        for r in (2.0, 2 * np.pi, 16.0):
            np.testing.assert_allclose(modcent(x, r), modcent_ref(x, r), atol=1e-14)


# ---------------------------------------------------------------------------
# fftindex – reference tests
# ---------------------------------------------------------------------------

class TestFFTIndexReference:
    """
    Tests for fftindex_ref().

    MATLAB counterpart: TestPrimitives (fftindex section).
    """

    @pytest.mark.parametrize("N", [8, 16, 32, 1024])
    def test_range(self, N):
        """fftindex(N) output in [-ceil(N/2)+1, floor(N/2)]."""
        idx = fftindex_ref(N)
        lo  = -(N + 1) // 2 + 1  # -ceil(N/2) + 1
        hi  = N // 2              # floor(N/2)
        assert np.all(idx >= lo) and np.all(idx <= hi), (
            f"N={N}: fftindex_ref out of range [{lo}, {hi}]"
        )

    @pytest.mark.parametrize("N", [8, 16, 32, 1024])
    def test_last_element_is_minus_one(self, N):
        """For even N ≥ 4, the last element of fftindex(N) is -1."""
        idx = fftindex_ref(N)
        assert idx[-1] == -1, f"N={N}: last element {idx[-1]} ≠ -1"
# ---------------------------------------------------------------------------
# floor23 – reference tests
# ---------------------------------------------------------------------------

class TestFloor23Reference:
    """
    Tests for floor23_ref().

    MATLAB counterpart: TestPrimitives (floor23 section).
    """

    def test_result_leq_input(self):
        """floor23(n) ≤ n."""
        rng = np.random.default_rng(0)
        for n in rng.integers(1, 10001, size=30):
            assert floor23_ref(int(n)) <= int(n)

    def test_result_is_23_smooth(self):
        """floor23(n) factors as 2^i * 3^j."""
        rng = np.random.default_rng(1)
        for n in rng.integers(1, 10001, size=30):
            v = floor23_ref(int(n))
            while v % 2 == 0:
                v //= 2
            while v % 3 == 0:
                v //= 3
            assert v == 1, f"floor23_ref({n}) not 2-3 smooth"

    def test_lower_bound(self):
        """floor23(n) > n/6 for n ≥ 100."""
        rng = np.random.default_rng(2)
        for n in rng.integers(100, 10001, size=30):
            result = floor23_ref(int(n))
            assert result > int(n) / 6, (
                f"floor23_ref({n}) = {result} ≤ n/6 = {int(n)/6:.1f}"
            )


@pytest.mark.requires_impl
class TestFloor23Impl:
    def test_matches_reference(self, needs_impl):
        from cool_frames.core import floor23  # type: ignore

        rng = np.random.default_rng(42)
        for n in rng.integers(1, 10001, size=30):
            n = int(n)
            assert floor23(n) == floor23_ref(n), f"floor23({n}) mismatch"
