"""
test_prop_involute.py
=====================
Python port of:
    layer0_ops/property/PropInvoluteAlgebra.m

Property tests for the involute operation (algebraic identities).
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import involute_ref

# ---------------------------------------------------------------------------
# Reference tests (unconditional)
# ---------------------------------------------------------------------------

class TestInvoluteAlgebraReference:
    """
    MATLAB counterpart: PropInvoluteAlgebra.
    """

    def test_double_application_identity(self):
        """involute(involute(x)) == x exactly."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            N = int(rng.integers(32, 513))
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            np.testing.assert_allclose(
                involute_ref(involute_ref(x)), x, atol=1e-12
            )

    def test_fft_relation(self):
        """fft(involute(x)) == conj(fft(x)) within 1e-12."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            N   = int(rng.integers(32, 513))
            x   = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            lhs = np.fft.fft(involute_ref(x))
            rhs = np.conj(np.fft.fft(x))
            np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_conjugate_commutes(self):
        """involute(conj(x)) == conj(involute(x))."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            N   = int(rng.integers(32, 513))
            x   = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            lhs = involute_ref(np.conj(x))
            rhs = np.conj(involute_ref(x))
            np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_real_input_fft_relation(self):
        """For real x: fft(involute(x)) == conj(fft(x))."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            N   = int(rng.integers(32, 513))
            x   = rng.standard_normal(N)          # real signal
            lhs = np.fft.fft(involute_ref(x))     # fft of involuted signal
            rhs = np.conj(np.fft.fft(x))          # conjugate of FFT
            np.testing.assert_allclose(lhs, rhs, atol=1e-12)


# ---------------------------------------------------------------------------
# Implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestInvoluteAlgebraImpl:
    """Verify cool_frames.layer0.involute matches reference on algebraic identities."""

    def test_double_application_identity(self, needs_impl):
        from cool_frames.core import involute  # type: ignore

        rng = np.random.default_rng(42)
        for _ in range(20):
            N = int(rng.integers(32, 257))
            x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            np.testing.assert_allclose(involute(involute(x)), x, atol=1e-12)

    def test_fft_relation(self, needs_impl):
        from cool_frames.core import involute  # type: ignore

        rng = np.random.default_rng(42)
        for _ in range(20):
            N   = int(rng.integers(32, 257))
            x   = rng.standard_normal(N) + 1j * rng.standard_normal(N)
            lhs = np.fft.fft(involute(x))
            rhs = np.conj(np.fft.fft(x))
            np.testing.assert_allclose(lhs, rhs, atol=1e-12)
