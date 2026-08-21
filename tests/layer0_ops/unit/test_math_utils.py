"""
test_math_utils.py
==================
Python port of:
    layer0_ops/unit/TestMathUtils.m

Covers: pderiv, psech

Test categories
---------------
  [Reference]  Always runs – uses numpy reference implementations from conftest.
  [Impl]       @pytest.mark.requires_impl – calls cool_frames.layer0.
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import pderiv_ref, psech_ref

# ---------------------------------------------------------------------------
# pderiv – reference tests (always run)
# ---------------------------------------------------------------------------

class TestPderivReference:
    """
    Structural and mathematical tests of pderiv_ref().

    MATLAB counterpart: TestMathUtils (pderiv section).
    """

    def test_output_length_vector(self):
        """pderiv must return a vector of the same length as input."""
        x  = np.random.default_rng(0).standard_normal(64)
        fd = pderiv_ref(x)
        assert len(fd) == len(x)

    def test_output_size_matrix(self):
        """For a 2-D input, pderiv_ref acts along axis-0 (columns)."""
        X  = np.random.default_rng(1).standard_normal((64, 4))
        # Apply column-wise
        FD = np.column_stack([pderiv_ref(X[:, j]) for j in range(X.shape[1])])
        assert FD.shape == X.shape

    def test_derivative_of_constant_is_zero(self):
        """Derivative of a constant signal is zero for all diff orders."""
        L = 64
        f = np.ones(L)
        for order in (2, 4, np.inf):
            fd = pderiv_ref(f, difforder=order)
            assert np.linalg.norm(fd) < 1e-10, (
                f"pderiv_ref(const, order={order}): norm = {np.linalg.norm(fd):.2e} > 1e-10"
            )

    def test_derivative_of_sine(self):
        """
        Derivative of sin(2π·k·x) on [0,1) = 2π·k · cos(2π·k·x).

        MATLAB counterpart: testPderivLinearSine
        """
        L = 128
        n = np.arange(L)
        f        = np.sin(2 * np.pi * n / L)
        fd       = pderiv_ref(f, difforder=np.inf)
        expected = 2 * np.pi * np.cos(2 * np.pi * n / L)
        rel_err  = np.linalg.norm(fd - expected) / np.linalg.norm(expected)
        assert rel_err < 1e-10, (
            f"pderiv_ref(sin, Inf): rel_err={rel_err:.2e} > 1e-10"
        )

    def test_order_inf_real_for_real_input(self):
        """Spectral derivative of a real signal must be real."""
        f  = np.random.default_rng(2).standard_normal(64)
        fd = pderiv_ref(f, difforder=np.inf)
        assert np.isrealobj(fd), "pderiv_ref(Inf): output must be real for real input"

    def test_order2_vs_order4_smooth_signal(self):
        """For a low-frequency signal, orders 2 and 4 agree within 2 %."""
        L  = 256
        n  = np.arange(L)
        f  = np.sin(2 * np.pi * 3 * n / L) + 0.5 * np.cos(2 * np.pi * 5 * n / L)
        fd2 = pderiv_ref(f, difforder=2)
        fd4 = pderiv_ref(f, difforder=4)
        rel_err = np.linalg.norm(fd2 - fd4) / (np.linalg.norm(fd4) + np.finfo(float).eps)
        assert rel_err < 0.02, (
            f"pderiv_ref orders 2 vs 4: rel_err={rel_err:.3f} > 0.02"
        )

    @pytest.mark.parametrize("order", [2, 4])
    def test_real_output_for_real_input(self, order):
        """Finite-difference orders must return real for real input."""
        f  = np.random.default_rng(3).standard_normal(64)
        fd = pderiv_ref(f, difforder=order)
        assert np.isrealobj(fd), f"pderiv_ref(order={order}): real input must give real output"


# ---------------------------------------------------------------------------
# psech – reference tests (always run)
# ---------------------------------------------------------------------------

class TestPsechReference:
    """
    Structural and mathematical tests of psech_ref().

    MATLAB counterpart: TestMathUtils (psech section).
    """

    @pytest.mark.parametrize("L", [32, 64, 128, 256])
    def test_output_length(self, L):
        """psech must return a vector of length L."""
        g = psech_ref(L)
        assert len(g) == L

    @pytest.mark.parametrize("L", [32, 64, 128])
    def test_unit_norm(self, L):
        """psech is normalised to unit L2-norm."""
        g = psech_ref(L)
        assert abs(np.linalg.norm(g) - 1.0) < 1e-4, (
            f"psech_ref(L={L}): norm={np.linalg.norm(g):.6f} ≠ 1.0"
        )

    def test_real_valued(self):
        """psech must return a real-valued signal."""
        g = psech_ref(64)
        assert np.isrealobj(g)

    def test_dft_invariance(self):
        """
        The canonical psech(L, tfr=1) is its own DFT:
            norm(g - fft(g)/sqrt(L)) ≈ 0

        MATLAB counterpart: testPsechDFTInvariance (uses LTFAT dft = fft/sqrt(L))
        """
        L   = 128
        g   = psech_ref(L)
        dft = np.fft.fft(g) / np.sqrt(L)    # LTFAT dft convention
        err = np.linalg.norm(g - dft)
        assert err < 1e-8, f"psech_ref DFT invariance: err={err:.2e} > 1e-8"

    @pytest.mark.parametrize("L", [32, 64, 128])
    def test_fft_real_valued(self, L):
        """fft(psech) must be real (whole-point even window)."""
        g = psech_ref(L)
        G = np.fft.fft(g)
        assert np.max(np.abs(np.imag(G))) < 1e-12, (
            f"psech_ref(L={L}): fft not real (max imag = {np.max(np.abs(np.imag(G))):.2e})"
        )

    def test_tfr_scaling(self):
        """
        Larger tfr → wider time window → lower peak value.

        MATLAB counterpart: testPsechTfrScaling
        """
        L  = 128
        g1 = psech_ref(L, tfr=1)
        g2 = psech_ref(L, tfr=4)
        assert np.max(np.abs(g2)) < np.max(np.abs(g1)), (
            "psech_ref: tfr=4 peak must be lower than tfr=1 peak"
        )

