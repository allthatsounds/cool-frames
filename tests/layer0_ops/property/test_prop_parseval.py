"""
test_prop_parseval.py
=====================
Python port of:
    layer0_ops/property/PropParsevalConsistency.m

Property tests for energy distribution in subbands.

For a tight frame with bound A:
    sum_m (1/a_m) * ||c_m||^2 = A * ||x||^2

For the synthetic rectangular full-DFT filterbank (fft_fb fixture) the
filters are not a tight frame, but:
  - Zero input -> zero subband energy
  - Scaling by alpha -> subband energy scales by |alpha|^2
  - Weighted energy ratio (divided by input energy) is in a sane range

Uses the fft_fb fixture (synthetic full-length DFT filters).
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Reference tests (unconditional) — algebraic energy properties
# ---------------------------------------------------------------------------

class TestParsevalReference:
    """
    Algebraic energy properties that hold for any linear filterbank,
    verified via pure-numpy DFT multiplication.
    """

    def test_zero_input_zero_energy(self, fft_fb):
        """Zero input -> zero subband energy at DFT level."""
        Ls = fft_fb["Ls"]
        G  = fft_fb["G"]
        a  = fft_fb["a"]
        F  = np.zeros(Ls, dtype=complex)

        for m, (g_m, a_m) in enumerate(zip(G, a)):
            N_m = Ls // a_m
            c_m = np.fft.ifft(F * g_m)[:N_m]
            energy = np.sum(np.abs(c_m) ** 2)
            assert energy == pytest.approx(0.0, abs=1e-20), \
                f"Band {m}: non-zero energy for zero input"

    def test_linear_scaling(self, fft_fb):
        """Subband energy scales with |alpha|^2 at DFT level."""
        rng   = np.random.default_rng(42)
        Ls    = fft_fb["Ls"]
        G     = fft_fb["G"]
        a     = fft_fb["a"]
        alpha = 3.7

        x  = rng.standard_normal(Ls)
        F1 = np.fft.fft(x)
        F2 = np.fft.fft(alpha * x)

        for m, (g_m, a_m) in enumerate(zip(G, a)):
            N_m = Ls // a_m
            c1  = np.fft.ifft(F1 * g_m)[:N_m]
            c2  = np.fft.ifft(F2 * g_m)[:N_m]
            e1  = np.sum(np.abs(c1) ** 2)
            e2  = np.sum(np.abs(c2) ** 2)
            assert e2 == pytest.approx(alpha ** 2 * e1, rel=1e-10), \
                f"Band {m}: energy does not scale with alpha^2"

    def test_weighted_energy_sane_range(self, fft_fb):
        """Weighted subband energy ratio is in (0.01, 200) for random input."""
        rng = np.random.default_rng(42)
        Ls  = fft_fb["Ls"]
        G   = fft_fb["G"]
        a   = fft_fb["a"]
        x   = rng.standard_normal(Ls)
        F   = np.fft.fft(x)

        weighted = sum(
            np.sum(np.abs(np.fft.ifft(F * g_m)[: Ls // a_m]) ** 2) / a_m
            for g_m, a_m in zip(G, a)
        )
        ratio = weighted / np.sum(x ** 2)
        assert 0.01 < ratio < 200, f"Weighted energy ratio {ratio:.4f} out of sane range"


# ---------------------------------------------------------------------------
# Implementation Parseval tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestParsevalConsistencyImpl:
    """
    MATLAB counterpart: PropParsevalConsistency.
    Uses comp_filterbank_fft.
    """

    @pytest.mark.parametrize("seed", [42, 123])
    def test_weighted_energy_real_noise(self, needs_impl, seed, fft_fb):
        """Weighted subband energy ratio is in (0.01, 200) for real noise."""
        from cool_frames.core import comp_filterbank_fft  # type: ignore

        rng = np.random.default_rng(seed)
        Ls  = fft_fb["Ls"]
        x   = rng.standard_normal(Ls)
        F   = np.fft.fft(x)
        c   = comp_filterbank_fft(F, fft_fb["G"], fft_fb["a"])

        weighted = sum(
            np.sum(np.abs(c_m) ** 2) / a_m
            for c_m, a_m in zip(c, fft_fb["a"])
        )
        ratio = weighted / np.sum(x ** 2)
        assert 0.01 < ratio < 200, f"seed={seed}: ratio {ratio:.4f} out of range"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_weighted_energy_complex_noise(self, needs_impl, seed, fft_fb):
        """Weighted subband energy ratio is in (0.01, 200) for complex noise."""
        from cool_frames.core import comp_filterbank_fft  # type: ignore

        rng = np.random.default_rng(seed)
        Ls  = fft_fb["Ls"]
        x   = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
        F   = np.fft.fft(x)
        c   = comp_filterbank_fft(F, fft_fb["G"], fft_fb["a"])

        weighted = sum(
            np.sum(np.abs(c_m) ** 2) / a_m
            for c_m, a_m in zip(c, fft_fb["a"])
        )
        ratio = weighted / np.sum(np.abs(x) ** 2)
        assert 0.01 < ratio < 200, f"seed={seed}: ratio {ratio:.4f} out of range"

    def test_zero_input_zero_energy(self, needs_impl, fft_fb):
        """Zero input -> zero subband energy."""
        from cool_frames.core import comp_filterbank_fft  # type: ignore

        F = np.fft.fft(fft_fb["zeros_sig"])
        c = comp_filterbank_fft(F, fft_fb["G"], fft_fb["a"])

        total = sum(np.sum(np.abs(c_m) ** 2) for c_m in c)
        assert total == pytest.approx(0.0, abs=1e-20)

    def test_energy_scales_with_alpha_squared(self, needs_impl, fft_fb):
        """filterbank(alpha*x) has energy alpha^2 * filterbank(x) energy."""
        from cool_frames.core import comp_filterbank_fft  # type: ignore

        rng   = np.random.default_rng(42)
        Ls    = fft_fb["Ls"]
        alpha = 3.7
        x     = rng.standard_normal(Ls)

        c1 = comp_filterbank_fft(np.fft.fft(x),         fft_fb["G"], fft_fb["a"])
        c2 = comp_filterbank_fft(np.fft.fft(alpha * x), fft_fb["G"], fft_fb["a"])

        e1 = sum(np.sum(np.abs(c_m) ** 2) for c_m in c1)
        e2 = sum(np.sum(np.abs(c_m) ** 2) for c_m in c2)

        assert e2 == pytest.approx(alpha ** 2 * e1, rel=1e-10), \
            "Subband energy does not scale with |alpha|^2"
