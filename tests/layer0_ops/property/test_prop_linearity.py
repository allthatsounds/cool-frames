"""
test_prop_linearity.py
======================
Python port of:
    layer0_ops/property/PropLinearity.m

Property tests for linearity of the filterbank analysis and synthesis paths.

Property (analysis):
    comp_filterbank_fft(alpha*F1 + F2, G, a)[m]
        == alpha * comp_filterbank_fft(F1, G, a)[m]
           + comp_filterbank_fft(F2, G, a)[m]

Property (synthesis):
    comp_ifilterbank_fft(alpha*c1 + c2, G, a)
        == alpha * comp_ifilterbank_fft(c1, G, a)
           + comp_ifilterbank_fft(c2, G, a)

Uses synthetic rectangular full-length DFT filters (fft_fb fixture).
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Reference linearity of DFT multiplication (unconditional)
# ---------------------------------------------------------------------------

class TestLinearityReference:
    """
    Verify the mathematical linearity property holds for pure DFT
    multiplication — no impl needed.
    """

    def test_analysis_dft_linearity(self, fft_fb):
        """alpha*F1 + F2 -> same result as alpha*result1 + result2 (DFT level)."""
        rng = np.random.default_rng(42)
        Ls  = fft_fb["Ls"]
        G   = fft_fb["G"]
        a   = fft_fb["a"]

        for _ in range(10):
            x1    = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            x2    = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            alpha = rng.standard_normal() + 1j * rng.standard_normal()

            F_comb = np.fft.fft(alpha * x1 + x2)
            F1     = np.fft.fft(x1)
            F2     = np.fft.fft(x2)

            # For each filter, check that pointwise multiply distributes
            for m, (g_m, a_m) in enumerate(zip(G, a)):
                N_m         = Ls // a_m
                # FFT-based subband: IFFT of (F * G) downsampled in freq domain
                c_comb_m    = np.fft.ifft(F_comb * g_m)[:N_m]
                c1_m        = np.fft.ifft(F1 * g_m)[:N_m]
                c2_m        = np.fft.ifft(F2 * g_m)[:N_m]
                expected    = alpha * c1_m + c2_m
                np.testing.assert_allclose(
                    c_comb_m, expected, atol=1e-10,
                    err_msg=f"DFT-level linearity failed for band {m}"
                )


# ---------------------------------------------------------------------------
# Implementation linearity tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestAnalysisLinearityImpl:
    """
    MATLAB counterpart: PropLinearity.testLinearityAnalysisPath
    Uses comp_filterbank_fft.
    """

    @pytest.mark.parametrize("seed", [42, 123, 7])
    def test_analysis_linearity(self, needs_impl, seed, fft_fb):
        """comp_filterbank_fft(alpha*F1 + F2) == alpha*c1 + c2 for 10 random trials."""
        from cool_frames.core import comp_filterbank_fft  # type: ignore

        rng   = np.random.default_rng(seed)
        Ls    = fft_fb["Ls"]
        G     = fft_fb["G"]
        a     = fft_fb["a"]

        for trial in range(10):
            x1    = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            x2    = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            alpha = rng.standard_normal() + 1j * rng.standard_normal()

            c_comb = comp_filterbank_fft(np.fft.fft(alpha * x1 + x2), G, a)
            c1     = comp_filterbank_fft(np.fft.fft(x1), G, a)
            c2     = comp_filterbank_fft(np.fft.fft(x2), G, a)

            scale = abs(alpha) * np.linalg.norm(x1) + np.linalg.norm(x2)
            tol   = 1e-10 * max(scale, 1.0)

            for m in range(len(c1)):
                expected = alpha * c1[m] + c2[m]
                np.testing.assert_allclose(
                    c_comb[m], expected, atol=tol,
                    err_msg=f"seed={seed}, trial={trial}, band={m}: analysis linearity failed"
                )


@pytest.mark.requires_impl
class TestSynthesisLinearityImpl:
    """
    MATLAB counterpart: PropLinearity.testLinearitySynthesisPath
    Uses comp_ifilterbank_fft.
    """

    @pytest.mark.parametrize("seed", [42, 123, 7])
    def test_synthesis_linearity(self, needs_impl, seed, fft_fb):
        """comp_ifilterbank_fft(alpha*c1 + c2) == alpha*x1 + x2 for 10 random trials."""
        from cool_frames.core import comp_ifilterbank_fft  # type: ignore

        rng  = np.random.default_rng(seed)
        Ls   = fft_fb["Ls"]
        G    = fft_fb["G"]
        a    = fft_fb["a"]
        M    = fft_fb["M"]

        for trial in range(10):
            alpha = rng.standard_normal() + 1j * rng.standard_normal()

            c1     = [rng.standard_normal(Ls // a_m) + 1j * rng.standard_normal(Ls // a_m)
                      for a_m in a]
            c2     = [rng.standard_normal(Ls // a_m) + 1j * rng.standard_normal(Ls // a_m)
                      for a_m in a]
            c_comb = [alpha * c1[m] + c2[m] for m in range(M)]

            x_comb  = comp_ifilterbank_fft(c_comb, G, a)
            x1_recon = comp_ifilterbank_fft(c1, G, a)
            x2_recon = comp_ifilterbank_fft(c2, G, a)

            expected = alpha * x1_recon + x2_recon
            scale    = abs(alpha) * np.linalg.norm(x1_recon) + np.linalg.norm(x2_recon)
            tol      = 1e-10 * max(scale, 1.0)

            np.testing.assert_allclose(
                x_comb, expected, atol=tol,
                err_msg=f"seed={seed}, trial={trial}: synthesis linearity failed"
            )
