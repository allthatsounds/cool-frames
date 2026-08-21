"""
test_fft_full.py
================
Python port of:
    layer0_ops/unit/TestFFTFull.m

Covers: comp_filterbank_fft, comp_ifilterbank_fft

Calling conventions
-------------------
comp_filterbank_fft(F, G, a)
    F : np.ndarray [L] or [L x W] — fft(f), NOT the signal itself
    G : list of M length-L DFT responses
    a : array-like of M integer subsampling factors (all must divide L)

comp_ifilterbank_fft(c, G, a)
    c : list of M coefficient arrays [N_m] or [N_m x W], N_m = L / a_m
    G : same list of length-L DFT responses
    a : same subsampling factors
    Returns: length-L frequency-domain array (apply ifft to get time domain)

The fixture fft_fb (from conftest) provides:
    Ls, M, a, G, noise_real, noise_stereo, zeros_sig
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Reference tests (no impl required) – structural / algebraic checks
# using the synthetic fixture built in conftest
# ---------------------------------------------------------------------------

class TestFFTFilterbankReference:
    """
    Structural checks that can be done with pure numpy using the fixture data.
    MATLAB counterpart: TestFFTFull.
    """

    def test_subsampling_divides_Ls(self, fft_fb):
        """All a(m) must divide Ls exactly (required by comp_filterbank_fft)."""
        Ls = fft_fb["Ls"]
        for a_m in fft_fb["a"]:
            assert Ls % a_m == 0, f"a={a_m} does not divide Ls={Ls}"

    def test_filter_lengths(self, fft_fb):
        """Each G[m] has length Ls."""
        Ls = fft_fb["Ls"]
        for m, g in enumerate(fft_fb["G"]):
            assert len(g) == Ls, f"G[{m}] has length {len(g)}, expected {Ls}"

    def test_num_bands(self, fft_fb):
        """Number of filters equals M."""
        assert len(fft_fb["G"]) == fft_fb["M"]

    def test_rectangular_filters_are_binary(self, fft_fb):
        """Synthetic rectangular filters contain only 0 and 1."""
        for g in fft_fb["G"]:
            assert set(np.unique(g)).issubset({0.0, 1.0})

    def test_noise_signal_length(self, fft_fb):
        """noise_real and zeros_sig have length Ls."""
        assert len(fft_fb["noise_real"]) == fft_fb["Ls"]
        assert len(fft_fb["zeros_sig"]) == fft_fb["Ls"]


# ---------------------------------------------------------------------------
# comp_filterbank_fft – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCompFilterbankFFTImpl:
    """
    MATLAB counterpart: TestFFTFull (comp_filterbank_fft section).
    """

    def test_zero_input(self, needs_impl, fft_fb):
        """Zero signal -> all subbands zero."""
        from cool_frames.core import comp_filterbank_fft  # type: ignore

        F = np.fft.fft(fft_fb["zeros_sig"])
        c = comp_filterbank_fft(F, fft_fb["G"], fft_fb["a"])

        for m, c_m in enumerate(c):
            np.testing.assert_allclose(
                c_m, np.zeros_like(c_m), atol=1e-14,
                err_msg=f"Subband {m} not zero for zero input"
            )

    def test_output_sizes(self, needs_impl, fft_fb):
        """Each c[m] has N_m = Ls/a[m] rows."""
        from cool_frames.core import comp_filterbank_fft  # type: ignore

        Ls = fft_fb["Ls"]
        F  = np.fft.fft(fft_fb["noise_real"])
        c  = comp_filterbank_fft(F, fft_fb["G"], fft_fb["a"])

        for m, (c_m, a_m) in enumerate(zip(c, fft_fb["a"])):
            expected = Ls // a_m
            assert c_m.shape[0] == expected, \
                f"Subband {m}: expected {expected} rows, got {c_m.shape[0]}"

    def test_real_noise_returns_M_bands(self, needs_impl, fft_fb):
        """Real noise: M subbands returned."""
        from cool_frames.core import comp_filterbank_fft  # type: ignore

        F = np.fft.fft(fft_fb["noise_real"])
        c = comp_filterbank_fft(F, fft_fb["G"], fft_fb["a"])
        assert len(c) == fft_fb["M"]

    def test_multichannel(self, needs_impl, fft_fb):
        """Multi-channel [Ls x 2]: each c[m] has 2 columns."""
        from cool_frames.core import comp_filterbank_fft  # type: ignore

        F = np.fft.fft(fft_fb["noise_stereo"], axis=0)
        c = comp_filterbank_fft(F, fft_fb["G"], fft_fb["a"])

        for m, c_m in enumerate(c):
            assert c_m.shape[1] == 2, \
                f"Subband {m}: expected 2 columns, got {c_m.shape[1]}"


# ---------------------------------------------------------------------------
# comp_ifilterbank_fft – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCompIFilterbankFFTImpl:
    """
    MATLAB counterpart: TestFFTFull (comp_ifilterbank_fft section).
    """

    def test_zero_coefficients(self, needs_impl, fft_fb):
        """Zero coefficients -> zero frequency-domain output."""
        from cool_frames.core import comp_ifilterbank_fft  # type: ignore

        Ls     = fft_fb["Ls"]
        c_zero = [np.zeros(Ls // a_m) for a_m in fft_fb["a"]]
        F_recon = comp_ifilterbank_fft(c_zero, fft_fb["G"], fft_fb["a"])

        np.testing.assert_allclose(F_recon, np.zeros(Ls), atol=1e-10)

    def test_allpass_roundtrip(self, needs_impl, fft_fb):
        """All-pass filter (G=[1,...,1], a=1): ifft(ifilterbank(filterbank(F))) ≈ f."""
        from cool_frames.core import comp_filterbank_fft, comp_ifilterbank_fft  # type: ignore

        Ls        = fft_fb["Ls"]
        f         = fft_fb["noise_real"]
        G_trivial = [np.ones(Ls, dtype=complex)]
        a_trivial = np.array([1])

        F       = np.fft.fft(f)
        c       = comp_filterbank_fft(F, G_trivial, a_trivial)
        F_recon = comp_ifilterbank_fft(c, G_trivial, a_trivial)
        f_recon = np.real(np.fft.ifft(F_recon))

        err = np.linalg.norm(f_recon - f) / np.linalg.norm(f)
        assert err < 1e-10, \
            f"All-pass roundtrip: relative error {err:.2e} exceeds 1e-10"


# ---------------------------------------------------------------------------
# combined forward+inverse – implementation test
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFFTRoundtripImpl:
    """Forward + inverse roundtrip consistency checks."""

    def test_forward_inverse_consistency(self, needs_impl, fft_fb):
        """comp_ifilterbank_fft(comp_filterbank_fft(F, G, a), G, a) is non-trivial."""
        from cool_frames.core import comp_filterbank_fft, comp_ifilterbank_fft  # type: ignore

        F       = np.fft.fft(fft_fb["noise_real"])
        c       = comp_filterbank_fft(F, fft_fb["G"], fft_fb["a"])
        F_recon = comp_ifilterbank_fft(c, fft_fb["G"], fft_fb["a"])

        # Output should have the right length and non-trivial energy
        assert len(F_recon) == fft_fb["Ls"]
        assert np.linalg.norm(F_recon) > 0
