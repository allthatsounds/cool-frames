"""
test_fft_bandlimited.py
=======================
Python port of:
    layer0_ops/unit/TestFFTBandlimited.m

Covers: comp_filterbank_fftbl

Calling convention
------------------
comp_filterbank_fftbl(F, G, foff, a, realonly)
    F        : np.ndarray [L] or [L x W] — fft(f)
    G        : list of M short DFT responses (len(G[m]) < L in general)
    foff     : array-like [M] — frequency offset (starting DFT bin) per band
    a        : subsampling factors [M] (integer or fractional [M x 2])
    realonly : array-like [M] bool — if True, add conjugate mirror band

The fixture fftbl_fb (from conftest) provides:
    Ls, M, bw, G_bl, foff, realonly, a, noise_real, noise_complex, zeros_sig
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Structural reference tests (no impl required)
# ---------------------------------------------------------------------------

class TestFFTBLReference:
    """
    Structural sanity checks on the synthetic bandlimited fixture.
    MATLAB counterpart: TestFFTBandlimited.
    """

    def test_filter_count(self, fftbl_fb):
        """Number of BL filters equals M."""
        assert len(fftbl_fb["G_bl"]) == fftbl_fb["M"]

    def test_filter_lengths(self, fftbl_fb):
        """Each G_bl[m] has length bw."""
        bw = fftbl_fb["bw"]
        for m, g in enumerate(fftbl_fb["G_bl"]):
            assert len(g) == bw, f"G_bl[{m}] length {len(g)}, expected {bw}"

    def test_foff_length(self, fftbl_fb):
        """foff vector has length M."""
        assert len(fftbl_fb["foff"]) == fftbl_fb["M"]

    def test_signal_lengths(self, fftbl_fb):
        """Input signals have length Ls."""
        Ls = fftbl_fb["Ls"]
        assert len(fftbl_fb["noise_real"])    == Ls
        assert len(fftbl_fb["noise_complex"]) == Ls
        assert len(fftbl_fb["zeros_sig"])     == Ls

    def test_foff_non_negative(self, fftbl_fb):
        """Frequency offsets are non-negative."""
        assert all(f >= 0 for f in fftbl_fb["foff"])


# ---------------------------------------------------------------------------
# comp_filterbank_fftbl – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCompFilterbankFFTBLImpl:
    """
    MATLAB counterpart: TestFFTBandlimited.
    """

    def test_zero_input(self, needs_impl, fftbl_fb):
        """Zero signal -> all subbands zero."""
        from cool_frames.core import comp_filterbank_fftbl  # type: ignore

        F = np.fft.fft(fftbl_fb["zeros_sig"])
        c = comp_filterbank_fftbl(F, fftbl_fb["G_bl"], fftbl_fb["foff"],
                                  fftbl_fb["a"], fftbl_fb["realonly"])

        for m, c_m in enumerate(c):
            np.testing.assert_allclose(
                c_m, np.zeros_like(c_m), atol=1e-14,
                err_msg=f"Subband {m} not zero for zero input"
            )

    def test_output_sizes_positive(self, needs_impl, fftbl_fb):
        """Each subband has at least 1 row."""
        from cool_frames.core import comp_filterbank_fftbl  # type: ignore

        F = np.fft.fft(fftbl_fb["noise_real"])
        c = comp_filterbank_fftbl(F, fftbl_fb["G_bl"], fftbl_fb["foff"],
                                  fftbl_fb["a"], fftbl_fb["realonly"])

        assert len(c) > 0, "No subbands returned"
        for m, c_m in enumerate(c):
            assert np.asarray(c_m).shape[0] >= 1, f"Subband {m} has zero rows"

    def test_real_signal_returns_M_bands(self, needs_impl, fftbl_fb):
        """Real input: M subbands returned."""
        from cool_frames.core import comp_filterbank_fftbl  # type: ignore

        F = np.fft.fft(fftbl_fb["noise_real"])
        c = comp_filterbank_fftbl(F, fftbl_fb["G_bl"], fftbl_fb["foff"],
                                  fftbl_fb["a"], fftbl_fb["realonly"])

        assert len(c) == fftbl_fb["M"]

    def test_complex_signal_returns_M_bands(self, needs_impl, fftbl_fb):
        """Complex input: M subbands returned."""
        from cool_frames.core import comp_filterbank_fftbl  # type: ignore

        F = np.fft.fft(fftbl_fb["noise_complex"])
        c = comp_filterbank_fftbl(F, fftbl_fb["G_bl"], fftbl_fb["foff"],
                                  fftbl_fb["a"], fftbl_fb["realonly"])

        assert len(c) == fftbl_fb["M"]

    def test_realonly_single_band(self, needs_impl, fftbl_fb):
        """realonly=1 accepted without error for a single band."""
        from cool_frames.core import comp_filterbank_fftbl  # type: ignore

        G_ro   = [fftbl_fb["G_bl"][0]]
        foff_ro = np.array([fftbl_fb["foff"][0]])
        a_ro   = np.array([fftbl_fb["a"][0]])
        ro_ro  = np.array([1])

        F = np.fft.fft(fftbl_fb["noise_real"])
        c = comp_filterbank_fftbl(F, G_ro, foff_ro, a_ro, ro_ro)

        assert len(c) == 1
