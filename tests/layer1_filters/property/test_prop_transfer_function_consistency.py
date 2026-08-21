"""
test_prop_transfer_function_consistency.py
==========================================
Python port of:
    layer1_filters/property/PropTransferFunctionConsistency.m

Consistency of comp_transferfunction across filter types.

1. BL filter: H_full[k] = 0 outside [foff, foff+bw-1]; BL segment = g.H(L).
2. FIR filter: H == fft(roll(postpad(h, L), offset)).
3. Biquad filter: H == g.H(L) exactly.
4. Freq filter: H_full constructed from BL segment at foff.
5. Delay: adding delay d multiplies H by exp(-2πi*d*k/L).
6. Peak bin scales correctly with L for BL filters.
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import postpad_ref


def _bl_manual(g, L: int) -> np.ndarray:
    """Reconstruct full-length response by placing BL segment at foff."""
    H_bl   = np.asarray(g["H"](L))
    foff   = int(g["foff"](L))
    Mbl    = len(H_bl)
    H_full = np.zeros(L, dtype=complex)
    idx    = np.mod(np.arange(foff, foff + Mbl), L)
    H_full[idx] = H_bl
    return H_full


# ---------------------------------------------------------------------------
# BL filter consistency
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestBlFilterConsistencyImpl:
    """
    MATLAB counterpart: PropTransferFunctionConsistency (BL section).
    """

    @pytest.mark.parametrize("L", [128, 256, 512, 1024])
    def test_blfilter_across_lengths(self, needs_impl, L):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g        = blfilter("hann", 0.1, 0.3)
        H_manual = _bl_manual(g, L)
        H_tf     = comp_transferfunction(g, L)
        np.testing.assert_allclose(H_tf, H_manual, atol=1e-10,
                                   err_msg=f"blfilter: mismatch at L={L}")

    def test_blfilter_dc_consistency(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = blfilter("hann", 0.05, 0)
        L = 256
        np.testing.assert_allclose(comp_transferfunction(g, L), _bl_manual(g, L), atol=1e-10)

    def test_blfilter_high_fc_wraparound(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = blfilter("hann", 0.1, 0.8)
        L = 512
        np.testing.assert_allclose(comp_transferfunction(g, L), _bl_manual(g, L), atol=1e-10)


# ---------------------------------------------------------------------------
# FIR filter consistency
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFirFilterConsistencyImpl:
    """
    MATLAB counterpart: PropTransferFunctionConsistency (FIR section).
    """

    @pytest.mark.parametrize("M,L", [(16, 128), (32, 256), (64, 512)])
    def test_firfilter_across_lengths(self, needs_impl, M, L):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g        = firfilter("hann", M)
        h        = np.asarray(g["h"])
        H_manual = np.fft.fft(np.roll(postpad_ref(h, L), g["offset"]))
        np.testing.assert_allclose(comp_transferfunction(g, L), H_manual, atol=1e-10)

    def test_firfilter_causal(self, needs_impl):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = firfilter("hann", 32, "causal")
        L = 256
        h = np.asarray(g["h"])
        H_manual = np.fft.fft(np.roll(postpad_ref(h, L), g["offset"]))
        np.testing.assert_allclose(comp_transferfunction(g, L), H_manual, atol=1e-10)

    def test_firfilter_truncgauss(self, needs_impl):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = firfilter("truncgauss", 48)
        L = 512
        h = np.asarray(g["h"])
        H_manual = np.fft.fft(np.roll(postpad_ref(h, L), g["offset"]))
        np.testing.assert_allclose(comp_transferfunction(g, L), H_manual, atol=1e-10)


# ---------------------------------------------------------------------------
# Biquad consistency
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestBiquadConsistencyImpl:
    """
    MATLAB counterpart: PropTransferFunctionConsistency (biquad section).
    """

    @pytest.mark.parametrize("L", [128, 256, 512, 1024])
    def test_biquadfilter_across_lengths(self, needs_impl, L):
        from cool_frames.filters import biquadfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = biquadfilter(0.3, 0.05, "peak")
        np.testing.assert_allclose(comp_transferfunction(g, L), g["H"](L), atol=1e-10,
                                   err_msg=f"biquadfilter mismatch at L={L}")


# ---------------------------------------------------------------------------
# Freq filter consistency
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFreqFilterConsistencyImpl:
    """
    MATLAB counterpart: PropTransferFunctionConsistency (freqfilter section).
    """

    def test_freqfilter_gauss(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 512
        fc = 0.3
        g  = freqfilter("gauss", 0.1, fc)
        H_bl     = np.asarray(g["H"](L))
        foff     = int(g["foff"](L))
        H_manual = np.roll(postpad_ref(H_bl, L), foff)
        np.testing.assert_allclose(comp_transferfunction(g, L), H_manual, atol=1e-10)

    def test_freqfilter_gammatone(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 512
        fc = 0.25
        g  = freqfilter("gammatone", 0.08, fc)
        H_bl     = np.asarray(g["H"](L))
        foff     = int(g["foff"](L))
        H_manual = np.roll(postpad_ref(H_bl, L), foff)
        np.testing.assert_allclose(comp_transferfunction(g, L), H_manual, atol=1e-10)


# ---------------------------------------------------------------------------
# Delay consistency
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestDelayConsistencyImpl:
    """
    MATLAB counterpart: PropTransferFunctionConsistency (delay section).
    """

    def test_delay_shifts_phase_blfilter(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 256
        d = 5
        k = np.arange(L)
        phase = np.exp(-2j * np.pi * d * k / L)
        g0 = blfilter("hann", 0.1, 0.3)
        gd = blfilter("hann", 0.1, 0.3, "delay", d)
        np.testing.assert_allclose(comp_transferfunction(gd, L),
                                   comp_transferfunction(g0, L) * phase, atol=1e-10)

    def test_delay_shifts_phase_freqfilter(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 256
        d = 5
        k = np.arange(L)
        phase = np.exp(-2j * np.pi * d * k / L)
        g0 = freqfilter("gauss", 0.1, 0.3)
        gd = freqfilter("gauss", 0.1, 0.3, "delay", d)
        np.testing.assert_allclose(comp_transferfunction(gd, L),
                                   comp_transferfunction(g0, L) * phase, atol=1e-10)


# ---------------------------------------------------------------------------
# Peak-bin scaling and energy consistency across L
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestLengthInvarianceImpl:
    """
    MATLAB counterpart: PropTransferFunctionConsistency (length invariance section).
    """

    def test_peak_bin_scales_with_L(self, needs_impl):
        """Peak bin (0-indexed) ≈ round(fc/2*L) for L=256 and L=512."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        fc       = 0.3
        g        = blfilter("hann", 0.05, fc)
        tol_bins = 2
        for L in (256, 512):
            H        = comp_transferfunction(g, L)
            peak_bin = int(np.argmax(np.abs(H)))
            expected = round(fc / 2 * L)
            assert abs(peak_bin - expected) <= tol_bins, \
                f"L={L}: peak_bin={peak_bin}, expected≈{expected}"

    def test_energy_norm_consistent_across_L(self, needs_impl):
        """(1/L)*sum|H|^2 should be consistent across different L values."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g        = blfilter("hann", 0.1, 0.3)
        energies = [np.sum(np.abs(comp_transferfunction(g, L)) ** 2) / L
                    for L in [128, 256, 512, 1024]]
        cv = np.std(energies) / np.mean(energies)
        assert cv < 0.1, f"Energy coefficient of variation {cv:.4f} too large"
