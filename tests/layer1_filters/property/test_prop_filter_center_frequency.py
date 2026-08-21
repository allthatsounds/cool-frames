"""
test_prop_filter_center_frequency.py
=====================================
Python port of:
    layer1_filters/property/PropFilterCenterFrequency.m

Centre-frequency accuracy for all filter constructors.

LTFAT frequency convention: fc in [0, 2], Nyquist = 1.
Peak bin (0-indexed) ≈ round(fc / 2 * L).
Tolerance: ±3 bins (covers rounding at bin boundaries).
"""

from __future__ import annotations

import pytest

import numpy as np

L         = 1024
TOL_BINS  = 3


def _peak_bin(H: np.ndarray) -> int:
    """0-indexed argmax of |H|."""
    return int(np.argmax(np.abs(H)))


def _expected_bin(fc: float, L: int = L) -> int:
    return round(fc / 2 * L)


# ---------------------------------------------------------------------------
# blfilter
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestBlFilterCentreFreqImpl:
    """
    MATLAB counterpart: PropFilterCenterFrequency (blfilter section).
    """

    @pytest.mark.parametrize("fc", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    def test_blfilter_centre_freq_sweep(self, needs_impl, fc):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = blfilter("hann", 0.08, fc, "peak")
        H = comp_transferfunction(g, L)
        assert abs(_peak_bin(H) - _expected_bin(fc)) < TOL_BINS, \
            f"blfilter fc={fc}: peak_bin={_peak_bin(H)}, expected≈{_expected_bin(fc)}"

    @pytest.mark.parametrize("fc_hz", [500, 1000, 2000, 3000])
    def test_blfilter_hz_centre_freq(self, needs_impl, fc_hz):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        fs   = 8000
        g    = blfilter("hann", 0.08, fc_hz, "peak", "fs", fs)
        H    = comp_transferfunction(g, L)
        fc_n = fc_hz / (fs / 2)
        assert abs(_peak_bin(H) - _expected_bin(fc_n)) < TOL_BINS, \
            f"blfilter fc_hz={fc_hz}: peak_bin={_peak_bin(H)}, expected≈{_expected_bin(fc_n)}"

    def test_dc_filter_peak_at_bin_0(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        H = comp_transferfunction(blfilter("hann", 0.1, 0, "peak"), L)
        assert _peak_bin(H) == 0

    def test_monotone_peak_with_increasing_fc(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        fcs       = np.arange(0.1, 0.75, 0.05)
        peak_bins = [_peak_bin(comp_transferfunction(blfilter("hann", 0.06, fc, "peak"), L))
                     for fc in fcs]
        assert np.all(np.diff(peak_bins) >= 0), \
            "Peak bin should be non-decreasing as fc increases"


# ---------------------------------------------------------------------------
# freqfilter
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFreqFilterCentreFreqImpl:
    """
    MATLAB counterpart: PropFilterCenterFrequency (freqfilter section).
    """

    @pytest.mark.parametrize("fc", [0.1, 0.2, 0.3, 0.5, 0.7])
    def test_freqfilter_centre_freq_sweep(self, needs_impl, fc):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        H = comp_transferfunction(freqfilter("gauss", 0.05, fc, "peak"), L)
        assert abs(_peak_bin(H) - _expected_bin(fc)) < TOL_BINS

    @pytest.mark.parametrize("fc", [0.2, 0.4, 0.6])
    def test_freqfilter_gammatone_centre_freq(self, needs_impl, fc):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        H = comp_transferfunction(freqfilter("gammatone", 0.05, fc, "peak"), L)
        assert abs(_peak_bin(H) - _expected_bin(fc)) < TOL_BINS

    def test_dc_freqfilter_peak_at_bin_0(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        H = comp_transferfunction(freqfilter("gauss", 0.05, 0, "peak"), L)
        assert _peak_bin(H) == 0


# ---------------------------------------------------------------------------
# firfilter
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFirFilterCentreFreqImpl:
    """
    MATLAB counterpart: PropFilterCenterFrequency (firfilter section).
    """

    @pytest.mark.parametrize("fc", [0.1, 0.25, 0.4])
    def test_firfilter_centre_freq_sweep(self, needs_impl, fc):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        H = comp_transferfunction(firfilter("hann", 128, fc, "peak"), L)
        assert abs(_peak_bin(H) - _expected_bin(fc)) < TOL_BINS


# ---------------------------------------------------------------------------
# biquadfilter
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestBiquadFilterCentreFreqImpl:
    """
    MATLAB counterpart: PropFilterCenterFrequency (biquadfilter section).
    Search positive-frequency half to handle the conjugate mirror.
    """

    @pytest.mark.parametrize("fc", [0.1, 0.2, 0.3, 0.4, 0.5])
    def test_biquadfilter_centre_freq_sweep(self, needs_impl, fc):
        from cool_frames.filters import biquadfilter  # type: ignore
        g     = biquadfilter(fc, 0.02)
        H     = g["H"](L)
        H_pos = H[: L // 2 + 1]   # positive-frequency half
        peak_bin = int(np.argmax(np.abs(H_pos)))
        assert abs(peak_bin - _expected_bin(fc)) < TOL_BINS, \
            f"biquadfilter fc={fc}: peak_bin={peak_bin}, expected≈{_expected_bin(fc)}"
