"""
test_freqfilter.py
==================
Python port of:
    layer1_filters/unit/TestFreqFilter.m

Covers: freqfilter (frequency-domain filter constructor)

API
---
freqfilter(name, bw)           -> Gaussian/Gammatone/Butterworth bandpass at DC
freqfilter(name, bw, fc)       -> same, shifted to normalised fc
freqfilter(name, bw, fc, norm) -> with explicit normalisation flag

The returned object is dict-like with same fields as blfilter:
    H, foff, delay, realonly
"""

from __future__ import annotations

import pytest

import numpy as np


@pytest.mark.requires_impl
class TestFreqFilterStructImpl:
    """
    MATLAB counterpart: TestFreqFilter (struct format section).
    """

    def test_has_required_fields(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        g = freqfilter("gauss", 0.1)
        for field in ("H", "foff", "delay", "realonly"):
            assert field in g, f"Missing field: {field}"

    def test_H_is_callable(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        g = freqfilter("gauss", 0.1)
        assert callable(g["H"])

    def test_default_realonly_is_zero(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        assert freqfilter("gauss", 0.1, 0.3)["realonly"] == 0

    def test_real_flag_sets_realonly(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        assert freqfilter("gauss", 0.1, 0.3, "real")["realonly"] == 1

    def test_default_delay_is_zero(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        assert freqfilter("gauss", 0.1, 0.3)["delay"] == 0

    def test_delay_parameter_stored(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        assert freqfilter("gauss", 0.1, 0.3, "delay", 7)["delay"] == 7


@pytest.mark.requires_impl
class TestFreqFilterTransferFunctionImpl:
    """
    MATLAB counterpart: TestFreqFilter (transfer function section).
    """

    @pytest.mark.parametrize("L", [128, 256, 1024])
    def test_transfer_function_length(self, needs_impl, L):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = freqfilter("gauss", 0.1, 0.3)
        assert len(comp_transferfunction(g, L)) == L

    def test_transfer_function_finite(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        H = comp_transferfunction(freqfilter("gauss", 0.1, 0.3), 256)
        assert np.all(np.isfinite(H))

    def test_transfer_function_not_all_zero(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        H = comp_transferfunction(freqfilter("gauss", 0.1, 0.3), 256)
        assert np.max(np.abs(H)) > 0

    def test_gammatone_transfer_function(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = freqfilter("gammatone", 0.1, 0.3)
        H = comp_transferfunction(g, 256)
        assert len(H) == 256 and np.all(np.isfinite(H))

    def test_butterworth_transfer_function(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = freqfilter("butterworth", 0.1, 0.3)
        H = comp_transferfunction(g, 256)
        assert len(H) == 256 and np.all(np.isfinite(H))


@pytest.mark.requires_impl
class TestFreqFilterNormImpl:
    """
    MATLAB counterpart: TestFreqFilter (energy normalisation section).
    """

    def test_energy_norm_default(self, needs_impl):
        """Default 'energy': (1/L)*sum|H|^2 == 1."""
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 512
        H = comp_transferfunction(freqfilter("gauss", 0.05, 0.3), L)
        assert np.sum(np.abs(H) ** 2) / L == pytest.approx(1.0, abs=1e-10)

    def test_peak_norm(self, needs_impl):
        """'peak': max|H| == 1."""
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 512
        H = comp_transferfunction(freqfilter("gauss", 0.05, 0.3, "peak"), L)
        assert np.max(np.abs(H)) == pytest.approx(1.0, abs=1e-10)

    def test_scal_multiplies_response(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 256
        s  = 4.0
        g1 = freqfilter("gauss", 0.05, 0.3)
        g2 = freqfilter("gauss", 0.05, 0.3, "scal", s)
        H1 = comp_transferfunction(g1, L)
        H2 = comp_transferfunction(g2, L)
        np.testing.assert_allclose(H2, s * H1, atol=1e-10 * np.linalg.norm(H1))


@pytest.mark.requires_impl
class TestFreqFilterCentreFreqImpl:
    """
    MATLAB counterpart: TestFreqFilter (centre frequency section).
    """

    def test_peak_near_centre_freq(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 1024
        fc = 0.3
        g  = freqfilter("gauss", 0.05, fc, "peak")
        H  = comp_transferfunction(g, L)
        peak_bin     = int(np.argmax(np.abs(H)))
        expected_bin = round(fc / 2 * L)
        assert abs(peak_bin - expected_bin) < 5

    def test_hz_input_consistency(self, needs_impl):
        """Hz and normalised inputs give the same filter."""
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        fs    = 8000
        fc_hz = 1000
        bw_hz = 400
        g_hz   = freqfilter("gauss", bw_hz,         fc_hz,          "fs", fs)
        g_norm = freqfilter("gauss", bw_hz / (fs/2), fc_hz / (fs/2))
        L = 512
        np.testing.assert_allclose(
            comp_transferfunction(g_hz,   L),
            comp_transferfunction(g_norm, L),
            atol=1e-10
        )
