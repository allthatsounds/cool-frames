"""
test_blfilter.py
================
Python port of:
    layer1_filters/unit/TestBlFilter.m

Covers: blfilter (band-limited filter constructor)

API
---
blfilter(name, fsupp)              -> BL filter at DC
blfilter(name, fsupp, fc)          -> BL filter at normalised fc
blfilter(name, fsupp, fc, norm)    -> with explicit normalisation
blfilter(name, fsupp, fc, 'fs', fs) -> fc and fsupp in Hz

The returned object is dict-like with keys:
    H        : callable  H(L) -> np.ndarray length-L DFT response (BL segment)
    foff     : callable  foff(L) -> int  starting bin of BL segment
    delay    : int
    realonly : int
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# blfilter – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestBlFilterStructImpl:
    """
    MATLAB counterpart: TestBlFilter (struct format section).
    """

    def test_has_required_fields(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        g = blfilter("hann", 0.1)
        for field in ("H", "foff", "delay", "realonly"):
            assert field in g, f"Missing field: {field}"

    def test_H_is_callable(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        g = blfilter("hann", 0.1)
        assert callable(g["H"]), "g['H'] must be callable"

    def test_foff_is_callable(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        g = blfilter("hann", 0.1, 0.3)
        assert callable(g["foff"]), "g['foff'] must be callable for shifted filter"

    def test_default_realonly_is_zero(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        g = blfilter("hann", 0.1, 0.3)
        assert g["realonly"] == 0

    def test_real_flag_sets_realonly(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        g = blfilter("hann", 0.1, 0.3, "real")
        assert g["realonly"] == 1

    def test_default_delay_is_zero(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        g = blfilter("hann", 0.1, 0.3)
        assert g["delay"] == 0

    def test_delay_parameter_stored(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        g = blfilter("hann", 0.1, 0.3, "delay", 5)
        assert g["delay"] == 5


@pytest.mark.requires_impl
class TestBlFilterTransferFunctionImpl:
    """
    MATLAB counterpart: TestBlFilter (comp_transferfunction section).
    """

    @pytest.mark.parametrize("L", [128, 256, 1024])
    def test_transfer_function_length(self, needs_impl, L):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = blfilter("hann", 0.1, 0.3)
        H = comp_transferfunction(g, L)
        assert len(H) == L

    def test_transfer_function_finite(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = blfilter("hann", 0.1, 0.3)
        H = comp_transferfunction(g, 256)
        assert np.all(np.isfinite(H))

    def test_transfer_function_not_all_zero(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = blfilter("hann", 0.1, 0.3)
        H = comp_transferfunction(g, 256)
        assert np.max(np.abs(H)) > 0


@pytest.mark.requires_impl
class TestBlFilterNormImpl:
    """
    MATLAB counterpart: TestBlFilter (energy normalisation section).
    """

    def test_energy_norm_default(self, needs_impl):
        """Default 'energy': (1/L)*sum|H|^2 == 1."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 512
        g = blfilter("hann", 0.1, 0.3)
        H = comp_transferfunction(g, L)
        assert np.sum(np.abs(H) ** 2) / L == pytest.approx(1.0, abs=1e-10)

    def test_energy_norm_explicit(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 256
        g = blfilter("hann", 0.1, 0.3, "energy")
        H = comp_transferfunction(g, L)
        assert np.sum(np.abs(H) ** 2) / L == pytest.approx(1.0, abs=1e-10)

    def test_peak_norm(self, needs_impl):
        """'peak': max|H| == 1."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 512
        g = blfilter("hann", 0.1, 0.3, "peak")
        H = comp_transferfunction(g, L)
        assert np.max(np.abs(H)) == pytest.approx(1.0, abs=1e-10)

    def test_scal_multiplies_response(self, needs_impl):
        """'scal' s: H_scal == s * H_default."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 256
        s  = 3.0
        g1 = blfilter("hann", 0.1, 0.3)
        g2 = blfilter("hann", 0.1, 0.3, "scal", s)
        H1 = comp_transferfunction(g1, L)
        H2 = comp_transferfunction(g2, L)
        np.testing.assert_allclose(H2, s * H1, atol=1e-10 * np.linalg.norm(H1))


@pytest.mark.requires_impl
class TestBlFilterCentreFreqImpl:
    """
    MATLAB counterpart: TestBlFilter (centre frequency section).
    """

    def test_peak_near_centre_freq(self, needs_impl):
        """Peak of |H| within 4 bins of fc=0.3."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 1024
        fc = 0.3
        g  = blfilter("hann", 0.1, fc, "peak")
        H  = comp_transferfunction(g, L)
        peak_bin     = int(np.argmax(np.abs(H)))
        expected_bin = round(fc / 2 * L)
        assert abs(peak_bin - expected_bin) < 4

    def test_dc_filter_peak_at_bin_0(self, needs_impl):
        """DC filter (fc=0): peak at bin 0 (0-indexed)."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 256
        g = blfilter("hann", 0.1, 0, "peak")
        H = comp_transferfunction(g, L)
        assert int(np.argmax(np.abs(H))) == 0

    def test_hz_input_consistency(self, needs_impl):
        """Hz and normalised inputs give the same filter."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        fs    = 8000
        fc_hz = 1000
        bw_hz = 400
        g_hz   = blfilter("hann", bw_hz,         fc_hz,          "fs", fs)
        g_norm = blfilter("hann", bw_hz / (fs/2), fc_hz / (fs/2))
        L = 512
        H_hz   = comp_transferfunction(g_hz,   L)
        H_norm = comp_transferfunction(g_norm, L)
        np.testing.assert_allclose(H_hz, H_norm, atol=1e-10)
