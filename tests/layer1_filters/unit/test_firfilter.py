"""
test_firfilter.py
=================
Python port of:
    layer1_filters/unit/TestFirFilter.m

Covers: firfilter (FIR filter constructor)

API
---
firfilter(name, M)           -> filter dict, DC-centred (fc=0)
firfilter(name, M, fc)       -> FIR filter at normalised fc
firfilter(name, M, fc, norm) -> with explicit normalisation flag

The returned object is expected to be dict-like with keys:
    h        : np.ndarray  impulse response
    offset   : int         = -floor(M/2) for non-causal, 0 for causal
    delay    : int         = 0 by default
    realonly : int         = 0 by default, 1 when 'real' flag given
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# firfilter – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFirfilterStructImpl:
    """
    MATLAB counterpart: TestFirFilter (struct format section).
    """

    def test_has_required_fields(self, needs_impl):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", 32)
        for field in ("h", "offset", "delay", "realonly"):
            assert field in g, f"Missing field: {field}"

    def test_h_is_numeric_vector(self, needs_impl):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", 32)
        h = np.asarray(g["h"])
        assert h.ndim == 1 and np.issubdtype(h.dtype, np.number)

    def test_default_realonly_is_zero(self, needs_impl):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", 32)
        assert g["realonly"] == 0

    def test_real_flag_sets_realonly(self, needs_impl):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", 32, 0, "real")
        assert g["realonly"] == 1


@pytest.mark.requires_impl
class TestFirfilterLengthImpl:
    """
    MATLAB counterpart: TestFirFilter (impulse response length section).
    """

    @pytest.mark.parametrize("M", [16, 32, 64, 128])
    def test_impulse_response_length(self, needs_impl, M):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", M)
        assert len(np.asarray(g["h"])) == M

    def test_sine_filter_length(self, needs_impl):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("sine", 48)
        assert len(np.asarray(g["h"])) == 48


@pytest.mark.requires_impl
class TestFirfilterNormImpl:
    """
    MATLAB counterpart: TestFirFilter (energy normalisation section).
    """

    @pytest.mark.parametrize("M", [16, 32, 64])
    def test_energy_norm_default(self, needs_impl, M):
        """Default 'energy': sum(h^2) == 1."""
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", M)
        h = np.asarray(g["h"], dtype=float)
        assert np.sum(h ** 2) == pytest.approx(1.0, abs=1e-10)

    def test_peak_norm(self, needs_impl):
        """'peak': max|h| == 1."""
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", 64, 0, "peak")
        h = np.asarray(g["h"])
        assert np.max(np.abs(h)) == pytest.approx(1.0, abs=1e-10)

    def test_scal_multiplies_impulse_response(self, needs_impl):
        """'scal' s: h_scaled == s * h_default."""
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        s  = 2.0
        g1 = firfilter("hann", 32)
        g2 = firfilter("hann", 32, 0, "scal", s)
        np.testing.assert_allclose(np.asarray(g2["h"]), s * np.asarray(g1["h"]),
                                   atol=1e-10 * np.linalg.norm(g1["h"]))


@pytest.mark.requires_impl
class TestFirfilterOffsetImpl:
    """
    MATLAB counterpart: TestFirFilter (offset / delay section).
    """

    def test_non_causal_offset_is_minus_floor_M_over_2(self, needs_impl):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        M = 32
        g = firfilter("hann", M)
        assert g["offset"] == -M // 2

    def test_causal_offset_is_zero(self, needs_impl):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", 32, 0, "causal")
        assert g["offset"] == 0

    def test_default_delay_is_zero(self, needs_impl):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", 32)
        assert g["delay"] == 0


@pytest.mark.requires_impl
class TestFirfilterTransferFunctionImpl:
    """
    MATLAB counterpart: TestFirFilter (comp_transferfunction section).
    """

    def test_impulse_response_is_real_at_dc(self, needs_impl):
        """DC-centred firfilter: h should be real."""
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", 32)
        h = np.asarray(g["h"])
        assert np.all(np.isreal(h)), "DC firfilter: impulse response should be real"

    def test_peak_near_centre_freq(self, needs_impl):
        """firfilter at fc=0.25: peak of |H| within 5 bins of expected."""
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 1024
        fc = 0.25
        g  = firfilter("hann", 64, fc, "peak")
        H  = comp_transferfunction(g, L)
        peak_bin     = int(np.argmax(np.abs(H)))
        expected_bin = round(fc / 2 * L)
        assert abs(peak_bin - expected_bin) < 5

    def test_filterbank_zero_input(self, needs_impl):
        """Zero input -> zero subband output."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        Ls = 512
        x  = np.zeros(Ls)
        g  = [firfilter("hann", 32)]
        a  = 4
        c  = filterbank(x, g, a)
        assert np.max(np.abs(c[0])) < 1e-12

    def test_filterbank_output_length(self, needs_impl):
        """filterbank subband has ceil(Ls/a) rows."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        Ls  = 512
        rng = np.random.default_rng(42)
        x   = rng.standard_normal(Ls)
        g   = [firfilter("hann", 32)]
        a   = 4
        c   = filterbank(x, g, a)
        assert c[0].shape[0] == int(np.ceil(Ls / a))
