"""
tests/layer1_filters/unit/test_warpedfilters.py
================================================
Python unit tests for warpedfilters and warpedblfilter.

Mirrors TestWarpedFilters.m, covering:
  1. Return-value structure
  2. Warping functions (linear, sqrt, ERB, constant-Q)
  3. Sampling modes
  4. Frame bounds at various redundancies
  5. warpedblfilter individual filter construction
  6. comp_warpedfreqresponse and comp_warpedfoff
"""
from __future__ import annotations

import pytest

import numpy as np
from cool_frames.numpy.filterbanks import filterbankresponse
from cool_frames.numpy.filters import warpedfilters
from cool_frames.numpy.filters._audscale import audtofreq, freqtoaud
from cool_frames.numpy.filters._warpedfilters import (
    comp_warpedfoff,
    comp_warpedfreqresponse,
    warpedblfilter,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FS = 16000
LS = 8000
FMIN = 100
FMAX = FS / 2

# ERB scale functions
freqtoscale_erb = lambda f: freqtoaud(f, "erb")
scaletofreq_erb = lambda s: audtofreq(s, "erb")


# ---------------------------------------------------------------------------
# 1. Return-value structure
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_returns_four_outputs(self):
        g, a, fc, L, _info = warpedfilters(
            freqtoscale_erb, scaletofreq_erb, FS, FMIN, FMAX, 1, LS
        )
        assert len(g) > 0
        assert a is not None
        assert len(fc) == len(g)
        assert L >= LS

    def test_filter_descriptor_has_H_and_foff(self):
        g, _, _, _, _info = warpedfilters(
            freqtoscale_erb, scaletofreq_erb, FS, FMIN, FMAX, 1, LS
        )
        for gi in g:
            assert "H" in gi
            assert "foff" in gi

    def test_filters_can_be_evaluated(self):
        g, _, _, L, _info = warpedfilters(
            freqtoscale_erb, scaletofreq_erb, FS, FMIN, FMAX, 1, LS
        )
        for i in [0, len(g) // 2, len(g) - 1]:
            H_val = g[i]["H"](L)
            assert len(H_val) > 0
            assert np.max(np.abs(H_val)) > 0


# ---------------------------------------------------------------------------
# 2. Warping functions
# ---------------------------------------------------------------------------

class TestWarpingFunctions:

    def test_erb_scale(self):
        g, _, _, _, _info = warpedfilters(
            freqtoscale_erb, scaletofreq_erb, FS, FMIN, FMAX, 1, LS
        )
        assert len(g) > 5

    def test_log_scale(self):
        warp = lambda x: 10 * np.log(np.maximum(x, 1e-10))
        inv = lambda x: np.exp(x / 10)
        g, _, _, _, _info = warpedfilters(warp, inv, FS, 50, FMAX, 4, LS)
        assert len(g) > 5

    def test_linear_scale(self):
        warp = lambda x: x
        inv = lambda x: x
        g, _, _, _, _info = warpedfilters(warp, inv, FS, FMIN, FMAX, 1, LS)
        assert len(g) > 0

    def test_sqrt_scale(self):
        warp = lambda x: np.sqrt(np.maximum(x, 0))
        inv = lambda x: x**2
        g, _, _, _, _info = warpedfilters(warp, inv, FS, FMIN, FMAX, 1, LS)
        assert len(g) > 0


# ---------------------------------------------------------------------------
# 3. Sampling modes
# ---------------------------------------------------------------------------

class TestSamplingModes:

    def test_regsampling(self):
        g, a, _, L, _info = warpedfilters(
            freqtoscale_erb, scaletofreq_erb, FS, FMIN, FMAX, 1, LS,
            sampling="regsampling"
        )
        assert L >= LS

    def test_fractional(self):
        g, a, _, L, _info = warpedfilters(
            freqtoscale_erb, scaletofreq_erb, FS, FMIN, FMAX, 1, LS,
            sampling="fractional"
        )
        assert L == LS
        assert np.asarray(a).shape[1] == 2

    def test_uniform(self):
        g, a, _, L, _info = warpedfilters(
            freqtoscale_erb, scaletofreq_erb, FS, FMIN, FMAX, 1, LS,
            sampling="uniform"
        )
        a_arr = np.asarray(a).ravel()
        assert np.all(a_arr == a_arr[0])


# ---------------------------------------------------------------------------
# 4. Frame bounds
# ---------------------------------------------------------------------------

class TestFrameBounds:

    @pytest.mark.parametrize("redmul", [1.0, 1.5, 2.0, 4.0])
    def test_positive_frame_response(self, redmul):
        g, a, _, L, _info = warpedfilters(
            freqtoscale_erb, scaletofreq_erb, FS, FMIN, FMAX, 1, LS,
            bwmul=1.5, redmul=redmul, sampling="fractional"
        )
        resp = filterbankresponse(g, a, L, real=True)
        assert np.min(resp) > 0, \
            f"Frame response has zeros at redmul={redmul}"


# ---------------------------------------------------------------------------
# 5. warpedblfilter
# ---------------------------------------------------------------------------

class TestWarpedBlFilter:

    def test_basic_construction(self):
        g = warpedblfilter(
            "hann", 2.0, 1000.0,
            fs=FS,
            freqtoscale=freqtoscale_erb,
            scaletofreq=scaletofreq_erb,
            norm="inf",
            scal=1.0,
        )
        assert "H" in g
        assert "foff" in g
        assert callable(g["H"])
        assert callable(g["foff"])

    def test_filter_has_nonzero_response(self):
        g = warpedblfilter(
            "hann", 2.0, 1000.0,
            fs=FS,
            freqtoscale=freqtoscale_erb,
            scaletofreq=scaletofreq_erb,
            norm="inf",
            scal=1.0,
        )
        H = g["H"](LS)
        assert np.max(np.abs(H)) > 0


# ---------------------------------------------------------------------------
# 6. comp_warpedfreqresponse and comp_warpedfoff
# ---------------------------------------------------------------------------

class TestWarpedHelpers:

    def test_comp_warpedfoff_basic(self):
        foff = comp_warpedfoff(
            1000.0, 2.0, FS, LS,
            freqtoscale_erb, scaletofreq_erb
        )
        assert isinstance(foff, int)
        assert 0 <= foff < LS

    def test_comp_warpedfreqresponse_basic(self):
        H = comp_warpedfreqresponse(
            "hann", 1000.0, 2.0, FS, LS,
            freqtoscale_erb, scaletofreq_erb
        )
        assert len(H) > 0
        assert np.max(np.abs(H)) > 0
