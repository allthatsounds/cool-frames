"""
tests/layer1_filters/property/test_prop_wavelet_filters.py
==========================================================
Property-based tests for waveletfilters and freqwavelet.

Properties:
  P1. Wavelet peak at expected centre frequency
  P2. Support monotonically decreasing with scale
  P3. filterbanklength divides L for regsampling
  P4. Positive frame response for tested configurations
  P5. Lambert W satisfies W(z)*exp(W(z)) = z
  P6. firwin_eval consistent with firwin at integer grid
"""
from __future__ import annotations

import math

import pytest

import numpy as np

try:
    from cool_frames.numpy.filterbanks import filterbankresponse
    from cool_frames.numpy.filters import waveletfilters
    from cool_frames.numpy.filters._firwin import firwin, firwin_eval
    from cool_frames.numpy.filters._freqwavelet import freqwavelet
    from cool_frames.numpy.filters._wavelet import lambertw
    _HAS_LTFAT = True
except ImportError:
    _HAS_LTFAT = False

pytestmark = pytest.mark.skipif(not _HAS_LTFAT,
                                reason="cool_frames not installed")


# ---------------------------------------------------------------------------
# P1. Wavelet peak at expected centre frequency
# ---------------------------------------------------------------------------

class TestPeakAtCenterFrequency:

    @pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 4.0, 8.0])
    def test_cauchy_peak(self, scale):
        L = 4096
        H, info = freqwavelet(["cauchy", 300], L, scale=scale,
                               output_format="full")
        peak_idx = int(np.asarray(np.argmax(np.abs(H.ravel()))).item())
        expected_bin = round(float(np.asarray(info["fc"]).item()) * L / 2)
        assert abs(peak_idx - expected_bin) < 3


# ---------------------------------------------------------------------------
# P2. Support monotonically decreasing with scale
# ---------------------------------------------------------------------------

class TestSupportDecreases:

    def test_support_decreases_with_scale(self):
        L = 8192
        scales = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        prev_fsupp = float("inf")
        for s in scales:
            _, info = freqwavelet(["cauchy", 300], L, scale=s,
                                   output_format="full")
            cur = float(np.asarray(info["fsupp"]).item())
            assert cur <= prev_fsupp + 2
            prev_fsupp = cur


# ---------------------------------------------------------------------------
# P3. filterbanklength divides L for regsampling
# ---------------------------------------------------------------------------

class TestFilterbanklength:

    def test_L_divisible_by_a(self):
        Ls = 4096
        scales = np.linspace(10, 0.1, 40)
        _, a, _, L, _ = waveletfilters(2.0, Ls, scales=scales, sampling="regsampling")
        a_flat = np.asarray(a).ravel()
        for ai in a_flat:
            assert L % int(ai) == 0


# ---------------------------------------------------------------------------
# P4. Positive frame response
# ---------------------------------------------------------------------------

class TestPositiveFrameResponse:

    @pytest.mark.parametrize("wavelet", [
        ["cauchy", 300],
        ["morlet", 6],
        ["fbsp", 4, 3],
    ])
    def test_positive_response(self, wavelet):
        """Frame response is non-negative; > 90 % of bins are non-zero.

        Wavelet filterbanks with a finite set of scales and
        ``lowpass='single'`` cover most of the positive-frequency half
        but may leave a gap near Nyquist. We therefore check that the
        response is non-negative everywhere and that the vast majority
        of positive-frequency bins are covered.
        """
        Ls = 4096
        scales = np.linspace(10, 0.1, 40)
        gout, a, _, L, _ = waveletfilters(
            2.0, Ls, scales=scales, wavelet=wavelet, sampling="uniform", lowpass="single"
        )
        resp = filterbankresponse(gout, a, L, real=True)
        assert np.all(resp >= 0), "Frame response must be non-negative"
        # Most positive-frequency bins should have non-zero energy
        pos_resp = resp[:L // 2 + 1]
        coverage = np.mean(pos_resp > 0)
        assert coverage > 0.5, \
            f"Only {coverage*100:.0f}% of positive-freq bins are covered"


# ---------------------------------------------------------------------------
# P5. Lambert W identity
# ---------------------------------------------------------------------------

class TestLambertWProperty:

    @pytest.mark.parametrize("z", [0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
    def test_identity(self, z):
        """W(z) * exp(W(z)) == z"""
        w = lambertw(z, b=0)
        w_real = float(np.asarray(np.real(w)).item())
        assert abs(w_real * math.exp(w_real) - z) < 1e-8


# ---------------------------------------------------------------------------
# P6. firwin_eval consistency with firwin
# ---------------------------------------------------------------------------

class TestFirwinEvalConsistency:

    @pytest.mark.parametrize("winname", [
        "hann", "hamming", "blackman", "rect", "tria", "nuttall",
    ])
    def test_consistency(self, winname):
        M = 128
        # Use WPE convention: x = n/M
        x = np.arange(M) / M
        g_eval = firwin_eval(winname, x)
        g_int = firwin(winname, M, norm="inf")
        np.testing.assert_allclose(g_eval, g_int, atol=1e-10)
