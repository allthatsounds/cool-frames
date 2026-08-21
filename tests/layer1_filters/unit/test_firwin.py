"""
test_firwin.py
==============
Python port of:
    layer1_filters/unit/TestFirwin.m

Covers: firwin, freqwin, freqwavelet

LTFAT frequency convention
---------------------------
Windows use a whole-point even (WPE) layout: peak at index 0, so that
w[k] = w[-k mod M] (i.e., w[1:] == w[1:][::-1]).

firwin 'hann' partition of unity: w + np.roll(fftshift(w))  — the Python
equivalent of MATLAB's fftshift is np.fft.fftshift.
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import firwin_ref

# ---------------------------------------------------------------------------
# firwin – reference tests (no impl required)
# ---------------------------------------------------------------------------

class TestFirwinReference:
    """
    Mathematical properties verified using firwin_ref.
    MATLAB counterpart: TestFirwin.
    """

    # ── output length ────────────────────────────────────────────────────────

    @pytest.mark.parametrize("M", [16, 32, 64, 128])
    def test_hann_output_length(self, M):
        assert len(firwin_ref("hann", M)) == M

    @pytest.mark.parametrize("M", [15, 33, 65])
    def test_hann_output_length_odd(self, M):
        assert len(firwin_ref("hann", M)) == M

    def test_sine_length(self):
        assert len(firwin_ref("sine", 64)) == 64

    def test_rect_length(self):
        assert len(firwin_ref("rect", 32)) == 32

    # ── WPE symmetry: w[1:] == w[1:][::-1] ───────────────────────────────────

    @pytest.mark.parametrize("name", ["hann", "sine", "rect", "tria"])
    @pytest.mark.parametrize("M", [18, 19, 20, 21, 64])
    def test_wpe_symmetry(self, name, M):
        w = firwin_ref(name, M)
        np.testing.assert_allclose(w[1:], w[1:][::-1], atol=1e-14,
                                   err_msg=f"{name} M={M}: not WPE symmetric")

    # ── non-negativity ────────────────────────────────────────────────────────

    def test_hann_non_negative(self):
        w = firwin_ref("hann", 64)
        assert np.min(w) >= -1e-14

    def test_sine_non_negative(self):
        w = firwin_ref("sine", 64)
        assert np.min(w) >= -1e-14

    def test_rect_odd_is_all_ones(self):
        """firwin('rect', odd M): all elements are 1."""
        w = firwin_ref("rect", 33)
        np.testing.assert_allclose(w, np.ones(33), atol=1e-14)

    # ── partition of unity: w + fftshift(w) = 1 ──────────────────────────────

    @pytest.mark.parametrize("M", [32, 64, 128, 256])
    def test_hann_pu(self, M):
        w   = firwin_ref("hann", M)
        pu  = w + np.fft.fftshift(w)
        np.testing.assert_allclose(pu, np.ones(M), atol=1e-14,
                                   err_msg=f"hann PU failed for M={M}")

    @pytest.mark.parametrize("M", [32, 64, 128, 256])
    def test_tria_pu(self, M):
        w   = firwin_ref("tria", M)
        pu  = w + np.fft.fftshift(w)
        np.testing.assert_allclose(pu, np.ones(M), atol=1e-14,
                                   err_msg=f"tria PU failed for M={M}")

    # ── sine tight frame: w^2 + fftshift(w^2) = 1 ────────────────────────────

    @pytest.mark.parametrize("M", [32, 64, 128])
    def test_sine_tight_frame(self, M):
        w   = firwin_ref("sine", M)
        tf  = w ** 2 + np.fft.fftshift(w ** 2)
        np.testing.assert_allclose(tf, np.ones(M), atol=1e-14,
                                   err_msg=f"sine tight frame failed for M={M}")

    def test_sine_is_sqrt_hann(self):
        """sine^2 == hann point-wise."""
        M    = 64
        sine = firwin_ref("sine", M)
        hann = firwin_ref("hann", M)
        np.testing.assert_allclose(sine ** 2, hann, atol=1e-14)

    # ── peak at index 0 ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("name", ["hann", "sine", "rect", "tria"])
    def test_peak_at_dc(self, name):
        w = firwin_ref(name, 64)
        assert np.argmax(w) == 0, f"{name}: peak should be at index 0"

    @pytest.mark.parametrize("name", ["hann", "sine", "rect", "tria"])
    def test_peak_value_one(self, name):
        """All windows should peak at 1 (WPE convention)."""
        w = firwin_ref(name, 64)
        np.testing.assert_allclose(w[0], 1.0, atol=1e-14,
                                   err_msg=f"{name}: peak should be 1")


# ---------------------------------------------------------------------------
# firwin – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFirwinImpl:
    """Verify cool_frames.layer1.firwin matches reference."""

    @pytest.mark.parametrize("name,M", [
        ("hann", 32), ("hann", 64), ("sine", 64), ("rect", 33), ("tria", 64)
    ])
    def test_matches_reference(self, needs_impl, name, M):
        from cool_frames.filters import firwin  # type: ignore

        np.testing.assert_allclose(firwin(name, M), firwin_ref(name, M), atol=1e-12,
                                   err_msg=f"{name}, M={M}")

    # -- Broader parametric tests (ported from ttest_firwin.m) --

    @pytest.mark.parametrize("name", [
        "hann", "hamming", "blackman", "rect", "tria", "nuttall",
    ])
    @pytest.mark.parametrize("M", [18, 19, 20, 21])
    def test_wpe_symmetry_all_windows(self, needs_impl, name, M):
        """WPE symmetry: w[1:] == w[1:][::-1] for all windows and lengths."""
        from cool_frames.filters import firwin  # type: ignore
        w = firwin(name, M, norm="inf")
        np.testing.assert_allclose(w[1:], w[1:][::-1], atol=1e-14,
                                   err_msg=f"{name} M={M}: not WPE symmetric")

    @pytest.mark.parametrize("name", [
        "hann", "hamming", "blackman", "rect", "tria", "nuttall",
    ])
    def test_peak_at_bin1_all_windows(self, needs_impl, name):
        """Peak should be at index 0 (DFT ordering) for all windows."""
        from cool_frames.filters import firwin  # type: ignore
        w = firwin(name, 64, norm="inf")
        np.testing.assert_allclose(w[0], 1.0, atol=1e-10,
                                   err_msg=f"{name}: peak should be 1")


# ---------------------------------------------------------------------------
# freqwin – implementation tests only
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFreqwinImpl:
    """
    MATLAB counterpart: TestFirwin (freqwin section).
    """

    def test_gauss_output_length(self, needs_impl):
        from cool_frames.filters import freqwin  # type: ignore
        assert len(freqwin("gauss", 128, 0.1)) == 128

    def test_gauss_peak_normalized(self, needs_impl):
        from cool_frames.filters import freqwin  # type: ignore
        w = freqwin("gauss", 256, 0.05)
        assert max(abs(w)) == pytest.approx(1.0, abs=1e-12)

    def test_gammatone_length(self, needs_impl):
        from cool_frames.filters import freqwin  # type: ignore
        assert len(freqwin("gammatone", 128, 0.1)) == 128

    def test_butterworth_length(self, needs_impl):
        from cool_frames.filters import freqwin  # type: ignore
        assert len(freqwin("butterworth", 128, 0.1)) == 128

    def test_non_negative(self, needs_impl):
        from cool_frames.filters import freqwin  # type: ignore
        w = freqwin("gauss", 128, 0.1)
        assert np.min(w) >= -1e-12


# ---------------------------------------------------------------------------
# freqwavelet – implementation tests only
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFreqwaveletImpl:
    """
    MATLAB counterpart: TestFirwin (freqwavelet section).
    """

    def test_cauchy_output_length(self, needs_impl):
        from cool_frames.filters import freqwavelet  # type: ignore
        H, _ = freqwavelet("cauchy", 256)
        assert len(H) == 256

    def test_cauchy_peak_normalized(self, needs_impl):
        from cool_frames.filters import freqwavelet  # type: ignore
        H, _ = freqwavelet("cauchy", 256, "peak")
        assert max(abs(H)) == pytest.approx(1.0, abs=1e-12)

    def test_morlet_length(self, needs_impl):
        from cool_frames.filters import freqwavelet  # type: ignore
        H, _ = freqwavelet("morlet", 256)
        assert len(H) == 256

    def test_morse_length(self, needs_impl):
        from cool_frames.filters import freqwavelet  # type: ignore
        H, _ = freqwavelet("morse", 256)
        assert len(H) == 256

    @pytest.mark.parametrize("name", ["cauchy", "morlet", "morse"])
    def test_finite_values(self, needs_impl, name):
        from cool_frames.filters import freqwavelet  # type: ignore
        H, _ = freqwavelet(name, 256)
        assert np.all(np.isfinite(H)), f"freqwavelet {name}: non-finite values"
