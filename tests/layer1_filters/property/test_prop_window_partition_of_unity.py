"""
test_prop_window_partition_of_unity.py
=======================================
Python port of:
    layer1_filters/property/PropWindowPartitionOfUnity.m

Partition-of-unity and tight-frame properties of firwin.

PU:            w + fftshift(w) = ones(M,)     → hann, tria (for even M)
Tight frame:   w^2 + fftshift(w^2) = ones(M,) → sine
rect:          binary boxcar, 0 at Nyquist bin (for even M)
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import firwin_ref

# ---------------------------------------------------------------------------
# Reference tests (unconditional)
# ---------------------------------------------------------------------------

class TestWindowPartitionOfUnityReference:
    """
    MATLAB counterpart: PropWindowPartitionOfUnity.
    All tests use firwin_ref — no impl required.
    """

    # ── hann PU ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("M", [32, 64, 128, 256])
    def test_hann_pu_multiple_lengths(self, M):
        w        = firwin_ref("hann", M)
        residual = np.linalg.norm(w + np.fft.fftshift(w) - np.ones(M))
        assert residual < 1e-13, f"hann PU failed for M={M}: residual={residual:.2e}"

    def test_hann_pu_elementwise(self):
        M  = 64
        w  = firwin_ref("hann", M)
        pu = w + np.fft.fftshift(w)
        np.testing.assert_allclose(pu, np.ones(M), atol=1e-14)

    # ── tria PU ─────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("M", [32, 64, 128, 256])
    def test_tria_pu(self, M):
        w        = firwin_ref("tria", M)
        residual = np.linalg.norm(w + np.fft.fftshift(w) - np.ones(M))
        assert residual < 1e-13, f"tria PU failed for M={M}"

    # ── rect structure ───────────────────────────────────────────────────────

    @pytest.mark.parametrize("M", [32, 64, 128, 256])
    def test_rect_binary_boxcar_even_M(self, M):
        """rect for even M: all 1 except Nyquist bin (index M/2) which is 0."""
        w        = firwin_ref("rect", M)
        expected = np.ones(M)
        expected[M // 2] = 0.0
        np.testing.assert_allclose(w, expected, atol=1e-14,
                                   err_msg=f"rect boxcar structure failed for M={M}")

    # ── sine tight frame ─────────────────────────────────────────────────────

    @pytest.mark.parametrize("M", [32, 64, 128, 256])
    def test_sine_tight_frame_multiple_lengths(self, M):
        w        = firwin_ref("sine", M)
        residual = np.linalg.norm(w ** 2 + np.fft.fftshift(w ** 2) - np.ones(M))
        assert residual < 1e-13, f"sine TF failed for M={M}: residual={residual:.2e}"

    def test_sine_is_sqrt_hann(self):
        M    = 64
        sine = firwin_ref("sine", M)
        hann = firwin_ref("hann", M)
        np.testing.assert_allclose(sine ** 2, hann, atol=1e-14)

    # ── symmetry is prerequisite of PU ───────────────────────────────────────

    @pytest.mark.parametrize("name", ["hann", "rect", "tria", "sine"])
    def test_all_pu_windows_wpe_symmetric(self, name):
        M = 64
        w = firwin_ref(name, M)
        np.testing.assert_allclose(w[1:], w[1:][::-1], atol=1e-13,
                                   err_msg=f"{name}: not WPE symmetric")

    # ── random even lengths stress test ──────────────────────────────────────

    def test_hann_pu_random_even_lengths(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            M        = 2 * int(rng.integers(8, 257))
            w        = firwin_ref("hann", M)
            residual = np.linalg.norm(w + np.fft.fftshift(w) - np.ones(M))
            assert residual < 1e-13, f"hann PU failed for random M={M}"

    def test_sine_tight_frame_random_even_lengths(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            M        = 2 * int(rng.integers(8, 257))
            w        = firwin_ref("sine", M)
            residual = np.linalg.norm(w ** 2 + np.fft.fftshift(w ** 2) - np.ones(M))
            assert residual < 1e-13, f"sine TF failed for random M={M}"


# ---------------------------------------------------------------------------
# Implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestWindowPartitionOfUnityImpl:
    """Verify cool_frames.layer1.firwin PU/TF properties."""

    @pytest.mark.parametrize("M", [32, 64, 128, 256])
    def test_hann_pu(self, needs_impl, M):
        from cool_frames.filters import firwin  # type: ignore
        w = np.asarray(firwin("hann", M), dtype=float)
        np.testing.assert_allclose(w + np.fft.fftshift(w), np.ones(M), atol=1e-13)

    @pytest.mark.parametrize("M", [32, 64, 128, 256])
    def test_sine_tight_frame(self, needs_impl, M):
        from cool_frames.filters import firwin  # type: ignore
        w = np.asarray(firwin("sine", M), dtype=float)
        np.testing.assert_allclose(w ** 2 + np.fft.fftshift(w ** 2), np.ones(M), atol=1e-13)

    def test_itersine_tight_frame(self, needs_impl):
        """itersine satisfies its own tight-frame condition and differs from sine."""
        from cool_frames.filters import firwin  # type: ignore
        M         = 64
        w_iter    = np.asarray(firwin("itersine", M), dtype=float)
        w_sine    = np.asarray(firwin("sine",     M), dtype=float)
        # itersine tight frame
        residual  = np.linalg.norm(w_iter ** 2 + np.fft.fftshift(w_iter ** 2) - np.ones(M))
        assert residual < 1e-13
        # itersine != sine
        assert np.linalg.norm(w_iter - w_sine) > 1e-3
