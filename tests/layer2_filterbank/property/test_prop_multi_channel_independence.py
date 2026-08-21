"""
test_prop_multi_channel_independence.py
========================================
Python port of:
    layer2_filterbank/property/PropMultiChannelIndependence.m

When filterbank receives an L × W matrix input, it applies the filterbank
independently to each column:

(1) filterbank([x₁ x₂], g, a)[m][:, w] == filterbank(xw, g, a)[m]
(2) Output coefficient arrays have the correct W-column shape.
(3) Linear combinations across channels are handled correctly.
"""

from __future__ import annotations

import pytest

import numpy as np


@pytest.mark.requires_impl
class TestTwoColumnIndependenceImpl:
    """PropMultiChannelIndependence: 2-column input matches separate calls."""

    def test_two_column_independence(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(42)
        for trial in range(50):
            x1 = rng.standard_normal(L)
            x2 = rng.standard_normal(L)
            c12 = filterbank(np.column_stack([x1, x2]), g, a)
            c1  = filterbank(x1, g, a)
            c2  = filterbank(x2, g, a)
            M = len(g)
            for m in range(M):
                arr12 = np.asarray(c12[m])
                arr1  = np.asarray(c1[m]).ravel()
                arr2  = np.asarray(c2[m]).ravel()
                col1  = arr12[:, 0] if arr12.ndim > 1 else arr12
                col2  = arr12[:, 1] if arr12.ndim > 1 else arr12
                e1 = np.linalg.norm(col1 - arr1) / (np.linalg.norm(arr1) + 1e-15)
                e2 = np.linalg.norm(col2 - arr2) / (np.linalg.norm(arr2) + 1e-15)
                assert max(e1, e2) < 1e-12, \
                    f"Trial {trial}, band {m}: independence error {max(e1,e2):.2e}"


@pytest.mark.requires_impl
class TestOutputChannelDimensionImpl:
    """PropMultiChannelIndependence: output has W columns."""

    @pytest.mark.parametrize("W", [1, 2, 3, 5])
    def test_output_channel_dim(self, needs_impl, W):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((L, W))
        c = filterbank(x, g, a)
        M = len(g)
        for m in range(M):
            arr = np.asarray(c[m])
            ncols = 1 if arr.ndim == 1 else arr.shape[1]
            assert ncols == W, \
                f"W={W}, band {m}: expected {W} cols, got {ncols}"


@pytest.mark.requires_impl
class TestCrossChannelLinearityImpl:
    """PropMultiChannelIndependence: linear combinations across channels."""

    def test_cross_channel_linearity(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        alpha = 2.5
        beta  = -1.3j
        rng = np.random.default_rng(1)
        for trial in range(30):
            x1 = rng.standard_normal(L)
            x2 = rng.standard_normal(L) + 1j * rng.standard_normal(L)
            c1  = filterbank(x1, g, a)
            c2  = filterbank(x2, g, a)
            ccc = filterbank(alpha * x1 + beta * x2, g, a)
            M = len(g)
            for m in range(M):
                expected = alpha * np.asarray(c1[m]) + beta * np.asarray(c2[m])
                denom    = np.linalg.norm(c1[m]) + np.linalg.norm(c2[m]) + 1e-15
                err      = np.linalg.norm(np.asarray(ccc[m]) - expected) / denom
                assert err < 1e-10, \
                    f"Trial {trial}, band {m}: cross-channel linearity {err:.2e}"


@pytest.mark.requires_impl
class TestThreeColumnIndependenceImpl:
    """PropMultiChannelIndependence: 3-column independence."""

    def test_three_column_independence(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(3)
        x1 = rng.standard_normal(L)
        x2 = rng.standard_normal(L) + 1j * rng.standard_normal(L)
        x3 = rng.standard_normal(L)
        c123 = filterbank(np.column_stack([x1, x2, x3]), g, a)
        c1   = filterbank(x1, g, a)
        c2   = filterbank(x2, g, a)
        c3   = filterbank(x3, g, a)
        for m in range(len(g)):
            arr = np.asarray(c123[m])
            e1  = np.linalg.norm(arr[:, 0] - np.asarray(c1[m]).ravel()) / \
                  (np.linalg.norm(c1[m]) + 1e-15)
            e2  = np.linalg.norm(arr[:, 1] - np.asarray(c2[m]).ravel()) / \
                  (np.linalg.norm(c2[m]) + 1e-15)
            e3  = np.linalg.norm(arr[:, 2] - np.asarray(c3[m]).ravel()) / \
                  (np.linalg.norm(c3[m]) + 1e-15)
            assert max(e1, e2, e3) < 1e-12, \
                f"Band {m}: 3-col independence error {max(e1,e2,e3):.2e}"
