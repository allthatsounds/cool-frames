"""
test_prop_analysis_linearity.py
================================
Python port of:
    layer2_filterbank/property/PropAnalysisLinearity.m

Property: filterbank(α·x + β·y, g, a) ≡ α·filterbank(x,g,a) + β·filterbank(y,g,a)

The filterbank analysis T is a linear map. Tested with random real and
complex inputs and random complex scalars α, β.
"""

from __future__ import annotations

import pytest

import numpy as np


@pytest.mark.requires_impl
class TestAnalysisLinearityRealImpl:
    """PropAnalysisLinearity: linearity with real inputs."""

    def test_linearity_real_inputs(self, needs_impl):
        """50 trials: α·x + β·y with complex scalars and real signals."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        rng = np.random.default_rng(42)
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        M = len(g)
        for trial in range(50):
            x     = rng.standard_normal(Ls)
            y     = rng.standard_normal(Ls)
            alpha = rng.standard_normal() + 1j * rng.standard_normal()
            beta  = rng.standard_normal() + 1j * rng.standard_normal()
            cx    = filterbank(x, g, a)
            cy    = filterbank(y, g, a)
            csum  = filterbank(alpha * x + beta * y, g, a)
            for m in range(M):
                expected = alpha * np.asarray(cx[m]) + beta * np.asarray(cy[m])
                denom    = np.linalg.norm(cx[m]) + np.linalg.norm(cy[m]) + 1e-15
                err      = np.linalg.norm(np.asarray(csum[m]) - expected) / denom
                assert err < 1e-10, \
                    f"Trial {trial}, band {m}: linearity error {err:.2e}"


@pytest.mark.requires_impl
class TestAnalysisLinearityComplexImpl:
    """PropAnalysisLinearity: linearity with complex inputs."""

    def test_linearity_complex_inputs(self, needs_impl):
        """50 trials: complex signals and complex scalars."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        rng = np.random.default_rng(43)
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        M = len(g)
        for trial in range(50):
            x     = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            y     = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            alpha = rng.standard_normal() + 1j * rng.standard_normal()
            beta  = rng.standard_normal() + 1j * rng.standard_normal()
            cx    = filterbank(x, g, a)
            cy    = filterbank(y, g, a)
            csum  = filterbank(alpha * x + beta * y, g, a)
            for m in range(M):
                expected = alpha * np.asarray(cx[m]) + beta * np.asarray(cy[m])
                denom    = np.linalg.norm(cx[m]) + np.linalg.norm(cy[m]) + 1e-15
                err      = np.linalg.norm(np.asarray(csum[m]) - expected) / denom
                assert err < 1e-10, \
                    f"Trial {trial}, band {m}: complex linearity error {err:.2e}"


@pytest.mark.requires_impl
class TestAnalysisLinearityScalarImpl:
    """PropAnalysisLinearity: scalar scaling and zero-input."""

    def test_scalar_multiple(self, needs_impl):
        """Scaling input by s must scale every subband by s."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        rng = np.random.default_rng(44)
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        M = len(g)
        for s in [2+0j, -1+0j, 1j, -3+2j, 0.5-0.5j]:
            x   = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            cx  = filterbank(x, g, a)
            csx = filterbank(s * x, g, a)
            for m in range(M):
                err = np.linalg.norm(np.asarray(csx[m]) - s * np.asarray(cx[m])) / \
                      (np.linalg.norm(cx[m]) + 1e-15)
                assert err < 1e-12, \
                    f"Scale s={s}, band {m}: error {err:.2e}"

    def test_zero_input_gives_zero(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        x0 = np.zeros(Ls)
        c0 = filterbank(x0, g, a)
        for m, cm in enumerate(c0):
            assert np.linalg.norm(np.asarray(cm)) < 1e-12, \
                f"filterbank(0) band {m}: not zero"
