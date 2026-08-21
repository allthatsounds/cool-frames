"""
test_prop_frame_bounds.py
==========================
Python port of:
    layer2_filterbank/property/PropFrameBounds.m

Property: A||x||² <= Σ_m (1/a_m)||c_m||² <= B||x||²

Tested with uniform (ERB) and non-uniform (CQT) subsampling across
100 random complex signals each.
"""

from __future__ import annotations

import pytest

import numpy as np


@pytest.mark.requires_impl
class TestFrameBoundsUniformImpl:
    """PropFrameBounds: ERB filterbank (uniform subsampling)."""

    def test_uniform_subsampling(self, needs_impl):
        from cool_frames.filterbanks import filterbank, filterbankbounds  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        # Complex test signals -> use the two-sided (complex) frame bounds.
        # The default real=True bounds fold the response (correct real-signal
        # bounds; their kappa matches the SVD ground truth) and would not
        # bracket complex-signal energy ratios.
        A, B   = filterbankbounds(g, a, L, real=False)
        a_arr  = np.asarray(a).ravel()
        rng    = np.random.default_rng(0)
        ratios = []
        for trial in range(100):
            x    = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            c    = filterbank(x, g, a)
            eTx  = sum(
                np.linalg.norm(np.asarray(cm).ravel()) ** 2 / float(am)
                for cm, am in zip(c, a_arr)
            )
            ex   = np.linalg.norm(x) ** 2
            ratio = eTx / ex
            ratios.append(ratio)
            assert ratio >= A - 1e-6, \
                f"Trial {trial}: ratio {ratio:.6f} < A={A:.6f}"
            assert ratio <= B + 1e-6, \
                f"Trial {trial}: ratio {ratio:.6f} > B={B:.6f}"
        assert min(ratios) >= A - 0.1
        assert max(ratios) <= B + 0.1


@pytest.mark.requires_impl
class TestFrameBoundsNonuniformImpl:
    """PropFrameBounds: CQT filterbank (non-uniform subsampling)."""

    def test_nonuniform_subsampling(self, needs_impl):
        try:
            from cool_frames.filterbanks import filterbank, filterbankbounds  # type: ignore
            from cool_frames.filters import (
                cqtfilters,  # type: ignore
                filterbanklength,  # type: ignore
            )
        except ImportError:
            pytest.skip("cqtfilters not available")
        Ls, fs = 1024, 8000
        try:
            g, a, fc, _L, _info = cqtfilters(fs, Ls, fmin=50, fmax=fs / 2, bins=12)
        except Exception:
            pytest.skip("cqtfilters raised an exception")
        L = filterbanklength(Ls, a)
        # Complex test signals -> two-sided (complex) frame bounds (see above).
        A, B   = filterbankbounds(g, a, L, real=False)
        a_arr  = np.asarray(a).ravel()
        rng    = np.random.default_rng(1)
        for trial in range(100):
            x   = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            c   = filterbank(x, g, a)
            eTx = sum(
                np.linalg.norm(np.asarray(cm).ravel()) ** 2 / float(am)
                for cm, am in zip(c, a_arr)
            )
            ex  = np.linalg.norm(x) ** 2
            ratio = eTx / ex
            assert ratio >= A - 1e-6, \
                f"CQT trial {trial}: ratio {ratio:.6f} < A={A:.6f}"
            assert ratio <= B + 1e-6, \
                f"CQT trial {trial}: ratio {ratio:.6f} > B={B:.6f}"
