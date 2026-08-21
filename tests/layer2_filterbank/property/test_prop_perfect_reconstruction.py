"""
test_prop_perfect_reconstruction.py
=====================================
Python port of:
    layer2_filterbank/property/PropPerfectReconstruction.m

Property: synthesis(analysis(x)) = x for dual and tight frame filterbanks.

NOTE: ifilterbank returns a signal of length L >= Ls. Reconstruction is
compared on the first Ls samples: xr[:Ls].

NOTE: The ERB filterbank is one-sided; use real signals to avoid
negative-frequency aliasing.
"""

from __future__ import annotations

import pytest

import numpy as np


def _make_signals(Ls: int, fs: int = 8000, seed: int = 42):
    rng = np.random.default_rng(seed)
    t   = np.arange(Ls) / fs
    signals = {
        "noise":   rng.standard_normal(Ls),
        "chirp":   np.sin(2 * np.pi * (100 + 1950 * t / t[-1]) * t),
        "impulse": np.zeros(Ls),
    }
    signals["impulse"][0] = 1.0
    return signals


@pytest.mark.requires_impl
class TestDualFramePRImpl:
    """PropPerfectReconstruction: dual-frame reconstruction, 100 real trials."""

    def test_dual_pr_100_trials(self, needs_impl):
        from cool_frames.filterbanks import filterbank, filterbankdual, ifilterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gd = filterbankdual(g, a, L)
        rng = np.random.default_rng(42)
        for trial in range(100):
            x  = rng.standard_normal(Ls)
            c  = filterbank(x, g, a)
            xr = np.asarray(ifilterbank(c, gd, a, real=True))
            rel_err = np.linalg.norm(x - xr[:Ls]) / np.linalg.norm(x)
            assert rel_err < 1e-10, \
                f"Trial {trial}: dual PR error {rel_err:.2e}"


@pytest.mark.requires_impl
class TestTightFramePRImpl:
    """PropPerfectReconstruction: tight-frame reconstruction, 100 real trials."""

    def test_tight_pr_100_trials(self, needs_impl):
        from cool_frames.filterbanks import filterbank, filterbankbounds, filterbanktight, ifilterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gt = filterbanktight(g, a, L)
        _, B = filterbankbounds(gt, a, L)
        rng = np.random.default_rng(43)
        for trial in range(100):
            x  = rng.standard_normal(Ls)
            c  = filterbank(x, gt, a)
            xr = np.asarray(ifilterbank(c, gt, a, real=True))
            rel_err = np.linalg.norm(x - xr[:Ls] / B) / np.linalg.norm(x)
            assert rel_err < 1e-10, \
                f"Trial {trial}: tight PR error {rel_err:.2e}"


@pytest.mark.requires_impl
class TestPRNamedSignalsImpl:
    """PropPerfectReconstruction: dual-frame PR with named signals."""

    @pytest.mark.parametrize("sig_name", ["noise", "chirp", "impulse"])
    def test_named_signal(self, needs_impl, sig_name):
        from cool_frames.filterbanks import filterbank, filterbankdual, ifilterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gd = filterbankdual(g, a, L)
        sigs = _make_signals(Ls, fs)
        x    = sigs[sig_name]
        c    = filterbank(x, g, a)
        xr   = np.asarray(ifilterbank(c, gd, a, real=True))
        rel_err = np.linalg.norm(x - xr[:Ls]) / (np.linalg.norm(x) + 1e-15)
        assert rel_err < 1e-10, \
            f"Signal '{sig_name}': dual PR error {rel_err:.2e}"
