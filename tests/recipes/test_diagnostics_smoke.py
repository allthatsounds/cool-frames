"""Smoke + sanity tests for cool_frames.diagnostics (recommend, spectrogram) and the
frame-quality numbers A/B/kappa from filterbankbounds.

`recommend_filterbank` is covered in test_recommend_filterbank.py. The former
`diagnose`/`print_diagnostic` wrappers were removed (2026-06-14) as redundant:
frame quality now comes from filterbankbounds(..., return_kappa=True).
"""
from __future__ import annotations

import numpy as np
import pytest


def _bank(fs=8000, Ls=4000):
    from cool_frames.numpy.filters import audfilters  # type: ignore
    g, a, fc, L, _info = audfilters(fs, Ls)
    return g, a, fc, L, fs


class TestFrameBoundsKappa:
    def test_filterbankbounds_return_kappa(self):
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        g, a, fc, L, fs = _bank()
        A, B, kappa = filterbankbounds(g, a, L, return_kappa=True)
        assert 0.0 < A <= B, f"expected 0 < A <= B, got A={A}, B={B}"
        assert kappa >= 1.0 - 1e-9, f"kappa must be >= 1, got {kappa}"
        assert abs(kappa - B / A) < 1e-9
        # default 2-tuple form unchanged
        A2, B2 = filterbankbounds(g, a, L)
        assert (A2, B2) == (A, B)


class TestSpectrogram:
    def test_filterbank_spectrogram_runs(self):
        from cool_frames.numpy.diagnostics import filterbank_spectrogram  # type: ignore
        f = np.random.default_rng(0).standard_normal(4000)
        spec = filterbank_spectrogram(f, fs=8000)
        assert isinstance(spec, dict) and len(spec) > 0
        assert any(isinstance(v, np.ndarray) and v.ndim >= 2 for v in spec.values()), \
            "expected at least one 2-D array (the spectrogram) in the result"

    @pytest.mark.parametrize("scale", ["erb", "cqt"])
    def test_filterbank_spectrogram_scales(self, scale):
        from cool_frames.numpy.diagnostics import filterbank_spectrogram  # type: ignore
        f = np.random.default_rng(1).standard_normal(4000)
        spec = filterbank_spectrogram(f, fs=8000, scale=scale)
        assert isinstance(spec, dict) and len(spec) > 0

    def test_reassigned_spectrogram_runs(self):
        from cool_frames.numpy.diagnostics import reassigned_spectrogram  # type: ignore
        f = np.random.default_rng(2).standard_normal(4000)
        spec = reassigned_spectrogram(f, fs=8000)
        assert isinstance(spec, dict) and len(spec) > 0
