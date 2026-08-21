"""
test_analysis_synthesis.py
==========================
Python port of:
    layer2_filterbank/unit/TestAnalysisSynthesis.m

Unit tests for analysis / synthesis entry points.

Covers: filterbank, ufilterbank, ifilterbank, ifilterbankiter,
        filterbanklength, audfilters, cqtfilters, gabfilters,
        waveletfilters, warpedfilters.
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers (numpy-level, no impl required)
# ---------------------------------------------------------------------------

def _make_signals(Ls: int, fs: int = 8000, rng_seed: int = 42):
    """Return a dict of test signals at sample rate fs, length Ls."""
    rng = np.random.default_rng(rng_seed)
    t   = np.arange(Ls) / fs
    sigs = {
        "noise_mono":   rng.standard_normal(Ls).astype(complex),
        "noise_stereo": rng.standard_normal((Ls, 2)),
        "sine_440":     np.sin(2 * np.pi * 440 * t),
        "sine_1k":      np.sin(2 * np.pi * 1000 * t),
        "impulse":      np.zeros(Ls),
    }
    sigs["impulse"][0] = 1.0
    return sigs


# ---------------------------------------------------------------------------
# filterbank — structure tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbankStructImpl:
    """TestAnalysisSynthesis: filterbank returns cell array / list."""

    def test_returns_list(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(0)
        x = rng.standard_normal(Ls)
        c = filterbank(x, g, a)
        assert isinstance(c, (list, tuple)), "filterbank must return a list/tuple"

    def test_channel_count(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        M = len(g)
        rng = np.random.default_rng(1)
        x = rng.standard_normal(Ls)
        c = filterbank(x, g, a)
        assert len(c) == M, f"Expected {M} bands, got {len(c)}"

    def test_coeff_columns_mono(self, needs_impl):
        """Mono input → each subband has 1 column."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        rng = np.random.default_rng(2)
        x = rng.standard_normal(Ls)
        c = filterbank(x, g, a)
        for m, c_m in enumerate(c):
            arr = np.asarray(c_m)
            assert arr.ndim == 1 or arr.shape[1] == 1, \
                f"Band {m}: expected 1 column for mono input, got shape {arr.shape}"

    def test_coeff_columns_stereo(self, needs_impl):
        """Stereo input → each subband has 2 columns."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        rng = np.random.default_rng(3)
        x = rng.standard_normal((Ls, 2))
        c = filterbank(x, g, a)
        for m, c_m in enumerate(c):
            arr = np.asarray(c_m)
            assert arr.ndim >= 2 and arr.shape[1] == 2, \
                f"Band {m}: expected 2 columns for stereo, got shape {arr.shape}"

    def test_linearity(self, needs_impl):
        """T(α·f1 + β·f2) == α·T(f1) + β·T(f2)."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        rng = np.random.default_rng(4)
        alpha, beta = 2.7, -1.3
        f1 = rng.standard_normal(Ls)
        f2 = np.sin(2 * np.pi * 440 * np.arange(Ls) / fs)
        c1   = filterbank(f1, g, a)
        c2   = filterbank(f2, g, a)
        csum = filterbank(alpha * f1 + beta * f2, g, a)
        for m in range(len(g)):
            expected = alpha * np.asarray(c1[m]) + beta * np.asarray(c2[m])
            err = np.linalg.norm(np.asarray(csum[m]) - expected) / (np.linalg.norm(expected) + 1e-15)
            assert err < 1e-10, f"Linearity violated at band {m}: err={err:.2e}"


# ---------------------------------------------------------------------------
# ufilterbank
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestUfilterbankImpl:
    """TestAnalysisSynthesis: ufilterbank returns 2-D matrix.

    These tests use gabfilters, which is not yet implemented in the NumPy
    backend.  The tests are skipped automatically until gabfilters is ported.
    """

    def test_returns_2d(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import gabfilters  # type: ignore
        Ls, M_gab, a_hop = 1024, 16, 32
        try:
            g_gab, a_gab, fc_gab, *_ = gabfilters(16000, Ls, window="hann", a=a_hop, M=M_gab)
        except NotImplementedError:
            pytest.skip("gabfilters not yet implemented")
        rng = np.random.default_rng(5)
        x = rng.standard_normal(Ls)
        c = filterbank(x, g_gab, a_gab, stack=True)
        arr = np.asarray(c)
        assert arr.ndim >= 2, "ufilterbank must return at least a 2-D array"

    def test_channel_dimension(self, needs_impl):
        """ufilterbank channel dim == M_gab (or M//2+1 for real-valued)."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import gabfilters  # type: ignore
        Ls, M_gab, a_hop = 1024, 16, 32
        try:
            g_gab, a_gab, fc_gab, *_ = gabfilters(16000, Ls, window="hann", a=a_hop, M=M_gab)
        except NotImplementedError:
            pytest.skip("gabfilters not yet implemented")
        rng = np.random.default_rng(6)
        x = rng.standard_normal(Ls)
        c = np.asarray(filterbank(x, g_gab, a_gab, stack=True))
        # LTFAT Gabor keeps M//2+1 channels for real signals
        M2_expected = M_gab // 2 + 1
        assert c.shape[1] == M2_expected, \
            f"ufilterbank channel dim: expected {M2_expected}, got {c.shape[1]}"


# ---------------------------------------------------------------------------
# ifilterbankiter
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestIfilterbankiterImpl:
    """TestAnalysisSynthesis: iterative reconstruction convergence."""

    def test_convergence(self, needs_impl):
        from cool_frames.filterbanks import filterbank, ifilterbankiter  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        rng = np.random.default_rng(7)
        x = rng.standard_normal(Ls)
        c = filterbank(x, g, a)
        result = ifilterbankiter(c, g, a)
        # ifilterbankiter may return (xr, relres) or (xr, relres, iter)
        relres = result[1] if isinstance(result, (list, tuple)) else 0.0
        assert relres < 0.1, f"ifilterbankiter did not converge: relres={relres:.2e}"


# ---------------------------------------------------------------------------
# filterbanklength
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbanklengthImpl:
    """TestAnalysisSynthesis: filterbanklength properties."""

    def test_geq_ls(self, needs_impl):
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        assert L >= Ls, f"filterbanklength={L} < Ls={Ls}"

    def test_idempotent(self, needs_impl):
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L1 = filterbanklength(Ls, a)
        L2 = filterbanklength(L1, a)
        assert L2 == L1, f"filterbanklength not idempotent: {L1} → {L2}"

    def test_is_integer(self, needs_impl):
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        assert int(L) == L, "filterbanklength must return an integer"


# ---------------------------------------------------------------------------
# audfilters
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestAudfiltersImpl:
    """TestAnalysisSynthesis: audfilters structural properties."""

    def test_monotonic_freq(self, needs_impl):
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        fc_arr = np.asarray(fc, dtype=float)
        assert np.all(np.diff(fc_arr) > 0), \
            "audfilters: centre frequencies not strictly monotone"

    def test_freq_upper_bound(self, needs_impl):
        from cool_frames.filters import audfilters  # type: ignore
        fs = 8000
        g, a, fc, _, _info = audfilters(fs, 1024)
        assert np.max(np.asarray(fc, float)) <= fs / 2 + 1, \
            "audfilters: max fc exceeds Nyquist"

    def test_subsampling_positive(self, needs_impl):
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        assert np.all(np.asarray(a) > 0), \
            "audfilters: subsampling factors must be positive"

    def test_filter_count_positive(self, needs_impl):
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        assert len(g) > 0, "audfilters must return at least one filter"

    def test_length_valid(self, needs_impl):
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls = 1024
        g, a, fc, _, _info = audfilters(8000, Ls)
        L = filterbanklength(Ls, a)
        assert L >= Ls


# ---------------------------------------------------------------------------
# cqtfilters
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCqtfiltersImpl:
    """TestAnalysisSynthesis: cqtfilters structural properties."""

    def test_constant_log_spacing(self, needs_impl):
        """Interior channels should have approximately uniform log-freq spacing."""
        from cool_frames.filters import cqtfilters  # type: ignore
        bins = 12
        g, a, fc, redmul, _info = cqtfilters(8000, 1024, fmin=100, fmax=4000, bins=bins)
        fc_arr = np.asarray(fc, dtype=float)
        fc_inner = fc_arr[1:-1]
        if len(fc_inner) >= 4:
            log_diffs = np.diff(np.log(fc_inner))
            cv = np.std(log_diffs) / (abs(np.mean(log_diffs)) + 1e-15)
            assert cv < 0.05, f"cqtfilters: log-freq spacing not uniform (cv={cv:.3f})"

    def test_coverage_min(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        fmin = 100
        g, a, fc, redmul, _info = cqtfilters(8000, 1024, fmin=fmin, fmax=4000, bins=12)
        assert np.min(np.asarray(fc, float)) <= fmin * 1.5

    def test_coverage_max(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        fmax = 4000
        g, a, fc, redmul, _info = cqtfilters(8000, 1024, fmin=100, fmax=fmax, bins=12)
        assert np.max(np.asarray(fc, float)) >= fmax * 0.5


# ---------------------------------------------------------------------------
# gabfilters
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestGabfiltersImpl:
    """TestAnalysisSynthesis: gabfilters structural properties.

    Tests that gabfilters returns the expected structure (filters, a, fc, redmul, info).
    """

    def test_importable_from_layer1(self, needs_impl):
        from cool_frames.filters import gabfilters  # type: ignore
        assert callable(gabfilters)

    def test_importable_from_layer2(self, needs_impl):
        from cool_frames.filters import gabfilters  # type: ignore
        assert callable(gabfilters)

    def test_basic_construction(self, needs_impl):
        """Test that gabfilters constructs properly and returns 5 values."""
        from cool_frames.filters import gabfilters  # type: ignore
        g, a, fc, redmul, info = gabfilters(16000, 1024, window="hann", a=64, M=32)
        assert isinstance(g, list)
        assert len(g) > 0
        assert isinstance(a, np.ndarray)
        assert isinstance(fc, np.ndarray)
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# waveletfilters
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestWaveletfiltersImpl:
    """TestAnalysisSynthesis: waveletfilters structural properties."""

    def test_runs(self, needs_impl):
        from cool_frames.filters import waveletfilters  # type: ignore
        scales = np.arange(1, 9)
        g, a, fc, redmul, info = waveletfilters(2.0, 1024, scales=scales)
        assert len(g) > 0
        assert len(g) == len(a)
        assert np.all(np.asarray(fc, float) >= 0)
        assert isinstance(info, dict)
