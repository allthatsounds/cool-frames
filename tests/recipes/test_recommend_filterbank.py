"""
Tests for cool_frames_additions.recipes.recommend_filterbank
==================================================
"""
from __future__ import annotations

import numpy as np
import pytest

from cool_frames.numpy.diagnostics.recommend_filterbank import (
    FilterbankRecommendation,
    _bandwidth_occupancy,
    _energy_frequency_range,
    _estimate_f0,
    _harmonicity_ratio,
    _onset_rate,
    _stationarity,
    recommend_filterbank,
)


@pytest.fixture
def fs():
    return 16000


@pytest.fixture
def pure_tone(fs):
    t = np.arange(fs * 2) / fs
    return np.sin(2 * np.pi * 440 * t)


@pytest.fixture
def white_noise(fs):
    return np.random.default_rng(42).standard_normal(fs * 2)


@pytest.fixture
def click_train(fs):
    x = np.zeros(fs * 2)
    for i in range(20):
        pos = int(i * fs * 0.1)
        if pos + 10 < len(x):
            x[pos : pos + 10] = 5.0
    return x


# ---------------------------------------------------------------------------
# Signal analysis helpers
# ---------------------------------------------------------------------------

class TestHarmonicity:
    def test_pure_tone_high(self, pure_tone, fs):
        h = _harmonicity_ratio(pure_tone, fs)
        assert h > 0.8, f"Pure tone harmonicity should be high, got {h:.3f}"

    def test_noise_low(self, white_noise, fs):
        h = _harmonicity_ratio(white_noise, fs)
        assert h < 0.3, f"Noise harmonicity should be low, got {h:.3f}"

    def test_silence_zero(self, fs):
        h = _harmonicity_ratio(np.zeros(fs), fs)
        assert h == 0.0

    def test_range(self, pure_tone, fs):
        h = _harmonicity_ratio(pure_tone, fs)
        assert 0.0 <= h <= 1.0


class TestBandwidthOccupancy:
    def test_tone_narrow(self, pure_tone, fs):
        bw = _bandwidth_occupancy(pure_tone, fs)
        assert bw < 0.2, f"Pure tone should be narrowband, got {bw:.3f}"

    def test_noise_broad(self, white_noise, fs):
        bw = _bandwidth_occupancy(white_noise, fs)
        assert bw > 0.5, f"White noise should be broadband, got {bw:.3f}"


class TestStationarity:
    def test_tone_stationary(self, pure_tone, fs):
        s = _stationarity(pure_tone, fs)
        assert s > 0.8, f"Pure tone should be stationary, got {s:.3f}"

    def test_clicks_nonstationary(self, click_train, fs):
        s = _stationarity(click_train, fs)
        assert s < 0.8, f"Click train should be non-stationary, got {s:.3f}"

    def test_range(self, pure_tone, fs):
        s = _stationarity(pure_tone, fs)
        assert 0.0 <= s <= 1.0


class TestOnsetRate:
    def test_clicks_high(self, click_train, fs):
        rate = _onset_rate(click_train, fs)
        assert rate > 5.0, f"Click train should have high onset rate, got {rate:.1f}"

    def test_silence_zero(self, fs):
        rate = _onset_rate(np.zeros(fs), fs)
        assert rate == 0.0


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

class TestRecommendFilterbank:
    def test_returns_dataclass(self, pure_tone, fs):
        rec = recommend_filterbank(pure_tone, fs)
        assert isinstance(rec, FilterbankRecommendation)
        assert isinstance(rec.designer, str)
        assert isinstance(rec.params, dict)
        assert isinstance(rec.rationale, str)
        assert isinstance(rec.metrics, dict)
        assert isinstance(rec.alternatives, list)

    def test_harmonic_gets_cqt(self, pure_tone, fs):
        rec = recommend_filterbank(pure_tone, fs)
        assert rec.designer == "cqtfilters"

    def test_noise_gets_audfilters(self, white_noise, fs):
        rec = recommend_filterbank(white_noise, fs)
        assert rec.designer == "audfilters"

    def test_transient_gets_timeadaptive(self, click_train, fs):
        rec = recommend_filterbank(click_train, fs)
        assert rec.designer == "timeadaptivefilters"

    def test_real_time_purpose(self, pure_tone, fs):
        rec = recommend_filterbank(pure_tone, fs, purpose="real_time")
        assert rec.designer == "audfilters"

    def test_phase_retrieval_high_redundancy(self, pure_tone, fs):
        rec = recommend_filterbank(pure_tone, fs, purpose="phase_retrieval")
        assert rec.params.get("redmul", 1) >= 4.0

    def test_modification_increases_redundancy(self, white_noise, fs):
        rec_analysis = recommend_filterbank(white_noise, fs, purpose="analysis")
        rec_mod = recommend_filterbank(white_noise, fs, purpose="modification")
        redmul_analysis = rec_analysis.params.get("redmul", 1.0)
        redmul_mod = rec_mod.params.get("redmul", 1.0)
        assert redmul_mod >= redmul_analysis

    def test_metrics_present(self, pure_tone, fs):
        rec = recommend_filterbank(pure_tone, fs)
        for key in ["harmonicity", "onset_rate_per_sec", "bandwidth_occupancy",
                     "stationarity", "duration_sec", "fs"]:
            assert key in rec.metrics

    def test_alternatives_present(self, pure_tone, fs):
        rec = recommend_filterbank(pure_tone, fs)
        assert len(rec.alternatives) >= 1
        for alt in rec.alternatives:
            assert "designer" in alt
            assert "reason" in alt

    def test_chord_gets_cqt(self, fs):
        """Chord (multiple harmonics) should be detected as harmonic → CQT."""
        t = np.arange(fs * 2) / fs
        x = np.sin(2*np.pi*261*t) + np.sin(2*np.pi*329*t) + np.sin(2*np.pi*392*t)
        rec = recommend_filterbank(x, fs)
        assert rec.designer == "cqtfilters"


# ---------------------------------------------------------------------------
# Parameter recommendation quality
# ---------------------------------------------------------------------------

class TestParameterRecommendations:
    def test_fmin_fmax_in_metrics(self, pure_tone, fs):
        """Metrics should include estimated frequency range."""
        rec = recommend_filterbank(pure_tone, fs)
        assert "fmin_hz" in rec.metrics
        assert "fmax_hz" in rec.metrics
        assert rec.metrics["fmin_hz"] > 0
        assert rec.metrics["fmax_hz"] <= fs / 2

    def test_f0_estimated_for_tone(self, pure_tone, fs):
        """A pure 440 Hz tone should have f0 close to 440."""
        rec = recommend_filterbank(pure_tone, fs)
        assert rec.metrics["f0_hz"] is not None
        assert abs(rec.metrics["f0_hz"] - 440) < 20

    def test_f0_none_for_noise(self, white_noise, fs):
        """White noise should have no detectable f0."""
        rec = recommend_filterbank(white_noise, fs)
        assert rec.metrics["f0_hz"] is None

    def test_cqt_params_include_bins(self, pure_tone, fs):
        """CQT recommendation should specify bins per octave."""
        rec = recommend_filterbank(pure_tone, fs)
        assert rec.designer == "cqtfilters"
        assert "bins" in rec.params
        assert rec.params["bins"] >= 12

    def test_cqt_fmin_below_f0(self, pure_tone, fs):
        """CQT fmin should be at or below the fundamental."""
        rec = recommend_filterbank(pure_tone, fs)
        assert rec.designer == "cqtfilters"
        assert rec.params["fmin"] <= 440

    def test_cqt_fmax_above_signal(self, pure_tone, fs):
        """CQT fmax should cover the signal's active band."""
        rec = recommend_filterbank(pure_tone, fs)
        assert "fmax" in rec.params
        assert rec.params["fmax"] >= 440

    def test_audfilters_fmin_fmax_set(self, white_noise, fs):
        """Audfilters recommendation should include fmin and fmax."""
        rec = recommend_filterbank(white_noise, fs)
        assert rec.designer == "audfilters"
        assert "fmin" in rec.params
        assert "fmax" in rec.params

    def test_low_tone_gets_fine_cqt(self, fs):
        """A low-frequency tone should get higher bins per octave."""
        t = np.arange(fs * 2) / fs
        x_low = np.sin(2 * np.pi * 80 * t)
        rec = recommend_filterbank(x_low, fs)
        if rec.designer == "cqtfilters":
            assert rec.params["bins"] >= 36


class TestEnergyFrequencyRange:
    def test_tone_narrow_range(self, pure_tone, fs):
        fmin, fmax = _energy_frequency_range(pure_tone, fs)
        assert fmin > 100  # well above 0
        assert fmax < 1000  # well below Nyquist
        assert fmin < 440 < fmax  # contains the tone

    def test_noise_wide_range(self, white_noise, fs):
        fmin, fmax = _energy_frequency_range(white_noise, fs)
        assert fmax > fs / 4  # covers a large range


class TestEstimateF0:
    def test_pure_tone(self, pure_tone, fs):
        f0 = _estimate_f0(pure_tone, fs)
        assert f0 is not None
        assert abs(f0 - 440) < 20

    def test_noise_returns_none(self, white_noise, fs):
        f0 = _estimate_f0(white_noise, fs)
        assert f0 is None

    def test_low_tone(self, fs):
        t = np.arange(fs * 2) / fs
        x = np.sin(2 * np.pi * 100 * t)
        f0 = _estimate_f0(x, fs)
        assert f0 is not None
        assert abs(f0 - 100) < 10
