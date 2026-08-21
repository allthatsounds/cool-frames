"""Tests for the general (non-auditory) scale conversions added 2026-06-12:
linear, log/octave, semitone (12-TET), third-octave (IEC 61260)."""
from __future__ import annotations

import pytest

import numpy as np
from cool_frames.numpy.filters import audfiltbw, audspace, audtofreq, freqtoaud

NEW = ["linear", "log", "semitone", "third-octave"]


@pytest.mark.parametrize("scale", NEW)
def test_roundtrip(scale):
    f = np.array([50.0, 100.0, 440.0, 1000.0, 8000.0])
    assert np.allclose(audtofreq(freqtoaud(f, scale), scale), f, rtol=1e-9)


@pytest.mark.parametrize("scale", NEW)
def test_monotonic(scale):
    f = np.linspace(20.0, 8000.0, 64)
    assert np.all(np.diff(freqtoaud(f, scale)) > 0)


def test_semitone_is_midi():
    assert abs(float(freqtoaud(440.0, "semitone")) - 69.0) < 1e-9      # A4
    assert abs(float(audtofreq(60.0, "semitone")) - 261.6256) < 1e-2   # C4


def test_log_octave_spacing():
    pts = audspace(100.0, 1600.0, 5, "log")
    assert np.allclose(pts, [100, 200, 400, 800, 1600], rtol=1e-9)


def test_linear_uniform_spacing():
    pts = audspace(0.0, 1000.0, 11, "linear")
    assert np.allclose(np.diff(pts), 100.0)


def test_bandwidth_constant_q():
    fc = np.array([500.0, 1000.0, 2000.0])
    assert np.allclose(audfiltbw(fc, "log"), np.log(2.0) * fc)
    assert np.allclose(audfiltbw(fc, "semitone"), (np.log(2.0) / 12.0) * fc)
    assert np.allclose(audfiltbw(fc, "third-octave"), (np.log(2.0) / 3.0) * fc)
    assert np.allclose(audfiltbw(fc, "linear"), 1.0)   # constant


def test_aliases():
    assert np.allclose(freqtoaud(1000.0, "octave"), freqtoaud(1000.0, "log"))
    assert np.allclose(freqtoaud(440.0, "midi"), freqtoaud(440.0, "semitone"))


def test_unknown_scale_raises():
    with pytest.raises(ValueError):
        freqtoaud(1000.0, "bogus")
