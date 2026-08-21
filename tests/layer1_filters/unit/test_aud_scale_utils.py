"""
test_aud_scale_utils.py
=======================
Python port of:
    layer1_filters/unit/TestAudScaleUtils.m

Covers: erbtofreq, freqtoerb, pfilt, setnorm
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# erbtofreq / freqtoerb
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestErbtofreqImpl:
    """
    MATLAB counterpart: TestAudScaleUtils (erbtofreq section).
    """

    def test_zero_erb_maps_to_near_zero_hz(self, needs_impl):
        from cool_frames.filters import erbtofreq  # type: ignore
        assert abs(erbtofreq(0)) < 1.0  # 0 ERB ≈ 0 Hz (tolerance 1 Hz)

    def test_positive_erb_gives_positive_hz(self, needs_impl):
        from cool_frames.filters import erbtofreq  # type: ignore
        freqs = erbtofreq(np.array([1.0, 5.0, 10.0, 20.0, 30.0]))
        assert np.all(freqs > 0)

    def test_monotonically_increasing(self, needs_impl):
        from cool_frames.filters import erbtofreq  # type: ignore
        erbs  = np.arange(0, 35.5, 0.5)
        freqs = erbtofreq(erbs)
        assert np.all(np.diff(freqs) > 0)

    def test_vector_length_preserved(self, needs_impl):
        from cool_frames.filters import erbtofreq  # type: ignore
        erbs  = np.linspace(1, 30, 50)
        freqs = erbtofreq(erbs)
        assert len(freqs) == 50

    def test_inverse_roundtrip(self, needs_impl):
        from cool_frames.filters import erbtofreq, freqtoerb  # type: ignore
        freqs_in  = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
        freqs_out = erbtofreq(freqtoerb(freqs_in))
        rel_err   = np.linalg.norm(freqs_out - freqs_in) / np.linalg.norm(freqs_in)
        assert rel_err < 1e-10


@pytest.mark.requires_impl
class TestFreqtoerbImpl:
    """
    MATLAB counterpart: TestAudScaleUtils (freqtoerb section).
    """

    def test_1000hz_maps_to_expected_range(self, needs_impl):
        from cool_frames.filters import freqtoerb  # type: ignore
        erb = freqtoerb(1000.0)
        assert 10 < erb < 25

    def test_monotonically_increasing(self, needs_impl):
        from cool_frames.filters import freqtoerb  # type: ignore
        freqs = np.array([100.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0])
        erbs  = freqtoerb(freqs)
        assert np.all(np.diff(erbs) > 0)

    def test_vector_length_preserved(self, needs_impl):
        from cool_frames.filters import freqtoerb  # type: ignore
        freqs = np.linspace(100, 8000, 40)
        assert len(freqtoerb(freqs)) == 40

    def test_positive_hz_gives_positive_erb(self, needs_impl):
        from cool_frames.filters import freqtoerb  # type: ignore
        erbs = freqtoerb(np.array([250.0, 1000.0, 4000.0]))
        assert np.all(erbs > 0)

    def test_inverse_roundtrip(self, needs_impl):
        from cool_frames.filters import erbtofreq, freqtoerb  # type: ignore
        erbs_in  = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
        erbs_out = freqtoerb(erbtofreq(erbs_in))
        rel_err  = np.linalg.norm(erbs_out - erbs_in) / np.linalg.norm(erbs_in)
        assert rel_err < 1e-10


# ---------------------------------------------------------------------------
# setnorm
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestSetnormImpl:
    """
    MATLAB counterpart: TestAudScaleUtils (setnorm section).
    """

    def test_l2_norm(self, needs_impl):
        from cool_frames.core import setnorm  # type: ignore
        f  = np.random.default_rng(42).standard_normal(64)
        fn, _ = setnorm(f, "2")
        assert np.linalg.norm(fn) == pytest.approx(1.0, abs=1e-12)

    def test_energy_is_same_as_l2(self, needs_impl):
        from cool_frames.core import setnorm  # type: ignore
        f  = np.random.default_rng(42).standard_normal(64)
        fn, _ = setnorm(f, "energy")
        assert np.linalg.norm(fn) == pytest.approx(1.0, abs=1e-12)

    def test_l1_norm(self, needs_impl):
        from cool_frames.core import setnorm  # type: ignore
        f  = np.random.default_rng(42).standard_normal(64)
        fn, _ = setnorm(f, "1")
        assert np.sum(np.abs(fn)) == pytest.approx(1.0, abs=1e-12)

    def test_linf_norm(self, needs_impl):
        from cool_frames.core import setnorm  # type: ignore
        f  = np.random.default_rng(42).standard_normal(64)
        fn, _ = setnorm(f, "inf")
        assert np.max(np.abs(fn)) == pytest.approx(1.0, abs=1e-12)

    def test_rms_norm(self, needs_impl):
        from cool_frames.core import setnorm  # type: ignore
        f   = np.random.default_rng(42).standard_normal(128)
        fn, _  = setnorm(f, "rms")
        rms = np.linalg.norm(fn) / np.sqrt(len(fn))
        assert rms == pytest.approx(1.0, abs=1e-12)

    def test_preserves_shape(self, needs_impl):
        from cool_frames.core import setnorm  # type: ignore
        f  = np.random.default_rng(42).standard_normal((32, 4))
        fn, _ = setnorm(f, "2")
        assert fn.shape == f.shape

    def test_second_output_is_original_norm(self, needs_impl):
        from cool_frames.core import setnorm  # type: ignore
        f             = np.random.default_rng(42).standard_normal(64) * 5.0
        original_norm = np.linalg.norm(f)
        _, fnorm      = setnorm(f, "2")
        assert fnorm == pytest.approx(original_norm, abs=1e-12)
