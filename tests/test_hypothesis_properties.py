"""
Hypothesis-based property tests for cool_frames.

These supplement the existing parametric property tests in
tests/layer2_filterbank/property/ with randomised input generation
via the Hypothesis library.

Run with::

    pytest tests/test_hypothesis_properties.py -v

Install hypothesis if needed::

    pip install hypothesis
"""

from __future__ import annotations

import pytest

import numpy as np

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Sample rates commonly used in audio processing
sample_rates = st.sampled_from([8000, 16000, 22050, 44100, 48000])

# Signal lengths: at least 2048 samples (need sufficient length for multi-channel filterbank frequency resolution), up to ~1 second at 16 kHz
signal_lengths = st.integers(min_value=2048, max_value=16000)

# Filter scales
filter_scales = st.sampled_from(["erb", "bark", "mel"])


def random_signal(length: int, seed: int) -> np.ndarray:
    """Generate a random real-valued signal."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(length)


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestPerfectReconstruction:
    """Analysis → synthesis with dual frame recovers the input."""

    @given(
        fs=sample_rates,
        Ls=signal_lengths,
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(
        max_examples=30,
        deadline=10_000,  # ms — filterbank design can be slow
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_dual_reconstruction(self, fs, Ls, seed):
        from cool_frames.numpy.filterbanks import (
            filterbank,
            filterbankdual,
            ifilterbank,
        )
        from cool_frames.numpy.filters import audfilters

        # Skip if Ls is too short for this sample rate
        assume(Ls >= 256)

        g, a, _fc, L, _info = audfilters(fs, Ls)
        gd = filterbankdual(g, a, L)

        x = random_signal(Ls, seed)
        c = filterbank(x, g, a, L=L)
        x_rec = np.asarray(ifilterbank(c, gd, a, Ls=Ls, real=True))

        err = np.max(np.abs(x - x_rec[:Ls]))
        assert err < 1e-6, f"Reconstruction error {err:.2e} at fs={fs}, Ls={Ls}"

    @given(
        fs=sample_rates,
        Ls=signal_lengths,
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(
        max_examples=20,
        deadline=10_000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_tight_reconstruction(self, fs, Ls, seed):
        from cool_frames.numpy.filterbanks import (
            filterbank,
            filterbanktight,
            ifilterbank,
        )
        from cool_frames.numpy.filters import audfilters

        assume(Ls >= 256)

        g, a, _fc, L, _info = audfilters(fs, Ls)
        gt = filterbanktight(g, a, L)

        x = random_signal(Ls, seed)
        c = filterbank(x, gt, a, L=L)
        x_rec = np.asarray(ifilterbank(c, gt, a, Ls=Ls, real=True))

        err = np.max(np.abs(x - x_rec[:Ls]))
        assert err < 1e-6, f"Tight reconstruction error {err:.2e} at fs={fs}, Ls={Ls}"


class TestAnalysisLinearity:
    """Filterbank analysis is a linear operator."""

    @given(
        fs=sample_rates,
        Ls=signal_lengths,
        alpha=st.floats(min_value=-10, max_value=10, allow_nan=False),
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(
        max_examples=20,
        deadline=10_000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_scaling(self, fs, Ls, alpha, seed):
        """filterbank(alpha * x) == alpha * filterbank(x)"""
        from cool_frames.numpy.filterbanks import filterbank
        from cool_frames.numpy.filters import audfilters

        assume(Ls >= 256)

        g, a, _fc, L, _info = audfilters(fs, Ls)
        x = random_signal(Ls, seed)

        c_scaled = filterbank(alpha * x, g, a, L=L)
        c_then_scale = [alpha * ci for ci in filterbank(x, g, a, L=L)]

        for k, (a_val, b_val) in enumerate(zip(c_scaled, c_then_scale)):
            err = np.max(np.abs(a_val - b_val))
            assert err < 1e-10, f"Linearity (scaling) failed in channel {k}: {err:.2e}"

    @given(
        fs=sample_rates,
        Ls=signal_lengths,
        seed1=st.integers(min_value=0, max_value=2**31),
        seed2=st.integers(min_value=0, max_value=2**31),
    )
    @settings(
        max_examples=20,
        deadline=10_000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_additivity(self, fs, Ls, seed1, seed2):
        """filterbank(x + y) == filterbank(x) + filterbank(y)"""
        from cool_frames.numpy.filterbanks import filterbank
        from cool_frames.numpy.filters import audfilters

        assume(Ls >= 256)

        g, a, _fc, L, _info = audfilters(fs, Ls)
        x = random_signal(Ls, seed1)
        y = random_signal(Ls, seed2)

        c_sum = filterbank(x + y, g, a, L=L)
        c_x = filterbank(x, g, a, L=L)
        c_y = filterbank(y, g, a, L=L)
        c_added = [cx + cy for cx, cy in zip(c_x, c_y)]

        for k, (a_val, b_val) in enumerate(zip(c_sum, c_added)):
            err = np.max(np.abs(a_val - b_val))
            assert err < 1e-10, f"Additivity failed in channel {k}: {err:.2e}"


class TestEnergyBounds:
    """Frame bounds constrain the energy ratio between signal and coefficients."""

    @given(
        fs=sample_rates,
        Ls=signal_lengths,
        seed=st.integers(min_value=0, max_value=2**31),
    )
    @settings(
        max_examples=15,
        deadline=15_000,
        suppress_health_check=[HealthCheck.too_slow],
    )
    def test_energy_within_frame_bounds(self, fs, Ls, seed):
        """A * ||x||^2  <=  sum(||c_k||^2)  <=  B * ||x||^2"""
        from cool_frames.numpy.filterbanks import (
            filterbank,
            filterbankbounds,
        )
        from cool_frames.numpy.filters import audfilters

        assume(Ls >= 256)

        g, a, _fc, L, _info = audfilters(fs, Ls)
        # The raw coefficient energy sum(|c_k|^2) is governed by the eigenvalues
        # of the frame operator S = D*D. filterbankbounds(real=True) returns the
        # *folded* real-frame bounds, which for these single-sided real-audio
        # banks are exactly 2x the operator eigenvalues (verified against
        # filterbankbounds_svd across fs/Ls). The SVD bounds give the operator
        # scale directly but are O(L^3) — far too slow for Ls up to 16000 — so
        # we halve the (O(L)) folded bounds instead.
        A_folded, B_folded = filterbankbounds(g, a, L)
        A, B = A_folded / 2.0, B_folded / 2.0

        x = random_signal(Ls, seed)
        c = filterbank(x, g, a, L=L)

        signal_energy = np.sum(x**2)
        coeff_energy = sum(np.sum(np.abs(ci) ** 2) for ci in c)

        # Allow small numerical slack
        slack = 1e-6 * signal_energy
        assert coeff_energy >= A * signal_energy - slack, (
            f"Below lower frame bound: {coeff_energy:.4f} < {A:.4f} * {signal_energy:.4f}"
        )
        assert coeff_energy <= B * signal_energy + slack, (
            f"Above upper frame bound: {coeff_energy:.4f} > {B:.4f} * {signal_energy:.4f}"
        )


class TestTightFrameNormalisation:
    """After tight-frame normalisation the folded frame response is ≈ 1.

    ``filterbanktight`` normalises using the *folded* (``real=True``)
    frame response, which sums each bin with its negative-frequency
    mirror.  After normalisation this folded response should be uniformly
    1.0, confirming a tight frame for real-valued signals.

    We verify this by computing ``filterbankresponse(gt, a, L, real=True)``
    and checking that its min and max are both ≈ 1.
    """

    @pytest.mark.parametrize(
        "fs,Ls",
        [
            (16000, 4096),
            (22050, 8000),
            (44100, 8000),
            (44100, 16000),
            (48000, 16000),
        ],
    )
    def test_tight_real_response_flat(self, fs, Ls):
        from cool_frames.numpy.filterbanks import (
            filterbankresponse,
            filterbanktight,
        )
        from cool_frames.numpy.filters import audfilters

        g, a, _fc, L, _info = audfilters(fs, Ls)
        gt = filterbanktight(g, a, L)
        resp = filterbankresponse(gt, a, L, real=True)

        A, B = float(np.min(resp)), float(np.max(resp))
        assert np.isfinite(A), f"Lower frame bound is not finite: {A}"
        assert np.isfinite(B), f"Upper frame bound is not finite: {B}"
        assert abs(A - 1.0) < 1e-10, f"Tight frame response min {A:.12f} should be 1.0"
        assert abs(B - 1.0) < 1e-10, f"Tight frame response max {B:.12f} should be 1.0"
