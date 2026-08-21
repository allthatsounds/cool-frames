"""
test_prop_filter_normalization.py
==================================
Python port of:
    layer1_filters/property/PropFilterNormalization.m

Normalization consistency across all filter constructors:
  'energy' norm: (1/L)*sum|H(L)|^2 = 1  for any valid L
  'peak'   norm: max|H(L)| = 1           for any valid L
  'scal'   s:    H_scal = s * H_default

These should hold across different signal lengths L.
"""

from __future__ import annotations

import pytest

import numpy as np

_LENGTHS = [128, 256, 512, 1024]


# ---------------------------------------------------------------------------
# firwin_ref baseline (no impl required)
# ---------------------------------------------------------------------------

class TestWindowNormalizationReference:
    """Verify firwin_ref PU/TF properties hold across lengths — no impl required."""

    from conftest import firwin_ref as _fw

    @pytest.mark.parametrize("M", _LENGTHS)
    def test_hann_energy_norm_per_length(self, M):
        """sum(w^2)/M for hann window."""
        from conftest import firwin_ref
        w = firwin_ref("hann", M)
        # For a PU window: ||w||^2 = M/2  (half energy), so sum/M = 0.5
        # This isn't normalised to 1; test just that it's constant across M.
        e = np.sum(w ** 2) / M
        assert 0.0 < e < 1.0  # sanity: not zero, not greater than 1


# ---------------------------------------------------------------------------
# Energy normalization – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestEnergyNormImpl:
    """
    MATLAB counterpart: PropFilterNormalization (energy norm section).
    """

    @pytest.mark.parametrize("L", _LENGTHS)
    def test_blfilter_energy_norm(self, needs_impl, L):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = blfilter("hann", 0.1, 0.3, "energy")
        H = comp_transferfunction(g, L)
        assert np.sum(np.abs(H) ** 2) / L == pytest.approx(1.0, abs=1e-8), \
            f"blfilter energy norm failed at L={L}"

    @pytest.mark.parametrize("L", _LENGTHS)
    def test_freqfilter_energy_norm(self, needs_impl, L):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = freqfilter("gauss", 0.05, 0.3, "energy")
        H = comp_transferfunction(g, L)
        assert np.sum(np.abs(H) ** 2) / L == pytest.approx(1.0, abs=1e-8), \
            f"freqfilter energy norm failed at L={L}"

    @pytest.mark.parametrize("M", [16, 32, 64])
    def test_firfilter_energy_norm(self, needs_impl, M):
        """firfilter energy: sum(h^2) == 1 (time-domain, length-independent)."""
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        g = firfilter("hann", M, 0, "energy")
        h = np.asarray(g["h"], dtype=float)
        assert np.sum(h ** 2) == pytest.approx(1.0, abs=1e-8), \
            f"firfilter energy norm failed for M={M}"

    @pytest.mark.parametrize("L", _LENGTHS)
    def test_biquadfilter_energy_norm(self, needs_impl, L):
        from cool_frames.filters import biquadfilter  # type: ignore
        g = biquadfilter(0.25, 0.04, "energy")
        H = g["H"](L)
        assert np.sum(np.abs(H) ** 2) / L == pytest.approx(1.0, abs=1e-8), \
            f"biquadfilter energy norm failed at L={L}"


# ---------------------------------------------------------------------------
# Peak normalization – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPeakNormImpl:
    """
    MATLAB counterpart: PropFilterNormalization (peak norm section).
    """

    @pytest.mark.parametrize("L", _LENGTHS)
    def test_blfilter_peak_norm(self, needs_impl, L):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = blfilter("hann", 0.1, 0.3, "peak")
        H = comp_transferfunction(g, L)
        assert np.max(np.abs(H)) == pytest.approx(1.0, abs=1e-10), \
            f"blfilter peak norm failed at L={L}"

    @pytest.mark.parametrize("L", _LENGTHS)
    def test_freqfilter_peak_norm(self, needs_impl, L):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = freqfilter("gauss", 0.05, 0.3, "peak")
        H = comp_transferfunction(g, L)
        assert np.max(np.abs(H)) == pytest.approx(1.0, abs=1e-10), \
            f"freqfilter peak norm failed at L={L}"

    @pytest.mark.parametrize("L", _LENGTHS)
    def test_biquadfilter_peak_norm(self, needs_impl, L):
        from cool_frames.filters import biquadfilter  # type: ignore
        g = biquadfilter(0.25, 0.04, "peak")
        H = g["H"](L)
        assert np.max(np.abs(H)) == pytest.approx(1.0, abs=1e-10), \
            f"biquadfilter peak norm failed at L={L}"


# ---------------------------------------------------------------------------
# Scal linearity – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestScalLinearityImpl:
    """
    MATLAB counterpart: PropFilterNormalization (scal linearity section).
    """

    @pytest.mark.parametrize("seed", [42, 7, 99])
    def test_blfilter_scal_linearity(self, needs_impl, seed):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        rng = np.random.default_rng(seed)
        L   = 512
        for _ in range(10):
            s  = 0.5 + 2 * rng.random()
            g1 = blfilter("hann", 0.1, 0.3)
            g2 = blfilter("hann", 0.1, 0.3, "scal", s)
            H1 = comp_transferfunction(g1, L)
            H2 = comp_transferfunction(g2, L)
            np.testing.assert_allclose(H2, s * H1,
                                       atol=1e-10 * np.linalg.norm(H1),
                                       err_msg=f"blfilter scal linearity failed for s={s:.3f}")

    @pytest.mark.parametrize("seed", [42, 7])
    def test_freqfilter_scal_linearity(self, needs_impl, seed):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        rng = np.random.default_rng(seed)
        L   = 512
        for _ in range(10):
            s  = 0.5 + 2 * rng.random()
            g1 = freqfilter("gauss", 0.05, 0.3)
            g2 = freqfilter("gauss", 0.05, 0.3, "scal", s)
            H1 = comp_transferfunction(g1, L)
            H2 = comp_transferfunction(g2, L)
            np.testing.assert_allclose(H2, s * H1,
                                       atol=1e-10 * np.linalg.norm(H1))


# ---------------------------------------------------------------------------
# Energy vs peak consistency – implementation test
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestEnergyPeakConsistencyImpl:
    """
    MATLAB counterpart: PropFilterNormalization (energy vs peak section).
    """

    def test_energy_and_peak_differ_by_global_constant(self, needs_impl):
        """H_energy and H_peak are the same shape scaled by a global constant."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 512
        ge = blfilter("hann", 0.1, 0.3, "energy")
        gp = blfilter("hann", 0.1, 0.3, "peak")
        He = comp_transferfunction(ge, L)
        Hp = comp_transferfunction(gp, L)
        # ratio only where |Hp| is non-trivial
        mask  = np.abs(Hp) > 1e-6 * np.max(np.abs(Hp))
        ratio = He[mask] / Hp[mask]
        cv    = np.std(np.abs(ratio)) / np.mean(np.abs(ratio))
        assert cv < 1e-8, "energy and peak responses should differ by a global constant"
