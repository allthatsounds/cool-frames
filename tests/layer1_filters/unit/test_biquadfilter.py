"""
test_biquadfilter.py
====================
Python port of:
    layer1_filters/unit/TestBiquadFilter.m

Covers: biquadfilter (IIR resonator constructor), comp_biquad

API
---
biquadfilter(fc, bw)             -> single filter dict
biquadfilter([fc1,fc2,...], bw)  -> list of filter dicts
biquadfilter(fc, bw, norm)       -> with normalisation flag ('energy', 'peak', 'scal')
biquadfilter(fc, bw, 'fs', fs)   -> fc and bw in Hz

Pole parametrization:
    r     = 1 - pi*bw/2   (pole radius)
    theta = pi * fc        (pole angle)
    rho   = logit(r)       (ML parametrization of radius)
    phi   = logit(theta/pi) (ML parametrization of angle)

comp_biquad(r, theta, L, norm) -> length-L DFT response
"""

from __future__ import annotations

import pytest

import numpy as np


@pytest.mark.requires_impl
class TestBiquadFilterStructImpl:
    """
    MATLAB counterpart: TestBiquadFilter (struct format section).
    """

    def test_has_required_fields(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        g = biquadfilter(0.25, 0.05)
        for field in ("H", "foff", "delay", "realonly", "r", "theta", "rho", "phi"):
            assert field in g, f"Missing field: {field}"

    def test_H_is_callable(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        g = biquadfilter(0.25, 0.05)
        assert callable(g["H"])

    def test_H_returns_correct_length(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        g = biquadfilter(0.25, 0.05)
        for L in (64, 256, 1024):
            assert len(g["H"](L)) == L

    def test_foff_is_callable_returning_zero(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        g = biquadfilter(0.25, 0.05)
        assert callable(g["foff"])
        assert g["foff"](1)   == 0
        assert g["foff"](512) == 0

    def test_vector_input_yields_list(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        gout = biquadfilter([0.1, 0.2, 0.3], [0.05, 0.05, 0.05])
        assert isinstance(gout, list) and len(gout) == 3
        for g in gout:
            assert "H" in g

    def test_scalar_input_returns_dict(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        g = biquadfilter(0.25, 0.05)
        assert isinstance(g, dict)

    def test_default_realonly_is_zero(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        assert biquadfilter(0.25, 0.05)["realonly"] == 0

    def test_real_flag_sets_realonly(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        assert biquadfilter(0.25, 0.05, "real")["realonly"] == 1

    def test_delay_parameter(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        assert biquadfilter(0.25, 0.05, "delay", 3)["delay"] == 3


@pytest.mark.requires_impl
class TestBiquadFilterPoleParamsImpl:
    """
    MATLAB counterpart: TestBiquadFilter (pole parameter section).
    """

    def test_pole_radius_from_bandwidth(self, needs_impl):
        """r ≈ 1 - pi*bw/2."""
        from cool_frames.filters import biquadfilter  # type: ignore
        bw = 0.04
        g  = biquadfilter(0.25, bw)
        assert abs(g["r"] - (1 - np.pi * bw / 2)) < 1e-10

    def test_pole_angle_from_center_freq(self, needs_impl):
        """theta == pi * fc."""
        from cool_frames.filters import biquadfilter  # type: ignore
        fc = 0.3
        g  = biquadfilter(fc, 0.05)
        assert abs(g["theta"] - np.pi * fc) < 1e-12

    @pytest.mark.parametrize("bw", [0.001, 0.01, 0.1, 0.5])
    def test_stability_constraint(self, needs_impl, bw):
        """Pole radius must be strictly in (0, 1)."""
        from cool_frames.filters import biquadfilter  # type: ignore
        g = biquadfilter(0.25, bw)
        assert 0 < g["r"] < 1

    def test_ml_parametrization_roundtrip(self, needs_impl):
        """rho = logit(r), phi = logit(theta/pi) — verify roundtrip."""
        from cool_frames.filters import biquadfilter  # type: ignore
        g = biquadfilter(0.3, 0.04)
        r_from_rho     = 1.0 / (1.0 + np.exp(-g["rho"]))
        theta_from_phi = np.pi / (1.0 + np.exp(-g["phi"]))
        assert abs(r_from_rho     - g["r"])     < 1e-12
        assert abs(theta_from_phi - g["theta"]) < 1e-12

    def test_ml_override_rho(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        rho_in     = 2.0
        r_expected = 1.0 / (1.0 + np.exp(-rho_in))
        g = biquadfilter(0.25, 0.05, "rho", rho_in)
        assert abs(g["r"] - r_expected) < 1e-12

    def test_ml_override_phi(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        phi_in      = 0.5
        th_expected = np.pi / (1.0 + np.exp(-phi_in))
        g = biquadfilter(0.25, 0.05, "phi", phi_in)
        assert abs(g["theta"] - th_expected) < 1e-12


@pytest.mark.requires_impl
class TestBiquadFilterFreqResponseImpl:
    """
    MATLAB counterpart: TestBiquadFilter (frequency response section).
    """

    def test_peak_near_centre_freq(self, needs_impl):
        """Peak in positive-frequency half within 3 bins of fc."""
        from cool_frames.filters import biquadfilter  # type: ignore
        L  = 1024
        fc = 0.3
        g  = biquadfilter(fc, 0.02, "peak")
        H  = g["H"](L)
        H_pos       = H[: L // 2 + 1]
        peak_bin    = int(np.argmax(np.abs(H_pos)))
        expected    = round(fc / 2 * L)
        assert abs(peak_bin - expected) < 3

    def test_energy_normalization(self, needs_impl):
        """'energy': (1/L)*sum|H|^2 ≈ 1."""
        from cool_frames.filters import biquadfilter  # type: ignore
        L = 512
        g = biquadfilter(0.25, 0.03, "energy")
        H = g["H"](L)
        assert np.sum(np.abs(H) ** 2) / L == pytest.approx(1.0, abs=1e-6)

    def test_peak_normalization(self, needs_impl):
        """'peak': max|H| ≈ 1."""
        from cool_frames.filters import biquadfilter  # type: ignore
        L = 512
        g = biquadfilter(0.25, 0.03, "peak")
        H = g["H"](L)
        assert np.max(np.abs(H)) == pytest.approx(1.0, abs=1e-6)

    def test_scal_parameter(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        L  = 256
        s  = 2.5
        g1 = biquadfilter(0.25, 0.03)
        g2 = biquadfilter(0.25, 0.03, "scal", s)
        H1 = g1["H"](L)
        H2 = g2["H"](L)
        assert np.max(np.abs(H2 - s * H1)) < 1e-6 * np.linalg.norm(H1)

    def test_hz_input_consistency(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        fs    = 8000
        fc_hz = 1000
        bw_hz = 200
        g_hz   = biquadfilter(fc_hz,         bw_hz,         "fs", fs)
        g_norm = biquadfilter(fc_hz / (fs/2), bw_hz / (fs/2))
        L = 512
        assert np.max(np.abs(g_hz["H"](L) - g_norm["H"](L))) < 1e-6


@pytest.mark.requires_impl
class TestBiquadFilterIntegrationImpl:
    """
    MATLAB counterpart: TestBiquadFilter (filterbank integration section).
    """

    def test_integration_with_filterbank(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import biquadfilter  # type: ignore
        rng   = np.random.default_rng(42)
        Ls    = 1024
        x     = rng.standard_normal(Ls)
        fc_v  = [0.1, 0.25, 0.4]
        g     = [biquadfilter(fc, 0.05) for fc in fc_v]
        a     = [4, 4, 4]
        c     = filterbank(x, g, a)
        assert len(c) == 3
        for k in range(3):
            assert c[k].shape[0] == int(np.ceil(Ls / a[k]))

    def test_zero_input_zero_output(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import biquadfilter  # type: ignore
        x = np.zeros(1024)
        g = [biquadfilter(0.25, 0.05)]
        c = filterbank(x, g, [2])
        assert np.max(np.abs(c[0])) < 1e-12


@pytest.mark.requires_impl
class TestCompBiquadImpl:
    """
    MATLAB counterpart: TestBiquadFilter (comp_biquad section).
    """

    @pytest.mark.parametrize("L", [32, 64, 512])
    def test_output_length(self, needs_impl, L):
        from cool_frames.numpy.filters._filters import comp_biquad  # type: ignore
        H = comp_biquad(0.9, np.pi / 4, L, "energy")
        assert len(H) == L

    def test_near_marginal_large_peak(self, needs_impl):
        """r close to 1 → very large response at the pole frequency."""
        from cool_frames.numpy.filters._filters import comp_biquad  # type: ignore
        r = 1 - 1e-6
        H = comp_biquad(r, np.pi / 4, 128, "none")
        assert np.max(np.abs(H)) > 100

    def test_stability_across_parameters(self, needs_impl):
        """For any r < 1, response must be finite."""
        from cool_frames.numpy.filters._filters import comp_biquad  # type: ignore
        rng = np.random.default_rng(42)
        for _ in range(50):
            r     = rng.random() * 0.99
            theta = rng.random() * np.pi
            H     = comp_biquad(r, theta, 128, "none")
            assert np.all(np.isfinite(H))

    def test_energy_normalization(self, needs_impl):
        """'energy': (1/L)*sum|H|^2 ≈ 1."""
        from cool_frames.numpy.filters._filters import comp_biquad  # type: ignore
        rng = np.random.default_rng(42)
        for _ in range(20):
            r     = 0.5 + rng.random() * 0.4
            theta = rng.random() * np.pi
            L     = 256
            H     = comp_biquad(r, theta, L, "energy")
            assert np.sum(np.abs(H) ** 2) / L == pytest.approx(1.0, abs=1e-6)
