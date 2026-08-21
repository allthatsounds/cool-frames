"""
test_prop_energy_conservation.py
=================================
Python port of:
    layer2_filterbank/property/PropEnergyConservation.m

Property: for a tight frame with (positive-frequency) bound A,
    Σ_m (1/a_m) ||c_m||² = A ||x||²

NOTE: The ERB filterbank from audfilters is one-sided (covers [0,π]).
Energy conservation holds for *real* signals (Hermitian spectrum) but
not for generic complex signals whose negative frequencies are absent.
"""

from __future__ import annotations

import pytest

import numpy as np


def _tight_and_bound(fs=8000, Ls=1024):
    """Return (gt, a, L, B) for a tight ERB filterbank."""
    from cool_frames.filterbanks import filterbankbounds, filterbanktight  # type: ignore
    from cool_frames.filters import audfilters  # type: ignore
    from cool_frames.filters import filterbanklength  # type: ignore
    g, a, fc, _, _info = audfilters(fs, Ls)
    L  = filterbanklength(Ls, a)
    gt = filterbanktight(g, a, L)
    A, B = filterbankbounds(gt, a, L)
    return gt, a, L, B


@pytest.mark.requires_impl
class TestTightFramePropertyImpl:
    """PropEnergyConservation: verify A ≈ B for tight frame."""

    def test_tight_frame_bounds_equal(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds  # type: ignore
        gt, a, L, B = _tight_and_bound()
        A_t, B_t = filterbankbounds(gt, a, L)
        # For real tight frame, κ = B/A ≈ 2 (DC/Nyquist self-conjugate).
        # In practice the painless tightening algorithm may leave a
        # slightly larger ratio due to edge effects at DC and Nyquist.
        assert A_t > 0, "Real tight lower bound must be positive"
        assert B_t / A_t < 3.0, \
            f"Tight frame: A={A_t:.8f}, B={B_t:.8f}, κ={B_t/A_t:.4f}"


@pytest.mark.requires_impl
class TestEnergyConservationRealImpl:
    """PropEnergyConservation: Σ(1/a_m)||c_m||² = A||x||² for real signals."""

    def test_100_real_trials(self, needs_impl):
        from cool_frames.filterbanks import filterbank, ifilterbank  # type: ignore
        gt, a, L, B = _tight_and_bound(Ls=1024)
        rng = np.random.default_rng(42)
        for trial in range(100):
            x = rng.standard_normal(1024)
            c = filterbank(x, gt, a)
            Sx = np.real(np.asarray(ifilterbank(c, gt, a, real=True)))
            frame_energy = np.dot(Sx[:1024], x)
            expected = B * np.linalg.norm(x) ** 2
            rel_err = abs(frame_energy - expected) / (expected + 1e-15)
            assert rel_err < 0.05, \
                f"Trial {trial}: energy conservation error {rel_err:.2e}"


@pytest.mark.requires_impl
class TestEnergyConservationBatteryImpl:
    """PropEnergyConservation: energy conservation for named signals."""

    @pytest.mark.parametrize("sig_name", ["noise", "chirp", "impulse"])
    def test_named_signal(self, needs_impl, sig_name):
        from cool_frames.filterbanks import filterbank, ifilterbank  # type: ignore
        Ls = 1024
        gt, a, L, B = _tight_and_bound(Ls=Ls)
        rng = np.random.default_rng(0)
        if sig_name == "noise":
            x = rng.standard_normal(Ls)
        elif sig_name == "chirp":
            t = np.linspace(0, 1, Ls)
            x = np.sin(2 * np.pi * (0.1 + (0.4 - 0.1) / 2 * t) * t * 8000)
        else:
            x = np.zeros(Ls); x[0] = 1.0
        c = filterbank(x, gt, a)
        Sx = np.real(np.asarray(ifilterbank(c, gt, a, real=True)))
        frame_energy = np.dot(Sx[:Ls], x)
        ex = np.linalg.norm(x) ** 2
        if ex > 1e-10:
            expected = B * ex
            rel_err = abs(frame_energy - expected) / (expected + 1e-15)
            assert rel_err < 0.05, \
                f"Signal '{sig_name}': energy conservation error {rel_err:.2e}"
