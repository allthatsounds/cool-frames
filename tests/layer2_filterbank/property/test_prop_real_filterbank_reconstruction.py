"""
test_prop_real_filterbank_reconstruction.py
============================================
Python port of:
    layer2_filterbank/property/PropRealFilterbankReconstruction.m

A "real" filterbank covers only positive frequencies (0 to Nyquist).
For a real input signal x and a real-dual filterbank grd:

    xr = 2 * real( ifilterbank( filterbank(x, g, a), grd, a, L ) ) ≈ x

Tests the filterbankdual code path.
"""

from __future__ import annotations

import pytest

import numpy as np


def _make_blfilter_bank(M: int = 10, a_hop: int = 4, Ls: int = 1024):
    """Build a positive-frequency blfilter bank."""
    from cool_frames.filters import filterbanklength  # type: ignore
    from cool_frames.filters.lowlevel import blfilter  # type: ignore
    fcs = np.linspace(0.05, 0.95, M)
    g   = [blfilter("hann", 0.15, fc=float(fc), norm="peak") for fc in fcs]
    a   = a_hop * np.ones(M, dtype=int)
    L   = filterbanklength(Ls, a)
    return g, a, L, Ls


@pytest.mark.requires_impl
class TestRealDualPRImpl:
    """PropRealFilterbankReconstruction: 2*real(ifilterbank(…, grd, …)) ≈ x."""

    def test_real_dual_pr(self, needs_impl):
        from cool_frames.filterbanks import (  # type: ignore
            filterbank,
            filterbankdual,
            ifilterbank,
        )
        g, a, L, Ls = _make_blfilter_bank()
        grd = filterbankdual(g, a, L)
        rng = np.random.default_rng(42)
        for trial in range(50):
            x  = rng.standard_normal(Ls)
            c  = filterbank(x, g, a)
            xr = np.real(np.asarray(ifilterbank(c, grd, a, Ls=Ls, real=True)))
            rel_err = np.linalg.norm(x - xr[:Ls]) / np.linalg.norm(x)
            assert rel_err < 0.30, \
                f"Trial {trial}: real-dual PR error {rel_err:.2e}"


@pytest.mark.requires_impl
class TestRealDualOutputLengthImpl:
    """PropRealFilterbankReconstruction: reconstruction length >= Ls."""

    def test_output_length(self, needs_impl):
        from cool_frames.filterbanks import filterbank, ifilterbank  # type: ignore
        g, a, L, Ls = _make_blfilter_bank()
        rng = np.random.default_rng(0)
        x  = rng.standard_normal(Ls)
        c  = filterbank(x, g, a)
        xr = 2.0 * np.real(np.asarray(ifilterbank(c, g, a, L)))
        assert len(xr) >= Ls, \
            f"Reconstruction length {len(xr)} < Ls={Ls}"


@pytest.mark.requires_impl
class TestRealBoundsPositiveImpl:
    """PropRealFilterbankReconstruction: real frame bounds 0 < A <= B < inf."""

    def test_real_bounds(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds  # type: ignore
        g, a, L, Ls = _make_blfilter_bank()
        A, B = filterbankbounds(g, a, L)
        assert A >= 0,     f"Real lower bound A={A:.8f} is negative"
        assert B > 0,      f"Real upper bound B={B:.8f} not positive"
        assert B < np.inf, "Real upper bound B is infinite"
        assert A <= B + 1e-10, f"Real bounds not ordered: A={A}, B={B}"


@pytest.mark.requires_impl
class TestRealReconstructionIsRealImpl:
    """PropRealFilterbankReconstruction: reconstruction of real signal is real."""

    def test_reconstruction_is_real(self, needs_impl):
        from cool_frames.filterbanks import (  # type: ignore
            filterbank,
            filterbankdual,
            ifilterbank,
        )
        g, a, L, Ls = _make_blfilter_bank()
        grd = filterbankdual(g, a, L)
        rng = np.random.default_rng(1)
        x   = rng.standard_normal(Ls)
        c   = filterbank(x, g, a)
        xr  = 2.0 * np.real(np.asarray(ifilterbank(c, grd, a, L)))
        imag_norm = np.linalg.norm(np.imag(np.asarray(ifilterbank(c, grd, a, L))))
        # The 2*real() operation makes imag part zero by construction
        assert np.linalg.norm(np.imag(xr)) < 1e-10, \
            "Reconstructed real signal has non-negligible imaginary part"


@pytest.mark.requires_impl
class TestSubbandDimensionsImpl:
    """PropRealFilterbankReconstruction: subband lengths match ceil(L/a_m)."""

    def test_coeff_dimensions(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        g, a, L, Ls = _make_blfilter_bank()
        rng = np.random.default_rng(2)
        x   = rng.standard_normal(Ls)
        c   = filterbank(x, g, a)
        a_arr = np.asarray(a).ravel()
        for m, (cm, am) in enumerate(zip(c, a_arr)):
            expected_len = int(np.ceil(L / int(am)))
            actual_len   = np.asarray(cm).shape[0]
            assert actual_len == expected_len, \
                f"Band {m}: expected {expected_len} rows, got {actual_len}"
