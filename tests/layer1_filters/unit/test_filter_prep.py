"""
test_filter_prep.py
===================
Python port of:
    layer1_filters/unit/TestFilterPrep.m

Covers: comp_transferfunction, comp_filterbank_pre

comp_transferfunction(g, L)
    Evaluates the full-length L-point frequency response of a filter struct.

comp_filterbank_pre(g_cell, a, L, crossover=0)
    Evaluates all function handles in a filter cell array, applies modulations,
    and canonicalises each filter to numeric form.
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import postpad_ref

# ---------------------------------------------------------------------------
# comp_transferfunction – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCompTransferFunctionLengthImpl:
    """
    MATLAB counterpart: TestFilterPrep (comp_transferfunction length section).
    """

    @pytest.mark.parametrize("L", [64, 256, 1024])
    def test_length_from_blfilter(self, needs_impl, L):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = blfilter("hann", 0.1, 0.3)
        assert len(comp_transferfunction(g, L)) == L

    @pytest.mark.parametrize("L", [64, 256, 512])
    def test_length_from_firfilter(self, needs_impl, L):
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = firfilter("hann", 32)
        assert len(comp_transferfunction(g, L)) == L

    def test_length_from_freqfilter(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = freqfilter("gauss", 0.1, 0.3)
        assert len(comp_transferfunction(g, 512)) == 512

    def test_length_from_biquadfilter(self, needs_impl):
        from cool_frames.filters import biquadfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g = biquadfilter(0.25, 0.05)
        assert len(comp_transferfunction(g, 256)) == 256


@pytest.mark.requires_impl
class TestCompTransferFunctionConsistencyImpl:
    """
    MATLAB counterpart: TestFilterPrep (consistency section).
    """

    def test_firfilter_matches_manual_fft(self, needs_impl):
        """comp_transferfunction == fft(roll(postpad(h, L), offset))."""
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 512
        g = firfilter("hann", 64)
        H_pre = comp_transferfunction(g, L)
        H_man = np.fft.fft(np.roll(postpad_ref(np.asarray(g["h"]), L), g["offset"]))
        np.testing.assert_allclose(H_pre, H_man, atol=1e-10)

    def test_biquadfilter_matches_direct_H(self, needs_impl):
        """comp_transferfunction == g.H(L) for biquadfilter."""
        from cool_frames.filters import biquadfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 512
        g = biquadfilter(0.3, 0.04, "peak")
        np.testing.assert_allclose(comp_transferfunction(g, L), g["H"](L), atol=1e-10)


@pytest.mark.requires_impl
class TestCompTransferFunctionRealonlyImpl:
    """
    MATLAB counterpart: TestFilterPrep (realonly symmetry section).

    NOTE (2026-06-13): these assert the LTFAT-faithful behaviour where a
    ``realonly`` filter's transfer function is Hermitian-symmetrised
    (``(H+involute)/2``). cool_frames deliberately diverges: ``filter_freqresp`` (hence
    ``comp_transferfunction``) reports the stored single-sided response and does
    NOT mirror ``realonly``, because the transform kernels ignore the flag and
    real reconstruction is done at synthesis via ``2*real(ifft)``. This is what
    made the canonical tight CQT frame read kappa=1. See
    research/companion/ltfat_divergences.tex. Marked xfail accordingly.
    """

    @pytest.mark.xfail(reason="cool_frames: realonly handled at synthesis, not mirrored "
                              "in comp_transferfunction (LTFAT divergence)")
    def test_realonly_enforces_hermitian_symmetry(self, needs_impl):
        """realonly=1: H[1:] == conj(H[1:][::-1]) (Hermitian)."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 256
        g  = blfilter("hann", 0.1, 0.3, "real")
        assert g["realonly"] == 1
        H  = comp_transferfunction(g, L)
        np.testing.assert_allclose(H[1:], np.conj(H[1:][::-1]), atol=1e-10)

    @pytest.mark.xfail(reason="cool_frames: realonly handled at synthesis, not mirrored "
                              "in comp_transferfunction (LTFAT divergence)")
    def test_realonly_time_domain_is_real(self, needs_impl):
        """Hermitian H → real IFFT."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 256
        g = blfilter("hann", 0.1, 0.3, "real")
        H = comp_transferfunction(g, L)
        h = np.fft.ifft(H)
        assert np.max(np.abs(np.imag(h))) < 1e-10


@pytest.mark.requires_impl
class TestCompTransferFunctionDelayImpl:
    """
    MATLAB counterpart: TestFilterPrep (delay property section).
    """

    def test_delay_shifts_phase(self, needs_impl):
        """delay d: H_delayed(k) = H(k) * exp(-2πi*d*k/L)."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L = 256
        d = 3
        g0 = blfilter("hann", 0.1, 0.3)
        gd = blfilter("hann", 0.1, 0.3, "delay", d)
        H0 = comp_transferfunction(g0, L)
        Hd = comp_transferfunction(gd, L)
        k  = np.arange(L)
        np.testing.assert_allclose(Hd, H0 * np.exp(-2j * np.pi * d * k / L), atol=1e-10)


# ---------------------------------------------------------------------------
# comp_filterbank_pre – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCompFilterbankPreImpl:
    """
    MATLAB counterpart: TestFilterPrep (comp_filterbank_pre section).
    """

    def test_pre_makes_H_numeric(self, needs_impl):
        """After comp_filterbank_pre, g['H'] must be numeric, not callable."""
        from cool_frames.filters.lowlevel import blfilter, freqfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_filterbank_pre  # type: ignore
        g_cell = [blfilter("hann", 0.1, 0.3), freqfilter("gauss", 0.1, 0.2)]
        a      = np.array([4, 4])
        L      = 512
        g_pre  = comp_filterbank_pre(g_cell, a, L, 0)
        for m, gp in enumerate(g_pre):
            assert not callable(gp["H"]), f"Filter {m}: H should be numeric after pre"

    def test_pre_preserves_positive_length(self, needs_impl):
        """After pre, g['H'] has positive length."""
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_filterbank_pre  # type: ignore
        g_cell = [blfilter("hann", 0.1, 0.3)]
        g_pre  = comp_filterbank_pre(g_cell, np.array([8]), 512, 0)
        assert not callable(g_pre[0]["H"])
        assert len(np.asarray(g_pre[0]["H"])) > 0

    def test_pre_idempotent_on_firfilter(self, needs_impl):
        """Running comp_filterbank_pre twice gives the same H."""
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_filterbank_pre  # type: ignore
        g_cell = [firfilter("hann", 32)]
        a      = np.array([4])
        L      = 256
        g_pre1 = comp_filterbank_pre(g_cell, a, L, 0)
        g_pre2 = comp_filterbank_pre(g_pre1, a, L, 0)
        np.testing.assert_allclose(np.asarray(g_pre2[0]["H"]),
                                   np.asarray(g_pre1[0]["H"]), atol=1e-10)

    def test_pre_firfilter_matches_comp_transferfunction(self, needs_impl):
        """comp_filterbank_pre + comp_transferfunction == direct comp_transferfunction."""
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        from cool_frames.numpy.filters._filters import comp_filterbank_pre, comp_transferfunction  # type: ignore
        L      = 256
        g0     = firfilter("hann", 32)
        g_pre  = comp_filterbank_pre([g0], np.array([4]), L, 0)
        H_tf   = comp_transferfunction(g0,      L)
        H_pre  = comp_transferfunction(g_pre[0], L)
        np.testing.assert_allclose(H_pre, H_tf, atol=1e-10)
