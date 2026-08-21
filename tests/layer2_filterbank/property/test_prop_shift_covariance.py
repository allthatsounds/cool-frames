"""
test_prop_shift_covariance.py
==============================
Python port of:
    layer2_filterbank/property/PropShiftCovariance.m

For a filterbank with hop sizes a(m), a covariant shift of k*a(m) samples
in the input shifts subband m by exactly k samples:

    filterbank(circshift(x, k*a(m)), g, a)[m]
        == circshift(filterbank(x, g, a)[m], k)

IMPORTANT: signals must have length L (the DFT length from audfilters), not
Ls. The filterbank is implemented as circular convolution with period L.
"""

from __future__ import annotations

import pytest

import numpy as np


@pytest.mark.requires_impl
class TestShiftCovarianceImpl:
    """PropShiftCovariance: covariant shift on selected subbands."""

    def test_shift_covariance(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        a_arr = np.asarray(a).ravel()
        # Test first, middle, and last subband
        subband_indices = [0, M // 2, M - 1]
        rng = np.random.default_rng(1)
        for sig_type in ["real", "complex"]:
            if sig_type == "real":
                x = rng.standard_normal(L)
            else:
                x = rng.standard_normal(L) + 1j * rng.standard_normal(L)
            c_orig = filterbank(x, g, a)
            for k in [1, 2, 3]:
                for m in subband_indices:
                    shift_amount = k * int(a_arr[m])
                    x_shifted    = np.roll(x, shift_amount)
                    c_shift      = filterbank(x_shifted, g, a)
                    c_m_expected = np.roll(np.asarray(c_orig[m]).ravel(), k)
                    c_m_actual   = np.asarray(c_shift[m]).ravel()
                    rel_err      = np.linalg.norm(c_m_actual - c_m_expected) / \
                                   (np.linalg.norm(c_m_expected) + 1e-10)
                    assert rel_err < 1e-6, \
                        f"{sig_type}, k={k}, band={m}: shift covariance err {rel_err:.2e}"


@pytest.mark.requires_impl
class TestShiftCovarianceRandomImpl:
    """PropShiftCovariance: 100 random complex trials on first subband."""

    def test_shift_covariance_random(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        a_arr = np.asarray(a).ravel()
        m  = 0
        k  = 1
        shift_amount = k * int(a_arr[m])
        rng = np.random.default_rng(42)
        for trial in range(100):
            x        = rng.standard_normal(L) + 1j * rng.standard_normal(L)
            c_orig   = filterbank(x, g, a)
            c_shift  = filterbank(np.roll(x, shift_amount), g, a)
            expected = np.roll(np.asarray(c_orig[m]).ravel(), k)
            actual   = np.asarray(c_shift[m]).ravel()
            rel_err  = np.linalg.norm(actual - expected) / \
                       (np.linalg.norm(expected) + 1e-10)
            assert rel_err < 1e-6, \
                f"Trial {trial}: shift covariance error {rel_err:.2e}"


@pytest.mark.requires_impl
class TestShiftCovarianceAllBandsImpl:
    """PropShiftCovariance: k=1,2,3 on every subband (single signal)."""

    def test_all_bands_multiple_shifts(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        a_arr = np.asarray(a).ravel()
        rng = np.random.default_rng(42)
        x      = rng.standard_normal(L) + 1j * rng.standard_normal(L)
        c_orig = filterbank(x, g, a)
        for k in [1, 2, 3]:
            for m in range(M):
                shift_amount = k * int(a_arr[m])
                c_shift      = filterbank(np.roll(x, shift_amount), g, a)
                expected     = np.roll(np.asarray(c_orig[m]).ravel(), k)
                actual       = np.asarray(c_shift[m]).ravel()
                rel_err      = np.linalg.norm(actual - expected) / \
                               (np.linalg.norm(expected) + 1e-10)
                assert rel_err < 1e-6, \
                    f"k={k}, band={m}: shift covariance error {rel_err:.2e}"
