"""
test_prop_ufilterbank_equivalence.py
=====================================
Python port of:
    layer2_filterbank/property/PropUfilterbankEquivalence.m

For a uniform (scalar) subsampling factor a, ufilterbank and filterbank
must produce identical coefficients:

    filterbank(x, g, a, stack=True)[:, m] == filterbank(x, g, a_vec)[m]

where a_vec = a * ones(M).
"""

from __future__ import annotations

import pytest

import numpy as np


def _make_bl_bank(M: int = 8, hop: int = 4, bw: float = 0.15, Ls: int = 1024):
    from cool_frames.filters import filterbanklength  # type: ignore
    from cool_frames.filters.lowlevel import blfilter  # type: ignore
    fcs   = np.linspace(0.1, 0.9, M)
    g     = [blfilter("hann", bw, fc=float(fc), norm="peak") for fc in fcs]
    a_vec = hop * np.ones(M, dtype=int)
    L     = filterbanklength(Ls, a_vec)
    return g, a_vec, hop, L


@pytest.mark.requires_impl
class TestUfilterbankEquivalenceRealImpl:
    """PropUfilterbankEquivalence: real signal."""

    def test_equivalence_real(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        g, a_vec, a_scalar, L = _make_bl_bank()
        rng = np.random.default_rng(0)
        x   = rng.standard_normal(L)
        c_cell   = filterbank(x, g, a_vec)
        c_matrix = np.asarray(filterbank(x, g, a_scalar, stack=True))
        M = len(g)
        for m in range(M):
            err = np.linalg.norm(np.asarray(c_cell[m]).ravel() - c_matrix[:, m]) / \
                  (np.linalg.norm(c_cell[m]) + 1e-15)
            assert err < 1e-12, \
                f"Band {m} (real): ufilterbank/filterbank mismatch {err:.2e}"


@pytest.mark.requires_impl
class TestUfilterbankEquivalenceComplexImpl:
    """PropUfilterbankEquivalence: complex signal."""

    def test_equivalence_complex(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        g, a_vec, a_scalar, L = _make_bl_bank(M=6, hop=8, bw=0.12)
        rng = np.random.default_rng(1)
        x   = rng.standard_normal(L) + 1j * rng.standard_normal(L)
        c_cell   = filterbank(x, g, a_vec)
        c_matrix = np.asarray(filterbank(x, g, a_scalar, stack=True))
        M = len(g)
        for m in range(M):
            err = np.linalg.norm(np.asarray(c_cell[m]).ravel() - c_matrix[:, m]) / \
                  (np.linalg.norm(c_cell[m]) + 1e-15)
            assert err < 1e-12, \
                f"Band {m} (complex): ufilterbank/filterbank mismatch {err:.2e}"


@pytest.mark.requires_impl
class TestUfilterbankOutputDimensionsImpl:
    """PropUfilterbankEquivalence: ufilterbank output has shape (L/a, M)."""

    def test_output_dimensions(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        M, a = 5, 4
        Ls   = 1024
        fcs  = np.linspace(0.1, 0.9, M)
        g    = [blfilter("hann", 0.15, fc=float(fc), norm="peak") for fc in fcs]
        L    = filterbanklength(Ls, a)
        rng  = np.random.default_rng(2)
        x    = rng.standard_normal(L)
        c    = np.asarray(filterbank(x, g, a, stack=True))
        assert c.shape[0] == L // a, \
            f"ufilterbank rows: expected {L//a}, got {c.shape[0]}"
        assert c.shape[1] == M, \
            f"ufilterbank cols: expected {M}, got {c.shape[1]}"


@pytest.mark.requires_impl
class TestUfilterbankAudfiltersImpl:
    """PropUfilterbankEquivalence: audfilters uniform-hop channels."""

    def test_audfilters_uniform_channels(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a_full, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a_full)
        a_arr    = np.asarray(a_full).ravel()
        a_scalar = int(a_arr[0])
        # Restrict to channels sharing the same hop size
        idx = np.where(a_arr == a_scalar)[0]
        g_sub   = [g[i] for i in idx]
        a_vec   = a_scalar * np.ones(len(g_sub), dtype=int)
        rng     = np.random.default_rng(3)
        x       = rng.standard_normal(L)
        c_cell  = filterbank(x, g_sub, a_vec)
        c_mat   = np.asarray(filterbank(x, g_sub, a_scalar, stack=True))
        for m in range(len(g_sub)):
            err = np.linalg.norm(np.asarray(c_cell[m]).ravel() - c_mat[:, m]) / \
                  (np.linalg.norm(c_cell[m]) + 1e-15)
            assert err < 1e-12, \
                f"audfilters band {m}: ufilterbank/filterbank mismatch {err:.2e}"
