"""
test_phasederiv.py
==================
Unit tests for second-order phase derivatives (filterbankphasederiv).

Tests are split into:
  1. Structural tests (pure numpy, no cool_frames install needed)
     — test the computation functions directly with synthetic data
  2. Integration tests (require cool_frames) — marked @requires_impl
     — test full pipeline with real filterbank designs

Structural tests
----------------
TestCompPhasederivFilters2nd:
    Filter construction: correct number of filters, callable H

TestCompFilterbankphasederiv:
    Core computation: output shapes, real-valued, known-signal properties

Integration tests
-----------------
TestFilterbankphasederivImpl:
    Full pipeline with audfilters: output structure, real-valued,
    pure tone derivatives ≈ 0, multiple derivatives at once
"""
from __future__ import annotations

import pathlib
import sys

import pytest

import numpy as np

_pkg = pathlib.Path(__file__).resolve().parents[3] / "cool_frames"
if str(_pkg.parent) not in sys.path:
    sys.path.insert(0, str(_pkg.parent))

from cool_frames.numpy.phase._phasederiv import (
    comp_filterbankphasederiv,
    comp_phasederivfilters_2nd,
)

# ---------------------------------------------------------------------------
# Helpers: build a minimal synthetic filterbank for structural tests
# ---------------------------------------------------------------------------

def _synth_filterbank(L: int = 256, M: int = 8):
    """Build a trivial box-filter filterbank for structural tests."""
    bin_width = L // M
    g = []
    for m in range(M):
        H = np.zeros(L, dtype=complex)
        lo = m * bin_width
        hi = (m + 1) * bin_width
        H[lo:hi] = 1.0
        g.append({"H": lambda L_, _H=H.copy(): _H, "foff": lambda L_, _lo=lo: _lo})
    a = np.ones((M, 2), dtype=int)
    return g, a, L, M


def _synth_coefficients(M: int, N: int, seed: int = 42):
    """Generate random complex coefficient lists for M channels, each length N."""
    rng = np.random.default_rng(seed)
    c = [rng.standard_normal(N) + 1j * rng.standard_normal(N) for _ in range(M)]
    return c


# ---------------------------------------------------------------------------
# TestCompPhasederivFilters2nd
# ---------------------------------------------------------------------------

class TestCompPhasederivFilters2nd:
    """Tests for the second-order derivative filter builder."""

    def test_returns_five_lists(self):
        g, a, L, M = _synth_filterbank()
        ch, cd, cd2, ch2, chd = comp_phasederivfilters_2nd(g, a, L)
        assert len(ch) == M
        assert len(cd) == M
        assert len(cd2) == M
        assert len(ch2) == M
        assert len(chd) == M

    def test_filters_are_callable(self):
        g, a, L, M = _synth_filterbank()
        _, _, cd2, ch2, chd = comp_phasederivfilters_2nd(g, a, L)
        for m in range(M):
            # Each filter dict should have callable H
            assert callable(cd2[m]["H"])
            assert callable(ch2[m]["H"])
            assert callable(chd[m]["H"])

    def test_filter_response_length(self):
        g, a, L, M = _synth_filterbank()
        _, _, cd2, ch2, chd = comp_phasederivfilters_2nd(g, a, L)
        for m in range(M):
            h_cd2 = cd2[m]["H"](L)
            h_ch2 = ch2[m]["H"](L)
            h_chd = chd[m]["H"](L)
            # Should be same length as original filter
            h_orig = g[m]["H"](L)
            assert len(h_cd2) == len(h_orig)
            assert len(h_ch2) == len(h_orig)
            assert len(h_chd) == len(h_orig)

    def test_fir_filters(self):
        """Test with FIR (time-domain) filters."""
        L = 256
        M = 4
        g = []
        for m in range(M):
            h = np.random.default_rng(m).standard_normal(16)
            g.append({"h": h})
        a = np.ones((M, 2), dtype=int)
        ch, cd, cd2, ch2, chd = comp_phasederivfilters_2nd(g, a, L)
        for m in range(M):
            assert "h" in cd2[m]
            assert "h" in ch2[m]
            assert "h" in chd[m]
            assert len(cd2[m]["h"]) == len(g[m]["h"])


# ---------------------------------------------------------------------------
# TestCompFilterbankphasederiv
# ---------------------------------------------------------------------------

class TestCompFilterbankphasederiv:
    """Tests for the core second-order derivative computation."""

    def test_output_structure(self):
        """Result dict has correct keys and each value has M arrays."""
        M, N = 8, 32
        c = _synth_coefficients(M, N, seed=1)
        # Use same coefficients for all derivative inputs (wrong values, but tests structure)
        result = comp_filterbankphasederiv(
            c, c, c, c, c, c, L=256, derivs=("tt", "ff", "tf"),
        )
        assert set(result.keys()) == {"tt", "ff", "tf"}
        for key in ("tt", "ff", "tf"):
            assert len(result[key]) == M

    def test_output_shapes(self):
        """Each output array has same length as corresponding input."""
        M = 5
        lengths = [100, 80, 60, 40, 20]
        rng = np.random.default_rng(42)
        c = [rng.standard_normal(n) + 1j * rng.standard_normal(n) for n in lengths]
        result = comp_filterbankphasederiv(
            c, c, c, c, c, c, L=512, derivs=("tt", "ff", "tf"),
        )
        for key in ("tt", "ff", "tf"):
            for m in range(M):
                assert len(result[key][m]) == lengths[m]

    def test_outputs_are_real(self):
        """Second-order derivatives must be real-valued."""
        M, N = 8, 32
        c = _synth_coefficients(M, N, seed=2)
        cd = _synth_coefficients(M, N, seed=3)
        ch = _synth_coefficients(M, N, seed=4)
        cd2 = _synth_coefficients(M, N, seed=5)
        ch2 = _synth_coefficients(M, N, seed=6)
        chd = _synth_coefficients(M, N, seed=7)
        result = comp_filterbankphasederiv(
            c, cd, ch, cd2, ch2, chd, L=256, derivs=("tt", "ff", "tf"),
        )
        for key in ("tt", "ff", "tf"):
            for m in range(M):
                assert np.isrealobj(result[key][m]), \
                    f"{key}[{m}] should be real"

    def test_subset_of_derivs(self):
        """Requesting only 'tt' should only compute 'tt'."""
        M, N = 4, 16
        c = _synth_coefficients(M, N, seed=8)
        result = comp_filterbankphasederiv(
            c, c, c, c, c, c, L=128, derivs=("tt",),
        )
        assert "tt" in result
        assert "ff" not in result
        assert "tf" not in result


# ---------------------------------------------------------------------------
# Integration tests (require cool_frames)
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbankphasederivImpl:
    """Full pipeline tests requiring cool_frames."""

    def test_output_structure(self, needs_impl):
        """Returns (dict, list) with correct keys and M channels."""
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.phase import filterbankphasederiv

        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(42)
        f = rng.standard_normal(Ls)

        result, c = filterbankphasederiv(f, g, a, derivs=["tt", "ff", "tf"], L=L)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"tt", "ff", "tf"}
        assert len(c) == M
        for key in ("tt", "ff", "tf"):
            assert len(result[key]) == M

    def test_cell_sizes_match_filterbank(self, needs_impl):
        """Derivative arrays must have same length as filterbank coefficients."""
        from cool_frames.filterbanks import filterbank as fb
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.phase import filterbankphasederiv

        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(43)
        f = rng.standard_normal(Ls)
        c_ref = fb(f, g, a)

        result, _ = filterbankphasederiv(f, g, a, derivs=["tt", "ff", "tf"], L=L)

        for key in ("tt", "ff", "tf"):
            for m in range(M):
                assert np.asarray(result[key][m]).shape == np.asarray(c_ref[m]).shape, \
                    f"{key}[{m}] shape mismatch"

    def test_derivatives_are_real(self, needs_impl):
        """All second-order derivatives must be real-valued."""
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.phase import filterbankphasederiv

        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(44)
        f = rng.standard_normal(Ls)

        result, _ = filterbankphasederiv(f, g, a, derivs=["tt", "ff", "tf"], L=L)

        for key in ("tt", "ff", "tf"):
            for m in range(M):
                assert np.isrealobj(np.asarray(result[key][m])), \
                    f"{key}[{m}] must be real"

    def test_single_deriv_string(self, needs_impl):
        """Passing a single string instead of list should work."""
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.phase import filterbankphasederiv

        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(45)
        f = rng.standard_normal(Ls)

        result, c = filterbankphasederiv(f, g, a, derivs="tt", L=L)
        assert "tt" in result
        assert len(result["tt"]) == len(g)

    def test_invalid_deriv_raises(self, needs_impl):
        """Unknown derivative name should raise ValueError."""
        from cool_frames.filters import audfilters
        from cool_frames.phase import filterbankphasederiv

        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        rng = np.random.default_rng(46)
        f = rng.standard_normal(Ls)

        with pytest.raises(ValueError, match="Unknown derivative"):
            filterbankphasederiv(f, g, a, derivs="xxx")

    def test_pure_tone_tt_near_zero(self, needs_impl):
        """For a pure tone, tt (chirp rate) should be near zero in high-energy bins."""
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.phase import filterbankphasederiv

        Ls, fs = 2048, 16000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)

        # 1 kHz pure tone
        t = np.arange(Ls) / fs
        f = np.sin(2 * np.pi * 1000 * t)

        result, c = filterbankphasederiv(f, g, a, derivs="tt", L=L)

        # Find channel closest to 1 kHz
        fc_hz = np.asarray(fc) * fs
        best_ch = int(np.argmin(np.abs(fc_hz - 1000)))

        # In the dominant channel, tt should be near zero
        cm = np.asarray(c[best_ch])
        sm = np.abs(cm) ** 2
        mask = sm > 0.1 * np.max(sm)
        if np.any(mask):
            tt_vals = np.asarray(result["tt"][best_ch])[mask]
            # Chirp rate of a pure tone is 0
            median_tt = np.median(np.abs(tt_vals))
            assert median_tt < 5.0, \
                f"Pure tone tt should be ≈0, got median |tt|={median_tt:.2f}"

    def test_works_without_L(self, needs_impl):
        """Omitting L should not raise."""
        from cool_frames.filters import audfilters
        from cool_frames.phase import filterbankphasederiv

        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        rng = np.random.default_rng(47)
        f = rng.standard_normal(Ls)

        try:
            result, c = filterbankphasederiv(f, g, a, derivs=["tt", "ff", "tf"])
        except Exception as exc:
            pytest.fail(f"filterbankphasederiv without L raised: {exc}")
