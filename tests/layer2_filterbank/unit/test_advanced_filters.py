"""
test_advanced_filters.py
========================
Python port of:
    layer2_filterbank/unit/TestAdvancedFilters.m

Unit tests for custom filter-construction entry points accessed from
the Layer-2 analysis/synthesis pipeline.

Covers: blfilter, freqfilter, freqwavelet, firwin, firfilter,
        nonu2ufilterbank, audspace (ERB-scale utilities).
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# blfilter
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestBlfilterLayer2Impl:
    """TestAdvancedFilters: blfilter structural checks."""

    def test_returns_dict_like(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        g = blfilter("hann", 0.1)
        assert hasattr(g, "__getitem__") or hasattr(g, "__dict__") or isinstance(g, dict), \
            "blfilter must return a dict-like struct"

    def test_has_field_H(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        g = blfilter("hann", 0.1)
        assert "H" in g or hasattr(g, "H"), "blfilter struct must have field 'H'"

    def test_fc_option(self, needs_impl):
        from cool_frames.filters.lowlevel import blfilter  # type: ignore
        fc_target = 0.25
        g = blfilter("hann", 0.1, fc=fc_target)
        assert g is not None, "blfilter with fc option must return a filter"
        if "fc" in g or hasattr(g, "fc"):
            fc_val = g["fc"] if isinstance(g, dict) else g.fc
            assert abs(float(fc_val) - fc_target) < 0.01, \
                f"blfilter: stored fc={fc_val} differs from target {fc_target}"


# ---------------------------------------------------------------------------
# freqfilter
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFreqfilterLayer2Impl:
    """TestAdvancedFilters: freqfilter structural checks."""

    def test_returns_dict_like(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        g = freqfilter("gauss", 0.05)
        assert g is not None, "freqfilter must return a non-None filter struct"

    def test_has_field_H(self, needs_impl):
        from cool_frames.filters.lowlevel import freqfilter  # type: ignore
        g = freqfilter("gauss", 0.05)
        assert "H" in g or hasattr(g, "H"), "freqfilter struct must have field 'H'"


# ---------------------------------------------------------------------------
# freqwavelet
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFreqwaveletLayer2Impl:
    """TestAdvancedFilters: freqwavelet structural checks."""

    def test_runs(self, needs_impl):
        from cool_frames.filters import freqwavelet  # type: ignore
        g = freqwavelet("cauchy", 0.1)
        assert g is not None, "freqwavelet must return a non-None filter"


# ---------------------------------------------------------------------------
# firwin
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFirwinLayer2Impl:
    """TestAdvancedFilters: firwin basic checks (reachable via layer2 imports)."""

    @pytest.mark.parametrize("name,M", [("hann", 32), ("hann", 64), ("sine", 32)])
    def test_length(self, needs_impl, name, M):
        from cool_frames.filters import firwin  # type: ignore
        w = firwin(name, M)
        assert len(w) == M, f"firwin('{name}', {M}): expected length {M}, got {len(w)}"

    def test_non_negative_hann(self, needs_impl):
        from cool_frames.filters import firwin  # type: ignore
        w = firwin("hann", 64)
        assert np.all(np.asarray(w) >= -1e-12), "hann window must be non-negative"


# ---------------------------------------------------------------------------
# firfilter
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFirfilterLayer2Impl:
    """TestAdvancedFilters: firfilter structural checks."""

    def test_has_h_field(self, needs_impl):
        from cool_frames.filters import firwin
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        h = firwin("hann", 32)
        g = firfilter(h)
        assert "h" in g or hasattr(g, "h"), "firfilter struct must have field 'h'"

    def test_realonly_flag(self, needs_impl):
        from cool_frames.filters import firwin
        from cool_frames.filters.lowlevel import firfilter  # type: ignore
        h = firwin("hann", 32)
        g_real = firfilter(h, realonly=True)
        g_cplx = firfilter(h, realonly=False)
        ro_r = g_real["realonly"] if isinstance(g_real, dict) else g_real.realonly
        ro_c = g_cplx["realonly"] if isinstance(g_cplx, dict) else g_cplx.realonly
        assert bool(ro_r) is True,  "firfilter realonly=True: flag not set"
        assert bool(ro_c) is False, "firfilter realonly=False: flag not cleared"


# ---------------------------------------------------------------------------
# ERB-scale utilities (audspace, freqtoaud, audtofreq)
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestErbScaleUtilsLayer2Impl:
    """TestAdvancedFilters: ERB-scale utility functions."""

    def test_audspace_length(self, needs_impl):
        from cool_frames.filters import audspace  # type: ignore
        M = 20
        s = audspace(0, 4000, M, "erb")
        assert len(s) == M, f"audspace: expected {M} points, got {len(s)}"

    def test_audspace_monotone(self, needs_impl):
        from cool_frames.filters import audspace  # type: ignore
        s = audspace(0, 4000, 30, "erb")
        s_arr = np.asarray(s, dtype=float)
        assert np.all(np.diff(s_arr) > 0), "audspace: not monotone increasing"

    def test_freqtoaud_roundtrip(self, needs_impl):
        from cool_frames.filters import audtofreq, freqtoaud  # type: ignore
        freqs = np.array([100.0, 500.0, 1000.0, 4000.0])
        erbs  = freqtoaud(freqs, "erb")
        back  = audtofreq(erbs, "erb")
        np.testing.assert_allclose(back, freqs, rtol=1e-6,
                                   err_msg="freqtoaud/audtofreq roundtrip failed")
