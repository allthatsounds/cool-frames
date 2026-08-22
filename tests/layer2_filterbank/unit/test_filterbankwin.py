"""
test_filterbankwin.py
=====================
Unit tests for filterbankwin – the filter-bank window evaluator/validator.

Python port of:
    layer2_filterbank/unit/TestFilterbankWin.m

filterbankwin evaluates a filterbank specification and returns a normalised
representation together with an info dict.

API under test
--------------
    filterbankwin(g, a, L=None) -> (g_out, a_out, info)

    g   : list of filter dicts  OR  ['realdual'|'dual'|'tight'|'realtight', g]
    a   : hop sizes (int scalar, 1-D array, or (M,2) fractional array)
    L   : system length (None means "don't evaluate bandlimited filters")

    Returns
    -------
    g_out : list of prepared filter dicts
    a_out : (M, 2) int array  [[a_num, a_den], ...]
    info  : dict with at least the keys:
              M           – number of channels
              ispainless  – True iff all filter supports ≤ L/a_m
              isfir       – True iff all filters are time-domain FIR
              L           – system length (only when L was provided)
              longestfilter – support of the longest filter (samples)
              gl          – list of per-channel filter lengths
"""

from __future__ import annotations

import pytest

import numpy as np

# ===========================================================================
# Helper
# ===========================================================================

def _build_audfilterbank(Ls: int = 1024, fs: int = 8000):
    """Return (g, a, L) for a standard audfilters filterbank."""
    from cool_frames.filters import (
        audfilters,  # type: ignore
        filterbanklength,  # type: ignore
    )
    g, a, fc, _, _info = audfilters(fs, Ls)
    L = filterbanklength(Ls, a)
    return g, a, L


# ===========================================================================
# 1. Return-value structure
# ===========================================================================

@pytest.mark.requires_impl
class TestFilterbankwinStructImpl:
    """filterbankwin return-value structure."""

    def test_returns_three_outputs(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        result = filterbankwin(g, a, L)
        assert len(result) == 3, \
            f"filterbankwin must return (g_out, a_out, info), got {len(result)} values"

    def test_g_out_is_list(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        g_out, a_out, info = filterbankwin(g, a, L)
        assert isinstance(g_out, list), "g_out must be a list"

    def test_g_out_same_length(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        g_out, a_out, info = filterbankwin(g, a, L)
        assert len(g_out) == len(g), \
            f"len(g_out)={len(g_out)} != len(g)={len(g)}"

    def test_a_out_is_2d(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        g_out, a_out, info = filterbankwin(g, a, L)
        a_arr = np.asarray(a_out)
        assert a_arr.ndim == 2, \
            f"a_out must be 2-D (M, 2), got shape {a_arr.shape}"
        assert a_arr.shape == (len(g), 2), \
            f"a_out shape must be ({len(g)}, 2), got {a_arr.shape}"

    def test_info_is_dict(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        _, _, info = filterbankwin(g, a, L)
        assert isinstance(info, dict), "info must be a dict"

    def test_info_has_required_keys(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        _, _, info = filterbankwin(g, a, L)
        for key in ("M", "ispainless", "isfir", "longestfilter", "gl"):
            assert key in info, f"info dict missing key '{key}'"

    def test_info_M_matches_len_g(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        _, _, info = filterbankwin(g, a, L)
        assert info["M"] == len(g), \
            f"info['M']={info['M']} != len(g)={len(g)}"

    def test_info_L_matches_input(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        _, _, info = filterbankwin(g, a, L)
        assert info.get("L") == L, \
            f"info['L']={info.get('L')} != L={L}"


# ===========================================================================
# 2. Without L (lazy evaluation)
# ===========================================================================

@pytest.mark.requires_impl
class TestFilterbankwinNoLImpl:
    """filterbankwin without L (lazy mode – no FFT evaluation)."""

    def test_returns_without_L(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        result = filterbankwin(g, a)          # no L
        assert len(result) == 3

    def test_g_out_unchanged_without_L(self, needs_impl):
        """Without L, filter dicts should be returned as-is (no FFT)."""
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        g_out, a_out, info = filterbankwin(g, a)
        assert len(g_out) == len(g)

    def test_info_M_without_L(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        _, _, info = filterbankwin(g, a)
        assert info["M"] == len(g)


# ===========================================================================
# 3. String-spec shortcuts  ('realdual', 'dual', 'tight', 'realtight')
# ===========================================================================

@pytest.mark.requires_impl
class TestFilterbankwinStringSpecImpl:
    """filterbankwin string-spec shortcuts."""

    def test_realdual_spec(self, needs_impl):
        """filterbankwin(['realdual', g], a, L) must return a dual filterbank."""
        from cool_frames.filterbanks import filterbankdual, filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        g_rd_spec, a_rd, info_rd = filterbankwin(["realdual", g], a, L)
        # Compare against direct filterbankdual call
        g_rd_direct = filterbankdual(g, a, L)
        assert len(g_rd_spec) == len(g_rd_direct), \
            "realdual spec: length mismatch with filterbankdual"

    def test_dual_spec(self, needs_impl):
        """filterbankwin(['dual', g], a, L) must return filterbankdual result."""
        from cool_frames.filterbanks import filterbankdual, filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        g_d_spec, _, _ = filterbankwin(["dual", g], a, L)
        g_d_direct = filterbankdual(g, a, L)
        assert len(g_d_spec) == len(g_d_direct), \
            "dual spec: length mismatch with filterbankdual"

    def test_tight_spec(self, needs_impl):
        """filterbankwin(['tight', g], a, L) must return filterbanktight result."""
        from cool_frames.filterbanks import filterbanktight, filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        g_t_spec, _, _ = filterbankwin(["tight", g], a, L)
        g_t_direct = filterbanktight(g, a, L)
        assert len(g_t_spec) == len(g_t_direct), \
            "tight spec: length mismatch with filterbanktight"

    def test_realtight_spec(self, needs_impl):
        """filterbankwin(['realtight', g], a, L) must return filterbanktight result."""
        from cool_frames.filterbanks import filterbanktight, filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        g_rt_spec, _, _ = filterbankwin(["realtight", g], a, L)
        g_rt_direct = filterbanktight(g, a, L)
        assert len(g_rt_spec) == len(g_rt_direct), \
            "realtight spec: length mismatch with filterbanktight"

    def test_realdual_perfect_reconstruction(self, needs_impl):
        """Using the realdual spec should yield near-perfect reconstruction."""
        from cool_frames.filterbanks import (  # type: ignore
            filterbank,
            filterbankwin,
            ifilterbank,
        )
        g, a, L = _build_audfilterbank()
        gd_spec, _, _ = filterbankwin(["realdual", g], a, L)

        rng = np.random.default_rng(99)
        x = rng.standard_normal(1024)
        c = filterbank(x, g, a)
        # This test folds manually with 2*real(...), so request the raw complex
        # synthesis (real=False); the default real=True would fold a second time.
        xr = 2 * np.real(ifilterbank(c, gd_spec, a, L, real=False))
        err = np.linalg.norm(xr[:1024] - x) / (np.linalg.norm(x) + 1e-15)
        assert err < 1e-10, \
            f"realdual spec: reconstruction error {err:.2e} exceeds 1e-10"


# ===========================================================================
# 4. Painless and FIR info flags
# ===========================================================================

@pytest.mark.requires_impl
class TestFilterbankwinInfoFlagsImpl:
    """filterbankwin info-dict flags."""

    def test_ispainless_type(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        _, _, info = filterbankwin(g, a, L)
        assert isinstance(info["ispainless"], (bool, np.bool_)), \
            "info['ispainless'] must be bool"

    def test_isfir_type(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        _, _, info = filterbankwin(g, a, L)
        assert isinstance(info["isfir"], (bool, np.bool_)), \
            "info['isfir'] must be bool"

    def test_gl_length(self, needs_impl):
        """info['gl'] must be a list/array of M per-channel filter lengths."""
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        _, _, info = filterbankwin(g, a, L)
        gl = info["gl"]
        assert len(gl) == len(g), \
            f"len(info['gl'])={len(gl)} != M={len(g)}"

    def test_longestfilter_positive_for_fir(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        _, _, info = filterbankwin(g, a, L)
        # For bandlimited (non-FIR) filterbanks, longestfilter may be 0
        # Only check for FIR filterbanks
        if info["isfir"]:
            assert info["longestfilter"] > 0, \
                "info['longestfilter'] must be positive for FIR filterbanks"

    def test_longestfilter_equals_max_gl(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        _, _, info = filterbankwin(g, a, L)
        assert info["longestfilter"] == max(info["gl"]), \
            "longestfilter must equal max(gl)"


# ===========================================================================
# 5. Hop-size normalisation
# ===========================================================================

@pytest.mark.requires_impl
class TestFilterbankwinHopNormImpl:
    """filterbankwin normalises scalar / 1-D / 2-D hop sizes to (M, 2)."""

    def test_scalar_hop_broadcasts(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        M = len(g)
        a_scalar = int(a.ravel()[0]) if hasattr(a, "ravel") else int(a[0])
        _, a_out, _ = filterbankwin(g, a_scalar, None)
        assert np.asarray(a_out).shape[0] == M, \
            "scalar hop: a_out must have M rows"

    def test_1d_hop_becomes_2d(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        a_1d = (a[:, 0] if np.asarray(a).ndim == 2 else np.asarray(a)).astype(int)
        _, a_out, _ = filterbankwin(g, a_1d, None)
        assert np.asarray(a_out).ndim == 2, \
            "1-D hop: a_out must be 2-D (M, 2)"

    def test_denominator_is_positive(self, needs_impl):
        from cool_frames.filterbanks import filterbankwin  # type: ignore
        g, a, L = _build_audfilterbank()
        _, a_out, _ = filterbankwin(g, a, L)
        a_arr = np.asarray(a_out)
        assert np.all(a_arr[:, 1] > 0), \
            "All hop denominators must be positive"
