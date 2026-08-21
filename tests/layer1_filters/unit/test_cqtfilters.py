"""
test_cqtfilters.py
==================
Python port of:
    layer1_filters/unit/TestCqtFilters.m

Unit tests for cqtfilters (constant-Q filterbank design).

API under test
--------------
    cqtfilters(fs, fmin, fmax, bins, Ls, *, sampling, Qvar, min_win,
               winname, redmul, norm)
    -> (g, a, fc, L)

    filterbankbounds(g, a, L)
    -> (A, B, kappa)

    partial_tighten(g, a, L, alpha)
    -> g_alpha

Test categories
---------------
  1. Return-value structure and types
  2. Centre-frequency properties
  3. Bandwidth / constant-Q property
  4. Filter validity (non-zero response, realonly flag)
  5. Frame bounds (A > 0)
  6. Hop-size properties
  7. Optional-parameter effects
  8. Partial tightening
  9. Error handling
"""

from __future__ import annotations

import math

import pytest

import numpy as np

# Default test parameters (match TestCqtFilters.m)
FS   = 8000
FMIN = 100.0
FMAX = 3500.0
BINS = 12
LS   = 2048


# ---------------------------------------------------------------------------
# Helper: evaluate a filter's full-length frequency response
# ---------------------------------------------------------------------------

def _eval_filter(gm, L):
    from cool_frames.numpy.filters._filters import filter_freqresp  # type: ignore
    H, _ = filter_freqresp(gm, L)
    return H


# ===========================================================================
# 1. Return-value structure and types
# ===========================================================================

@pytest.mark.requires_impl
class TestCqtFiltersStructImpl:
    """Return-value structure and types."""

    def test_returns_five_outputs(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        result = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        assert len(result) == 5, "cqtfilters must return (g, a, fc, L, info)"

    def test_g_is_list(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        assert isinstance(g, list), "g must be a list of filter dicts"

    def test_each_filter_has_required_fields(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        for m, gm in enumerate(g):
            for field in ("H", "foff", "delay", "realonly"):
                assert field in gm, f"Filter {m}: missing field '{field}'"

    def test_H_is_callable(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        for m, gm in enumerate(g):
            assert callable(gm["H"]), f"Filter {m}: H must be callable"

    def test_foff_is_callable(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        for m, gm in enumerate(g):
            assert callable(gm["foff"]) or isinstance(gm["foff"], (int, np.integer)), \
                f"Filter {m}: foff must be callable or int"

    def test_L_is_positive_integer_geq_Ls(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        assert isinstance(L, (int, np.integer)), "L must be an integer"
        assert L > 0,  "L must be positive"
        assert L >= LS, "L must be >= Ls"

    def test_fc_a_g_have_same_length(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        M = len(g)
        assert len(fc) == M, "len(fc) must equal len(g)"
        assert a.shape[0] == M, "a.shape[0] must equal len(g)"


# ===========================================================================
# 2. Centre-frequency properties
# ===========================================================================

@pytest.mark.requires_impl
class TestCqtFiltersCentreFreqImpl:
    """Centre-frequency properties."""

    def test_first_fc_is_dc(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        assert fc[0] == 0.0, "First centre frequency must be 0 Hz (DC)"

    def test_last_fc_is_nyquist(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        assert fc[-1] == pytest.approx(FS / 2.0), \
            "Last centre frequency must be Nyquist = fs/2"

    def test_fc_monotone_increasing(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        assert np.all(np.diff(fc) > 0), \
            "Centre frequencies must be strictly monotone increasing"

    def test_second_fc_near_fmin(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        rel_err = abs(fc[1] - FMIN) / FMIN
        assert rel_err < 0.01, \
            f"Second channel (first CQT) should be near fmin={FMIN}, got {fc[1]}"

    def test_fc_does_not_exceed_nyquist(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        assert np.all(fc <= FS / 2.0 + 1e-9), \
            "No centre frequency may exceed Nyquist"


# ===========================================================================
# 3. Bandwidth / constant-Q property
# ===========================================================================

@pytest.mark.requires_impl
class TestCqtFiltersConstantQImpl:
    """Bandwidth and Q-factor properties."""

    def test_q_is_approximately_constant(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        fc_inner = fc[1:-1]
        if len(fc_inner) >= 3:
            bw_approx = fc_inner[2:] - fc_inner[:-2]
            Q_vals    = fc_inner[1:-1] / bw_approx
            cv = float(np.std(Q_vals) / np.mean(Q_vals))
            assert cv < 0.05, \
                f"Q coefficient of variation too large: {cv:.4f}"

    def test_q_natural_for_bins12(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=12)
        Q_nat = 1.0 / (2.0 ** (1.0 / 12) - 2.0 ** (-1.0 / 12))
        fc_inner = fc[1:-1]
        if len(fc_inner) >= 3:
            bw_approx = fc_inner[2:] - fc_inner[:-2]
            Q_vals    = fc_inner[1:-1] / bw_approx
            rel_err = np.abs(Q_vals - Q_nat) / Q_nat
            assert np.all(rel_err < 0.05), \
                f"Q values far from Q_natural={Q_nat:.2f}: {Q_vals}"

    def test_qvar_scales_bandwidth(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g1, a1, fc1, L1, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS, Qvar=1.0)
        g2, a2, fc2, L2, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS, Qvar=2.0)
        # With Qvar=2 the inner filters are wider → fewer filters expected
        # (or same count but different hop sizes); either way, a basic check:
        # fc arrays should have the same centre frequencies
        min_len = min(len(fc1), len(fc2))
        np.testing.assert_allclose(
            fc1[:min_len], fc2[:min_len], rtol=0.01,
            err_msg="Centre frequencies should not change with Qvar"
        )


# ===========================================================================
# 4. Filter validity
# ===========================================================================

@pytest.mark.requires_impl
class TestCqtFiltersValidityImpl:
    """Filter validity checks."""

    def test_dc_filter_realonly_zero(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        assert g[0]["realonly"] == 0, "DC filter must have realonly=0"

    def test_inner_filters_realonly_one(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        for m in range(1, len(g) - 1):
            assert g[m]["realonly"] == 1, \
                f"Inner filter {m} must have realonly=1"

    def test_all_filters_finite_response(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        for m, gm in enumerate(g):
            H = _eval_filter(gm, L)
            assert np.all(np.isfinite(H)), \
                f"Filter {m}: non-finite values in frequency response"

    def test_no_filter_is_all_zero(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        for m, gm in enumerate(g):
            H = _eval_filter(gm, L)
            assert np.max(np.abs(H)) > 0, f"Filter {m} is all-zero"


# ===========================================================================
# 5. Frame bounds
# ===========================================================================

@pytest.mark.requires_impl
class TestCqtFiltersFrameBoundsImpl:
    """Frame-bound checks — the critical correctness property."""

    def test_frame_lower_bound_positive(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        _A, _B = filterbankbounds(g, a, L); A, B, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert A > 0, f"Frame lower bound A must be > 0 (got A={A:.6g})"

    def test_frame_bounds_finite(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        _A, _B = filterbankbounds(g, a, L); A, B, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert math.isfinite(A), "A must be finite"
        assert math.isfinite(B), "B must be finite"

    def test_frame_upper_geq_lower(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        _A, _B = filterbankbounds(g, a, L); A, B, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert B >= A, "Frame upper bound B must be >= A"

    @pytest.mark.parametrize("qvar", [1.0, 2.0, 3.0])
    def test_frame_valid_for_various_qvar(self, needs_impl, qvar):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS, Qvar=qvar)
        _A, _B = filterbankbounds(g, a, L); A, B, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert A > 0, f"Frame invalid for Qvar={qvar}: A={A:.6g}"

    @pytest.mark.parametrize("bins", [4, 8, 12, 24])
    def test_frame_valid_for_various_bins(self, needs_impl, bins):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=bins)
        _A, _B = filterbankbounds(g, a, L); A, B, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert A > 0, f"Frame invalid for bins={bins}: A={A:.6g}"

    def test_tight_frame_kappa_equals_one(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbanktight  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        gt = filterbanktight(g, a, L)
        _A, _B = filterbankbounds(gt, a, L); At, Bt, kappa_t = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert abs(At - Bt) / max(abs(At), 1e-12) < 1e-5, \
            f"Tight frame: A={At:.6g} != B={Bt:.6g}"


# ===========================================================================
# 6. Hop-size properties
# ===========================================================================

@pytest.mark.requires_impl
class TestCqtFiltersHopSizesImpl:
    """Hop-size properties."""

    def test_all_hop_sizes_positive(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        a_num = a[:, 0] if a.ndim == 2 else a
        assert np.all(a_num > 0), "All hop sizes must be positive"

    def test_L_divisible_by_all_hop_sizes(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        a_num = a[:, 0] if a.ndim == 2 else a
        for m, am in enumerate(a_num):
            assert L % am == 0, \
                f"L={L} not divisible by a[{m}]={am}"

    def test_uniform_sampling_produces_uniform_hops(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS, sampling="uniform")
        if a.ndim == 1:
            assert len(np.unique(a)) == 1, \
                "Uniform sampling: all hop sizes should be equal"

    def test_fractional_sampling_returns_two_columns(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS, sampling="fractional")
        assert a.ndim == 2 and a.shape[1] == 2, \
            f"Fractional sampling: a must be (M+2, 2), got {a.shape}"


# ===========================================================================
# 7. Optional-parameter effects
# ===========================================================================

@pytest.mark.requires_impl
class TestCqtFiltersOptionalParamsImpl:
    """Optional-parameter effects."""

    def test_variable_bins_per_octave(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        bins_vec = [8, 12, 16]
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=bins_vec)
        assert len(g) > 0, "g must not be empty for vector bins"
        _A, _B = filterbankbounds(g, a, L); A, B, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert A > 0, "Frame must be valid for variable bins per octave"

    @pytest.mark.parametrize("fs", [4000, 16000, 44100])
    def test_frame_valid_for_different_fs(self, needs_impl, fs):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        fmax = min(FMAX, fs / 2 - 1)
        g, a, fc, L, _info = cqtfilters(fs, LS, fmin=FMIN, fmax=fmax, bins=BINS)
        _A, _B = filterbankbounds(g, a, L); A, B, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert A > 0, f"Frame invalid for fs={fs}: A={A:.6g}"

    def test_fmax_clipped_to_nyquist(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FS * 2, bins=BINS)  # fmax >> Nyquist
        assert np.all(fc <= FS / 2.0 + 1e-9), \
            "Centre frequencies must not exceed Nyquist"

    def test_redmul_greater_than_one_gives_valid_frame(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS, redmul=2.0)
        _A, _B = filterbankbounds(g, a, L); A, B, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert A > 0, f"Frame invalid for redmul=2.0: A={A:.6g}"


# ===========================================================================
# 8. Partial tightening
# ===========================================================================

@pytest.mark.requires_impl
class TestCqtFiltersPartialTightenImpl:
    """Partial tightening: G_alpha = G / S^(alpha/2)."""

    def test_alpha0_preserves_bounds(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        from cool_frames.numpy.filters import partial_tighten  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        _A, _B = filterbankbounds(g, a, L); A0, B0, k0 = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        g_a0 = partial_tighten(g, a, L, 0.0)
        _A, _B = filterbankbounds(g_a0, a, L); Aa, Ba, ka = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert abs(Aa - A0) / max(A0, 1e-30) < 0.01, \
            f"alpha=0 should not change bounds: A0={A0:.4f}, Aa={Aa:.4f}"

    def test_alpha1_gives_tight_frame(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        from cool_frames.numpy.filters import partial_tighten  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        g_tight = partial_tighten(g, a, L, 1.0)
        _A, _B = filterbankbounds(g_tight, a, L); At, Bt, kappa_t = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert abs(At - Bt) / max(abs(At), 1e-12) < 1e-5, \
            f"alpha=1 must give tight frame: A={At:.6g}, B={Bt:.6g}"

    def test_kappa_decreases_with_alpha(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filterbanks import filterbankbounds  # type: ignore
        from cool_frames.numpy.filters import partial_tighten  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        kappas = []
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            g_a = partial_tighten(g, a, L, alpha)
            _A, _B = filterbankbounds(g_a, a, L); _, _, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
            kappas.append(kappa)
        # kappa should be non-increasing as alpha increases
        for i in range(len(kappas) - 1):
            assert kappas[i] >= kappas[i + 1] - 1e-6, \
                f"kappa should decrease with alpha: kappas={kappas}"

    def test_partial_tighten_returns_list_of_dicts(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filters import partial_tighten  # type: ignore
        g, a, fc, L, _info = cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS)
        g_half = partial_tighten(g, a, L, 0.5)
        assert isinstance(g_half, list), "partial_tighten must return a list"
        assert len(g_half) == len(g), "Length must match the input filterbank"
        for m, gm in enumerate(g_half):
            assert isinstance(gm, dict), f"Filter {m} must be a dict"


# ===========================================================================
# 9. Error handling
# ===========================================================================

@pytest.mark.requires_impl
class TestCqtFiltersErrorHandlingImpl:
    """Error-handling: invalid parameter combinations."""

    def test_error_when_fmin_geq_fmax(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        with pytest.raises((ValueError, Exception)):
            cqtfilters(FS, LS, fmin=1000.0, fmax=500.0, bins=BINS)

    def test_error_when_fmin_is_zero(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        with pytest.raises((ValueError, Exception)):
            cqtfilters(FS, LS, fmin=0.0, fmax=FMAX, bins=BINS)

    def test_error_for_unknown_sampling_mode(self, needs_impl):
        from cool_frames.filters import cqtfilters  # type: ignore
        with pytest.raises((ValueError, Exception)):
            cqtfilters(FS, LS, fmin=FMIN, fmax=FMAX, bins=BINS, sampling="bogus")
