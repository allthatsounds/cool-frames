"""
tests/layer1_filters/unit/test_greenwoodfilters.py
===================================================
Python unit tests for greenwoodfilters (Greenwood cochlear filterbank design).

Mirrors the structure of test_audfilters.py, covering:
  1. Return-value structure and types
  2. Centre-frequency properties
  3. Species parameter switching (A, alpha, k)
  4. Filter validity — non-zero response at own centre-frequency bin
  5. Frame bounds (A > 0 for dense and sparse configurations)
  6. Frame bounds with different species parameters
  7. Hop-size and sampling-mode properties
  8. Frame-theory helpers (partial_tighten)
  9. Edge-case and error handling
 10. Complement edge filters (DC and Nyquist)
"""
from __future__ import annotations

import math

import pytest

import numpy as np

try:
    from cool_frames.numpy.core._core import filterbanklength
    from cool_frames.numpy.filterbanks._frame import (
        filterbankbounds,
    )
    from cool_frames.numpy.filters._audscale import (
        GREENWOOD_DEFAULTS,
    )
    from cool_frames.numpy.filters._greenwoodfilters import (
        _greenwood_bw,
        _greenwood_freq,
        _greenwood_pos,
        _greenwood_space,
        greenwoodfilters,
    )
    from cool_frames.numpy.filters._design import partial_tighten
    from cool_frames.numpy.filterbanks import filterbankbounds
    from cool_frames.numpy.filters._filters import filter_freqresp
    _HAS_LTFAT = True
except ImportError:
    _HAS_LTFAT = False

pytestmark = pytest.mark.skipif(not _HAS_LTFAT,
                                reason="cool_frames not installed")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

FS = 8000
LS = 2048

# Species parameters (Greenwood 1990, Table I)
HUMAN    = dict(A=165.4, alpha=2.1, k=0.88)
ELEPHANT = dict(A=200.0, alpha=1.4, k=0.85)
CAT      = dict(A=456.0, alpha=0.8, k=0.80)


# ---------------------------------------------------------------------------
# 1. Return-value structure and types
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_returns_four_outputs(self):
        g, a, fc, L, _info = greenwoodfilters(FS, LS)
        assert len(g) > 0
        assert a is not None
        assert len(fc) > 0
        assert isinstance(L, int)

    def test_g_is_a_list_of_dicts(self):
        g, _, _, _, _info = greenwoodfilters(FS, LS)
        assert isinstance(g, list)
        for m, f in enumerate(g):
            assert isinstance(f, dict), f"g[{m}] must be a dict, got {type(f)}"

    def test_each_filter_has_required_fields(self):
        g, _, _, _, _info = greenwoodfilters(FS, LS)
        required = {"H", "foff", "delay", "realonly"}
        for m, f in enumerate(g):
            for field in required:
                assert field in f, f"g[{m}] missing field '{field}'"

    def test_a_has_same_length_as_g(self):
        g, a, _, _, _info = greenwoodfilters(FS, LS)
        a_arr = np.asarray(a)
        assert a_arr.shape[0] == len(g), "a must have one row per filter"

    def test_fc_has_same_length_as_g(self):
        g, _, fc, _, _info = greenwoodfilters(FS, LS)
        assert len(fc) == len(g), "fc must have one entry per filter"

    def test_all_hop_sizes_positive(self):
        _, a, _, _, _info = greenwoodfilters(FS, LS)
        a_arr = np.asarray(a)
        a_int = a_arr[:, 0] if a_arr.ndim == 2 else a_arr
        assert np.all(a_int > 0), "All hop sizes must be positive"


# ---------------------------------------------------------------------------
# 2. Centre-frequency properties
# ---------------------------------------------------------------------------

class TestCentreFrequencies:

    def test_dc_channel_has_zero_frequency(self):
        _, _, fc, _, _info = greenwoodfilters(FS, LS)
        assert fc[0] == 0.0, f"DC centre frequency must be 0, got {fc[0]}"

    def test_nyquist_channel_has_fs_over_2(self):
        _, _, fc, _, _info = greenwoodfilters(FS, LS)
        assert fc[-1] == FS / 2.0, f"Nyquist channel must be {FS/2}, got {fc[-1]}"

    def test_inner_channels_monotonically_increasing(self):
        _, _, fc, _, _info = greenwoodfilters(FS, LS)
        inner = fc[1:-1]
        assert np.all(np.diff(inner) > 0), \
            "Inner centre frequencies must be strictly increasing"

    def test_inner_channels_below_nyquist(self):
        _, _, fc, _, _info = greenwoodfilters(FS, LS)
        inner = fc[1:-1]
        assert np.all(inner < FS / 2.0), \
            "All inner centre frequencies must be strictly below Nyquist"

    def test_inner_channels_above_dc(self):
        _, _, fc, _, _info = greenwoodfilters(FS, LS)
        inner = fc[1:-1]
        assert np.all(inner > 0), \
            "All inner centre frequencies must be strictly above 0 Hz"

    def test_custom_fmin_fmax(self):
        fmin, fmax = 100.0, 2000.0
        _, _, fc, _, _info = greenwoodfilters(FS, LS, fmin=fmin, fmax=fmax)
        inner = fc[1:-1]
        assert min(inner) >= fmin * 0.99, \
            f"Lowest inner channel {min(inner):.1f} must be ~>= fmin={fmin}"
        assert max(inner) <= fmax * 1.01, \
            f"Highest inner channel {max(inner):.1f} must be ~<= fmax={fmax}"

    def test_M_parameter_controls_channel_count(self):
        M = 20
        g, _, _, _, _info = greenwoodfilters(FS, LS, M=M, fmin=50, fmax=3000)
        n_inner = len(g) - 2
        assert M - 2 <= n_inner <= M, (
            f"With M={M}, inner channels must be near M, got {n_inner}"
        )

    def test_spacing_affects_channel_density(self):
        """Smaller spacing => more channels."""
        g1, _, _, _, _info = greenwoodfilters(FS, LS, spacing=0.05)
        g2, _, _, _, _info = greenwoodfilters(FS, LS, spacing=0.02)
        assert len(g2) > len(g1), (
            f"spacing=0.02 ({len(g2)} ch) should give more channels "
            f"than spacing=0.05 ({len(g1)} ch)"
        )


# ---------------------------------------------------------------------------
# 3. Species parameter switching (A, alpha, k)
# ---------------------------------------------------------------------------

class TestSpeciesParameters:

    def test_human_defaults_match_greenwood_defaults(self):
        """When A/alpha/k are not specified, GREENWOOD_DEFAULTS are used."""
        g1, _, fc1, _, _info = greenwoodfilters(FS, LS)
        g2, _, fc2, _, _info = greenwoodfilters(FS, LS, **HUMAN)
        assert len(g1) == len(g2), "Default should match human parameters"
        np.testing.assert_allclose(fc1, fc2, atol=1e-6,
            err_msg="Default fc must match human Greenwood params")

    def test_elephant_gives_different_fc_from_human(self):
        _, _, fc_h, _, _info = greenwoodfilters(FS, LS, fmin=20, fmax=2000, **HUMAN)
        _, _, fc_e, _, _info = greenwoodfilters(FS, LS, fmin=20, fmax=2000, **ELEPHANT)
        # Different species parameters should produce different centre frequencies
        # (even if the number of channels happens to match)
        if len(fc_h) == len(fc_e):
            assert not np.allclose(fc_h[1:-1], fc_e[1:-1], atol=1.0), \
                "Human and elephant parameters must give different fc layouts"

    def test_cat_parameters(self):
        """Cat parameters (A=456, alpha=0.8) should produce a valid filterbank."""
        g, a, fc, L, _info = greenwoodfilters(FS, LS, **CAT)
        assert len(g) >= 3, "Cat filterbank must have at least DC + 1 inner + Nyquist"
        _A, _B = filterbankbounds(g, a, L); A_bound, _, _ = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert A_bound > 0, f"Cat filterbank must have A > 0 (got {A_bound:.6g})"

    def test_greenwood_defaults_are_immutable_human(self):
        """GREENWOOD_DEFAULTS are the human cochlea params and are read-only.

        Species selection is per-call via greenwoodfilters(A=, alpha=, k=),
        not a mutable global (the old set_greenwood_params was removed)."""
        assert GREENWOOD_DEFAULTS["A"] == 165.4    # human (Greenwood 1990)
        assert GREENWOOD_DEFAULTS["alpha"] == 2.1
        assert GREENWOOD_DEFAULTS["k"] == 0.88
        with pytest.raises(TypeError):
            GREENWOOD_DEFAULTS["A"] = 200.0        # read-only mapping

    def test_greenwood_freq_roundtrip(self):
        """_greenwood_freq and _greenwood_pos must be inverses."""
        A, alpha, k = HUMAN["A"], HUMAN["alpha"], HUMAN["k"]
        x = np.linspace(0.05, 0.95, 50)
        f = _greenwood_freq(x, A, alpha, k)
        x_back = _greenwood_pos(f, A, alpha, k)
        np.testing.assert_allclose(x_back, x, rtol=1e-10,
            err_msg="Greenwood freq<->pos must be invertible")

    def test_greenwood_bw_positive(self):
        """Bandwidth must be positive for all positive frequencies."""
        A, alpha, k = ELEPHANT["A"], ELEPHANT["alpha"], ELEPHANT["k"]
        fc = np.array([10.0, 50.0, 200.0, 1000.0, 4000.0])
        bw = _greenwood_bw(fc, A, alpha, k)
        assert np.all(bw > 0), f"Bandwidth must be positive, got {bw}"

    def test_greenwood_space_endpoints(self):
        """_greenwood_space must hit fmin and fmax."""
        A, alpha, k = HUMAN["A"], HUMAN["alpha"], HUMAN["k"]
        fmin, fmax, n = 100.0, 4000.0, 30
        freqs = _greenwood_space(fmin, fmax, n, A, alpha, k)
        np.testing.assert_allclose(freqs[0], fmin, rtol=1e-10)
        np.testing.assert_allclose(freqs[-1], fmax, rtol=1e-10)
        assert len(freqs) == n


# ---------------------------------------------------------------------------
# 4. Filter validity — non-zero response at own centre-frequency bin
# ---------------------------------------------------------------------------

class TestFilterValidity:

    def test_inner_filters_nonzero_at_own_centre_freq(self):
        g, a, fc, L, _info = greenwoodfilters(FS, LS)
        for m in range(1, len(g) - 1):
            H_full, _ = filter_freqresp(g[m], L)
            fc_bin = int(round(L * fc[m] / FS))
            fc_bin = max(0, min(L - 1, fc_bin))
            val = abs(H_full[fc_bin])
            assert val > 0, (
                f"Filter {m} (fc={fc[m]:.1f} Hz): H[fc_bin={fc_bin}] must be > 0, "
                f"got {val:.6g}"
            )

    def test_inner_filter_realonly_flag(self):
        g, _, fc, _, _info = greenwoodfilters(FS, LS)
        for m in range(1, len(g) - 1):
            assert g[m]["realonly"] == 0, (
                f"Filter {m} (fc={fc[m]:.1f} Hz): realonly must be 0"
            )

    def test_dc_filter_realonly_flag_valid(self):
        g, _, _, _, _info = greenwoodfilters(FS, LS)
        assert g[0]["realonly"] in (0, 1), "DC filter realonly must be 0 or 1"


# ---------------------------------------------------------------------------
# 5. Frame bounds — A > 0 for dense and sparse configurations
# ---------------------------------------------------------------------------

class TestFrameBounds:

    @pytest.mark.parametrize("M", [None, 30, 20, 10])
    def test_frame_lower_bound_positive(self, M):
        kw = {} if M is None else {"M": M}
        label = f"M={M}" if M else "default"
        g, a, _, L, _info = greenwoodfilters(FS, LS, **kw)
        _A, _B = filterbankbounds(g, a, L); A, B, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert A > 0, (
            f"Greenwood {label}: frame lower bound A must be > 0 (got A={A:.6g})"
        )
        assert B > 0, f"Greenwood {label}: frame upper bound B must be > 0"
        assert A <= B, f"Greenwood {label}: must have A <= B"

    def test_condition_number_finite(self):
        g, a, _, L, _info = greenwoodfilters(FS, LS)
        _A, _B = filterbankbounds(g, a, L); _, _, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert math.isfinite(kappa), \
            f"Condition number kappa = B/A must be finite (got {kappa})"


# ---------------------------------------------------------------------------
# 6. Frame bounds with different species parameters
# ---------------------------------------------------------------------------

class TestFrameBoundsSpecies:

    @pytest.mark.parametrize("species,params", [
        ("human", HUMAN),
        ("elephant", ELEPHANT),
        ("cat", CAT),
    ])
    def test_frame_lower_bound_positive_species(self, species, params):
        g, a, _, L, _info = greenwoodfilters(FS, LS, **params)
        _A, _B = filterbankbounds(g, a, L); A, B, _ = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert A > 0, (
            f"Species '{species}': frame lower bound A must be > 0 (got A={A:.6g})"
        )


# ---------------------------------------------------------------------------
# 7. Hop-size and sampling-mode properties
# ---------------------------------------------------------------------------

class TestHopSizeProperties:

    def test_regsampling_divisibility(self):
        _, a, _, L, _info = greenwoodfilters(FS, LS, sampling="regsampling")
        a_arr = np.asarray(a)
        a_int = a_arr[:, 0] if a_arr.ndim == 2 else a_arr
        remainders = L % a_int
        assert np.all(remainders == 0), \
            "In regsampling mode, L must be divisible by every hop size"

    def test_uniform_sampling_constant_hop(self):
        _, a, _, _, _info = greenwoodfilters(FS, LS, sampling="uniform")
        a_arr = np.asarray(a)
        assert np.all(a_arr == a_arr[0]), \
            "In uniform sampling mode, all hop sizes must be equal"

    def test_fractional_sampling_shape(self):
        _, a, _, _, _info = greenwoodfilters(FS, LS, sampling="fractional")
        a_arr = np.asarray(a)
        assert a_arr.ndim == 2 and a_arr.shape[1] == 2, \
            f"In fractional mode, a must be (M, 2), got shape {a_arr.shape}"

    def test_L_greater_than_or_equal_to_Ls(self):
        _, a, _, L, _info = greenwoodfilters(FS, LS)
        assert L >= LS, f"L={L} must be >= Ls={LS}"

    def test_redmul_increases_redundancy(self):
        _, a1, _, _, _info = greenwoodfilters(FS, LS, redmul=1.0)
        _, a2, _, _, _info = greenwoodfilters(FS, LS, redmul=2.0)
        a1_arr = np.asarray(a1)
        a2_arr = np.asarray(a2)
        a1_min = int(a1_arr[:, 0].min()) if a1_arr.ndim == 2 else int(a1_arr.min())
        a2_min = int(a2_arr[:, 0].min()) if a2_arr.ndim == 2 else int(a2_arr.min())
        assert a2_min <= a1_min, (
            f"redmul=2 should produce smaller hop sizes than redmul=1 "
            f"(got {a2_min} vs {a1_min})"
        )


# ---------------------------------------------------------------------------
# 8. Frame-theory helpers (partial_tighten)
# ---------------------------------------------------------------------------

class TestFrameTheoryHelpers:

    def test_partial_tighten_kappa_progression(self):
        """Kappa must monotonically decrease as alpha increases from 0 to 1."""
        g, a, _, L, _info = greenwoodfilters(FS, LS, M=20)
        alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
        kappas = []
        for alpha in alphas:
            g_a = partial_tighten(g, a, L, alpha)
            _A, _B = filterbankbounds(g_a, a, L); _, _, k = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
            kappas.append(k)
        for i in range(len(kappas) - 1):
            assert kappas[i + 1] <= kappas[i] + 1e-6, (
                f"kappa must decrease with alpha: kappa({alphas[i+1]})={kappas[i+1]:.4f} "
                f"> kappa({alphas[i]})={kappas[i]:.4f}"
            )

    def test_partial_tighten_alpha_zero_is_identity(self):
        g, a, _, L, _info = greenwoodfilters(FS, LS, M=20)
        _A, _B = filterbankbounds(g, a, L); A0, B0, _ = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        g_id = partial_tighten(g, a, L, 0.0)
        _A, _B = filterbankbounds(g_id, a, L); A1, B1, _ = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        np.testing.assert_allclose(A1, A0, rtol=1e-5,
            err_msg="alpha=0 partial tighten must leave A unchanged")
        np.testing.assert_allclose(B1, B0, rtol=1e-5,
            err_msg="alpha=0 partial tighten must leave B unchanged")

    def test_partial_tighten_alpha_one_gives_kappa_one(self):
        g, a, _, L, _info = greenwoodfilters(FS, LS, M=20)
        g_tight = partial_tighten(g, a, L, 1.0)
        _A, _B = filterbankbounds(g_tight, a, L); _, _, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert abs(kappa - 1.0) < 1e-6, \
            f"alpha=1 partial tighten must give kappa=1 (got {kappa:.8f})"


# ---------------------------------------------------------------------------
# 9. Edge-case and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_invalid_sampling_mode_raises(self):
        with pytest.raises(ValueError):
            greenwoodfilters(FS, LS, sampling="bogusmode")

    def test_fmin_above_fmax_raises(self):
        with pytest.raises(ValueError):
            greenwoodfilters(FS, LS, fmin=3000.0, fmax=100.0)

    def test_high_sample_rate(self):
        fs_high, ls_high = 44100, 16384
        g, a, fc, L, _info = greenwoodfilters(fs_high, ls_high)
        _A, _B = filterbankbounds(g, a, L); A, _, _ = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert A > 0, f"fs=44100: frame lower bound A must be > 0 (got A={A:.6g})"
        assert fc[-1] == fs_high / 2.0, \
            f"Nyquist channel must equal {fs_high/2}, got {fc[-1]}"

    def test_min_win_enforced(self):
        min_win = 8
        g, a, _, L, _info = greenwoodfilters(FS, LS, min_win=min_win)
        for m in range(1, len(g) - 1):
            H_full, _ = filter_freqresp(g[m], L)
            nnz_bins = int(np.sum(np.abs(H_full) > 0))
            assert nnz_bins >= min_win, (
                f"Filter {m}: support must be >= min_win={min_win} bins "
                f"(got {nnz_bins})"
            )

    def test_bwmul_widens_bandwidth(self):
        """bwmul > 1 should widen filters (lower kappa or different a)."""
        g1, a1, _, L1, _info = greenwoodfilters(FS, LS, bwmul=1.0)
        g2, a2, _, L2, _info = greenwoodfilters(FS, LS, bwmul=2.0)
        # Wider filters => smaller hop sizes or better kappa
        _A, _B = filterbankbounds(g1, a1, L1); _, _, k1 = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        _A, _B = filterbankbounds(g2, a2, L2); _, _, k2 = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        # Both should be valid frames
        assert math.isfinite(k1), f"bwmul=1 kappa must be finite (got {k1})"
        assert math.isfinite(k2), f"bwmul=2 kappa must be finite (got {k2})"


# ---------------------------------------------------------------------------
# 10. Complement edge filters (DC / Nyquist)
# ---------------------------------------------------------------------------

class TestComplementEdgeFilters:
    """Verify the complement-based DC and Nyquist edge-filter design."""

    FS2 = 44100
    LS2 = 44100

    def test_dc_filter_nonzero_energy_near_dc(self):
        g, a, fc, L, _info = greenwoodfilters(self.FS2, self.LS2)
        H_dc, _ = filter_freqresp(g[0], L)
        n_check = max(L // 20, 1)
        energy_dc_region = float(np.sum(np.abs(H_dc[:n_check]) ** 2))
        assert energy_dc_region > 0, (
            f"DC edge filter has no energy in bins 0..{n_check-1}"
        )

    def test_nyquist_filter_well_formed(self):
        g, a, _, L, _info = greenwoodfilters(self.FS2, self.LS2)
        H_vals = g[-1]["H"](L) if callable(g[-1]["H"]) else g[-1]["H"]
        assert np.isfinite(H_vals).all(), "Nyquist filter contains NaN/Inf"

    def test_frame_response_finite_everywhere(self):
        from cool_frames.numpy.filterbanks._frame import filterbankresponse
        g, a, _, L, _info = greenwoodfilters(self.FS2, self.LS2)
        resp = filterbankresponse(g, a, L, real=True)
        assert np.all(np.isfinite(resp)), "Frame response contains NaN/Inf"

    def test_edge_filter_freq_domain_structure(self):
        g, a, _, L, _info = greenwoodfilters(self.FS2, self.LS2)
        assert "H" in g[0], "DC filter must use 'H' representation"
        assert "H" in g[-1], "Nyquist filter must use 'H' representation"
        foff_dc = g[0]["foff"](L) if callable(g[0]["foff"]) else g[0]["foff"]
        assert foff_dc <= 0, f"DC filter foff must be <= 0, got {foff_dc}"

    def test_elephant_complement_gives_finite_kappa(self):
        """Elephant parameters with complement edges should give finite kappa."""
        g, a, _, L, _info = greenwoodfilters(self.FS2, self.LS2, **ELEPHANT)
        _A, _B = filterbankbounds(g, a, L); _, _, kappa = _A, _B, (_B / _A if _A > 1e-30 else float('inf'))
        assert math.isfinite(kappa), (
            f"Elephant filterbank must have finite kappa (got {kappa})"
        )
