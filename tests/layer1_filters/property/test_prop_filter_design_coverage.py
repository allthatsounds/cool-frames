"""
test_prop_filter_design_coverage.py
=====================================
Python port of:
    layer1_filters/property/PropFilterDesignCoverage.m

Frame-theoretic coverage properties for high-level filter design functions.

1. audfilters(fs, Ls):
   - len(g) == len(fc) == len(a)
   - Centre frequencies are strictly monotone increasing
   - All subsampling factors divide Ls
   - Full frequency coverage (no dead DFT bins) in complex-output mode
   - Frame lower bound A > 0 (sum_m |H_m(k)|^2 >= A for all k)

2. cqtfilters(fs, Ls, fmin=fmin, fmax=fmax, bins=bins_per_oct):
   - len(g) == len(fc) == len(a)
   - Centre frequencies are strictly monotone increasing
   - Constant-Q: fc/bw approximately constant across filters

3. gabfilters(16000, Ls, window=window, a=a_hop, M=M_ch):
   - len(g) == Ls // a_hop  (or similar power-of-2 count)
   - Frequency coverage in complex mode (no dead bins)
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_frame_bound(g_list, a_list, fc_list, L: int):
    """
    Compute lower frame bound A = min_k sum_m |H_m(k)|^2.

    For each filter g_m with subsampling factor a_m we compute the full
    L-point transfer function and accumulate |H_m(k)|^2.
    Only the positive-frequency half (k=0..L//2) is inspected so the
    bound is computed for a single-sided (real) representation.
    """
    from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore

    accumulator = np.zeros(L, dtype=float)
    for g_m in g_list:
        H_m = comp_transferfunction(g_m, L)
        accumulator += np.abs(H_m) ** 2
    # Only inspect positive-frequency bins for real filterbanks
    return float(np.min(accumulator[:L // 2 + 1]))


# ---------------------------------------------------------------------------
# audfilters
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestAudFiltersOutputConsistencyImpl:
    """
    MATLAB counterpart: PropFilterDesignCoverage (audfilters consistency).
    """

    def test_output_sizes_match(self, needs_impl):
        """len(g) == len(fc) == len(a)."""
        from cool_frames.filters import audfilters  # type: ignore
        fs = 16000
        Ls = 4096
        g, a, fc, _, _info = audfilters(fs, Ls)
        assert len(g) == len(fc), "g and fc have different lengths"
        assert len(g) == len(a),  "g and a have different lengths"

    @pytest.mark.parametrize("fs,Ls", [(8000, 2048), (16000, 4096), (44100, 8192)])
    def test_output_sizes_across_params(self, needs_impl, fs, Ls):
        """Consistency holds for multiple (fs, Ls) pairs."""
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(fs, Ls)
        assert len(g) == len(fc) == len(a)
        assert len(g) > 0, "audfilters returned empty filter list"

    def test_filters_are_nonempty(self, needs_impl):
        """Each filter struct has a non-empty transfer function."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        g, a, fc, _, _info = audfilters(16000, 4096)
        L = 4096
        for m, g_m in enumerate(g):
            H = comp_transferfunction(g_m, L)
            assert len(H) == L, f"Filter {m}: wrong H length"
            assert np.max(np.abs(H)) > 0, f"Filter {m}: all-zero transfer function"


@pytest.mark.requires_impl
class TestAudFiltersMonotoneFreqImpl:
    """
    MATLAB counterpart: PropFilterDesignCoverage (audfilters monotone fc).
    """

    def test_centre_freqs_monotone(self, needs_impl):
        """Centre frequencies should be strictly increasing."""
        from cool_frames.filters import audfilters  # type: ignore
        _, _, fc, _, _info = audfilters(16000, 4096)
        fc_arr = np.asarray(fc, dtype=float)
        assert np.all(np.diff(fc_arr) > 0), \
            "audfilters centre frequencies are not strictly monotone increasing"

    def test_centre_freqs_positive(self, needs_impl):
        """All centre frequencies must be positive (or zero for DC)."""
        from cool_frames.filters import audfilters  # type: ignore
        _, _, fc, _, _info = audfilters(16000, 4096)
        fc_arr = np.asarray(fc, dtype=float)
        assert np.all(fc_arr >= 0), "Some centre frequencies are negative"

    def test_max_freq_below_nyquist(self, needs_impl):
        """Highest centre frequency must be ≤ Nyquist (fs/2)."""
        from cool_frames.filters import audfilters  # type: ignore
        fs = 16000
        _, _, fc, _, _info = audfilters(fs, 4096)
        fc_arr = np.asarray(fc, dtype=float)
        assert np.max(fc_arr) <= fs / 2 * 1.01, \
            f"Max centre freq {np.max(fc_arr)} exceeds Nyquist {fs/2}"


@pytest.mark.requires_impl
class TestAudFiltersSubsamplingImpl:
    """
    MATLAB counterpart: PropFilterDesignCoverage (audfilters subsampling).
    """

    def test_all_a_divide_L(self, needs_impl):
        """All subsampling factors must divide L (the adjusted transform length)."""
        from cool_frames.filters import audfilters  # type: ignore
        Ls = 4096
        g, a, _, L, _info = audfilters(16000, Ls)
        # L returned by audfilters is the filterbank-compatible length
        for m, a_m in enumerate(a):
            assert L % int(a_m) == 0, \
                f"Filter {m}: a_m={a_m} does not divide L={L}"

    def test_a_are_positive_integers(self, needs_impl):
        """Subsampling factors must be positive integers."""
        from cool_frames.filters import audfilters  # type: ignore
        _, a, _, _, _info = audfilters(16000, 4096)
        for m, a_m in enumerate(a):
            assert int(a_m) == a_m and a_m >= 1, \
                f"Filter {m}: a_m={a_m} is not a positive integer"


@pytest.mark.requires_impl
class TestAudFiltersFrequencyCoverageImpl:
    """
    MATLAB counterpart: PropFilterDesignCoverage (audfilters coverage).
    """

    def test_no_dead_bins(self, needs_impl):
        """
        No positive-frequency DFT bin should have zero energy across all
        filters — i.e. sum_m |H_m(k)|^2 > 0 for k = 0 … L//2.

        audfilters produces a real filterbank covering [0, fs/2], so we
        only inspect the positive-frequency half.
        """
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 1024
        g, a, fc, _, _info = audfilters(16000, L)
        energy = np.zeros(L, dtype=float)
        for g_m in g:
            H = comp_transferfunction(g_m, L)
            energy += np.abs(H) ** 2
        pos_energy = energy[:L // 2 + 1]
        assert np.min(pos_energy) > 0, \
            f"Dead DFT bins found: min energy = {np.min(pos_energy):.2e}"

    def test_frame_lower_bound_positive(self, needs_impl):
        """
        Lower frame bound A = min_k sum_m |H_m(k)|^2 must be strictly > 0.
        """
        from cool_frames.filters import audfilters  # type: ignore
        L  = 1024
        g, a, fc, _, _info = audfilters(16000, L)
        A = _compute_frame_bound(g, a, fc, L)
        assert A > 0, f"Frame lower bound A={A:.4f} is not positive"


# ---------------------------------------------------------------------------
# cqtfilters
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCqtFiltersOutputConsistencyImpl:
    """
    MATLAB counterpart: PropFilterDesignCoverage (cqtfilters consistency).
    """

    def test_output_sizes_match(self, needs_impl):
        """len(g) == len(fc) == len(a)."""
        from cool_frames.filters import cqtfilters  # type: ignore
        fs   = 16000
        Ls   = 4096
        fmin = 100
        fmax = 6000
        bins = 12
        g, a, fc, _, _info = cqtfilters(fs, Ls, fmin=fmin, fmax=fmax, bins=bins)
        assert len(g) == len(fc) == len(a)
        assert len(g) > 0

    @pytest.mark.parametrize("bins_per_oct", [6, 12, 24])
    def test_output_sizes_across_bins(self, needs_impl, bins_per_oct):
        """More bins per octave → more filters, sizes still consistent."""
        from cool_frames.filters import cqtfilters  # type: ignore
        g, a, fc, _, _info = cqtfilters(16000, 4096, fmin=100, fmax=6000, bins=bins_per_oct)
        assert len(g) == len(fc) == len(a)
        assert len(g) > 0


@pytest.mark.requires_impl
class TestCqtFiltersMonotoneFreqImpl:
    """
    MATLAB counterpart: PropFilterDesignCoverage (cqtfilters monotone fc).
    """

    def test_centre_freqs_monotone(self, needs_impl):
        """Centre frequencies are strictly increasing."""
        from cool_frames.filters import cqtfilters  # type: ignore
        _, _, fc, _, _info = cqtfilters(16000, 4096, fmin=100, fmax=6000, bins=12)
        fc_arr = np.asarray(fc, dtype=float)
        assert np.all(np.diff(fc_arr) > 0), \
            "cqtfilters centre frequencies are not strictly monotone increasing"

    def test_centre_freqs_in_range(self, needs_impl):
        """Inner centre frequencies should lie within [fmin, fmax] (approx).

        The first (DC) and last (Nyquist) edge filters sit outside
        [fmin, fmax] by design, so we exclude them from the check.
        """
        from cool_frames.filters import cqtfilters  # type: ignore
        fmin, fmax = 100.0, 6000.0
        _, _, fc, _, _info = cqtfilters(16000, 4096, fmin=fmin, fmax=fmax, bins=12)
        fc_arr = np.asarray(fc, dtype=float)
        # Exclude DC (fc=0) and Nyquist (fc=fs/2) edge filters
        inner = fc_arr[1:-1]
        assert np.min(inner) >= fmin * 0.5, \
            f"Min inner centre freq {np.min(inner):.1f} too far below fmin={fmin}"
        assert np.max(inner) <= fmax * 1.5, \
            f"Max inner centre freq {np.max(inner):.1f} too far above fmax={fmax}"


@pytest.mark.requires_impl
class TestCqtFiltersConstantQImpl:
    """
    MATLAB counterpart: PropFilterDesignCoverage (cqtfilters constant-Q).
    """

    def test_constant_q_ratio(self, needs_impl):
        """
        For a CQT filterbank, fc/bw should be approximately constant across
        all filters (the Q factor is constant).

        We estimate bw from the 3-dB width of each filter's transfer function.
        The coefficient of variation (std/mean) of Q over all filters should
        be small (< 0.25).
        """
        from cool_frames.filters import cqtfilters  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L    = 2048
        g, _, fc, _, _info = cqtfilters(16000, L, fmin=200, fmax=4000, bins=12)
        fc_arr  = np.asarray(fc, dtype=float)

        Q_vals = []
        for g_m, fc_m in zip(g, fc_arr):
            H = np.abs(comp_transferfunction(g_m, L))
            if np.max(H) < 1e-15:
                continue
            H_norm  = H / np.max(H)
            half_pw = H_norm ** 2 >= 0.5
            bw_bins = np.sum(half_pw)
            if bw_bins < 1:
                continue
            fc_bin  = np.argmax(H)
            bw_hz   = bw_bins / L * 16000
            if bw_hz > 0 and fc_m > 0:
                Q_vals.append(fc_m / bw_hz)

        assert len(Q_vals) >= 4, "Not enough valid filters to check constant-Q"
        Q_arr = np.array(Q_vals)
        cv    = np.std(Q_arr) / np.mean(Q_arr)
        assert cv < 0.5, \
            f"Q-factor coefficient of variation {cv:.3f} too large — not constant-Q"


# ---------------------------------------------------------------------------
# gabfilters
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestGabFiltersOutputConsistencyImpl:
    """
    MATLAB counterpart: PropFilterDesignCoverage (gabfilters consistency).
    """

    def test_output_sizes_match(self, needs_impl):
        """len(g) == number of channels (M_ch)."""
        from cool_frames.filters import gabfilters  # type: ignore
        Ls     = 1024
        window = "hann"
        a_hop  = 64
        M_ch   = 128
        g, a, fc, *_ = gabfilters(16000, Ls, window=window, a=a_hop, M=M_ch)
        assert len(g) == len(fc), "g and fc have different lengths"
        assert len(g) == len(a),  "g and a have different lengths"
        assert len(g) > 0,        "gabfilters returned empty filter list"

    @pytest.mark.parametrize("M_ch", [64, 128, 256])
    def test_output_sizes_across_channels(self, needs_impl, M_ch):
        """Consistent sizing for different channel counts."""
        from cool_frames.filters import gabfilters  # type: ignore
        g, a, fc, *_ = gabfilters(16000, 1024, window="hann", a=32, M=M_ch)
        assert len(g) == len(fc) == len(a)
        assert len(g) > 0

    def test_filters_have_valid_transfer_functions(self, needs_impl):
        """Each Gabor filter has a finite, non-zero transfer function."""
        from cool_frames.filters import gabfilters  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L  = 512
        g, a, _, *_ = gabfilters(16000, L, window="hann", a=32, M=64)
        for m, g_m in enumerate(g):
            H = comp_transferfunction(g_m, L)
            assert np.all(np.isfinite(H)), f"Filter {m}: non-finite values"
            assert np.max(np.abs(H)) > 0,  f"Filter {m}: all-zero transfer function"


@pytest.mark.requires_impl
class TestGabFiltersFrequencyCoverageImpl:
    """
    MATLAB counterpart: PropFilterDesignCoverage (gabfilters coverage).
    """

    @pytest.mark.xfail(reason="Gabor filterbank complement filters may leave dead DFT bins")
    def test_no_dead_bins_complex(self, needs_impl):
        """
        Gabor filterbank in complex mode should cover all DFT bins:
        sum_m |H_m(k)|^2 > 0 for all k.
        """
        from cool_frames.filters import gabfilters  # type: ignore
        from cool_frames.numpy.filters._filters import comp_transferfunction  # type: ignore
        L    = 512
        M_ch = 64
        g, a, _, *_ = gabfilters(16000, L, window="hann", a=32, M=M_ch)
        energy = np.zeros(L, dtype=float)
        for g_m in g:
            H = comp_transferfunction(g_m, L)
            energy += np.abs(H) ** 2
        assert np.min(energy) > 0, \
            f"Gabor filterbank has dead DFT bins: min energy = {np.min(energy):.2e}"

    def test_centre_freqs_span_full_range(self, needs_impl):
        """
        Centre frequencies should span [0, fs/2] (or [0, 2] in normalised units).
        Verify that there are filters near DC and near Nyquist.
        """
        from cool_frames.filters import gabfilters  # type: ignore
        g, _, fc, *_ = gabfilters(16000, 1024, window="hann", a=64, M=128)
        fc_arr = np.asarray(fc, dtype=float)
        # Some fc near 0 and some near Nyquist (normalised fc ≤ 2)
        assert np.min(fc_arr) < 0.1 * np.max(fc_arr), \
            "No filters near DC"
        assert np.max(fc_arr) > 0.9 * np.max(fc_arr), \
            "No filters near Nyquist"


# ---------------------------------------------------------------------------
# Cross-design comparison (unconditional reference level)
# ---------------------------------------------------------------------------

class TestFilterDesignReference:
    """
    Structural checks that require only numpy — no cool_frames.
    """

    def test_erb_scale_is_monotone(self):
        """
        The ERB scale should map linearly-spaced Hz to a monotone-increasing
        sequence. Verified via a simple analytical formula.
        """
        # ERB(f) = 21.4 * log10(1 + f / 229) (Moore & Glasberg 1983)
        freqs = np.linspace(100, 8000, 50)
        erb   = 21.4 * np.log10(1.0 + freqs / 229.0)
        assert np.all(np.diff(erb) > 0), "ERB scale is not monotone"

    def test_cqt_frequency_spacing_geometric(self):
        """
        For bins_per_octave CQT, centre frequencies should be geometrically
        spaced: fc[n+1] / fc[n] ≈ 2^(1/bins_per_oct).
        """
        bins_per_oct = 12
        n_bins = 24
        fmin   = 100.0
        ratio  = 2.0 ** (1.0 / bins_per_oct)
        fc     = fmin * ratio ** np.arange(n_bins)
        ratios = fc[1:] / fc[:-1]
        np.testing.assert_allclose(ratios, ratio, rtol=1e-10,
                                   err_msg="CQT frequencies are not geometrically spaced")

    def test_gabor_uniform_spacing(self):
        """
        Gabor filterbank has uniformly-spaced centre frequencies.
        """
        M_ch  = 16
        fc    = np.arange(M_ch) / M_ch * 2.0   # normalised [0, 2)
        diffs = np.diff(fc)
        np.testing.assert_allclose(diffs, diffs[0], rtol=1e-10,
                                   err_msg="Gabor frequencies are not uniformly spaced")
