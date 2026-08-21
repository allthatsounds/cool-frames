"""
test_analysis_module.py
=======================
Tests for the matrix-spectral analyzer module
(``cool_frames.numpy.layer2._analysis``).

Covers all three public entry-points:
  - ``analyze_coefficients``  (coefficient-level analysis)
  - ``analyze_filterbank``    (full filterbank analysis)
  - ``analyze_frame_operator``(materialised frame operator)
  - ``print_report``          (pretty-print formatter)
"""
from __future__ import annotations

import io

import pytest

import numpy as np

# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def aud_fb():
    """Return (g, a, L, A, B, M) for an auditory filterbank."""
    from cool_frames.filterbanks import filterbankbounds
    from cool_frames.filters import audfilters, filterbanklength

    g, a, fc, _, _info = audfilters(8000, 256)
    L = filterbanklength(256, a)
    A, B = filterbankbounds(g, a, L)
    M = len(g)
    return g, a, L, A, B, M


@pytest.fixture
def aud_coeffs(aud_fb):
    """Return (c, x, g, a, L) — coefficients and test signal."""
    from cool_frames.filterbanks import filterbank

    g, a, L, A, B, M = aud_fb
    t = np.arange(L, dtype=float) / 8000
    x = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1000 * t)
    c = filterbank(x, g, a)
    return c, x, g, a, L


# ===================================================================
# 1.  analyze_coefficients
# ===================================================================

@pytest.mark.requires_impl
class TestAnalyzeCoefficients:
    """Tests for ``analyze_coefficients``."""

    def test_returns_dict_with_expected_keys(self, needs_impl, aud_coeffs):
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        c, x, g, a, L = aud_coeffs
        report = analyze_coefficients(c, a)

        assert isinstance(report, dict)
        for key in ('shape', 'energy', 'sparsity', 'dynamics', 'coherence'):
            assert key in report, f"Missing top-level key '{key}'"

    def test_shape_section(self, needs_impl, aud_coeffs):
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        c, x, g, a, L = aud_coeffs
        report = analyze_coefficients(c, a)
        sh = report['shape']

        assert sh['M'] == len(g)
        assert sh['Nsum'] == sum(len(np.asarray(cm).ravel()) for cm in c)
        assert isinstance(sh['uniform'], bool)
        assert len(sh['N_list']) == sh['M']

    def test_energy_positive(self, needs_impl, aud_coeffs):
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        c, x, g, a, L = aud_coeffs
        report = analyze_coefficients(c, a)
        en = report['energy']

        assert en['total'] > 0, "Total energy must be positive for non-zero signal"
        assert 0.0 <= en['peak_fraction'] <= 1.0
        assert 0 <= en['peak_channel'] < report['shape']['M']
        assert en['entropy'] >= 0.0

    def test_sparsity_in_range(self, needs_impl, aud_coeffs):
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        c, x, g, a, L = aud_coeffs
        report = analyze_coefficients(c, a)
        sp = report['sparsity']

        assert 0.0 <= sp['hoyer'] <= 1.0, f"Hoyer sparsity {sp['hoyer']} out of [0,1]"
        assert 0.0 <= sp['active_fraction'] <= 1.0
        assert sp['l1_norm'] >= sp['l2_norm'], "L1 >= L2 for any vector"

    def test_gini_in_range(self, needs_impl, aud_coeffs):
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        c, x, g, a, L = aud_coeffs
        report = analyze_coefficients(c, a)

        gini = report['energy']['gini']
        assert -0.1 <= gini <= 1.0, f"Gini coefficient {gini} out of range"

    def test_dynamics_finite(self, needs_impl, aud_coeffs):
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        c, x, g, a, L = aud_coeffs
        report = analyze_coefficients(c, a)
        dy = report['dynamics']

        assert dy['max_abs'] > 0
        assert np.isfinite(dy['dynamic_range_dB']) or dy['dynamic_range_dB'] == float('inf')

    def test_signal_energy_enables_frame_section(self, needs_impl, aud_coeffs):
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        c, x, g, a, L = aud_coeffs
        report_no_e = analyze_coefficients(c, a)
        report_e = analyze_coefficients(c, a, signal_energy=float(np.dot(x, x)))

        assert 'frame' not in report_no_e
        assert 'frame' in report_e
        assert report_e['frame']['energy_ratio'] > 0

    def test_uniform_input(self, needs_impl):
        """A stacked array is accepted, in the layout ``filterbank`` produces.

        ``filterbank(..., stack=True)`` returns ``(N, M)`` — time down the rows,
        channels across the columns, the same convention ``plotfilterbank``
        uses.  Until v0.1.1 ``analyze_coefficients`` read it as ``(M, N)``, so
        every per-channel statistic was computed over time slices; this test
        asserted the wrong convention against an arbitrary random array, which
        is why it did not catch it.  Anchoring to the producer removes the
        ambiguity.
        """
        from cool_frames.numpy.filterbanks import filterbank
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients
        from cool_frames.numpy.filters import audfilters

        g, _a, _fc, L, _info = audfilters(4000, 512)
        x = np.random.default_rng(0).standard_normal(512)
        C = filterbank(x, g, np.full(len(g), 32), L=L, stack=True)

        report = analyze_coefficients(C)

        assert report['shape']['uniform'] is True
        assert report['shape']['M'] == C.shape[1] == len(g)
        assert report['shape']['N'] == C.shape[0]
        assert report['coherence']['mean_abs_correlation'] is not None

    def test_zero_signal(self, needs_impl):
        """All-zero coefficients should not crash and report zero energy."""
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        c_zero = [np.zeros(32) for _ in range(5)]
        report = analyze_coefficients(c_zero)

        assert report['energy']['total'] == 0.0
        assert report['sparsity']['hoyer'] == 0.0


# ===================================================================
# 2.  analyze_filterbank
# ===================================================================

@pytest.mark.requires_impl
class TestAnalyzeFilterbank:
    """Tests for ``analyze_filterbank``."""

    def test_returns_dict_with_expected_keys(self, needs_impl, aud_fb):
        from cool_frames.numpy.filterbanks._analysis import analyze_filterbank

        g, a, L, A, B, M = aud_fb
        report = analyze_filterbank(g, a, L, materialise=False)

        for key in ('filterbank', 'frame', 'operator', 'coherence', 'coefficients'):
            assert key in report, f"Missing top-level key '{key}'"

    def test_frame_bounds_match(self, needs_impl, aud_fb):
        from cool_frames.numpy.filterbanks._analysis import analyze_filterbank

        g, a, L, A, B, M = aud_fb
        report = analyze_filterbank(g, a, L, materialise=False)

        assert abs(report['frame']['A'] - A) < 1e-8
        assert abs(report['frame']['B'] - B) < 1e-8
        assert report['frame']['kappa'] == pytest.approx(B / A, rel=1e-8)

    def test_filterbank_metadata(self, needs_impl, aud_fb):
        from cool_frames.numpy.filterbanks._analysis import analyze_filterbank

        g, a, L, A, B, M = aud_fb
        report = analyze_filterbank(g, a, L, materialise=False)
        fb = report['filterbank']

        assert fb['M'] == M
        assert fb['L'] == L
        assert fb['redundancy'] > 1.0, "Overcomplete filterbank should have redundancy > 1"

    def test_operator_lipschitz(self, needs_impl, aud_fb):
        from cool_frames.numpy.filterbanks._analysis import analyze_filterbank

        g, a, L, A, B, M = aud_fb
        report = analyze_filterbank(g, a, L, materialise=False)
        op = report['operator']

        assert op['lipschitz_analysis'] > 0
        assert op['lipschitz_frame_op'] == pytest.approx(B, rel=1e-8)
        assert op['frame_op_ratio_min'] > 0
        assert op['frame_op_ratio_max'] >= op['frame_op_ratio_min']

    def test_mutual_coherence_in_range(self, needs_impl, aud_fb):
        from cool_frames.numpy.filterbanks._analysis import analyze_filterbank

        g, a, L, A, B, M = aud_fb
        report = analyze_filterbank(g, a, L, materialise=False)

        mu = report['coherence']['mutual_coherence']
        assert 0.0 <= mu <= 1.0, f"Mutual coherence {mu} out of [0, 1]"

    def test_materialise_adds_eigenvalue_section(self, needs_impl, aud_fb):
        from cool_frames.numpy.filterbanks._analysis import analyze_filterbank

        g, a, L, A, B, M = aud_fb
        report_no = analyze_filterbank(g, a, L, materialise=False)
        report_yes = analyze_filterbank(g, a, L, materialise=True)

        assert 'eigenvalues' not in report_no
        assert 'eigenvalues' in report_yes
        assert report_yes['eigenvalues']['positive_definite'] is True

    def test_coefficients_section_present(self, needs_impl, aud_fb):
        from cool_frames.numpy.filterbanks._analysis import analyze_filterbank

        g, a, L, A, B, M = aud_fb
        report = analyze_filterbank(g, a, L, materialise=False)

        coeff = report['coefficients']
        assert 'shape' in coeff
        assert 'energy' in coeff
        assert coeff['shape']['M'] == M

    def test_optimal_learning_rate_positive(self, needs_impl, aud_fb):
        from cool_frames.numpy.filterbanks._analysis import analyze_filterbank

        g, a, L, A, B, M = aud_fb
        report = analyze_filterbank(g, a, L, materialise=False)

        eta = report['frame']['optimal_learning_rate']
        assert eta > 0
        assert eta == pytest.approx(2.0 / (A + B), rel=1e-8)


# ===================================================================
# 3.  analyze_frame_operator
# ===================================================================

@pytest.mark.requires_impl
class TestAnalyzeFrameOperator:
    """Tests for ``analyze_frame_operator``."""

    def test_returns_expected_keys(self, needs_impl):
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.numpy.filterbanks._analysis import analyze_frame_operator

        g, a, fc, _, _info = audfilters(8000, 128)
        L = filterbanklength(128, a)
        report = analyze_frame_operator(g, a, L)

        for key in ('eigenvalues', 'erank', 'nuclear_norm', 'frobenius_norm',
                     'operator_norm', 'spectral_gap', 'diag_energy_ratio',
                     'truncation_90', 'symmetry_error', 'positive_definite'):
            assert key in report, f"Missing key '{key}'"

    def test_eigenvalues_array(self, needs_impl):
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.numpy.filterbanks._analysis import analyze_frame_operator

        g, a, fc, _, _info = audfilters(8000, 128)
        L = filterbanklength(128, a)
        report = analyze_frame_operator(g, a, L)

        eigvals = report['eigenvalues']
        assert isinstance(eigvals, np.ndarray)
        assert len(eigvals) == L
        assert np.all(np.diff(eigvals) >= -1e-10), "Eigenvalues should be sorted"

    def test_positive_definite(self, needs_impl):
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.numpy.filterbanks._analysis import analyze_frame_operator

        g, a, fc, _, _info = audfilters(8000, 128)
        L = filterbanklength(128, a)
        report = analyze_frame_operator(g, a, L)

        assert report['positive_definite'] is True
        assert report['eigenvalues_min'] > 0

    def test_symmetry(self, needs_impl):
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.numpy.filterbanks._analysis import analyze_frame_operator

        g, a, fc, _, _info = audfilters(8000, 128)
        L = filterbanklength(128, a)
        report = analyze_frame_operator(g, a, L)

        assert report['symmetry_error'] < 1e-10, \
            f"Frame operator not symmetric: err={report['symmetry_error']:.2e}"

    def test_nuclear_norm_positive(self, needs_impl):
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.numpy.filterbanks._analysis import analyze_frame_operator

        g, a, fc, _, _info = audfilters(8000, 128)
        L = filterbanklength(128, a)
        report = analyze_frame_operator(g, a, L)

        assert report['nuclear_norm'] > 0
        assert report['frobenius_norm'] > 0
        assert report['operator_norm'] > 0

    def test_erank_in_range(self, needs_impl):
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.numpy.filterbanks._analysis import analyze_frame_operator

        g, a, fc, _, _info = audfilters(8000, 128)
        L = filterbanklength(128, a)
        report = analyze_frame_operator(g, a, L)

        assert 1.0 <= report['erank'] <= L
        assert 1 <= report['truncation_90'] <= L


# ===================================================================
# 4.  print_report
# ===================================================================

@pytest.mark.requires_impl
class TestPrintReport:
    """Tests for ``print_report``."""

    def test_coeff_report_prints(self, needs_impl, aud_coeffs):
        from cool_frames.numpy.filterbanks._analysis import (
            analyze_coefficients,
            print_report,
        )

        c, x, g, a, L = aud_coeffs
        report = analyze_coefficients(c, a)

        buf = io.StringIO()
        print_report(report, file=buf)
        output = buf.getvalue()

        assert len(output) > 100, "Report too short"
        assert "Total energy" in output
        assert "Hoyer sparsity" in output

    def test_filterbank_report_prints(self, needs_impl, aud_fb):
        from cool_frames.numpy.filterbanks._analysis import (
            analyze_filterbank,
            print_report,
        )

        g, a, L, A, B, M = aud_fb
        report = analyze_filterbank(g, a, L, materialise=False)

        buf = io.StringIO()
        print_report(report, file=buf)
        output = buf.getvalue()

        assert "Filterbank structure" in output
        assert "Frame bounds" in output
        assert "Condition number" in output

    def test_eigenvalue_report_prints(self, needs_impl):
        from cool_frames.filters import audfilters, filterbanklength
        from cool_frames.numpy.filterbanks._analysis import (
            analyze_frame_operator,
            print_report,
        )

        g, a, fc, _, _info = audfilters(8000, 128)
        L = filterbanklength(128, a)
        report = analyze_frame_operator(g, a, L)

        buf = io.StringIO()
        print_report(report, file=buf)
        output = buf.getvalue()

        assert "eigenvalue" in output.lower()
        assert "Effective rank" in output

    def test_full_report_with_materialise(self, needs_impl, aud_fb):
        from cool_frames.numpy.filterbanks._analysis import (
            analyze_filterbank,
            print_report,
        )

        g, a, L, A, B, M = aud_fb
        report = analyze_filterbank(g, a, L, materialise=True)

        buf = io.StringIO()
        print_report(report, file=buf)
        output = buf.getvalue()

        # Should contain all major sections
        assert "Filterbank structure" in output
        assert "Frame bounds" in output
        assert "Operator properties" in output
        assert "eigenvalue" in output.lower()

    def test_no_crash_on_zero_signal(self, needs_impl):
        from cool_frames.numpy.filterbanks._analysis import (
            analyze_coefficients,
            print_report,
        )

        c_zero = [np.zeros(16) for _ in range(3)]
        report = analyze_coefficients(c_zero)

        buf = io.StringIO()
        print_report(report, file=buf)
        assert len(buf.getvalue()) > 50


# ===================================================================
# 5.  Edge cases and non-uniform inputs
# ===================================================================

@pytest.mark.requires_impl
class TestEdgeCases:
    """Edge cases: non-uniform coefficients, single channel, etc."""

    def test_non_uniform_coefficients(self, needs_impl):
        """List of arrays with different lengths (non-uniform filterbank)."""
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        c = [np.random.randn(32), np.random.randn(16), np.random.randn(64)]
        report = analyze_coefficients(c)

        assert report['shape']['uniform'] is False
        assert report['shape']['M'] == 3
        assert report['shape']['N'] is None
        assert report['coherence']['mean_abs_correlation'] is None

    def test_single_channel(self, needs_impl):
        """Single-channel filterbank should work."""
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        c = [np.random.randn(100)]
        report = analyze_coefficients(c)

        assert report['shape']['M'] == 1
        assert report['energy']['peak_channel'] == 0
        assert report['energy']['peak_fraction'] == 1.0

    def test_complex_coefficients(self, needs_impl):
        """Complex coefficients should be handled correctly."""
        from cool_frames.numpy.filterbanks._analysis import analyze_coefficients

        rng = np.random.default_rng(0)
        c = [rng.standard_normal(32) + 1j * rng.standard_normal(32)
             for _ in range(5)]
        report = analyze_coefficients(c)

        assert report['energy']['total'] > 0
        assert report['sparsity']['l2_norm'] > 0

    def test_cqt_filterbank(self, needs_impl):
        """CQT filterbank should analyse without error."""
        from cool_frames.filters import cqtfilters, filterbanklength
        from cool_frames.numpy.filterbanks._analysis import analyze_filterbank

        g, a, fc, _, _info = cqtfilters(8000, 256, fmin=50, fmax=3900, bins=12)
        L = filterbanklength(256, a)
        report = analyze_filterbank(g, a, L, materialise=False)

        assert report['frame']['A'] > 0
        assert report['frame']['kappa'] >= 1.0
        assert report['filterbank']['M'] > 0
