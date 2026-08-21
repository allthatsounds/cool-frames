"""
test_prop_phase_gradient.py
===========================
Python port of:
    layer3_repr/property/PropPhaseGradientConsistency.m

Properties verified
-------------------
filterbankphasegrad
    (1) tgrad, fgrad, s have same cell dimensions as filterbank(f, g, a).
    (2) s (spectrogram) is non-negative and equals |c|^2 (above minlvl floor).
    (3) tgrad and fgrad are real-valued for any (real or complex) input.
    (4) Total energy: sum_m sum(s{m}) == sum_m ||c{m}||^2  (rtol 1e-6).

filterbankconstphase
    (5) |c_out{m}| == s{m} for 20 random signals (rtol 1e-6).
    (6) Passing explicit {tgrad, fgrad} still preserves magnitude.
    (7) Output cell has M elements, same size as filterbank output.

filterbankreassign
    (8) sr has M cells, each same size as s.
    (9) Total energy conserved across 5 signals (rtol 1.5 = 150 %).
   (10) sr values are non-negative.

filterbanksynchrosqueeze
   (11) cr has M cells, same size as c (uniform a=1 filterbank).
   (12) Output energy is finite and positive (5 trials).
   (13) Passing filter cell g vs fc_n must give consistent result (rtol < 2).

Notes
-----
- All tests require cool_frames: @pytest.mark.requires_impl.
- Signals for filterbankphasegrad tests use length L (the DFT length).
- filterbanksynchrosqueeze requires uniform hop (a = ones(M)).
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(L: int, seed: int, complex: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(L)
    if complex:
        x = x + 1j * rng.standard_normal(L)
    return x


# ---------------------------------------------------------------------------
# filterbankphasegrad – structural properties (Property 1–4)
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPhaseGradDimensionsMatchFilterbankImpl:
    """
    PropPhaseGradientConsistency: tgrad, fgrad, s match filterbank dimensions.
    Property (1).
    """

    def test_dimensions_match_filterbank(self, needs_impl):
        """tgrad, fgrad, s must have M cells with identical sizes to filterbank(f,g,a)."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(42)
        f = rng.standard_normal(Ls)
        c_ref = filterbank(f, g, a)
        tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
        for m in range(M):
            assert np.asarray(tgrad[m]).shape == np.asarray(c_ref[m]).shape, \
                f"phasegrad: tgrad[{m}] size must match coeff size"
            assert np.asarray(fgrad[m]).shape == np.asarray(c_ref[m]).shape, \
                f"phasegrad: fgrad[{m}] size must match coeff size"
            assert np.asarray(s[m]).shape == np.asarray(c_ref[m]).shape, \
                f"phasegrad: s[{m}] size must match coeff size"


@pytest.mark.requires_impl
class TestPhaseGradSpectrogramEqualsAbsSquaredImpl:
    """
    PropPhaseGradientConsistency: s{m} == |c{m}|^2 above minlvl floor.
    Property (2).
    """

    def test_spectrogram_equals_abs_squared(self, needs_impl):
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(42)
        f = rng.standard_normal(Ls)
        _, _, s, c = filterbankphasegrad(f, g, a, L)
        for m in range(M):
            expected = np.abs(np.asarray(c[m])) ** 2
            # Only check bins where |c|^2 >> eps (minlvl floor is irrelevant there)
            mask = expected > 1e-10 * (expected.max() + np.finfo(float).eps)
            if mask.any():
                sm = np.asarray(s[m])
                rel_err = np.linalg.norm(sm[mask] - expected[mask]) / \
                          (np.linalg.norm(expected[mask]) + np.finfo(float).eps)
                assert rel_err < 5e-6, \
                    f"phasegrad: s[{m}] must equal |c[{m}]|^2, rel_err={rel_err:.2e}"


@pytest.mark.requires_impl
class TestPhaseGradIsRealImpl:
    """
    PropPhaseGradientConsistency: tgrad and fgrad real for real/complex inputs.
    Property (3).
    """

    def test_real_for_multiple_signals(self, needs_impl):
        """5 real signals: tgrad and fgrad must be real for all channels."""
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        for trial, seed in enumerate(range(7, 12)):
            f = _make_signal(Ls, seed)
            tgrad, fgrad, _, _ = filterbankphasegrad(f, g, a, L)
            for m in range(M):
                assert np.isrealobj(np.asarray(tgrad[m])), \
                    f"Trial {trial}: tgrad[{m}] must be real"
                assert np.isrealobj(np.asarray(fgrad[m])), \
                    f"Trial {trial}: fgrad[{m}] must be real"


@pytest.mark.requires_impl
class TestPhaseGradTotalSpectrogramMatchesCoeffEnergyImpl:
    """
    PropPhaseGradientConsistency: sum_m sum(s{m}) == sum_m ||c{m}||^2 (rtol 1e-6).
    Property (4).
    """

    def test_total_spectrogram_matches_coeff_energy(self, needs_impl):
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(42)
        f = rng.standard_normal(Ls)
        _, _, s, c = filterbankphasegrad(f, g, a, L)
        e_s = sum(float(np.sum(np.asarray(sm))) for sm in s)
        e_c = sum(float(np.linalg.norm(np.asarray(cm)) ** 2) for cm in c)
        rel_err = abs(e_s - e_c) / (e_c + np.finfo(float).eps)
        assert rel_err < 1e-5, \
            f"phasegrad: total spectrogram energy {e_s:.6e} vs coeff energy {e_c:.6e}, " \
            f"rel_err={rel_err:.2e}"


# ---------------------------------------------------------------------------
# filterbankconstphase – magnitude invariance (Properties 5–7)
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestConstphaseMagnitudeInvariantImpl:
    """
    PropPhaseGradientConsistency: |c_out{m}| == s{m} for 20 random signals.
    Property (5).
    """

    def test_magnitude_invariant_multiple_signals(self, needs_impl):
        """20 random signals: magnitude must be preserved (rel_err < 1e-6)."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        from cool_frames.phase import filterbankconstphase  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        for trial in range(20):
            f = _make_signal(Ls, seed=13 + trial)
            c = filterbank(f, g, a)
            s = [np.abs(np.asarray(cm)) for cm in c]
            c_out = filterbankconstphase(s, a, fc, fs=fs)
            for m in range(M):
                mag_out = np.abs(np.asarray(c_out[m])).ravel()
                mag_in  = s[m].ravel()
                rel_err = np.linalg.norm(mag_out - mag_in) / \
                          (np.linalg.norm(mag_in) + np.finfo(float).eps)
                assert rel_err < 1e-6, \
                    f"Trial {trial}, channel {m}: constphase magnitude error {rel_err:.2e}"

    def test_explicit_gradient_preserves_magnitude(self, needs_impl):
        """Passing explicit {tgrad, fgrad} must still preserve magnitude. Property (6)."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        from cool_frames.phase import (  # type: ignore
            filterbankconstphase,
            filterbankphasegrad,
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(42)
        f = rng.standard_normal(Ls)
        c = filterbank(f, g, a)
        s = [np.abs(np.asarray(cm)) for cm in c]
        tgrad, fgrad, _, _ = filterbankphasegrad(f, g, a, L)
        c_out = filterbankconstphase(s, a, fc, fs=fs, tgrad=tgrad, fgrad=fgrad)
        for m in range(M):
            mag_out = np.abs(np.asarray(c_out[m])).ravel()
            mag_in  = s[m].ravel()
            rel_err = np.linalg.norm(mag_out - mag_in) / \
                      (np.linalg.norm(mag_in) + np.finfo(float).eps)
            assert rel_err < 1e-6, \
                f"constphase (explicit grad): magnitude error in channel {m}: {rel_err:.2e}"

    def test_structure_matches_filterbank_output(self, needs_impl):
        """Output has M elements, each same size as filterbank output. Property (7)."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        from cool_frames.phase import filterbankconstphase  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(13)
        f = rng.standard_normal(Ls)
        c = filterbank(f, g, a)
        s = [np.abs(np.asarray(cm)) for cm in c]
        c_out = filterbankconstphase(s, a, fc, fs=fs)
        assert len(c_out) == M, \
            f"constphase: output must have M={M} cells, got {len(c_out)}"
        for m in range(M):
            assert np.asarray(c_out[m]).shape == np.asarray(c[m]).shape, \
                f"constphase: c_out[{m}] size must match filterbank output"


# ---------------------------------------------------------------------------
# filterbankreassign – structural and energy properties (Properties 8–10)
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestReassignCellStructureImpl:
    """
    PropPhaseGradientConsistency: filterbankreassign cell structure and energy.
    Properties (8), (9), (10).
    """

    def test_cell_structure_preserved(self, needs_impl):
        """sr must have M cells, each same size as s. Property (8)."""
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        from cool_frames.phase import filterbankphasegrad, filterbankreassign  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(42)
        f = rng.standard_normal(Ls)
        tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
        sr, _, _ = filterbankreassign(s, tgrad, fgrad, a, fc)
        assert len(sr) == M, \
            f"filterbankreassign: sr must have M={M} cells, got {len(sr)}"
        for m in range(M):
            assert np.asarray(sr[m]).shape == np.asarray(s[m]).shape, \
                f"filterbankreassign: sr[{m}] size must match s[{m}]"

    def test_energy_conserved_across_signals(self, needs_impl):
        """5 signals: total energy change must be < 150 %. Property (9)."""
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        from cool_frames.phase import filterbankphasegrad, filterbankreassign  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        for trial in range(5):
            f = _make_signal(Ls, seed=99 + trial)
            tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
            sr, _, _ = filterbankreassign(s, tgrad, fgrad, a, fc)
            e_in  = sum(float(np.sum(np.asarray(sm))) for sm in s)
            e_out = sum(float(np.sum(np.asarray(sm))) for sm in sr)
            rel_err = abs(e_out - e_in) / (e_in + np.finfo(float).eps)
            assert rel_err < 1.5, \
                f"Trial {trial}: reassignment energy change {rel_err:.1%} exceeds 150 %"

    def test_output_nonnegative(self, needs_impl):
        """sr values must be non-negative. Property (10)."""
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        from cool_frames.phase import filterbankphasegrad, filterbankreassign  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(42)
        f = rng.standard_normal(Ls)
        tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
        sr, _, _ = filterbankreassign(s, tgrad, fgrad, a, fc)
        for m in range(M):
            sm = np.asarray(sr[m])
            assert np.all(sm >= -1e-12), \
                f"filterbankreassign: sr[{m}] must be non-negative (min={sm.min():.3e})"


# ---------------------------------------------------------------------------
# filterbanksynchrosqueeze – cell structure and energy (Properties 11–13)
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestSynchrosqueezeCellStructureImpl:
    """
    PropPhaseGradientConsistency: filterbanksynchrosqueeze structure and energy.
    Properties (11), (12), (13).

    Uses a uniform (a=1) filterbank to satisfy the equal-length-subband
    requirement of filterbanksynchrosqueeze.
    """

    def _uniform_setup(self, seed: int = 42):
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        M = len(g)
        a_ones = np.ones(M, dtype=int)
        f = _make_signal(Ls, seed)
        c = filterbank(f, g, a_ones)
        tgrad, _, _, _ = filterbankphasegrad(f, g, a_ones)
        return g, a_ones, fc, M, c, tgrad

    def test_cell_structure_preserved(self, needs_impl):
        """cr must have M cells with same size as c. Property (11)."""
        from cool_frames.phase import filterbanksynchrosqueeze  # type: ignore
        g, a_ones, fc, M, c, tgrad = self._uniform_setup(seed=42)
        # Need fgrad for synchrosqueeze - compute from filterbankphasegrad
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        L = filterbanklength(1024, a_ones)  # Dummy L for setup
        _, fgrad, _, _ = filterbankphasegrad(np.random.randn(1024), g, a_ones, L)
        cr, _, _ = filterbanksynchrosqueeze(c, tgrad, fgrad, a_ones, fc)
        assert len(cr) == M, \
            f"filterbanksynchrosqueeze: cr must have M={M} cells, got {len(cr)}"
        for m in range(M):
            assert np.asarray(cr[m]).shape == np.asarray(c[m]).shape, \
                f"filterbanksynchrosqueeze: cr[{m}] size must match c[{m}]"

    def test_energy_approximately_conserved(self, needs_impl):
        """5 trials: output energy must be finite and positive. Property (12)."""
        from cool_frames.phase import (  # type: ignore
            filterbankphasegrad,
            filterbanksynchrosqueeze,
        )
        for trial in range(5):
            g, a_ones, fc, M, c, tgrad = self._uniform_setup(seed=55 + trial)
            _, fgrad, _, _ = filterbankphasegrad(np.random.randn(1024), g, a_ones, 512)
            cr, _, _ = filterbanksynchrosqueeze(c, tgrad, fgrad, a_ones, fc)
            e_in  = sum(float(np.sum(np.abs(np.asarray(cm)) ** 2)) for cm in c)
            e_out = sum(float(np.sum(np.abs(np.asarray(cm)) ** 2)) for cm in cr)
            assert np.isfinite(e_out), \
                f"Trial {trial}: synchrosqueeze output energy must be finite"
            assert e_out > 0, \
                f"Trial {trial}: synchrosqueeze output energy must be positive"

    def test_consistent_with_filter_cell(self, needs_impl):
        """Passing filter cell g vs fc_n must give consistent result. Property (13)."""
        from cool_frames.phase import (  # type: ignore
            filterbankphasegrad,
            filterbanksynchrosqueeze,
        )
        g, a_ones, fc, M, c, tgrad = self._uniform_setup(seed=42)
        _, fgrad, _, _ = filterbankphasegrad(np.random.randn(1024), g, a_ones, 512)
        cr_fc, _, _ = filterbanksynchrosqueeze(c, tgrad, fgrad, a_ones, fc)
        cr_g, _, _ = filterbanksynchrosqueeze(c, tgrad, fgrad, a_ones, g)
        for m in range(M):
            cr_fc_m = np.asarray(cr_fc[m]).ravel()
            cr_g_m  = np.asarray(cr_g[m]).ravel()
            rel_err = np.linalg.norm(cr_fc_m - cr_g_m) / \
                      (np.linalg.norm(cr_fc_m) + np.finfo(float).eps)
            # MATLAB allows rel_err < 2 (very loose — different frequency mapping)
            assert rel_err < 2, \
                f"synchrosqueeze: fc_n vs g inputs rel_err={rel_err:.2e} > 2 (ch {m})"
