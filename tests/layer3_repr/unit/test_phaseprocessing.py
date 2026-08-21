"""
test_phaseprocessing.py
=======================
Python port of:
    layer3_repr/unit/TestPhaseProcessing.m

Covers
------
    filterbankphasegrad   – output count, shapes, dtypes, spectrogram value,
                            coefficient accuracy, omitting L argument
    filterbankconstphase  – output structure, magnitude invariance,
                            explicit phase-gradient input, no-error smoke
    filterbankreassign    – output structure, cell sizes preserved,
                            three-output form with repos/Lc,
                            accepts filter cell, energy approximately conserved
    filterbanksynchrosqueeze – output structure, cell sizes preserved,
                               three-output form, accepts filter cell

Notes
-----
- All tests require the cool_frames package: @pytest.mark.requires_impl.
- filterbanksynchrosqueeze requires equal-length subbands (a = ones(M, 1)).
- Signals used with filterbankreassign / filterbanksynchrosqueeze must have
  length L (the DFT length), not Ls.
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# filterbankphasegrad
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbankphasegradImpl:
    """Port of TestPhaseProcessing (filterbankphasegrad section)."""

    def test_output_count_M(self, needs_impl):
        """All four outputs must be returned and each have M cells."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(0)
        f = rng.standard_normal(Ls)
        tgrad, fgrad, s, c = filterbankphasegrad(f, g, a, L)
        assert len(tgrad) == M, f"tgrad must have M={M} cells, got {len(tgrad)}"
        assert len(fgrad) == M, f"fgrad must have M={M} cells, got {len(fgrad)}"
        assert len(s) == M,     f"s must have M={M} cells, got {len(s)}"
        assert len(c) == M,     f"c must have M={M} cells, got {len(c)}"

    def test_cell_sizes_match_filterbank(self, needs_impl):
        """tgrad, fgrad, s must have the same shape as filterbank(f, g, a)."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(1)
        f = rng.standard_normal(Ls)
        c_ref = filterbank(f, g, a)
        tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
        for m in range(M):
            assert np.asarray(tgrad[m]).shape == np.asarray(c_ref[m]).shape, \
                f"tgrad[{m}] shape mismatch"
            assert np.asarray(fgrad[m]).shape == np.asarray(c_ref[m]).shape, \
                f"fgrad[{m}] shape mismatch"
            assert np.asarray(s[m]).shape == np.asarray(c_ref[m]).shape, \
                f"s[{m}] shape mismatch"

    def test_gradient_is_real(self, needs_impl):
        """Phase gradients are real-valued (they are phase derivatives)."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(2)
        f = rng.standard_normal(Ls)
        tgrad, fgrad, _, _ = filterbankphasegrad(f, g, a, L)
        for m in range(M):
            assert np.isrealobj(np.asarray(tgrad[m])), \
                f"tgrad[{m}] must be real-valued"
            assert np.isrealobj(np.asarray(fgrad[m])), \
                f"fgrad[{m}] must be real-valued"

    def test_spectrogram_nonnegative(self, needs_impl):
        """The spectrogram s = |c|^2 must be non-negative."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(3)
        f = rng.standard_normal(Ls)
        _, _, s, _ = filterbankphasegrad(f, g, a, L)
        for m in range(M):
            sm = np.asarray(s[m])
            assert np.all(sm >= -1e-12), \
                f"s[{m}] must be non-negative (min={sm.min():.3e})"

    def test_coefficients_match_filterbank(self, needs_impl):
        """The returned c must agree with filterbank(f, g, a)."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(4)
        f = rng.standard_normal(Ls)
        c_ref = filterbank(f, g, a)
        _, _, _, c_pg = filterbankphasegrad(f, g, a, L)
        for m in range(M):
            c_ref_m = np.asarray(c_ref[m]).ravel()
            c_pg_m  = np.asarray(c_pg[m]).ravel()
            rel_err = np.linalg.norm(c_pg_m - c_ref_m) / \
                      (np.linalg.norm(c_ref_m) + np.finfo(float).eps)
            assert rel_err < 1e-10, \
                f"c[{m}]: filterbankphasegrad vs filterbank mismatch {rel_err:.2e}"

    def test_works_without_L(self, needs_impl):
        """Calling without L should not raise an error."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        rng = np.random.default_rng(5)
        f = rng.standard_normal(Ls)
        # Should run without error — L is inferred from signal
        try:
            filterbankphasegrad(f, g, a)
        except Exception as exc:
            pytest.fail(f"filterbankphasegrad without L raised: {exc}")


# ---------------------------------------------------------------------------
# filterbankconstphase
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbankconstphaseImpl:
    """Port of TestPhaseProcessing (filterbankconstphase section)."""

    def test_returns_list(self, needs_impl):
        """Output must be a list (cell array in MATLAB)."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import (  # type: ignore
            filterbankconstphase,
            filterbankphasegrad,
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(0)
        f = rng.standard_normal(Ls)
        c = filterbank(f, g, a)
        s = [np.abs(np.asarray(cm)) for cm in c]
        _, _, _, _pg_c = filterbankphasegrad(f, g, a, L)
        # tfr can be obtained from audfilters info; fall back to fc as proxy
        c_out = filterbankconstphase(s, a, fc)
        assert isinstance(c_out, (list, tuple)), \
            "filterbankconstphase: output must be a list/tuple"

    def test_output_length_M(self, needs_impl):
        """Output must contain M cells."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.phase import filterbankconstphase  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        M = len(g)
        rng = np.random.default_rng(1)
        f = rng.standard_normal(Ls)
        c = filterbank(f, g, a)
        s = [np.abs(np.asarray(cm)) for cm in c]
        c_out = filterbankconstphase(s, a, fc)
        assert len(c_out) == M, \
            f"filterbankconstphase: output must have M={M} cells, got {len(c_out)}"

    def test_magnitude_preserved(self, needs_impl):
        """|c_out[m]| must equal the input magnitude s[m] (rel err < 1e-6)."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.phase import filterbankconstphase  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        M = len(g)
        rng = np.random.default_rng(2)
        f = rng.standard_normal(Ls)
        c = filterbank(f, g, a)
        s = [np.abs(np.asarray(cm)) for cm in c]
        c_out = filterbankconstphase(s, a, fc)
        for m in range(M):
            mag_out = np.abs(np.asarray(c_out[m])).ravel()
            mag_in  = s[m].ravel()
            rel_err = np.linalg.norm(mag_out - mag_in) / \
                      (np.linalg.norm(mag_in) + np.finfo(float).eps)
            assert rel_err < 1e-6, \
                f"filterbankconstphase: |c_out[{m}]| magnitude error {rel_err:.2e}"

    def test_cell_size_preserved(self, needs_impl):
        """Each output cell must have the same shape as the input magnitude."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.phase import filterbankconstphase  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        M = len(g)
        rng = np.random.default_rng(3)
        f = rng.standard_normal(Ls)
        c = filterbank(f, g, a)
        s = [np.abs(np.asarray(cm)) for cm in c]
        c_out = filterbankconstphase(s, a, fc)
        for m in range(M):
            assert np.asarray(c_out[m]).shape == s[m].shape, \
                f"filterbankconstphase: c_out[{m}] shape mismatch"

    def test_with_explicit_gradient(self, needs_impl):
        """Passing {tgrad, fgrad} instead of tfr must preserve magnitude."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import (  # type: ignore
            filterbankconstphase,
            filterbankphasegrad,
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(4)
        f = rng.standard_normal(Ls)
        c = filterbank(f, g, a)
        s = [np.abs(np.asarray(cm)) for cm in c]
        tgrad, fgrad, _, _ = filterbankphasegrad(f, g, a, L)
        c_out = filterbankconstphase(s, a, fc, [tgrad, fgrad])
        for m in range(M):
            mag_out = np.abs(np.asarray(c_out[m])).ravel()
            mag_in  = s[m].ravel()
            rel_err = np.linalg.norm(mag_out - mag_in) / \
                      (np.linalg.norm(mag_in) + np.finfo(float).eps)
            assert rel_err < 1e-6, \
                f"filterbankconstphase (explicit grad): magnitude error in cell {m}: {rel_err:.2e}"

    def test_no_error(self, needs_impl):
        """filterbankconstphase must run without raising any exception."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.phase import filterbankconstphase  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        rng = np.random.default_rng(5)
        f = rng.standard_normal(Ls)
        c = filterbank(f, g, a)
        s = [np.abs(np.asarray(cm)) for cm in c]
        try:
            filterbankconstphase(s, a, fc)
        except Exception as exc:
            pytest.fail(f"filterbankconstphase raised unexpectedly: {exc}")


# ---------------------------------------------------------------------------
# filterbankreassign
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbankreassignImpl:
    """Port of TestPhaseProcessing (filterbankreassign section)."""

    def test_output_is_list(self, needs_impl):
        """sr output must be a list (cell array)."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad, filterbankreassign  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(0)
        f = rng.standard_normal(Ls)
        tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
        sr = filterbankreassign(s, tgrad, fgrad, a, fc)
        assert isinstance(sr, (list, tuple)), \
            "filterbankreassign: sr must be a list/tuple"

    def test_output_length_M(self, needs_impl):
        """sr must have M cells."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad, filterbankreassign  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(1)
        f = rng.standard_normal(Ls)
        tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
        sr, _, _ = filterbankreassign(s, tgrad, fgrad, a, fc)
        assert len(sr) == M, \
            f"filterbankreassign: sr must have M={M} cells, got {len(sr)}"

    def test_cell_sizes_preserved(self, needs_impl):
        """The reassigned spectrogram must be the same size as the input."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad, filterbankreassign  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(2)
        f = rng.standard_normal(Ls)
        tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
        sr, _, _ = filterbankreassign(s, tgrad, fgrad, a, fc)
        for m in range(M):
            assert np.asarray(sr[m]).shape == np.asarray(s[m]).shape, \
                f"filterbankreassign: sr[{m}] size must match s[{m}]"

    def test_three_output_form(self, needs_impl):
        """Three-output form [sr, repos, Lc] must run; sum(Lc) == len(repos)."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad, filterbankreassign  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        rng = np.random.default_rng(3)
        f = rng.standard_normal(Ls)
        tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
        sr, repos, Lc = filterbankreassign(s, tgrad, fgrad, a, fc)
        Lc_arr = np.asarray(Lc).ravel()
        assert len(Lc_arr) == M, \
            f"filterbankreassign: Lc must have M={M} elements, got {len(Lc_arr)}"
        assert int(np.sum(Lc_arr)) == len(np.asarray(repos).ravel()), \
            "filterbankreassign: numel(repos) must equal sum(Lc)"
        # sr must be non-negative
        for m in range(M):
            sm = np.asarray(sr[m])
            assert np.all(sm >= -1e-12), \
                f"filterbankreassign: sr[{m}] must be non-negative"

    def test_accepts_filter_cell(self, needs_impl):
        """filterbankreassign must accept the filter cell g as 5th argument."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad, filterbankreassign  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(4)
        f = rng.standard_normal(Ls)
        tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
        try:
            filterbankreassign(s, tgrad, fgrad, a, g)
        except Exception as exc:
            pytest.fail(
                f"filterbankreassign with filter cell as 5th arg raised: {exc}"
            )

    def test_energy_approximately_conserved(self, needs_impl):
        """Total reassigned energy must be positive and change < 150 %."""
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad, filterbankreassign  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(5)
        f = rng.standard_normal(Ls)
        tgrad, fgrad, s, _ = filterbankphasegrad(f, g, a, L)
        sr, _, _ = filterbankreassign(s, tgrad, fgrad, a, fc)
        e_in  = sum(float(np.sum(np.asarray(sm))) for sm in s)
        e_out = sum(float(np.sum(np.asarray(sm))) for sm in sr)
        assert e_out > 0, \
            "filterbankreassign: total reassigned energy must be positive"
        rel_err = abs(e_out - e_in) / (e_in + np.finfo(float).eps)
        assert rel_err < 1.5, \
            f"filterbankreassign: total energy change {rel_err:.1%} exceeds 150 %"


# ---------------------------------------------------------------------------
# filterbanksynchrosqueeze
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbanksynchrosqueezeImpl:
    """
    Port of TestPhaseProcessing (filterbanksynchrosqueeze section).

    NOTE: filterbanksynchrosqueeze requires equal-length subbands.
    We use a_ones = ones(M) to satisfy this requirement.
    """

    def _setup(self):
        """Build ERB filterbank with a_ones=ones(M) for synchrosqueeze."""
        from cool_frames.filterbanks import filterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.phase import filterbankphasegrad  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        M = len(g)
        L = filterbanklength(Ls, a)
        a_ones = np.ones(M, dtype=int)
        rng = np.random.default_rng(0)
        f = rng.standard_normal(Ls)
        c_ns = filterbank(f, g, a_ones)
        tgrad_ns, fgrad_ns, _, _ = filterbankphasegrad(f, g, a_ones)
        return g, a_ones, fc, M, L, c_ns, tgrad_ns, fgrad_ns

    def test_output_is_list(self, needs_impl):
        """cr must be a list (cell array)."""
        from cool_frames.phase import filterbanksynchrosqueeze  # type: ignore
        g, a_ones, fc, M, L, c_ns, tgrad_ns, fgrad_ns = self._setup()
        cr, _, _ = filterbanksynchrosqueeze(c_ns, tgrad_ns, fgrad_ns, a_ones, fc)
        assert isinstance(cr, (list, tuple)), \
            "filterbanksynchrosqueeze: cr must be a list/tuple"

    def test_output_length_M(self, needs_impl):
        """cr must have M cells."""
        from cool_frames.phase import filterbanksynchrosqueeze  # type: ignore
        g, a_ones, fc, M, L, c_ns, tgrad_ns, fgrad_ns = self._setup()
        cr, _, _ = filterbanksynchrosqueeze(c_ns, tgrad_ns, fgrad_ns, a_ones, fc)
        assert len(cr) == M, \
            f"filterbanksynchrosqueeze: cr must have M={M} cells, got {len(cr)}"

    def test_cell_sizes_preserved(self, needs_impl):
        """Synchrosqueezed output cells must have the same size as input c."""
        from cool_frames.phase import filterbanksynchrosqueeze  # type: ignore
        g, a_ones, fc, M, L, c_ns, tgrad_ns, fgrad_ns = self._setup()
        cr, _, _ = filterbanksynchrosqueeze(c_ns, tgrad_ns, fgrad_ns, a_ones, fc)
        for m in range(M):
            assert np.asarray(cr[m]).shape == np.asarray(c_ns[m]).shape, \
                f"filterbanksynchrosqueeze: cr[{m}] size must match c[{m}]"

    def test_three_output_form(self, needs_impl):
        """Three-output form [cr, repos, Lc]: sum(Lc) == len(repos)."""
        from cool_frames.phase import filterbanksynchrosqueeze  # type: ignore
        g, a_ones, fc, M, L, c_ns, tgrad_ns, fgrad_ns = self._setup()
        cr, repos, Lc = filterbanksynchrosqueeze(c_ns, tgrad_ns, fgrad_ns, a_ones, fc)
        Lc_arr = np.asarray(Lc).ravel()
        assert len(Lc_arr) == M, \
            f"filterbanksynchrosqueeze: Lc must have M={M} elements"
        assert int(np.sum(Lc_arr)) == len(np.asarray(repos).ravel()), \
            "filterbanksynchrosqueeze: numel(repos) must equal sum(Lc)"

    def test_accepts_filter_cell(self, needs_impl):
        """Must accept the filter cell g as 5th argument."""
        from cool_frames.phase import filterbanksynchrosqueeze  # type: ignore
        g, a_ones, fc, M, L, c_ns, tgrad_ns, fgrad_ns = self._setup()
        try:
            filterbanksynchrosqueeze(c_ns, tgrad_ns, fgrad_ns, a_ones, g)
        except Exception as exc:
            pytest.fail(
                f"filterbanksynchrosqueeze with filter cell as 5th arg raised: {exc}"
            )
