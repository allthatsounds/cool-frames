"""
test_utilities.py
=================
Python port of:
    layer2_filterbank/unit/TestUtilities.m

Unit tests for format-conversion and length-inference utilities.

Covers: nonu2ucfmt, u2nonucfmt, filterbanklengthcoef, center_freqs.
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# filterbanklengthcoef
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbanklengthcoefImpl:
    """TestUtilities: filterbanklengthcoef infers system length from coefficients."""

    def test_matches_filterbanklength(self, needs_impl):
        from cool_frames.filterbanks import filterbank, filterbanklengthcoef  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(2)
        x = rng.standard_normal(Ls)
        c = filterbank(x, g, a)
        L_from_coef = filterbanklengthcoef(c, a)
        assert L_from_coef == L, \
            f"filterbanklengthcoef: got {L_from_coef}, expected {L}"

    def test_is_positive_integer(self, needs_impl):
        from cool_frames.filterbanks import filterbank, filterbanklengthcoef  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        rng = np.random.default_rng(3)
        x = rng.standard_normal(1024)
        c = filterbank(x, g, a)
        L = filterbanklengthcoef(c, a)
        assert L > 0, "filterbanklengthcoef: output must be positive"
        assert int(L) == L, "filterbanklengthcoef: output must be an integer"

    def test_stereo_consistent(self, needs_impl):
        """Stereo input should give the same L as mono."""
        from cool_frames.filterbanks import filterbank, filterbanklengthcoef  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        Ls = 1024
        g, a, fc, _, _info = audfilters(8000, Ls)
        rng = np.random.default_rng(4)
        c_stereo = filterbank(rng.standard_normal((Ls, 2)), g, a)
        c_mono   = filterbank(rng.standard_normal(Ls),     g, a)
        L_s = filterbanklengthcoef(c_stereo, a)
        L_m = filterbanklengthcoef(c_mono,   a)
        assert L_s == L_m, \
            f"filterbanklengthcoef: stereo gives L={L_s}, mono gives L={L_m}"


# ---------------------------------------------------------------------------
# center_freqs
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCentFreqsImpl:
    """TestUtilities: center_freqs returns normalised centre frequencies."""

    def test_length(self, needs_impl):
        from cool_frames.diagnostics import center_freqs  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls = 1024
        g, a, fc, _, _info = audfilters(8000, Ls)
        L = filterbanklength(Ls, a)
        cfreq = center_freqs(g, L)
        assert len(cfreq) == len(g), \
            f"center_freqs: expected {len(g)} entries, got {len(cfreq)}"

    def test_normalised_range(self, needs_impl):
        """Normalised frequencies must lie in (-1, 1]."""
        from cool_frames.diagnostics import center_freqs  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls = 1024
        g, a, fc, _, _info = audfilters(8000, Ls)
        L = filterbanklength(Ls, a)
        cfreq = np.asarray(center_freqs(g, L), dtype=float)
        assert np.all(np.abs(cfreq) <= 1.0 + 1e-12), \
            f"center_freqs: values out of (-1,1]: max|cf|={np.max(np.abs(cfreq)):.4f}"

    def test_monotone_interior(self, needs_impl):
        """Interior centre frequencies (excluding last boundary filter) are non-decreasing."""
        from cool_frames.diagnostics import center_freqs  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls = 1024
        g, a, fc, _, _info = audfilters(8000, Ls)
        L = filterbanklength(Ls, a)
        cfreq = np.asarray(center_freqs(g, L), dtype=float)
        interior = cfreq[:-1]
        if len(interior) >= 2:
            assert np.all(np.diff(interior) > -1e-10), \
                "center_freqs: interior frequencies not non-decreasing"

    def test_consistent_with_audfilters_fc(self, needs_impl):
        """center_freqs (normalised) must correlate strongly with audfilters fc/Nyquist."""
        from cool_frames.diagnostics import center_freqs  # type: ignore
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        cfreq   = np.asarray(center_freqs(g, L), dtype=float)
        fc_norm = np.asarray(fc, dtype=float) / (fs / 2)
        M = len(g)
        interior = slice(1, M - 1)
        if M - 2 >= 2:
            corr_matrix = np.corrcoef(cfreq[interior], fc_norm[interior])
            assert corr_matrix[0, 1] > 0.99, \
                f"center_freqs: low correlation {corr_matrix[0,1]:.4f} with audfilters fc"


# ---------------------------------------------------------------------------
# pack_coefficients / unpack_coefficients (dense+mask batching)
# ---------------------------------------------------------------------------

class TestPackCoefficients:
    def test_roundtrip_nonuniform(self):
        import numpy as np
        from cool_frames.filterbanks import filterbank, pack_coefficients, unpack_coefficients
        from cool_frames.filters import audfilters
        g, a, fc, L, _info = audfilters(8000, 4000)
        c = filterbank(np.random.default_rng(0).standard_normal(L), g, a, L)
        dense, mask = pack_coefficients(c)
        assert dense.shape == (len(c), max(np.asarray(cm).size for cm in c))
        assert dense.shape == mask.shape
        c2 = unpack_coefficients(dense, mask)
        assert len(c2) == len(c)
        assert all(np.array_equal(np.asarray(c[m]).ravel(), c2[m]) for m in range(len(c)))

    def test_mask_counts_match_lengths(self):
        import numpy as np
        from cool_frames.filterbanks import pack_coefficients
        c = [np.arange(3.0), np.arange(5.0), np.arange(1.0)]
        dense, mask = pack_coefficients(c)
        assert dense.shape == (3, 5)
        assert list(mask.sum(axis=1)) == [3, 5, 1]
