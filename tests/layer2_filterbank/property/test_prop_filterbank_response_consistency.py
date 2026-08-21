"""
test_prop_filterbank_response_consistency.py
=============================================
Python port of:
    layer2_filterbank/property/PropFilterbankResponseConsistency.m

Properties:
(1) filterbankresponse(g, a, L) satisfies A <= gf(k) <= B for all k.
(2) The 'total' output equals the column-sum of the 'individual' output.
(3) For a tight frame, gf is approximately constant at positive frequencies.
(4) gf is everywhere non-negative.
(5) Tightening preserves the bound structure.
"""

from __future__ import annotations

import pytest

import numpy as np


@pytest.mark.requires_impl
class TestResponseWithinBoundsImpl:
    """PropFilterbankResponseConsistency: response lies in [A, B]."""

    def test_response_within_frame_bounds(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds, filterbankresponse  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        A, B = filterbankbounds(g, a, L)
        # Folded (real=True) response matches the folded real-signal bounds.
        gf   = np.real(np.asarray(filterbankresponse(g, a, L, real=True)))
        assert np.min(gf) >= A - 1e-6, \
            f"min(filterbankresponse)={np.min(gf):.6f} < A={A:.6f}"
        assert np.max(gf) <= B + 1e-6, \
            f"max(filterbankresponse)={np.max(gf):.6f} > B={B:.6f}"


@pytest.mark.requires_impl
class TestResponseTotalVsIndividualImpl:
    """PropFilterbankResponseConsistency: total == sum of individual."""

    def test_total_equals_column_sum(self, needs_impl):
        from cool_frames.filterbanks import filterbankfreqz, filterbankresponse  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.numpy.filterbanks._utils import normalise_a  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        gf_total = np.real(np.asarray(filterbankresponse(g, a, L)))
        # Compute individual responses manually: |H_m(k)|^2 / a_m
        M = len(g)
        a_norm = normalise_a(a, M)
        afrac = a_norm[:, 0] / a_norm[:, 1]
        H = filterbankfreqz(g, a_norm, L)
        gf_indiv = np.real(H * H.conj()) / afrac[np.newaxis, :]
        gf_sum = gf_indiv.sum(axis=1)
        err = np.linalg.norm(gf_total - gf_sum) / (np.linalg.norm(gf_total) + 1e-15)
        assert err < 1e-12, \
            f"total vs sum-of-individual mismatch: {err:.2e}"


@pytest.mark.requires_impl
class TestTightResponseConstantImpl:
    """PropFilterbankResponseConsistency: tight frame response is constant."""

    def test_tight_response_constant(self, needs_impl):
        from cool_frames.filterbanks import filterbankresponse, filterbanktight  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gt = filterbanktight(g, a, L)
        # Folded (real=True) response of a canonical tight frame is constant.
        gf     = np.real(np.asarray(filterbankresponse(gt, a, L, real=True)))
        At     = float(np.min(gf))
        Bt     = float(np.max(gf))
        assert At > 0, "Tight frame lower bound must be positive"
        assert abs(At - Bt) / (At + 1e-15) < 0.01, \
            f"Tight frame not constant: A={At:.8f}, B={Bt:.8f}"
        deviation = np.max(np.abs(gf - At)) / At
        assert deviation < 0.01, \
            f"Tight frame: max deviation {deviation:.2e}"


@pytest.mark.requires_impl
class TestResponseNonNegativeImpl:
    """PropFilterbankResponseConsistency: response is non-negative."""

    def test_non_negative(self, needs_impl):
        from cool_frames.filterbanks import filterbankresponse  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gf = np.real(np.asarray(filterbankresponse(g, a, L)))
        assert np.min(gf) >= -1e-12, \
            f"filterbankresponse must be non-negative; min={np.min(gf):.2e}"


@pytest.mark.requires_impl
class TestResponseBoundMonotoneImpl:
    """PropFilterbankResponseConsistency: tightening narrows the response range."""

    def test_tightening_narrows_range(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds, filterbankresponse, filterbanktight  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        A_orig, B_orig = filterbankbounds(g, a, L)
        gt = filterbanktight(g, a, L)
        At, Bt = filterbankbounds(gt, a, L)
        gf_orig  = np.real(np.asarray(filterbankresponse(g, a, L, real=True)))
        gf_tight = np.real(np.asarray(filterbankresponse(gt, a, L, real=True)))
        assert np.min(gf_orig)  >= A_orig - 1e-6
        assert np.max(gf_orig)  <= B_orig + 1e-6
        assert np.min(gf_tight) >= At     - 1e-6
        assert np.max(gf_tight) <= Bt     + 1e-6
