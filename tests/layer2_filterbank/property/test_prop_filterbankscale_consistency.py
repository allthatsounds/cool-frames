"""
test_prop_filterbankscale_consistency.py
=========================================
Python port of:
    layer2_filterbank/property/PropFilterbankscaleConsistency.m

Properties:
(1) filterbankscale(g, s) multiplies filterbankresponse by s².
(2) Scaling analysis by s and synthesis by 1/s leaves PR error unchanged.
(3) Per-filter scale vector changes each band independently;
    individual filterbankresponse reflects the per-filter changes.
(4) Scale by 1 is identity.
"""

from __future__ import annotations

import pytest

import numpy as np


@pytest.mark.requires_impl
class TestScaleMultipliesResponseImpl:
    """PropFilterbankscaleConsistency: scalar scale → response × s²."""

    @pytest.mark.parametrize("s", [0.5, 2.0, float(np.sqrt(2)), 3.0])
    def test_scalar_scale_response(self, needs_impl, s):
        from cool_frames.filterbanks import filterbankresponse, filterbankscale  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        gf_orig   = np.real(np.asarray(filterbankresponse(g, a, L)))
        gs        = filterbankscale(g, s)
        gf_scaled = np.real(np.asarray(filterbankresponse(gs, a, L)))
        err = np.linalg.norm(gf_scaled - s ** 2 * gf_orig) / \
              (np.linalg.norm(gf_orig) + 1e-15)
        assert err < 1e-10, \
            f"Scale s={s:.4f}: response scaling error {err:.2e}"


@pytest.mark.requires_impl
class TestScaleAndInverseScalePRImpl:
    """PropFilterbankscaleConsistency: analysis·s and synthesis·(1/s) preserves PR."""

    @pytest.mark.parametrize("s", [0.5, 2.0, -1.0, 3.0])
    def test_scale_inverse_cancel(self, needs_impl, s):
        from cool_frames.filterbanks import filterbank, filterbankdual, filterbankscale, ifilterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        gd = filterbankdual(g, a, L)
        gs = filterbankscale(g, s)
        gds = filterbankscale(gd, 1.0 / s)
        rng = np.random.default_rng(0)
        for trial in range(10):
            x = rng.standard_normal(Ls)
            c = filterbank(x, gs, a)
            xr = np.real(np.asarray(ifilterbank(c, gds, a, real=True)))
            rel_err = np.linalg.norm(x - xr[:Ls]) / np.linalg.norm(x)
            assert rel_err < 1e-10, \
                f"Scale s={s}, trial {trial}: PR error {rel_err:.2e}"


@pytest.mark.requires_impl
class TestPerFilterScaleImpl:
    """PropFilterbankscaleConsistency: per-filter scale changes individual responses."""

    def test_per_filter_scale_individual(self, needs_impl):
        from cool_frames.filterbanks import filterbankfreqz, filterbankscale  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        from cool_frames.numpy.filterbanks._utils import normalise_a  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        M = len(g)
        s_vec = 1.0 + 0.1 * np.arange(1, M + 1)   # row vector
        gs = filterbankscale(g, s_vec)
        # Compute per-filter responses using filterbankfreqz
        a_norm = normalise_a(a, M)
        afrac = a_norm[:, 0] / a_norm[:, 1]
        H_orig = filterbankfreqz(g, a_norm, L)
        H_scaled = filterbankfreqz(gs, a_norm, L)
        gf_orig_ind = np.real(H_orig * H_orig.conj()) / afrac[np.newaxis, :]
        gf_scaled_ind = np.real(H_scaled * H_scaled.conj()) / afrac[np.newaxis, :]
        for m in range(M):
            expected = s_vec[m] ** 2 * gf_orig_ind[:, m]
            actual = gf_scaled_ind[:, m]
            mask = gf_orig_ind[:, m] > 1e-6 * np.max(gf_orig_ind[:, m])
            if np.any(mask):
                rel_err = np.linalg.norm(actual[mask] - expected[mask]) / \
                          (np.linalg.norm(expected[mask]) + 1e-15)
                assert rel_err < 1e-4, \
                    f"Band {m}: per-filter scale error {rel_err:.2e}"


@pytest.mark.requires_impl
class TestScaleByOneIdentityImpl:
    """PropFilterbankscaleConsistency: scale by 1 is identity."""

    def test_scale_one_identity(self, needs_impl):
        from cool_frames.filterbanks import filterbankresponse, filterbankscale  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        gf_orig   = np.real(np.asarray(filterbankresponse(g,                          a, L)))
        gf_scaled = np.real(np.asarray(filterbankresponse(filterbankscale(g, 1.0), a, L)))
        err = np.linalg.norm(gf_orig - gf_scaled) / (np.linalg.norm(gf_orig) + 1e-15)
        assert err < 1e-12, f"Scale by 1: identity error {err:.2e}"
