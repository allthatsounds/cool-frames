"""
test_prop_parseval_identity.py
================================
Python port of:
    layer2_filterbank/property/PropParsevalIdentity.m

For a tight frame T with bound A, the Parseval identity:
    Σ_m (1/a_m) · <(Tx)_m, (Ty)_m>  =  A · <x, y>

Properties:
(1) Symmetry: Σ_m (1/a_m)<(Tx)_m,(Ty)_m> = conj(Σ_m (1/a_m)<(Ty)_m,(Tx)_m>)
(2) x=y reduces to energy conservation: Σ(1/a_m)||c_m||² = A||x||²
"""

from __future__ import annotations

import pytest

import numpy as np


def _tight_and_bound(fs=8000, Ls=1024):
    from cool_frames.filterbanks import filterbankbounds, filterbanktight  # type: ignore
    from cool_frames.filters import audfilters  # type: ignore
    from cool_frames.filters import filterbanklength  # type: ignore
    g, a, fc, _, _info = audfilters(fs, Ls)
    L  = filterbanklength(Ls, a)
    gt = filterbanktight(g, a, L)
    _, B = filterbankbounds(gt, a, L)
    return gt, a, L, B


@pytest.mark.requires_impl
class TestParsevalSymmetryImpl:
    """PropParsevalIdentity: weighted inner product is conjugate-symmetric."""

    def test_symmetry(self, needs_impl):
        from cool_frames.filterbanks import filterbank  # type: ignore
        gt, a, L, _ = _tight_and_bound()
        a_arr = np.asarray(a).ravel()
        Ls    = 1024
        rng   = np.random.default_rng(42)
        for trial in range(30):
            x = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            y = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            cx = filterbank(x, gt, a)
            cy = filterbank(y, gt, a)
            ip_xy = sum(
                np.dot(np.asarray(cx[m]).ravel(), np.conj(np.asarray(cy[m]).ravel())) / float(am)
                for m, am in enumerate(a_arr)
            )
            ip_yx = sum(
                np.dot(np.asarray(cy[m]).ravel(), np.conj(np.asarray(cx[m]).ravel())) / float(am)
                for m, am in enumerate(a_arr)
            )
            err = abs(ip_xy - np.conj(ip_yx)) / (abs(ip_xy) + 1e-15)
            assert err < 1e-10, \
                f"Trial {trial}: symmetry error {err:.2e}"


@pytest.mark.requires_impl
class TestParsevalReducesToEnergyImpl:
    """PropParsevalIdentity: x=y case equals energy conservation."""

    def test_energy_conservation(self, needs_impl):
        from cool_frames.filterbanks import filterbank, ifilterbank  # type: ignore
        gt, a, L, B = _tight_and_bound()
        Ls    = 1024
        rng   = np.random.default_rng(43)
        for trial in range(50):
            x  = rng.standard_normal(Ls)   # real signal (one-sided ERB)
            cx = filterbank(x, gt, a)
            Sx = np.real(np.asarray(ifilterbank(cx, gt, a, real=True)))
            frame_energy = np.dot(Sx[:Ls], x)
            expected = B * np.linalg.norm(x) ** 2
            err = abs(frame_energy - expected) / (expected + 1e-15)
            assert err < 0.05, \
                f"Trial {trial}: Parseval energy error {err:.2e}"
