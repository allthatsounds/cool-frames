"""Tests for the general (SVD/eigen) frame bounds, filterbankbounds_svd."""
from __future__ import annotations
import numpy as np
import pytest
from cool_frames.numpy.filters import audfilters, cqtfilters
from cool_frames.numpy.filterbanks import filterbankbounds_svd, filterbanktight


@pytest.mark.parametrize("bank", [
    ("aud", lambda: audfilters(2000, 256)),
    ("cqt", lambda: cqtfilters(2000, 256, fmin=80, fmax=900, bins=6)),
])
def test_bounds_ordered_and_positive(bank):
    _, mk = bank
    g, a, fc, L, _info = mk()
    A, B = filterbankbounds_svd(g, a, L)
    assert 0.0 < A <= B + 1e-9


def test_tight_frame_has_unit_condition_via_svd():
    # The TRUE (eigen) condition number of a canonical tight frame is 1,
    # regardless of how each filter stores its negative half (realonly).
    g, a, fc, L, _info = audfilters(2000, 256)
    gt = filterbanktight(g, a, L)
    A, B = filterbankbounds_svd(gt, a, L)
    assert abs(B / A - 1.0) < 1e-6, f"tight frame must have kappa=1, got {B/A}"


def test_svd_bounds_sane_vs_signal_energy():
    # For any signal, A||f||^2 <= ||Df||^2 <= B||f||^2 must hold (definition).
    from cool_frames.numpy.filterbanks import filterbank
    g, a, fc, L, _info = audfilters(2000, 256)
    A, B = filterbankbounds_svd(g, a, L)
    rng = np.random.default_rng(0)
    for _ in range(5):
        f = rng.standard_normal(L)
        c = filterbank(f, g, a, L)
        energy = sum(float(np.sum(np.abs(np.asarray(cm)) ** 2)) for cm in c)
        nf2 = float(np.sum(f ** 2))
        assert A * nf2 - 1e-6 <= energy <= B * nf2 + 1e-6
