"""Regression: frame verdicts and bounds for banks that are not painless.

Found 2026-09-28 while checking the admissibility verdicts after the
``gabfilters`` fix.

1. The designers published ``info["admissible"]["is_frame"] = True`` for
   banks with aliasing channels.  The covering theorem behind that verdict
   holds for painless banks only; for others covering is necessary, not
   sufficient.  ``waveletfilters(16000, 512, painless=False, fmax=4000,
   highpass='auto')`` covers every bin and has lower frame bound 0.
2. ``filterbankbounds`` returned the diagonal (painless) response for a
   non-uniform bank that is not painless, silently: ``(0.98, 3.73)`` on that
   bank.  It now solves the sparse frame operator exactly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cool_frames.numpy.filterbanks import filterbankbounds, filterbankbounds_svd
from cool_frames.numpy.filters import (
    audfilters,
    cqtfilters,
    greenwoodfilters,
    warpedfilters,
    waveletfilters,
)


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def _kappa(A, B):
    return B / A if A > 0 else np.inf


def test_wavelet_nonframe_that_covers_every_bin_gets_no_frame_verdict():
    g, a, fc, L, info = _quiet(waveletfilters, 16000, 512, painless=False,
                               fmax=4000, highpass="auto")
    assert info["painless"] is False
    assert info["admissible"] is None
    A, B = filterbankbounds(g, a, L)
    assert A == 0.0
    As, Bs = filterbankbounds_svd(g, a, L)
    assert As == 0.0
    # Folded convention: the real bounds are twice the SVD's (see
    # filterbankbounds), the complex ones equal.
    assert B == pytest.approx(2 * Bs, rel=1e-10)
    Ac, Bc = filterbankbounds(g, a, L, real=False)
    Asc, Bsc = filterbankbounds_svd(g, a, L, real=False)
    assert Ac == 0.0 and Asc == 0.0
    assert Bc == pytest.approx(Bsc, rel=1e-10)


@pytest.mark.parametrize("make", [
    lambda: waveletfilters(16000, 384, painless=False),
    lambda: audfilters(8000, 384, redmul=0.5),
    lambda: greenwoodfilters(8000, 384, redmul=0.5),
    lambda: cqtfilters(8000, 384, redmul=0.6, fmin=100),
    lambda: warpedfilters(np.sqrt, np.square, 8000, 50, 3800, 4, 256, redmul=0.5),
], ids=["wavelet", "aud", "greenwood", "cqt", "warped"])
def test_nonuniform_nonpainless_bounds_are_exact(make):
    g, a, fc, L, info = _quiet(make)
    assert info["painless"] is False
    assert info["admissible"] is None or info["admissible"]["is_frame"] is False
    for real in (True, False):
        A, B = filterbankbounds(g, a, L, real=real)
        As, Bs = filterbankbounds_svd(g, a, L, real=real)
        assert (A == 0.0) == (As < 1e-10 * Bs)
        k, ks = _kappa(A, B), _kappa(As, Bs)
        if np.isfinite(ks):
            assert k == pytest.approx(ks, rel=1e-8)


def test_not_a_frame_verdict_survives_aliasing():
    # Uncovered bins annihilate their exponentials whatever the hops, so the
    # "not a frame" verdict stands on a non-painless bank.
    g, a, fc, L, info = _quiet(audfilters, 8000, 512, M=8, redmul=0.5)
    assert info["painless"] is False
    assert info["admissible"]["is_frame"] is False
    assert filterbankbounds(g, a, L)[0] == 0.0


@pytest.mark.parametrize("make", [
    lambda: audfilters(8000, 1024),
    lambda: greenwoodfilters(8000, 1024),
    lambda: cqtfilters(8000, 1024, fmin=100),
    lambda: warpedfilters(np.sqrt, np.square, 8000, 50, 3800, 4, 1024),
    lambda: waveletfilters(16000, 512),
], ids=["aud", "greenwood", "cqt", "warped", "wavelet"])
def test_painless_banks_keep_their_verdict(make):
    g, a, fc, L, info = _quiet(make)
    assert info["painless"] is True
    assert info["admissible"]["is_frame"] is True
    assert filterbankbounds(g, a, L)[0] > 0


def test_large_block_path_matches_dense():
    # A block above the dense threshold goes through Lanczos and LOBPCG.
    from cool_frames.numpy.filterbanks import _frame

    g, a, fc, L, info = _quiet(waveletfilters, 16000, 4096, painless=False)
    A, B = filterbankbounds(g, a, L)
    old = _frame._DENSE_BLOCK
    try:
        _frame._DENSE_BLOCK = 10 ** 6
        Ad, Bd = filterbankbounds(g, a, L)
    finally:
        _frame._DENSE_BLOCK = old
    assert A == pytest.approx(Ad, rel=1e-9)
    assert B == pytest.approx(Bd, rel=1e-9)


def test_large_singular_block_reads_zero():
    from cool_frames.numpy.filterbanks import _frame

    g, a, fc, L, info = _quiet(waveletfilters, 16000, 512, painless=False,
                               fmax=4000, highpass="auto")
    old = _frame._DENSE_BLOCK
    try:
        _frame._DENSE_BLOCK = 50
        A, B = filterbankbounds(g, a, L)
    finally:
        _frame._DENSE_BLOCK = old
    assert A == 0.0
    assert B == pytest.approx(filterbankbounds(g, a, L)[1], rel=1e-9)
