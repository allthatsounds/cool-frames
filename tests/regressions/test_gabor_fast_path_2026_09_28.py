"""Regression: a gabfilters bank is analysed and synthesised through the DGT.

Since f8ad9e0 every ``gabfilters`` channel stores the whole L-point transform
of its window, as LTFAT's does, so the generic filter-bank kernels cost
O(M2 * L) per call: 350 ms for an analysis and 1.2 s for a synthesis of a
1024/256 bank at L = 2**16, and 53 s for 32 fast Griffin--Lim iterations
where the truncated bank of 5d4c2b3 took 3.3 s.  An unedited bank now takes
the Gabor transform (``c_k[n] = s_k exp(2j pi k a n / M) dgt(f)[k, n]`` and
its adjoint), a few milliseconds.  These tests pin the fast path to the
generic kernels, which a copy of the bank with writeable responses still
takes.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np
from cool_frames.numpy.filterbanks import (
    filterbank,
    filterbankdual,
    filterbanktight,
    ifilterbank,
    ifilterbankiter,
)
from cool_frames.numpy.filterbanks._utils import _gabor_fast, normalise_a
from cool_frames.numpy.filters import gabfilters


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def _generic(g):
    """The same bank with writeable copies of its responses: no longer
    recognised as a Gabor bank, so it takes the generic kernels."""
    out = []
    for gm in g:
        d = dict(gm)
        d["H"] = np.array(gm["H"], copy=True)
        out.append(d)
    return out


CASES = [
    dict(M=16, a=4, Ls=64, window="hann", real=True),
    dict(M=15, a=5, Ls=75, window="hann", real=True),
    dict(M=12, a=3, Ls=50, window="gauss", real=True),
    dict(M=16, a=4, Ls=64, window="blackman", real=False),
    dict(M=32, a=8, Ls=300, window="hann", real=True),
    dict(M=24, a=16, Ls=200, window="hann", real=True),  # a > M/2: low redundancy
    dict(M=20, a=5, Ls=97, window="hamming", real=False),
]


def _bank(case):
    return _quiet(
        gabfilters,
        1000,
        case["Ls"],
        M=case["M"],
        a=case["a"],
        window=case["window"],
        real=case["real"],
    )


@pytest.mark.parametrize(
    "case", CASES, ids=lambda c: f"M{c['M']}a{c['a']}{c['window']}{'R' if c['real'] else 'C'}"
)
def test_fast_path_is_taken_and_matches_generic(case):
    g, a, _fc, L, _ = _bank(case)
    assert _gabor_fast(g, normalise_a(a, len(g)), L) is not None
    assert _gabor_fast(_generic(g), normalise_a(a, len(g)), L) is None
    rng = np.random.default_rng(7)
    x = rng.standard_normal((case["Ls"], 2))
    if not case["real"]:
        x = x + 1j * rng.standard_normal((case["Ls"], 2))
    c_fast = filterbank(x, g, a, L)
    c_gen = filterbank(x, _generic(g), a, L)
    for cf, cg in zip(c_fast, c_gen):
        assert cf.shape == cg.shape
        assert np.max(np.abs(cf - cg)) <= 1e-12 * max(1.0, np.max(np.abs(cg)))
    # synthesis with the bank itself (the adjoint) and with its canonical dual
    for syn in (g, filterbankdual(g, a, L, real=case["real"])):
        y_fast = ifilterbank(c_gen, syn, a, L, real=case["real"])
        y_gen = ifilterbank(c_gen, _generic(syn), a, L, real=case["real"])
        assert np.max(np.abs(y_fast - y_gen)) <= 1e-12 * max(1.0, np.max(np.abs(y_gen)))


@pytest.mark.parametrize("case", CASES[:4], ids=lambda c: f"M{c['M']}a{c['a']}")
def test_round_trip_exact_with_dual_and_tight(case):
    g, a, _fc, L, _ = _bank(case)
    x = np.random.default_rng(3).standard_normal(case["Ls"])
    if not case["real"]:
        x = x + 1j * np.random.default_rng(4).standard_normal(case["Ls"])
    c = filterbank(x, g, a, L)
    y = ifilterbank(
        c, filterbankdual(g, a, L, real=case["real"]), a, case["Ls"], real=case["real"]
    )
    assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-12
    gt = filterbanktight(g, a, L, real=case["real"])
    y = ifilterbank(filterbank(x, gt, a, L), gt, a, case["Ls"], real=case["real"])
    assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-12


def test_adjoint_identity():
    """<A x, c> = <x, A^* c> with the fast analysis and synthesis."""
    g, a, _fc, L, _ = _bank(CASES[0])
    rng = np.random.default_rng(11)
    x = rng.standard_normal(L)
    c = [
        rng.standard_normal(cm.shape) + 1j * rng.standard_normal(cm.shape)
        for cm in filterbank(x, g, a, L)
    ]
    lhs = sum(np.vdot(cm, am) for cm, am in zip(c, filterbank(x, g, a, L)))
    # ifilterbank(real=True) returns 2 Re(A^* c): the adjoint of the analysis
    # restricted to real signals
    rhs = np.vdot(ifilterbank(c, g, a, L, real=True), x)
    assert abs(2 * lhs.real - rhs.real) <= 1e-10 * abs(rhs)


def test_edited_bank_takes_generic_path():
    g, a, _fc, L, _ = _bank(CASES[0])
    edited = [dict(gm) for gm in g]
    edited[3]["H"] = np.array(g[3]["H"], copy=True) * 2.0
    assert _gabor_fast(edited, normalise_a(a, len(g)), L) is None
    x = np.random.default_rng(1).standard_normal(L)
    c = filterbank(x, edited, a, L)
    c_ref = filterbank(x, _generic(g), a, L)
    assert np.allclose(c[3], 2.0 * c_ref[3])
    assert np.allclose(c[2], c_ref[2])


def test_iterative_inverse_with_fast_bank():
    g, a, _fc, L, _ = _bank(CASES[4])
    x = np.random.default_rng(5).standard_normal(CASES[4]["Ls"])
    xr, _relres, _it = ifilterbankiter(
        filterbank(x, g, a, L), g, a, Ls=len(x), real=True, maxit=100, tol=1e-12
    )
    assert np.linalg.norm(x - xr) / np.linalg.norm(x) < 1e-9


def test_mismatched_real_flag_still_warns():
    g, a, _fc, L, _ = _bank(CASES[0])  # single-sided bank
    c = filterbank(np.random.default_rng(2).standard_normal(L), g, a, L)
    with pytest.warns(UserWarning, match="appear single-sided"):
        ifilterbank(c, g, a, L, real=False)


@pytest.mark.parametrize("window, M, a", [("hamming", 16, 4), ("gauss", 12, 3), ("gauss", 16, 8)])
def test_even_window_with_non_zero_middle_sample(window, M, a):
    """G11: the Gabor closed forms extend the window as ``gabfilters`` (and
    LTFAT 2.6's ``fir2long``) do, so such a bank's dual is exact and its
    bounds are the dense operator's.  (Hamming 16/4 reconstructed to 7e-4
    and read kappa 1.0040 for 1.0000.)"""
    from cool_frames.numpy.filterbanks import filterbankbounds, filterbankbounds_svd

    g, aa, _fc, L, _ = _quiet(gabfilters, 1000, 60, M=M, a=a, window=window)
    w = np.asarray(g[0]["gabor"]["window"])
    assert len(w) % 2 == 0 and w[len(w) // 2] != 0
    x = np.random.default_rng(9).standard_normal(L)
    y = ifilterbank(filterbank(x, g, aa, L), filterbankdual(g, aa, L), aa, L)
    assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-12
    A, B = filterbankbounds(g, aa, L)
    A2, B2 = filterbankbounds_svd(_generic(g), aa, L)
    assert abs(B / A - B2 / A2) < 1e-9 * (B2 / A2)


@pytest.mark.parametrize(
    "M, a, Ls, window, real, startphase",
    [
        (32, 8, 300, "hann", True, "zhu"),
        (24, 6, 240, "gauss", True, "zhu"),
        (32, 8, 256, "hann", True, "rand"),
        (16, 4, 128, "hann", False, "zero"),
    ],
)
def test_rtisila_filter_bank_in_time_matches_spectral_engine(M, a, Ls, window, real, startphase):
    """P2: the RTISI-LA filter-bank engine works on a Gabor bank in time
    (``_GaborClass``); it equals the spectral engine, which the same bank
    with writeable responses takes."""
    from cool_frames.numpy.phase._rtisila_fb import FbFrames
    from cool_frames.phase import rtisila

    g, aa, _fc, L, _ = _quiet(gabfilters, 1000, Ls, M=M, a=a, window=window, real=real)
    rng = np.random.default_rng(1)
    x = rng.standard_normal(Ls)
    if not real:
        x = x + 1j * rng.standard_normal(Ls)
    s = [np.abs(cm) for cm in filterbank(x, g, aa, L)]
    kw = dict(L=L, Ls=Ls, real=real, startphase=startphase, seed=3)
    c1, f1, r1, _ = _quiet(rtisila, s, g, aa, **kw)
    c2, f2, r2, _ = _quiet(rtisila, s, _generic(g), aa, **kw)
    scale = max(np.max(np.abs(v)) for v in c2)
    assert max(np.max(np.abs(u - v)) for u, v in zip(c1, c2)) < 1e-10 * scale
    assert np.max(np.abs(f1 - f2)) < 1e-10 * np.max(np.abs(f2))
    assert abs(r1 - r2) < 1e-10
    gd = filterbankdual(g, aa, L, real=real)
    assert _quiet(
        FbFrames, g, gd, normalise_a(aa, len(g)), L, [L // a] * len(g), real, a
    ).timedomain


@pytest.mark.parametrize(
    "M, a, Ls, window, real",
    [
        (32, 8, 300, "hann", True),
        (24, 6, 240, "gauss", True),
        (16, 4, 128, "hann", False),
        (20, 5, 200, "hamming", True),
    ],
)
def test_phase_gradient_through_the_dgt_matches_derivative_banks(M, a, Ls, window, real):
    """P3: ``filterbankphasegrad`` on a Gabor bank takes the derivative
    coefficients through the DGT; they equal the derivative banks'."""
    from cool_frames.numpy.phase import filterbankphasegrad
    from cool_frames.numpy.phase._phasegrad import _gabor_derivative_coefficients

    g, aa, _fc, L, _ = _bank(dict(M=M, a=a, Ls=Ls, window=window, real=real))
    rng = np.random.default_rng(1)
    x = rng.standard_normal(Ls)
    if not real:
        x = x + 1j * rng.standard_normal(Ls)
    assert _gabor_derivative_coefficients(x, g, normalise_a(aa, len(g)), L) is not None
    assert _gabor_derivative_coefficients(x, _generic(g), normalise_a(aa, len(g)), L) is None
    fast = _quiet(filterbankphasegrad, x, g, aa, L)
    slow = _quiet(filterbankphasegrad, x, _generic(g), aa, L)
    for u, v in zip(fast, slow):
        scale = max(float(np.max(np.abs(w))) for w in v)
        assert max(float(np.max(np.abs(p - q))) for p, q in zip(u, v)) < 1e-11 * scale
