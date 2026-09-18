"""
Tests for the discrete Gabor transform (cool_frames.numpy.gabor).

Every test checks a function against the definition it claims, built as an
explicit matrix: the Gabor atoms g_{m,n}[l] = g[l - a*n] * exp(2j*pi*m*l/M)
(LTFAT's frequency-invariant convention) as the columns of G, and the frame
operator S = G G^H.  The lattices cover integer oversampling (p = 1),
rational oversampling (p > 1), critical sampling (a = M) and d = N/q > 1, the
index paths the Walnut factorisation takes separately.
"""
from __future__ import annotations

import numpy as np
import pytest

from cool_frames.gabor import (
    dgt,
    dgtlength,
    dgtreal,
    gabdual,
    gabframebounds,
    gabframediag,
    gabtight,
    idgt,
    idgtreal,
)
from cool_frames.numpy.core import middlepad

# (L, a, M): p = a / gcd(a, M), d = (L / a) / (M / gcd(a, M))
LATTICES = [
    (24, 4, 8),    # p = 1, d = 3
    (24, 6, 8),    # p = 3
    (30, 6, 10),   # p = 3, d = 1
    (36, 4, 12),   # p = 1
    (24, 6, 6),    # critical sampling
    (48, 4, 6),    # p = 2, d = 4
    (60, 10, 12),  # p = 5
    (72, 6, 9),    # p = 2, odd M
]
TOL = 1e-12


def _atoms(g, a, M):
    L = g.shape[0]
    l = np.arange(L)
    cols = [np.roll(g, a * n) * np.exp(2j * np.pi * m * l / M)
            for n in range(L // a) for m in range(M)]
    return np.stack(cols, axis=1)


def _relerr(x, ref):
    return np.linalg.norm(np.asarray(x) - np.asarray(ref)) / np.linalg.norm(ref)


@pytest.fixture(params=LATTICES, ids=lambda p: "L{}-a{}-M{}".format(*p))
def lattice(request):
    L, a, M = request.param
    rng = np.random.default_rng(L * 1000 + a * 10 + M)
    g = rng.standard_normal(L)
    G = _atoms(g, a, M)
    return dict(L=L, a=a, M=M, N=L // a, g=g, G=G, S=G @ G.conj().T, rng=rng)


# ---------------------------------------------------------------------------
# Transforms against their definitions
# ---------------------------------------------------------------------------
def test_dgt_is_the_ltfat_definition(lattice):
    L, a, M, N, G, rng = (lattice[k] for k in ("L", "a", "M", "N", "G", "rng"))
    f = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    ref = (G.conj().T @ f).reshape(N, M).T
    c = dgt(f, lattice["g"], a, M)
    assert c.shape == (M, N)
    assert _relerr(c, ref) < TOL


def test_dgt_complex_window(lattice):
    L, a, M, N, rng = (lattice[k] for k in ("L", "a", "M", "N", "rng"))
    g = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    f = rng.standard_normal(L)
    ref = (_atoms(g, a, M).conj().T @ f).reshape(N, M).T
    assert _relerr(dgt(f, g, a, M), ref) < TOL


def test_idgt_is_the_synthesis_operator(lattice):
    a, M, N, G, rng = (lattice[k] for k in ("a", "M", "N", "G", "rng"))
    c = rng.standard_normal((M, N)) + 1j * rng.standard_normal((M, N))
    ref = G @ c.T.reshape(-1)
    assert _relerr(idgt(c, lattice["g"], a), ref) < TOL


def test_multichannel_is_columnwise(lattice):
    L, a, M, g, rng = (lattice[k] for k in ("L", "a", "M", "g", "rng"))
    F = rng.standard_normal((L, 3))
    C = dgt(F, g, a, M)
    assert C.shape == (M, L // a, 3)
    for w in range(3):
        assert _relerr(C[:, :, w], dgt(F[:, w], g, a, M)) < TOL
    back = idgt(C, g, a)
    for w in range(3):
        assert _relerr(back[:, w], idgt(C[:, :, w], g, a)) < TOL


# ---------------------------------------------------------------------------
# Frame windows and bounds against S
# ---------------------------------------------------------------------------
def test_gabdual_is_canonical(lattice):
    L, a, M, g, S = (lattice[k] for k in ("L", "a", "M", "g", "S"))
    assert _relerr(gabdual(g, a, M, L), np.linalg.solve(S, g).real) < 1e-10


def test_gabtight_is_canonical(lattice):
    L, a, M, g, S = (lattice[k] for k in ("L", "a", "M", "g", "S"))
    w, V = np.linalg.eigh(S)
    ref = (V @ np.diag(w ** -0.5) @ V.conj().T @ g).real
    assert _relerr(gabtight(g, a, M, L), ref) < 1e-10


def test_framebounds_are_extreme_eigenvalues(lattice):
    L, a, M, g, S = (lattice[k] for k in ("L", "a", "M", "g", "S"))
    w = np.linalg.eigvalsh(S)
    A, B = gabframebounds(g, a, M, L)
    assert abs(A - w[0]) < 1e-10 * w[-1]
    assert abs(B - w[-1]) < 1e-10 * w[-1]


def test_framediag_is_the_diagonal_of_S(lattice):
    L, a, M, g, S = (lattice[k] for k in ("L", "a", "M", "g", "S"))
    assert _relerr(gabframediag(g, a, M, L), np.diag(S).real) < TOL


@pytest.mark.parametrize("window", ["dual", "tight"])
def test_perfect_reconstruction(lattice, window):
    L, a, M, g, rng = (lattice[k] for k in ("L", "a", "M", "g", "rng"))
    f = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    if window == "dual":
        c, gs = dgt(f, g, a, M), gabdual(g, a, M, L)
    else:
        gt = gabtight(g, a, M, L)
        c, gs = dgt(f, gt, a, M), gt
    assert _relerr(idgt(c, gs, a), f) < 1e-10


# ---------------------------------------------------------------------------
# Real-signal pair
# ---------------------------------------------------------------------------
def test_dgtreal_is_the_nonnegative_half(lattice):
    L, a, M, g, rng = (lattice[k] for k in ("L", "a", "M", "g", "rng"))
    f = rng.standard_normal(L)
    c = dgt(f, g, a, M)
    cr = dgtreal(f, g, a, M)
    assert cr.shape == (M // 2 + 1, L // a)
    assert np.array_equal(cr, c[: M // 2 + 1])
    m = np.arange(1, M)
    assert _relerr(c[M - m], np.conj(c[m])) < TOL


def test_idgtreal_reconstructs(lattice):
    L, a, M, g, rng = (lattice[k] for k in ("L", "a", "M", "g", "rng"))
    f = rng.standard_normal(L)
    out = idgtreal(dgtreal(f, g, a, M), gabdual(g, a, M, L), a, M)
    assert np.isrealobj(out)
    assert _relerr(out, f) < 1e-10


# ---------------------------------------------------------------------------
# Lengths and FIR windows
# ---------------------------------------------------------------------------
def test_dgtlength():
    assert dgtlength(1, 4, 6) == 12
    assert dgtlength(12, 4, 6) == 12
    assert dgtlength(13, 4, 6) == 24
    assert dgtlength(100, 8, 16) == 112


def test_signal_is_padded_and_truncated_back():
    rng = np.random.default_rng(7)
    a, M, Ls = 6, 8, 50
    L = dgtlength(Ls, a, M)
    g = rng.standard_normal(L)
    f = rng.standard_normal(Ls)
    c = dgt(f, g, a, M)
    assert c.shape == (M, L // a)
    assert _relerr(c, dgt(np.concatenate([f, np.zeros(L - Ls)]), g, a, M)) < TOL
    out = idgt(c, gabdual(g, a, M, L), a, Ls=Ls)
    assert out.shape == (Ls,)
    assert _relerr(out, f) < 1e-10


def test_short_window_is_fir2long_extended():
    rng = np.random.default_rng(8)
    a, M, L = 4, 12, 48
    g = rng.standard_normal(11)
    f = rng.standard_normal(L)
    assert _relerr(dgt(f, g, a, M), dgt(f, middlepad(g, L), a, M)) < TOL


@pytest.mark.parametrize("gl, a, M", [(10, 4, 12), (11, 4, 12), (16, 4, 16), (12, 3, 12)])
def test_painless_fir_dual_and_tight(gl, a, M):
    from cool_frames.filters import firwin

    g = firwin("hann", gl)
    gd, gt = gabdual(g, a, M), gabtight(g, a, M)
    assert gd.shape == (gl,) and gt.shape == (gl,)
    L = dgtlength(60, a, M)
    S = (lambda G: G @ G.conj().T)(_atoms(middlepad(g, L), a, M))
    assert _relerr(middlepad(gd, L), np.linalg.solve(S, middlepad(g, L)).real) < 1e-10
    f = np.random.default_rng(gl).standard_normal(L)
    assert _relerr(idgt(dgt(f, g, a, M), gd, a), f) < 1e-10
    assert _relerr(idgt(dgt(f, gt, a, M), gt, a), f) < 1e-10


def test_painless_fir_without_exact_fir_dual_is_refused():
    g = np.roll(np.hanning(12)[1:-1], -5)  # even length, non-zero middle sample
    with pytest.raises(ValueError, match="no exact FIR dual"):
        gabdual(g, 4, 12)
    assert gabdual(g, 4, 12, L=24).shape == (24,)


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
def test_not_a_frame_is_refused():
    with pytest.raises(ValueError, match="not a frame"):
        gabdual(np.ones(3), 4, 12)
    with pytest.raises(ValueError, match="not a frame"):
        gabtight(np.ones(3), 4, 12)


def test_invalid_arguments_are_refused():
    with pytest.raises(ValueError, match="complex windows"):
        gabdual(np.ones(8) + 1j, 4, 8)
    with pytest.raises(ValueError, match="longer than L"):
        dgt(np.ones(10), np.ones(30), 4, 8, L=24)
    with pytest.raises(ValueError, match="multiple of lcm"):
        dgt(np.ones(10), np.ones(10), 4, 6, L=18)
    with pytest.raises(ValueError, match="pass L"):
        gabdual(np.ones(18), 3, 12)
    with pytest.raises(ValueError, match="real signal"):
        dgtreal(np.ones(12) + 1j, np.ones(12), 4, 6)
    with pytest.raises(ValueError, match="rows"):
        idgtreal(np.ones((6, 3)), np.ones(12), 4, 6)
