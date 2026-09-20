"""The RTISI-LA family on a Gabor frame: ports of PHASERET (internal).

``rtisila``, ``gsrtisila`` and ``lertisila`` dispatch here when they are
given a window and a number of channels (``rtisila(s, g, a, M)``, PHASERET's
calling convention) instead of a filter bank.  These are line-by-line ports
of PHASERET's ``gabor/rtisila.m``, ``gabor/gsrtisila.m`` and
``gabor/lertisila.m`` with their ``comp_*`` helpers, and of the C library
wherever the MATLAB fallback and the C code (which PHASERET runs when its MEX
files are compiled) disagree.  They are checked against PHASERET in Octave
(``tests/regressions/test_rtisila_family.py`` holds the reference values).

Where this code departs from PHASERET on purpose:

- ``relres`` is ``|| |dgtreal(f)| - s || / ||s||``.  PHASERET's ``rtisila``
  and ``gsrtisila`` take the norm of the complex difference
  ``dgtreal(f) - s``, which measures the phase, not the consistency
  (``lertisila`` gets it right).  It is computed on the full-length signal
  the coefficients describe, before any cropping to ``Ls``.
- ``niter`` is the number of updates each frame receives,
  ``maxit * (lookahead + 1)``, which PHASERET's documentation states; its
  code returns ``maxit * lookahead``.
- ``gsrtisila``: the newest frame's initial coefficients are synthesised into
  its time frame before it is updated, as in PHASERET's C library.  The
  MATLAB fallback never reads them, which makes every initialisation other
  than ``'zeros'`` a no-op there.  ``'input'`` also initialises the first
  ``lookahead`` frames (the MATLAB line that means to do this has mismatched
  sizes and raises).
- ``lertisila``: the newest frame is written at its own column of the
  buffer.  PHASERET writes it at column ``2 * lookback + 1``, which is the
  newest frame's column only when ``lookahead == lookback`` (the default).
  ``'unwrap'`` is computed as a phase, ``unwrappar * |c_new| *
  exp(1j * (2 arg c_{n-1} - arg c_{n-2}))``, which is what PHASERET's ratio
  of coefficients is except where a coefficient is zero, where the ratio is
  0/0.
- The coefficients are returned in cool-frames' (LTFAT's default)
  frequency-invariant phase convention, so that ``c`` matches
  ``gabor.dgtreal``; ``phase='timeinv'`` returns PHASERET's default layout.
  The signal is the same either way.
- ``gsrtisila``'s ``'rtpghi'`` initialisation is not ported.
"""

from __future__ import annotations

from math import ceil, gcd

import numpy as np

from ..core import middlepad

_TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Arguments and windows
# ---------------------------------------------------------------------------
def _setup(s, g, a: int, M: int, phase: str):
    from ..gabor._dgt import _as_window, _check_lattice, _check_length, _real_window

    a, M = _check_lattice(a, M)
    if phase not in ("freqinv", "timeinv"):
        raise ValueError(f"phase must be 'freqinv' or 'timeinv', got {phase!r}")
    s = np.asarray(s)
    M2 = M // 2 + 1
    if s.ndim != 2 or s.shape[0] != M2:
        raise ValueError(
            f"s must be a (M // 2 + 1, N) = ({M2}, N) array of dgtreal "
            f"coefficients for M={M}, got shape {s.shape}"
        )
    N = s.shape[1]
    L = _check_length(N * a, a, M)
    if isinstance(g, str):
        from ..filters import firwin

        g = firwin(g, M)
    g = _real_window(np.asarray(g))
    if g.ndim != 1:
        raise ValueError(f"the window must be one-dimensional, got shape {g.shape}")
    if g.shape[0] > M:
        raise ValueError(
            f"the RTISI-LA family needs a painless Gabor system, a window no longer "
            f"than M={M} samples; got {g.shape[0]} (PHASERET refuses the same)"
        )
    gnum = _as_window(g, M)
    return s, g, gnum, a, M, M2, N, L


def _dual(g, a: int, M: int, L: int) -> tuple[np.ndarray, np.ndarray]:
    """The canonical dual at length ``L`` and cut to ``M`` samples (exact:
    the system is painless, so the dual has the window's support)."""
    from ..gabor import gabdual

    gd = gabdual(g, a, M, L)
    return gd, middlepad(gd, M)


def _check_lookahead(lookahead, default: int, N: int) -> int:
    if lookahead is None:
        return default
    la = int(lookahead)
    if la != lookahead or la < 0 or la > N - 1:
        raise ValueError(f"lookahead must be an integer in [0, {N - 1}], got {lookahead!r}")
    return la


def _lock(c: np.ndarray, a: int, M: int, sign: int) -> np.ndarray:
    """LTFAT ``phaselockreal`` (``sign=+1``, frequency- to time-invariant) or
    ``phaseunlockreal`` (``sign=-1``)."""
    M2, N = c.shape
    ph = np.mod(np.arange(M2)[:, None] * (np.arange(N) * a)[None, :], M)
    return c * np.exp(sign * 2j * np.pi * ph / M)


def _finish(
    c_fi: np.ndarray, g, gd: np.ndarray, a: int, M: int, s_abs: np.ndarray, Ls, phase: str
):
    """Synthesis, relres and the output convention, shared by the three."""
    from ..core import postpad
    from ..gabor import dgtreal, idgtreal

    f = idgtreal(c_fi, gd, a, M)
    norm_s = float(np.linalg.norm(s_abs))
    resid = np.abs(dgtreal(f, g, a, M, f.shape[0])) - s_abs
    relres = float(np.linalg.norm(resid) / norm_s) if norm_s > 0 else 0.0
    if Ls is not None:
        f = postpad(f, int(Ls))
    c = _lock(c_fi, a, M, +1) if phase == "timeinv" else c_fi
    return c, f, relres


# ---------------------------------------------------------------------------
# Frame buffers (PHASERET's M-sample frames, centred at floor(M/2))
# ---------------------------------------------------------------------------
def _overlay_frames(cols: np.ndarray, a: int, M: int) -> np.ndarray:
    """``comp_overlayframes``: overlap-add FIR-layout columns (rows of
    ``cols``) centred at ``floor(M/2) + i a``."""
    nw = cols.shape[0]
    out = np.zeros(nw * a - (a - 1) + M - 1)
    rng = M // 2 + np.concatenate([np.arange(0, -(-M // 2)), np.arange(-(M // 2), 0)])
    for i in range(nw):
        out[i * a + rng] += cols[i]
    return out


def _overlay_nth(frames: np.ndarray, n: int, a: int, M: int) -> np.ndarray:
    """``overlaynthframe``: frame ``n`` plus the overlapping parts of the
    other frames of the buffer."""
    out = frames[n].copy()
    for i in range(n + 1, frames.shape[0]):
        j = (i - n) * a
        if j >= M:
            break
        out[j:] += frames[i, : M - j]
    for i in range(n - 1, -1, -1):
        j = (n - i) * a
        if j >= M:
            break
        out[: M - j] += frames[i, j:]
    return out


class _Frames:
    """Analysis and synthesis of one centred frame (``phaseupdate``)."""

    def __init__(self, gdc: np.ndarray, M: int):
        self.gdc = gdc
        self.M = M
        self.h = M // 2

    def analyse(self, prd: np.ndarray) -> np.ndarray:
        return np.fft.rfft(np.roll(prd, -self.h))

    def synth(self, c: np.ndarray) -> np.ndarray:
        return self.gdc * np.roll(np.fft.irfft(c, self.M), self.h) * self.M

    def update(self, prd: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = s * np.exp(1j * np.angle(self.analyse(prd)))
        return self.synth(c), c


# ---------------------------------------------------------------------------
# RTISI-LA (Zhu, Beauregard and Wyse 2007)
# ---------------------------------------------------------------------------
def rtisila_gabor(s, g, a, M, *, Ls=None, maxit=5, lookahead=None, phase="freqinv"):
    s, g, gnum, a, M, M2, N, L = _setup(s, g, a, M, phase)
    maxit = _check_maxit(maxit)
    base = ceil(M / a) - 1
    la = _check_lookahead(lookahead, min(base, N - 1), N)
    lookback = max(base, la)
    abss = np.abs(s)
    gd, gdnum = _dual(g, a, M, L)

    # comp_rtisilawins: the newest frame is analysed with the overlap of the
    # dual windows of its neighbours (specg1: without its own, on the first
    # iteration), every other frame with the analysis window.
    wins = np.tile(gdnum, (2 * base + 1, 1))
    specg2 = _overlay_frames(wins, a, M)[:M]
    wins[0] = 0.0
    specg1 = _overlay_frames(wins, a, M)[:M]
    gc = np.fft.fftshift(gnum)
    fr = _Frames(np.fft.fftshift(gdnum), M)

    nbuf = lookback + la + 1
    frames = np.zeros((nbuf, M))
    sfr = np.zeros((la + 1, M2))
    sfr[1:] = abss[:, :la].T
    c = np.zeros((M2, N), dtype=complex)
    coef = np.zeros(M2, dtype=complex)
    for n in range(N):
        nxt = (n + la) % N
        frames[:-1] = frames[1:]
        frames[-1] = 0.0
        sfr[:-1] = sfr[1:]
        sfr[-1] = abss[:, nxt]
        for it in range(maxit):
            for nb in range(la, -1, -1):
                idx = lookback + nb
                if nb == la:
                    win = specg1 if it == 0 else specg2
                else:
                    win = gc
                frames[idx], coef = fr.update(_overlay_nth(frames, idx, a, M) * win, sfr[nb])
        c[:, n] = coef
    c_fi = _lock(c, a, M, -1)
    c_out, f, relres = _finish(c_fi, g, gd, a, M, abss, Ls, phase)
    return c_out, f, relres, maxit * (la + 1)


# ---------------------------------------------------------------------------
# GSRTISI-LA (Gnann and Spiertz 2008, 2010)
# ---------------------------------------------------------------------------
def _princarg(x: np.ndarray) -> np.ndarray:
    return x - _TWO_PI * np.round(x / _TWO_PI)


def _spsi_column(sabs: np.ndarray, a: int, M: int, phase: np.ndarray) -> np.ndarray:
    """PHASERET's ``comp_spsi`` MEX (libphaseret ``spsiupdate``) for one
    frame, starting from ``phase``.  The MATLAB copy differs only where a
    peak is exactly centred (``p == 0``), where it reuses the previous peak's
    neighbours; the C code, used here, spreads nothing from such a peak."""
    M2 = M // 2 + 1
    ph = phase.copy()
    tiny = np.finfo(float).tiny
    for m in range(1, M2 - 1):
        if not (sabs[m] > sabs[m - 1] and sabs[m] > sabs[m + 1]):
            continue
        al = np.log(sabs[m - 1] + tiny)
        be = np.log(sabs[m] + tiny)
        ga = np.log(sabs[m + 1] + tiny)
        den = al - 2 * be + ga
        p = 0.5 * (al - ga) / den if den != 0 else 0.0
        peak = ph[m] + _TWO_PI * a * (m + p) / M
        ph[m] = peak
        up = down = m
        if p > 0:
            ph[m + 1] = peak
            up, down = m + 2, m - 1
        if p < 0:
            ph[m - 1] = peak
            up, down = m + 1, m - 2
        b = down
        while b > 0 and sabs[b] < sabs[b + 1]:
            ph[b] = peak
            b -= 1
        b = up
        while b < M2 - 1 and sabs[b] < sabs[b - 1]:
            ph[b] = peak
            b += 1
    return sabs * np.exp(1j * ph)


_GS_STARTPHASE = ("zhu", "zeros", "input", "unwrap", "spsi")


def gsrtisila_gabor(
    s,
    g,
    a,
    M,
    *,
    Ls=None,
    maxit=5,
    lookahead=None,
    startphase="zhu",
    unwrappar=0.3,
    phase="freqinv",
):
    if startphase not in _GS_STARTPHASE:
        raise ValueError(
            f"gsrtisila: startphase must be one of {_GS_STARTPHASE} on a Gabor frame, "
            f"got {startphase!r} ('rtpghi' is not ported)"
        )
    s, g, gnum, a, M, M2, N, L = _setup(s, g, a, M, phase)
    maxit = _check_maxit(maxit)
    base = ceil(M / a) - 1
    la = _check_lookahead(lookahead, min(base, N - 1), N)
    lookback = max(base, la)
    abss = np.abs(s)
    gd, gdnum = _dual(g, a, M, L)

    # comp_gsrtisilawins: frame k of the look-ahead is analysed with g over
    # the sum of g gd M of the frames present around it.
    wins = np.tile(gnum * gdnum * M, (base + 1 + la, 1))
    winsum = _overlay_frames(wins, a, M)
    small = np.abs(winsum) < 1e-3
    winsum[small & (winsum > 0)] = 1e-3
    winsum[small & (winsum < 0)] = -1e-3
    winsum[winsum == 0] = 1.0
    gc = np.fft.fftshift(gnum)
    gana = np.stack([gc / winsum[a * (k + base) : a * (k + base) + M] for k in range(la + 1)])
    fr = _Frames(np.fft.fftshift(gdnum), M)

    nbuf = lookback + la + 1
    frames = np.zeros((nbuf, M))
    coefs = np.zeros((nbuf, M2), dtype=complex)
    sfr = np.zeros((la + 1, M2))
    sfr[1:] = abss[:, :la].T
    if startphase == "input":
        coefs[nbuf - la :] = s[:, :la].T
        for i in range(nbuf - la, nbuf):
            frames[i] = fr.synth(coefs[i])
    omega = _TWO_PI * a * np.arange(M2) / M
    c = np.zeros((M2, N), dtype=complex)
    for n in range(N):
        nxt = (n + la) % N
        for buf in (frames, coefs, sfr):
            buf[:-1] = buf[1:]
            buf[-1] = 0
        sfr[-1] = abss[:, nxt]
        if startphase == "spsi":
            coefs[-1] = _spsi_column(sfr[-1], a, M, np.angle(coefs[-2]))
        elif startphase == "unwrap":
            p2 = np.angle(coefs[-3])
            p1 = np.angle(coefs[-2])
            p0 = p1 + omega + _princarg(p1 - p2 - omega)
            coefs[-1] = unwrappar * sfr[-1] * np.exp(1j * p0)
        elif startphase == "input":
            coefs[-1] = s[:, nxt]
        frames[-1] = fr.synth(coefs[-1])
        for _it in range(maxit):
            for nb in range(la, -1, -1):
                idx = lookback + nb
                frames[idx], coefs[idx] = fr.update(
                    _overlay_nth(frames, idx, a, M) * gana[nb], sfr[nb]
                )
        c[:, n] = coefs[lookback]
    c_fi = _lock(c, a, M, -1)
    c_out, f, relres = _finish(c_fi, g, gd, a, M, abss, Ls, phase)
    return c_out, f, relres, maxit * (la + 1)


# ---------------------------------------------------------------------------
# LERTISI-LA / TF-RTISI-LA (Le Roux, Kameoka, Ono and Sagayama 2010)
# ---------------------------------------------------------------------------
def _legla_column(
    win: np.ndarray, K: np.ndarray, s: np.ndarray, M: int, onthefly: bool
) -> np.ndarray:
    """PHASERET ``comp_leglaupdatesinglecol`` (libphaseret
    ``leglaupdate_col_execute`` with ``EXT_UPDOWN``): the truncated
    projection of the middle column of ``win`` (``M2 x kernw``) through the
    half kernel ``K`` (``kernh2 x kernw``; the rows above the centre are its
    conjugates), borders extended by the conjugate symmetry of a real
    signal's spectrum, then the magnitude ``s`` imposed."""
    M2 = M // 2 + 1
    kh2, kw = K.shape
    o = kh2 - 1
    ext = np.empty((M2 + 2 * o, kw), dtype=complex)
    ext[o : o + M2] = win
    if o:
        r = np.arange(1, kh2)
        ext[o - r] = np.conj(win[r])
        m = np.arange(o)
        ext[o + M2 + m] = np.conj(win[M2 - 2 + M % 2 - m])
    Kc = np.conj(K)
    if not onthefly:
        acc = ext[o : o + M2] @ K[0]
        for r in range(1, kh2):
            acc += ext[o - r : o - r + M2] @ K[r] + ext[o + r : o + r + M2] @ Kc[r]
        return s * np.exp(1j * np.angle(acc))
    # Coefficient by coefficient, each update seen by the rows after it
    # through the middle column (the extension rows keep their values).
    mid = kw // 2
    other = np.ones(kw, dtype=bool)
    other[mid] = False
    acc = ext[o : o + M2][:, other] @ K[0, other] + ext[o : o + M2, mid] * K[0, mid]
    for r in range(1, kh2):
        acc += ext[o - r : o - r + M2][:, other] @ K[r, other]
        acc += ext[o + r : o + r + M2] @ Kc[r]
    col = ext[:, mid].copy()
    out = np.empty(M2, dtype=complex)
    Kmid = K[1:, mid]
    for m in range(M2):
        v = acc[m] + np.dot(Kmid, col[o + m - np.arange(1, kh2)])
        v = s[m] * np.exp(1j * np.angle(v))
        out[m] = v
        col[o + m] = v
    return out


_LE_STARTPHASE = ("zhu", "zero", "rand", "input", "unwrap")


def lertisila_gabor(
    s,
    g,
    a,
    M,
    *,
    Ls=None,
    maxit=5,
    lookahead=None,
    freqneighs=None,
    startphase="zhu",
    seed=None,
    variant="trunc",
    energy_order=False,
    asymwin=True,
    onthefly=False,
    unwrappar=0.3,
    phase="freqinv",
):
    from ..gabor import dgt, idgt

    if startphase not in _LE_STARTPHASE:
        raise ValueError(
            f"lertisila: startphase must be one of {_LE_STARTPHASE}, got {startphase!r}"
        )
    if variant not in ("trunc", "modtrunc"):
        raise ValueError(f"variant must be 'trunc' or 'modtrunc', got {variant!r}")
    s, g, _gnum, a, M, M2, N, L = _setup(s, g, a, M, phase)
    maxit = _check_maxit(maxit)
    base = ceil(M / a) - 1
    la = _check_lookahead(lookahead, base, N)
    lookback = max(la, base)
    fn = lookback if freqneighs is None else int(freqneighs)
    if fn < 0 or fn > M2 - 1:
        raise ValueError(f"freqneighs must be in [0, {M2 - 1}], got {freqneighs!r}")
    kh2, kw = fn + 1, 2 * lookback + 1
    if kw + 1 > N:
        raise ValueError(
            f"lertisila needs at least 2 * lookback + 2 = {kw + 1} frames for its "
            f"kernels, got N = {N}"
        )
    abss = np.abs(s)
    if startphase == "input":
        cfull = np.asarray(s, dtype=complex)
    elif startphase == "rand":
        rng = np.random.default_rng(seed)
        cfull = abss * np.exp(_TWO_PI * 1j * rng.random(abss.shape))
    else:
        cfull = abss.astype(complex)

    gd, _gdnum = _dual(g, a, M, L)

    def small(kern: np.ndarray) -> np.ndarray:
        if variant == "modtrunc":
            kern = kern.copy()
            kern[0, 0] = 0.0
        return middlepad(kern[:kh2].T, kw).T

    kern = small(dgt(gd, g, a, M))
    spec = []
    for first in (1, 0):
        ct = np.zeros((M, N))
        ct[0, first : kw + 1] = 1.0
        spec.append(small(dgt(np.real_if_close(idgt(ct, gd, a)), gd, a, M)))
    kNo = M // gcd(M, a)
    rows = np.arange(kh2)[:, None]

    def modulated(k0: np.ndarray, k: int) -> np.ndarray:
        return np.fft.fftshift(k0 * np.exp(-2j * np.pi * k * rows * a / M), axes=1)

    kerns = [modulated(kern, k) for k in range(kNo)]
    kspec1 = [modulated(spec[0], k) for k in range(kNo)]
    kspec2 = [modulated(spec[1], k) for k in range(kNo)]

    cbuf = np.zeros((M2, 3 * lookback + 1), dtype=complex)
    cbuf[:, : la + lookback + 1] = cfull[:, np.mod(-1 + np.arange(-lookback, la + 1), N)]
    c = np.zeros((M2, N), dtype=complex)
    h = kw // 2
    newest = lookback + la

    def upd(pos: int, K: np.ndarray, frame: int) -> None:
        cbuf[:, pos] = _legla_column(
            cbuf[:, pos - h : pos + h + 1], K, abss[:, frame % N], M, onthefly
        )

    for n in range(N):
        cbuf[:, : kw - 1] = cbuf[:, 1:kw]
        cbuf[:, kw - 1 :] = 0.0
        nxt = (n + la) % N
        if startphase == "unwrap":
            p1, p2 = cfull[:, (nxt - 1) % N], cfull[:, (nxt - 2) % N]
            cbuf[:, newest] = (
                unwrappar * np.abs(cfull[:, nxt]) * np.exp(1j * (2 * np.angle(p1) - np.angle(p2)))
            )
        elif startphase != "zhu":
            cbuf[:, newest] = cfull[:, nxt]
        k_new = (n + la) % kNo
        if asymwin:
            upd(newest, (kspec1 if startphase == "zhu" else kspec2)[k_new], nxt)
        else:
            upd(newest, kerns[k_new], nxt)
        for nb in range(la - 1, -1, -1):
            upd(lookback + nb, kerns[(n + nb) % kNo], n + nb)
        order = list(range(la, -1, -1))
        if energy_order:
            e = np.sum(np.abs(cbuf[:, lookback : lookback + la + 1]) ** 2, axis=0)
            order = [int(i) for i in np.argsort(-e, kind="stable")]
        for _it in range(1, maxit):
            for nb in order:
                if asymwin and nb == la:
                    K = kspec2[k_new]
                else:
                    K = kerns[(n + nb) % kNo]
                upd(lookback + nb, K, n + nb)
        c[:, n] = cbuf[:, lookback]
    c_out, f, relres = _finish(c, g, gd, a, M, abss, Ls, phase)
    return c_out, f, relres, maxit * (la + 1)


def _check_maxit(maxit) -> int:
    m = int(maxit)
    if m != maxit or m < 1:
        raise ValueError(f"maxit must be a positive integer, got {maxit!r}")
    return m
