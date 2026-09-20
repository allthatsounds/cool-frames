"""The filter-bank DGT / IDGT for windows shorter than the signal (internal).

Port of LTFAT's ``comp_dgt_fb`` / ``comp_idgt_fb`` and their ``real``
variants: for a window supported on ``gl < L`` samples, each column of the
DGT is a windowed segment of the signal folded to ``M`` samples and
transformed by one length-``M`` FFT, which costs ``O(N (gl + M log M))``
instead of the long-window factorisation's ``O(L ...)`` per window.  LTFAT
uses it for every window shorter than ``L``; cool-frames had only ported the
factorisation (``_walnut.py``), which made ``gabor.dgtreal`` 41x slower than
ltfatpy on a 3 s signal with a 1024-sample Hann window.

Both functions follow LTFAT's definition exactly, with no extra factor:

    c[m, n] = sum_l f[l] conj(g[l - a n]) exp(-2 pi i m l / M)
    f[l]    = sum_{m,n} c[m, n] g[l - a n] exp(2 pi i m l / M)

The window is passed as its support on the circle: ``wv[t] = g[first + t]``
(indices mod ``L``) for ``t = 0 .. gl-1``, where ``g`` is the length-``L``
window.  Taking the support from the *extended* window, rather than from the
FIR window LTFAT's ``fir2long`` would pad, keeps the two paths identical
whatever the padding does with the middle sample of an even-length window.

Vectorised as LTFAT's C code is organised: the signal is viewed as ``N``
blocks of ``a`` samples, so a window position is ``k`` whole blocks
(``gather``, ``overlap-add``), and the fold's rotation by ``(a n) mod M``
repeats with period ``M / gcd(a, M)`` in ``n``, so it is one ``roll`` per
residue class.  Frames are processed in slices so that no intermediate
exceeds a few million entries.
"""

from __future__ import annotations

from math import gcd

import numpy as np

_BLOCK_ENTRIES = 4_000_000


def window_support(gl_long: np.ndarray) -> tuple[int, np.ndarray]:
    """Shortest circular run holding every non-zero sample of a length-``L``
    window, as ``(first, values)``; ``values.size == L`` when there is none
    shorter."""
    L = gl_long.shape[0]
    nz = np.flatnonzero(gl_long)
    if nz.size == 0:
        return 0, np.zeros(1, dtype=gl_long.dtype)
    gaps = np.diff(nz)
    wrap = int(nz[0] + L - nz[-1])
    i = int(np.argmax(gaps)) if gaps.size else -1
    if i < 0 or wrap >= gaps[i]:
        first, last = int(nz[0]), int(nz[-1])
    else:
        first, last = int(nz[i + 1]), int(nz[i]) + L
    n = last - first + 1
    if n >= L:
        return 0, gl_long.copy()
    return first, gl_long[(first + np.arange(n)) % L].copy()


def _layout(wv: np.ndarray, first: int, a: int, M: int, L: int):
    """The window padded to whole blocks of ``a`` samples starting at a
    block boundary: ``(q0, k, wpad)`` with ``first = a q0 + r0`` and
    ``wpad[r0 + t] = wv[t]``, ``wpad.size = k a``; and the fold length
    ``Gp`` (a multiple of ``M``)."""
    q0, r0 = divmod(int(first) % L, a)
    k = -(-(wv.shape[0] + r0) // a)
    wpad = np.zeros(k * a, dtype=wv.dtype)
    wpad[r0: r0 + wv.shape[0]] = wv
    Gp = -(-k * a // M) * M
    return q0, k, wpad, Gp


def _slices(N: int, width: int):
    step = max(1, _BLOCK_ENTRIES // max(width, 1))
    for n0 in range(0, N, step):
        yield n0, min(N, n0 + step)


def dgt_fb(f: np.ndarray, wv: np.ndarray, first: int, a: int, M: int,
           onesided: bool = False) -> np.ndarray:
    """DGT of ``f`` (shape ``(L,)`` or ``(L, W)``) with the window support
    ``wv`` starting at ``first``.  Returns ``(M, N[, W])``, or the first
    ``M // 2 + 1`` rows when ``onesided`` (real ``f`` and window)."""
    squeeze = f.ndim == 1
    F = f[:, None] if squeeze else f
    L, W = F.shape
    N = L // a
    q0, k, wpad, Gp = _layout(wv, first, a, M, L)
    wconj = np.conj(wpad)
    period = M // gcd(a, M)
    rows_out = M // 2 + 1 if onesided else M
    dtype = np.result_type(F.dtype, wv.dtype, np.complex64)
    out = np.empty((rows_out, N, W), dtype=dtype)
    blocks = F.reshape(N, a, W)
    ka = k * a
    for n0, n1 in _slices(N, Gp * W):
        n = np.arange(n0, n1)
        U = np.zeros((n.size, Gp, W), dtype=np.result_type(F.dtype, wconj.dtype))
        for j in range(k):
            U[:, j * a:(j + 1) * a] = blocks[(n + q0 + j) % N]
        U[:, :ka] *= wconj[None, :, None]
        folded = U.reshape(n.size, Gp // M, M, W).sum(axis=1)
        # folded[n, j] belongs to residue (a (n + q0) + j) mod M
        V = np.empty_like(folded)
        for cls in range(min(period, n.size)):
            sel = np.arange(cls, n.size, period)
            s = (a * (n0 + cls + q0)) % M
            V[sel] = np.roll(folded[sel], s, axis=1)
        C = np.fft.rfft(V.real, axis=1) if onesided else np.fft.fft(V, axis=1)
        out[:, n0:n1, :] = np.transpose(C, (1, 0, 2))
    return out[:, :, 0] if squeeze else out


def idgt_fb(c: np.ndarray, wv: np.ndarray, first: int, L: int, a: int, M: int,
            onesided: bool = False) -> np.ndarray:
    """Synthesis ``f = sum c[m, n] g[l - a n] exp(2 pi i m l / M)`` with the
    window support ``wv`` starting at ``first``.  ``c`` is ``(M, N[, W])``,
    or ``(M // 2 + 1, N[, W])`` with ``onesided`` (a real signal from the
    non-negative frequencies of a real window's DGT; the output is real)."""
    squeeze = c.ndim == 2
    C = c[:, :, None] if squeeze else c
    N, W = C.shape[1], C.shape[2]
    Z = np.fft.irfft(C, n=M, axis=0) * M if onesided else np.fft.ifft(C, axis=0) * M
    Zt = np.transpose(Z, (1, 0, 2))  # (N, M, W)
    q0, k, wpad, _Gp = _layout(wv, first, a, M, L)
    ka = k * a
    period = M // gcd(a, M)
    real_out = onesided or (not np.iscomplexobj(Z) and not np.iscomplexobj(wv))
    out_blocks = np.zeros((N, a, W), dtype=float if real_out else complex)
    t = np.arange(ka)
    for n0, n1 in _slices(N, ka * W):
        n = np.arange(n0, n1)
        vals = np.empty((n.size, ka, W), dtype=np.result_type(Zt.dtype, wpad.dtype))
        for cls in range(min(period, n.size)):
            sel = np.arange(cls, n.size, period)
            s = (a * (n0 + cls + q0)) % M
            vals[sel] = Zt[n0 + sel][:, (s + t) % M, :]
        vals *= wpad[None, :, None]
        if real_out:
            vals = vals.real
        for j in range(k):
            out_blocks[(n + q0 + j) % N] += vals[:, j * a:(j + 1) * a]
    out = out_blocks.reshape(N * a, W)
    return out[:, 0] if squeeze else out
