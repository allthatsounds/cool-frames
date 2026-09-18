"""Window factorisation and the Walnut-form DGT / IDGT (internal).

This is the long-window algorithm of Søndergaard [1] (after Strohmer [2]),
the same factorisation LTFAT uses in ``comp_wfac`` / ``comp_dgt_walnut`` /
``comp_idgt_fac``:

  1. ``comp_wfac(g, a, M)``           -- window factorisation (once per g, a, M)
  2. ``comp_dgt_walnut(f, gf, ...)``  -- fold, short FFTs, block matrix products
  3. a final DFT over the frequency index.

With ``c = gcd(a, M)``, ``p = a/c``, ``q = M/c``, ``N = L/a`` and ``d = N/q``;
integer oversampling (``p == 1``) takes a simpler indexing path than rational
oversampling (``p > 1``).

**Normalisation.** ``_dgt_long`` and ``_idgt_long`` each carry a factor
``1/sqrt(M)`` relative to LTFAT's definition (LTFAT puts ``sqrt(M)`` into
``comp_wfac`` and divides it out again; this code does not).  They are private
for that reason: the public functions in ``_dgt.py`` rescale to LTFAT's
convention and are the only callers.

Ported on 2026-09-19 from ``cola_additions.dgt.numpy_backend.dgt`` (previously
``3d_studio/sondergaard_dgt``), unchanged apart from names and this docstring.

References
----------
[1] P. L. Søndergaard, "An efficient algorithm for the discrete Gabor
    transform using full-length windows," 2007.
[2] T. Strohmer, "Numerical algorithms for discrete Gabor expansions,"
    in Gabor Analysis and Algorithms, 1998.
"""
from __future__ import annotations

from math import gcd

import numpy as np

# ---------------------------------------------------------------------------
# Extended GCD
# ---------------------------------------------------------------------------

def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm. Returns (g, x, y) with a*x + b*y = g."""
    if a == 0:
        return b, 0, 1
    g, x, y = _extended_gcd(b % a, a)
    return g, y - (b // a) * x, x


# ---------------------------------------------------------------------------
# comp_wfac — Window factorisation
# ---------------------------------------------------------------------------

def comp_wfac(g: np.ndarray, a: int, M: int) -> tuple[np.ndarray, dict]:
    """Compute the Gabor window factorisation.

    Parameters
    ----------
    g : (L,) or (L, R) — analysis/synthesis window(s)
    a : int — time shift
    M : int — number of channels

    Returns
    -------
    gf : (p, q*R, c, d) complex array — factorised window (4-D)
    params : dict with lattice parameters {c, p, q, d, h_a, N, L, R}
    """
    g = np.asarray(g, dtype=complex)
    if g.ndim == 1:
        g = g[:, np.newaxis]
    L, R = g.shape

    N = L // a
    c = gcd(a, M)
    p = a // c
    q = M // c
    d = N // q

    _, h_a_raw, _ = _extended_gcd(a, M)
    h_a = -h_a_raw

    gf = np.zeros((p, q * R, c, d), dtype=g.dtype)

    if p == 1:
        # Integer oversampling
        if c == 1 and d == 1 and R == 1:
            for l in range(q):
                gf[0, l, 0, 0] = g[(-l) % L, 0]
        else:
            for w in range(R):
                for s in range(d):
                    for l in range(q):
                        idx = (np.arange(c) + (-l * a + s * p * M) % L) % L
                        gf[0, l + q * w, :, s] = g[idx, w]
    else:
        # Rational oversampling
        for w in range(R):
            for s in range(d):
                for l in range(q):
                    for k in range(p):
                        idx = np.arange(c) + c * ((k * q - l * p + s * p * q) % (d * p * q))
                        idx = idx % L
                        gf[k, l + q * w, :, s] = g[idx, w]

    # FFT along d-dimension
    if d > 1:
        gf = np.fft.fft(gf, axis=3)

    params = dict(c=c, p=p, q=q, d=d, h_a=h_a, N=N, L=L, R=R)
    return gf, params


# ---------------------------------------------------------------------------
# comp_dgt_walnut — Walnut representation step
# ---------------------------------------------------------------------------

def comp_dgt_walnut(f: np.ndarray, gf: np.ndarray, a: int, M: int,
                    params: dict) -> np.ndarray:
    """First step of the full-window factorisation of the Gabor matrix.

    Parameters
    ----------
    f      : (L, W) — input signal(s)
    gf     : (p, q*R, c, d) — factorised window (from comp_wfac)
    a      : int — time shift
    M      : int — number of channels
    params : lattice parameters from comp_wfac

    Returns
    -------
    cout : (M, N, W) — intermediate coefficients (before final FFT)
    """
    f = np.asarray(f, dtype=complex)
    if f.ndim == 1:
        f = f[:, np.newaxis]
    L, W = f.shape

    c = params["c"]
    p = params["p"]
    q = params["q"]
    d = params["d"]
    h_a = params["h_a"]
    N = params["N"]
    R = params["R"]

    # --- Fold the signal into ff[p, q*W, c, d] ---
    ff = np.zeros((p, q * W, c, d), dtype=f.dtype)

    if p == 1:
        # Integer oversampling
        if c == 1 and d == 1 and W == 1:
            ff[0, :, 0, 0] = f[:, 0]
        else:
            for s in range(d):
                for r_idx in range(c):
                    for l in range(q):
                        ff[0, l::q, r_idx, s] = f[r_idx + s * M + l * c, :]
    else:
        # Rational oversampling
        for w in range(W):
            for s in range(d):
                for l in range(q):
                    for k in range(p):
                        idx = (np.arange(c) + (k * M + s * p * M - l * h_a * a) % L) % L
                        ff[k, l + w * q, :, s] = f[idx, w]

    # FFT along d-dimension
    if d > 1:
        ff = np.fft.fft(ff, axis=3)

    # --- Matrix multiply: C[q*R, q*W, c, d] = gf^H @ ff ---
    C = np.zeros((q * R, q * W, c, d), dtype=f.dtype)

    for r_idx in range(c):
        for s in range(d):
            GM = gf[:, :, r_idx, s]  # (p, q*R)
            FM = ff[:, :, r_idx, s]  # (p, q*W)
            C[:, :, r_idx, s] = GM.conj().T @ FM

    # Inverse FFT along d-dimension
    if d > 1:
        C = np.fft.ifft(C, axis=3)

    # --- Place into output cout[M, N, R, W] ---
    cout = np.zeros((M, N, R, W), dtype=f.dtype)

    if p == 1:
        # Integer oversampling
        if c == 1 and d == 1 and W == 1 and R == 1:
            for l in range(q):
                indices = (np.arange(q) + l) % N
                cout[l, indices, 0, 0] = C[:, l, 0, 0]
        else:
            for rw in range(R):
                for w in range(W):
                    for s in range(d):
                        for l in range(q):
                            for u in range(q):
                                time_idx = (u + s * q + l) % N
                                cout[l * c:l * c + c, time_idx, rw, w] = \
                                    C[u + rw * q, l + w * q, :, s]
    else:
        # Rational oversampling
        for rw in range(R):
            for w in range(W):
                for s in range(d):
                    for l in range(q):
                        for u in range(q):
                            time_idx = (u + s * q - l * h_a) % N
                            cout[l * c:l * c + c, time_idx, rw, w] = \
                                C[u + rw * q, l + w * q, :, s]

    # Collapse R into W for single-window case (R=1)
    cout = cout.reshape(M, N, R * W)
    if R == 1:
        cout = cout.reshape(M, N, W)

    return cout


# ---------------------------------------------------------------------------
# dgt_long — full forward DGT
# ---------------------------------------------------------------------------

def _dgt_long(f: np.ndarray, g: np.ndarray, a: int, M: int,
             gf: np.ndarray | None = None,
             _params: dict | None = None) -> np.ndarray:
    """Compute the DGT using the Sondergaard factorisation.

    Parameters
    ----------
    f  : (L,) or (L, W) — input signal(s)
    g  : (L,) — analysis window
    a  : int — time shift (hop size)
    M  : int — number of frequency channels
    gf : optional precomputed window factorisation (from comp_wfac)
    _params : optional lattice parameters (from comp_wfac)

    Returns
    -------
    c  : (M, N) or (M, N, W) — DGT coefficients, N = L/a
    """
    f = np.asarray(f, dtype=complex)
    g = np.asarray(g, dtype=complex)

    single_signal = (f.ndim == 1)
    if f.ndim == 1:
        f = f[:, np.newaxis]


    # Window factorisation
    if gf is None or _params is None:
        gf, _params = comp_wfac(g, a, M)

    # Walnut representation
    cout = comp_dgt_walnut(f, gf, a, M, _params)  # (M, N, W)

    # Final FFT modulation along frequency axis
    cout = np.fft.fft(cout, axis=0) / np.sqrt(M)

    if single_signal:
        cout = cout[:, :, 0]

    return np.asarray(cout)


# ---------------------------------------------------------------------------
# comp_idgt_fac — inverse factorisation step
# ---------------------------------------------------------------------------

def comp_idgt_fac(coef: np.ndarray, gf: np.ndarray,
                  L: int, a: int, M: int, params: dict) -> np.ndarray:
    """Inverse DGT using the Sondergaard factorisation.

    Parameters
    ----------
    coef   : (M, N, W) — DGT coefficients
    gf     : (p, q*R, c, d) — factorised synthesis window
    L      : signal length
    a      : time shift
    M      : number of channels
    params : lattice parameters from comp_wfac

    Returns
    -------
    f : (L, W) — reconstructed signal(s)
    """
    coef = np.asarray(coef, dtype=complex)
    if coef.ndim == 2:
        coef = coef[:, :, np.newaxis]

    _, N, W = coef.shape

    c = params["c"]
    p = params["p"]
    q = params["q"]
    d = params["d"]
    h_a = params["h_a"]
    R = params["R"]

    # Apply inverse FFT to the coefficients (undo final FFT modulation)
    coef_ifft = np.fft.ifft(coef, axis=0) * np.sqrt(M)

    coef_4d = coef_ifft.reshape(M, N, 1, W)  # R=1 for single window

    # --- Collect into C[q*R, q*W, c, d] ---
    C = np.zeros((q * R, q * W, c, d), dtype=coef.dtype)

    if p == 1:
        for rw in range(R):
            for w in range(W):
                for s in range(d):
                    for l in range(q):
                        for u in range(q):
                            time_idx = (u + s * q + l) % N
                            C[u + rw * q, l + w * q, :, s] = \
                                coef_4d[l * c:l * c + c, time_idx, rw, w]
    else:
        for rw in range(R):
            for w in range(W):
                for s in range(d):
                    for l in range(q):
                        for u in range(q):
                            time_idx = (u + s * q - l * h_a) % N
                            C[u + rw * q, l + w * q, :, s] = \
                                coef_4d[l * c:l * c + c, time_idx, rw, w]

    # FFT along d-dimension
    if d > 1:
        C = np.fft.fft(C, axis=3)

    # --- Matrix multiply: ff = GM @ C ---
    ff = np.zeros((p, q * W, c, d), dtype=coef.dtype)

    for r_idx in range(c):
        for s in range(d):
            GM = gf[:, :, r_idx, s]           # (p, q*R)
            CM = C[:, :, r_idx, s]             # (q*R, q*W)
            ff[:, :, r_idx, s] = GM @ CM

    # Inverse FFT along d-dimension
    if d > 1:
        ff = np.fft.ifft(ff, axis=3)

    # --- Place back into signal f[L, W] ---
    f_out = np.zeros((L, W), dtype=coef.dtype)

    if p == 1:
        for s in range(d):
            for w in range(W):
                for l in range(q):
                    idx = (np.arange(c) + (s * M + l * a) % L) % L
                    f_out[idx, w] += ff[0, l + w * q, :, s]
    else:
        for w in range(W):
            for s in range(d):
                for l in range(q):
                    for k in range(p):
                        idx = (np.arange(c) + (k * M + s * p * M - l * h_a * a) % L) % L
                        f_out[idx, w] += ff[k, l + w * q, :, s]

    return f_out


# ---------------------------------------------------------------------------
# idgt_long — full inverse DGT
# ---------------------------------------------------------------------------

def _idgt_long(coef: np.ndarray, g: np.ndarray, L: int, a: int, M: int,
              gf: np.ndarray | None = None,
              _params: dict | None = None) -> np.ndarray:
    """Compute the inverse DGT using the Sondergaard factorisation.

    Parameters
    ----------
    coef : (M, N) or (M, N, W) — DGT coefficients
    g    : (L,) — synthesis window (typically the canonical dual window)
    L    : signal length
    a    : time shift
    M    : number of frequency channels
    gf   : optional precomputed window factorisation
    _params : optional lattice parameters (from comp_wfac)

    Returns
    -------
    f : (L,) or (L, W) — reconstructed signal
    """
    coef = np.asarray(coef, dtype=complex)
    g = np.asarray(g, dtype=complex)

    single = (coef.ndim == 2)

    if gf is None or _params is None:
        gf, _params = comp_wfac(g, a, M)

    f = comp_idgt_fac(coef, gf, L, a, M, _params)

    if single:
        f = f[:, 0]

    return f
