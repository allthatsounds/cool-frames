"""Frame diagonal, frame bounds, canonical dual and tight windows (internal).

Provides (all private; the public, LTFAT-convention functions are in
``_dgt.py``):
    _gabframediag          -- diagonal of the Gabor frame operator (LTFAT scale)
    _gabframebounds        -- frame bounds A, B (LTFAT scale)
    _gabdual_normalised    -- canonical dual for the 1/sqrt(M) DGT of _walnut.py
    _gabtight_normalised   -- canonical tight window for that DGT

``_gabdual_normalised`` returns ``M`` times LTFAT's canonical dual and
``_gabtight_normalised`` returns ``sqrt(M)`` times LTFAT's tight window,
because the transforms in ``_walnut.py`` carry ``1/sqrt(M)`` each; the
note below explains where that factor enters.  Real windows only.

Ported on 2026-09-19 from ``cola_additions.dgt.numpy_backend.gabdual``,
unchanged apart from names, imports and this header.

Two algorithmic paths:

  1. **Painless** (M >= window length): the frame operator is diagonal in
     the time domain with period *a*.  Compute the diagonal, invert, and
     multiply with the window.  O(gl) — very fast.

  2. **Factorised / general** (M < window length): use the Sondergaard
     window factorisation (comp_wfac) to convert the problem to c·d
     independent p×p linear systems solved via Cholesky decomposition.
     O(L·p²) — still fast for typical oversampling ratios.

Implementation challenges — DGT normalisation and the factor-M pitfall
----------------------------------------------------------------------
Our DGT uses a 1/√M factor in *both* forward and inverse transforms:

    DGT:   c[m,n] = ⟨f, g_{m,n}⟩ / √M
    IDGT:  f = Σ c[m,n] · gs_{m,n} / √M

This means IDGT(DGT(f, g), gd) = (1/M) · S_{gd,g} · f, where S_{gd,g}
is the physical cross frame operator.  For perfect reconstruction we need
S_{gd,g} = M·I, NOT S_{gd,g} = I.

This creates a **different normalisation requirement** for each path:

  * **Painless path:** gabframediag returns the *physical* frame diagonal
    d[n] = M · Σ_k |g(n+ka)|², matching LTFAT convention.  The painless
    dual formula becomes  gd = g · M / d  (NOT  g / d).  Equivalently
    the effective frame operator is d/M.  Missing this factor causes a
    systematic M-fold amplitude error in the reconstructed signal.

  * **Factorised path:** the wfac/iwfac domain absorbs the normalisation
    into its structure.  The per-block frame operator Sf = Gf · Gf^H
    (p×p) and dual Gf_d = Sf^{-1} · Gf already produce a dual window
    that gives S_{gd,g} = M·I after iwfac.  NO extra M scaling needed.

This asymmetry between paths is non-obvious and was the main source of
bugs during implementation.

A second subtlety: the LTFAT C code stores the factorised window as
(qR, p) — transposed relative to our (p, qR) convention.  This means
LTFAT's "Gf^T · Gf → (p,p)" corresponds to our "Gf · Gf^H → (p,p)".
Getting this wrong gives a (qR, qR) Gram matrix which is rank-deficient
for the common p=1 (integer oversampling) case and produces garbage.

References
----------
  [1] P. L. Søndergaard, "An efficient algorithm for the discrete Gabor
      transform using full-length windows," 2007.
  [2] LTFAT source: gabdual_painless.c, gabdual_fac.c, gabdual.c
"""
from __future__ import annotations

import numpy as np

from ..core import middlepad
from ._walnut import comp_wfac

# ---------------------------------------------------------------------------
# gabframediag — frame operator diagonal
# ---------------------------------------------------------------------------

def _gabframediag(
    g: np.ndarray,
    a: int,
    M: int,
    L: int | None = None,
) -> np.ndarray:
    """Diagonal of the Gabor frame operator.

    For a painless Gabor system (M ≥ gl), the frame operator is a
    multiplication operator with an *a*-periodic diagonal.

    Parameters
    ----------
    g : ndarray, shape (gl,)
        Analysis window.
    a : int
        Hop size.
    M : int
        Number of frequency channels.
    L : int, optional
        If given, return the full diagonal of length L (tiled from the
        a-periodic pattern).  If None, return only the first period
        (length a).

    Returns
    -------
    d : ndarray, shape (a,) or (L,)
        Frame operator diagonal.  Shape (a,) when L is None; shape (L,)
        when L is given.

    Notes
    -----
    This returns the *physical* frame operator diagonal (LTFAT convention),
    which includes the factor M.  When used with a normalised DGT that has
    a 1/√M factor in both forward and inverse transforms, the effective
    frame operator is ``d / M``.

    The diagonal can be used as a **preconditioner** for iterative frame
    methods (franaiter / frsyniter).
    """
    g = np.asarray(g, dtype=np.float64)
    gl = len(g)

    if L is not None and gl < L:
        g = middlepad(g, L)
        gl = L

    g2 = np.abs(g) ** 2

    # Simple, vectorised accumulation of |g(n)|² with a-periodicity
    d: np.ndarray = np.zeros(a, dtype=np.float64)
    for i in range(gl):
        d[i % a] += g2[i]

    d *= M

    if L is not None:
        d = np.tile(d, L // a)

    return d


# ---------------------------------------------------------------------------
# gabframebounds — frame bounds via factorisation
# ---------------------------------------------------------------------------

def _gabframebounds(
    g: np.ndarray,
    a: int,
    M: int,
    L: int | None = None,
) -> tuple[float, float]:
    """Compute the frame bounds (A, B) of a Gabor system.

    Two algorithmic paths:

    1. **Painless** (M ≥ gl): the frame operator is diagonal with period
       *a*.  Frame bounds are ``A = min(d)``, ``B = max(d)`` where *d* is
       the diagonal from :func:`gabframediag`.

    2. **Factorised** (M < gl): use the Sondergaard window factorisation
       to decompose the frame operator into c·d independent p×p Hermitian
       blocks ``Sf = Gf · Gf^H``.  Compute eigenvalues of each block;
       ``A = min(all eigenvalues)``, ``B = max(all eigenvalues)``.

    Parameters
    ----------
    g : ndarray, shape (gl,)
        Analysis window.
    a : int
        Hop size.
    M : int
        Number of frequency channels.
    L : int, optional
        Transform length.  Required when gl > M (general case).
        Defaults to gl for the painless case.

    Returns
    -------
    A : float
        Lower frame bound.
    B : float
        Upper frame bound.

    Notes
    -----
    The bounds are for the **normalised** DGT convention used in this
    module (1/√M factor in both forward and inverse transforms).

    The condition number ``B / A`` is 1 for a tight frame and ∞ for a
    non-frame (A = 0).  Useful for checking whether lattice parameters
    give a numerically well-conditioned system.
    """
    g = np.asarray(g, dtype=np.float64)
    gl = len(g)

    if gl <= M:
        # Painless case: bounds from diagonal
        d = _gabframediag(g, a, M)
        return float(np.min(d)), float(np.max(d))

    # General case: factorised eigenvalue computation
    if L is None:
        L = gl
    if gl < L:
        g = middlepad(g, L)

    gf, params = comp_wfac(g, a, M)
    c = params["c"]
    d_param = params["d"]

    AF = np.inf
    BF = 0.0

    for r in range(c):
        for s in range(d_param):
            Gf = gf[:, :, r, s]  # (p, q*R)
            Sf = Gf @ Gf.conj().T  # (p, p), Hermitian
            eigvals = np.linalg.eigvalsh(Sf)
            AF = min(AF, float(np.min(eigvals)))
            BF = max(BF, float(np.max(eigvals)))

    # Scale by M to match the physical frame operator convention used by
    # the painless path (gabframediag includes the factor M).  See the
    # "factor-M problem" note in the module docstring.
    AF *= M
    BF *= M

    return AF, BF


# ---------------------------------------------------------------------------
# gabdual — canonical dual window
# ---------------------------------------------------------------------------

def _gabdual_normalised(
    g: np.ndarray,
    a: int,
    M: int,
    L: int | None = None,
) -> np.ndarray:
    """Compute the canonical dual window of a Gabor frame.

    The canonical dual window gd satisfies:

        f = IDGT(DGT(f, g, a, M), gd, L, a, M)

    for all signals f of length L, provided that (g, a, M) forms a frame.

    Parameters
    ----------
    g : ndarray, shape (gl,)
        Analysis window.
    a : int
        Hop size.
    M : int
        Number of frequency channels.
    L : int or None
        Signal length.  If None, uses len(g).

    Returns
    -------
    gd : ndarray, shape (gl,) or (L,)
        Canonical dual window (same length as input, or L if specified).
    """
    g = np.asarray(g, dtype=np.float64)
    gl = len(g)

    if L is None:
        L = gl

    if gl <= M and gl == L:
        # Painless case: frame operator is diagonal
        return _gabdual_painless(g, a, M)
    else:
        # General case: use factorisation
        return _gabdual_long(g, a, M, L)


def _gabdual_painless(
    g: np.ndarray,
    a: int,
    M: int,
) -> np.ndarray:
    """Painless gabdual: M ≥ gl, diagonal frame operator.

    Port of LTFAT gabdual_painless.c, adapted for normalised DGT
    (which has 1/√M in both forward and inverse transforms, so the
    effective frame operator is S_physical / M).
    """
    gl = len(g)
    d = _gabframediag(g, a, M)  # physical diagonal (includes factor M)

    # For normalised DGT: gd = g * M / d  (compensate for 1/M in roundtrip)
    d_inv = M / d

    # Apply the a-periodic diagonal to the window
    gd = np.zeros_like(g)
    for i in range(gl):
        gd[i] = g[i] * d_inv[i % a]

    return gd


def _gabdual_long(
    g: np.ndarray,
    a: int,
    M: int,
    L: int,
) -> np.ndarray:
    """General gabdual via Sondergaard factorisation + Cholesky solve.

    Port of LTFAT gabdual.c / gabdual_fac.c:
      1. Factorise window:  gf = wfac(g, a, M)
      2. For each of c·d blocks: solve  Sf·X = Gf  where Sf = Gf^H·Gf
      3. Inverse-factorise:  gd = iwfac(X, a, M)
    """
    gl_orig = len(g)

    # Extend window to L if needed
    if len(g) != L:
        g_long = middlepad(g, L)
    else:
        g_long = g.copy()

    # Factorise
    gf, params = comp_wfac(g_long, a, M)

    c = params["c"]
    d = params["d"]

    # Solve the factorised dual: for each of c*d blocks,
    # compute frame operator Sf = Gf @ Gf^H (p×p), then
    # dual_gf = Sf^{-1} @ Gf  (p × q*R)
    #
    # LTFAT convention: factorised window is (qR, p) and uses
    # Gf^T @ Gf → (p, p). Our convention stores (p, qR), so
    # the p×p frame operator is Gf @ Gf^H.
    gdf = gf.copy()

    for r in range(c):
        for s in range(d):
            Gf = gf[:, :, r, s]  # (p, q*R)
            # Frame operator in factorised domain: (p, p)
            Sf = Gf @ Gf.conj().T  # (p, p), Hermitian positive definite
            if np.abs(Sf).max() < 1e-20:
                gdf[:, :, r, s] = 0  # Zero block → zero dual
            else:
                try:
                    X = np.linalg.solve(Sf, Gf)  # (p, q*R)
                except np.linalg.LinAlgError:
                    X = np.linalg.pinv(Sf) @ Gf
                gdf[:, :, r, s] = X

    # Inverse-factorise to get the dual window
    gd = _comp_iwfac(gdf, a, M, params)

    # Truncate back to original window length if needed
    if gl_orig < L:
        gd = middlepad(gd, gl_orig)

    return gd.real if np.max(np.abs(gd.imag)) < 1e-10 else gd


def _comp_iwfac(
    gf: np.ndarray,
    a: int,
    M: int,
    params: dict,
) -> np.ndarray:
    """Inverse window factorisation: recover window from factorised form.

    This is the inverse of comp_wfac.

    Parameters
    ----------
    gf : (p, q*R, c, d) complex array
    a, M : lattice parameters
    params : dict from comp_wfac

    Returns
    -------
    g : (L,) complex array
    """
    c = params["c"]
    p = params["p"]
    q = params["q"]
    d = params["d"]
    L = params["L"]
    R = params["R"]

    # Undo FFT along d-dimension
    if d > 1:
        gf = np.fft.ifft(gf, axis=3)

    g = np.zeros((L, R), dtype=gf.dtype)

    if p == 1:
        for w in range(R):
            for s in range(d):
                for l in range(q):
                    idx = (np.arange(c) + (-l * a + s * p * M) % L) % L
                    g[idx, w] += gf[0, l + q * w, :, s]
    else:
        for w in range(R):
            for s in range(d):
                for l in range(q):
                    for k in range(p):
                        idx = np.arange(c) + c * ((k * q - l * p + s * p * q) % (d * p * q))
                        idx = idx % L
                        g[idx, w] += gf[k, l + q * w, :, s]

    # The FFT in comp_wfac is undone by ifft above (which includes 1/d),
    # so no additional normalisation is needed.

    if R == 1:
        g = g[:, 0]

    return g


# ---------------------------------------------------------------------------
# gabtight — canonical tight window
# ---------------------------------------------------------------------------

def _gabtight_normalised(
    g: np.ndarray,
    a: int,
    M: int,
    L: int | None = None,
) -> np.ndarray:
    """Compute the canonical tight window of a Gabor frame.

    The tight window gt satisfies:

        f = IDGT(DGT(f, gt, a, M), gt, L, a, M)

    (same window for analysis and synthesis).

    Parameters
    ----------
    g : ndarray, shape (gl,)
        Analysis window.
    a : int
        Hop size.
    M : int
        Number of frequency channels.
    L : int or None
        Signal length.  If None, uses len(g).

    Returns
    -------
    gt : ndarray, shape (gl,) or (L,)
        Canonical tight window.
    """
    g = np.asarray(g, dtype=np.float64)
    gl = len(g)

    if L is None:
        L = gl

    if gl <= M and gl == L:
        return _gabtight_painless(g, a, M)
    else:
        # General case: compute dual, then take sqrt of frame operator
        # For the general case, use factorisation with sqrt
        return _gabtight_long(g, a, M, L)


def _gabtight_painless(
    g: np.ndarray,
    a: int,
    M: int,
) -> np.ndarray:
    """Painless gabtight: M ≥ gl, adapted for normalised DGT."""
    gl = len(g)
    d = _gabframediag(g, a, M)  # physical diagonal

    # For normalised DGT: gt = g * sqrt(M) / sqrt(d)
    d_inv_sqrt = np.sqrt(M) / np.sqrt(d)

    gt = np.zeros_like(g)
    for i in range(gl):
        gt[i] = g[i] * d_inv_sqrt[i % a]

    return gt


def _gabtight_long(
    g: np.ndarray,
    a: int,
    M: int,
    L: int,
) -> np.ndarray:
    """General gabtight via factorisation + Cholesky + matrix sqrt."""
    gl_orig = len(g)

    if len(g) != L:
        g_long = middlepad(g, L)
    else:
        g_long = g.copy()

    gf, params = comp_wfac(g_long, a, M)

    c = params["c"]
    d = params["d"]

    gtf = gf.copy()

    for r in range(c):
        for s in range(d):
            Gf = gf[:, :, r, s]  # (p, q*R)
            # Frame operator: Sf = Gf @ Gf^H  (p, p)
            Sf = Gf @ Gf.conj().T  # (p, p)
            # Sf^{-1/2} via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(Sf)
            eigvals = np.maximum(eigvals, 1e-30)  # numerical safety
            Sf_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.conj().T
            gtf[:, :, r, s] = Sf_inv_sqrt @ Gf  # (p, q*R)

    gt = _comp_iwfac(gtf, a, M, params)

    if gl_orig < L:
        gt = middlepad(gt, gl_orig)

    return gt.real if np.max(np.abs(gt.imag)) < 1e-10 else gt
