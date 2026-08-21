"""
numpy.operators._framemul
=========================
Frame multiplier operators for non-uniform filterbanks.

A frame multiplier M_sigma is the operator:

    M_sigma f = D_s* diag(sigma) D_a f

where D_a is the analysis operator (filterbank), D_s* is the synthesis
operator (ifilterbank), and sigma is a time-frequency symbol (one value
per coefficient). When D_a and D_s form a dual pair, M with sigma=1 is
the identity. When they form a tight frame pair (D_a = D_s = D_t),
M with constant sigma = alpha scales the signal by alpha.

Port of the LTFAT operator framework:
  ltfat_2.0/inst/operators/framemul.m
  ltfat_2.0/inst/operators/iframemul.m
  ltfat_2.0/inst/operators/framemuladj.m
  ltfat_2.0/inst/operators/framemulappr.m
  ltfat_2.0/inst/operators/framemuleigs.m

See MATH_REFERENCE.md §15a for the mathematical background.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from ..filterbanks._core import filterbank, ifilterbank


# ---------------------------------------------------------------------------
# framemul — forward multiplier
# ---------------------------------------------------------------------------

def framemul(f: np.ndarray,
             g_analysis: list[dict],
             g_synthesis: list[dict],
             a,
             sigma: list[np.ndarray],
             L: int,
             *,
             real: bool = True) -> np.ndarray:
    """Apply a frame multiplier to a signal.

    Computes M_sigma f = ifilterbank(sigma * filterbank(f, g_a, a, L),
                                      g_s, a, L).

    On a *tight* analysis frame (with its self-dual synthesis) the multiplier's
    eigenvalues equal the symbol ``sigma`` itself, so the frame adds no colour
    of its own --- the edit you hear is exactly the edit applied to the
    coefficients (cf. :func:`framemuleigs`).

    Parameters
    ----------
    f : ndarray, shape (L,)
        Input signal.
    g_analysis : list of M filter dicts
        Analysis frame filters.
    g_synthesis : list of M filter dicts
        Synthesis frame filters (dual, tight, or other).
    a : array-like
        Hop sizes, shape (M,) or (M, 2).
    sigma : list of M ndarrays
        Time-frequency symbol. Each sigma[m] has the same length as the
        m-th subband coefficient vector.
    L : int
        Transform length.
    real : bool
        If True (default), use real-signal reconstruction mode
        (``ifilterbank(..., real=True)``). This is correct for auditory
        and other single-sided filterbanks that cover only positive
        frequencies. Set to False for full-spectrum (complex) filterbanks.

    Returns
    -------
    result : ndarray, shape (L,)
        M_sigma f.

    See Also
    --------
    framemuladj : adjoint operator
    framemulinv : inverse (PCG)
    MATH_REFERENCE.md §15a
    """
    f = np.asarray(f, dtype=float)
    c = filterbank(f, g_analysis, a, L)
    c_masked = [c[m] * sigma[m] for m in range(len(c))]
    result = ifilterbank(c_masked, g_synthesis, a, L, real=real)
    return np.real(result)


# ---------------------------------------------------------------------------
# framemuladj — adjoint
# ---------------------------------------------------------------------------

def framemuladj(f: np.ndarray,
                g_analysis: list[dict],
                g_synthesis: list[dict],
                a,
                sigma: list[np.ndarray],
                L: int,
                *,
                real: bool = True) -> np.ndarray:
    """Apply the adjoint of a frame multiplier.

    M*_sigma f = ifilterbank(conj(sigma) * filterbank(f, g_s, a, L),
                              g_a, a, L).

    This is equivalent to framemul(f, g_synthesis, g_analysis, a,
    conj(sigma), L) — i.e., swap the frames and conjugate the symbol.

    Parameters
    ----------
    f, g_analysis, g_synthesis, a, sigma, L, real
        Same as :func:`framemul`.

    Returns
    -------
    result : ndarray, shape (L,)
        M*_sigma f.
    """
    sigma_conj = [np.conj(s) for s in sigma]
    return framemul(f, g_synthesis, g_analysis, a, sigma_conj, L, real=real)


# ---------------------------------------------------------------------------
# framemulinv — inverse via PCG
# ---------------------------------------------------------------------------

def framemulinv(f: np.ndarray,
                g_analysis: list[dict],
                g_synthesis: list[dict],
                a,
                sigma: list[np.ndarray],
                L: int,
                *,
                real: bool = True,
                tol: float = 1e-9,
                maxit: int = 100) -> tuple[np.ndarray, dict]:
    """Invert a frame multiplier via preconditioned conjugate gradient.

    Solves M_sigma x = f by applying PCG to the normal equation
    M*_sigma M_sigma x = M*_sigma f.

    Requires sigma[m][n] != 0 for all m, n (otherwise M is singular).

    Parameters
    ----------
    f : ndarray, shape (L,)
        Right-hand side (the output of framemul to invert).
    g_analysis, g_synthesis : list of M filter dicts
        Analysis and synthesis frame filters.
    a : array-like
        Hop sizes.
    sigma : list of M ndarrays
        Time-frequency symbol (must be nonzero everywhere).
    L : int
        Transform length.
    tol : float
        Relative residual tolerance for convergence.
    maxit : int
        Maximum number of PCG iterations.

    Returns
    -------
    x : ndarray, shape (L,)
        Recovered signal such that M_sigma x ≈ f.
    info : dict
        Convergence info with keys 'relres' (final relative residual),
        'iter' (number of iterations), 'converged' (bool).
    """
    f = np.asarray(f, dtype=float)

    def _apply_normal(x):
        """Apply M* M to x."""
        Mx = framemul(x, g_analysis, g_synthesis, a, sigma, L, real=real)
        return framemuladj(Mx, g_analysis, g_synthesis, a, sigma, L, real=real)

    # Right-hand side of normal equation
    rhs = framemuladj(f, g_analysis, g_synthesis, a, sigma, L, real=real)

    # Conjugate gradient
    x = np.zeros(L, dtype=float)
    r = rhs - _apply_normal(x)
    p = r.copy()
    rsold = np.dot(r, r)
    rhs_norm = np.linalg.norm(rhs)
    if rhs_norm == 0:
        return x, {'relres': 0.0, 'iter': 0, 'converged': True}

    converged = False
    for k in range(maxit):
        Ap = _apply_normal(p)
        pAp = np.dot(p, Ap)
        if pAp <= 0:
            break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = np.dot(r, r)
        relres = np.sqrt(rsnew) / rhs_norm
        if relres < tol:
            converged = True
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    else:
        relres = np.sqrt(rsold) / rhs_norm

    return x, {'relres': float(relres), 'iter': k + 1, 'converged': converged}


# ---------------------------------------------------------------------------
# framemulappr — best HS approximation
# ---------------------------------------------------------------------------

def _frame_matrices(g_analysis, g_synthesis, a, L, *, channel=None):
    """Return (A, Gamma, Nm): the analysis and synthesis matrices.

    ``A[lam, k] = <e_k, g_lam>`` — row ``lam`` is the analysis atom, built
    with one filterbank() call per basis vector (L calls total, all
    channels at once).

    ``Gamma[lam, :]`` is the synthesis atom, obtained by synthesising a
    unit coefficient in slot ``lam``. It is *measured* from ifilterbank
    rather than assumed to be conj(A), so whatever convention the
    synthesis side uses is captured exactly.

    ``Nm`` is the per-channel coefficient count, used to flatten and
    unflatten symbols.

    ``channel=m`` restricts the rows to channel ``m`` only. The diagonal
    method needs one channel at a time, and materialising the whole
    (Lambda x L) pair for a realistic filterbank is hundreds of megabytes
    -- which would defeat the purpose of having a cheap fallback.
    """
    M = len(g_analysis)
    c0 = filterbank(np.zeros(L), g_analysis, a, L)
    Nm = [len(np.asarray(c0[m]).ravel()) for m in range(M)]

    if channel is None:
        rows = list(range(M))
    else:
        rows = [channel]
    Lam = int(sum(Nm[m] for m in rows))
    offset = int(sum(Nm[m] for m in range(rows[0])))

    A = np.zeros((Lam, L), dtype=complex)
    for k in range(L):
        ek = np.zeros(L)
        ek[k] = 1.0
        ck = filterbank(ek, g_analysis, a, L)
        A[:, k] = np.concatenate([np.asarray(ck[m]).ravel() for m in rows])

    Gamma = np.zeros((Lam, L), dtype=complex)
    # real=False is deliberate: we want the raw complex synthesis atom, not
    # the folded real signal. ifilterbank warns about that for single-sided
    # banks, which is exactly the case here, so the warning is expected and
    # would otherwise fire once per coefficient.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*single-sided.*", category=UserWarning)
        for lam in range(Lam):
            cc = [np.zeros(n, dtype=complex) for n in Nm]
            i = lam + offset
            for m, n in enumerate(Nm):
                if i < n:
                    cc[m][i] = 1.0
                    break
                i -= n
            Gamma[lam, :] = np.asarray(
                ifilterbank(cc, g_synthesis, a, L, real=False)
            ).ravel()

    return A, Gamma, Nm


def _appr_system(A, Gamma, T, *, real=True):
    """Normal equations for the best Hilbert-Schmidt approximation.

    The multiplier is M_sigma = sum_lam sigma_lam P_lam with generator
    P_lam = gamma_lam (x) g_lam. Minimising ||T - M_sigma||_HS gives

        sum_mu <P_mu, P_lam>_HS sigma_mu = <T, P_lam>_HS

    With P_lam[i, j] = Gamma[lam, i] * A[lam, j] the Gram entries and the
    right-hand side both collapse to products of small matrices, so
    neither the generators nor the L x L operator ever have to be
    materialised per index.

    For ``real=True`` the synthesis folds the single-sided coefficients
    back to a real signal, so the effective generator is
    Q_lam = P_lam + conj(P_lam) = 2 Re(P_lam) and the system picks up the
    extra transpose terms below.
    """
    if real:
        gram = (2.0 * np.real((Gamma @ Gamma.T) * (A @ A.T))
                + 2.0 * np.real((Gamma @ Gamma.conj().T) * (A @ A.conj().T)))
        rhs = 2.0 * np.real(np.einsum("li,ij,lj->l", Gamma, T, A, optimize=True))
    else:
        gram = ((Gamma.conj() @ Gamma.T).T
                * (A.conj() @ A.T).T)
        rhs = np.einsum("li,ij,lj->l", Gamma.conj(), T, A.conj(), optimize=True)
    return gram, rhs


def framemulappr(T: np.ndarray,
                 g_analysis: list[dict],
                 g_synthesis: list[dict],
                 a,
                 L: int,
                 *,
                 real: bool = True,
                 method: str = "auto",
                 max_gram: int = 2000,
                 rcond: float | None = None) -> list[np.ndarray]:
    """Find the frame multiplier symbol that best approximates an operator.

    Computes the symbol sigma minimising the Hilbert-Schmidt norm
    ||T - M_sigma||_HS, following Balazs (2007). This is the filterbank
    generalisation of LTFAT's ``framemulappr`` / ``gabmulappr``.

    Parameters
    ----------
    T : ndarray, shape (L, L)
        The linear operator to approximate, as a matrix.
    g_analysis, g_synthesis : list of M filter dicts
        Analysis and synthesis frame filters.
    a : array-like
        Hop sizes.
    L : int
        Transform length.
    real : bool
        Must match the ``real`` flag used when applying the multiplier.
        With ``real=True`` the synthesis folds single-sided coefficients
        back to a real signal, which changes the generators and hence the
        normal equations.
    method : {'auto', 'full', 'diagonal'}
        ``'full'`` solves the complete normal equations and is the
        LTFAT-equivalent result. ``'diagonal'`` keeps only the Gram
        diagonal: far cheaper, but only valid when the generators are
        close to orthogonal, which for a redundant frame they are not.
        ``'auto'`` uses ``'full'`` when the total coefficient count is at
        most ``max_gram`` and falls back to ``'diagonal'`` with a warning
        otherwise.
    max_gram : int
        Coefficient-count ceiling for the full solve under ``'auto'``.
        The dense Gram is (Lambda x Lambda); the default keeps it to a
        few tens of megabytes.
    rcond : float or None
        Cutoff passed to ``numpy.linalg.lstsq`` for small singular
        values. ``None`` uses the NumPy default.

    Returns
    -------
    sigma : list of M ndarrays
        Symbol for each subband.

    Notes
    -----
    **The symbol need not be unique.** For a redundant frame the map
    sigma -> M_sigma has a non-trivial null space, so several symbols
    describe the same operator; the Gram is then singular. The full
    method returns the minimum-norm solution via ``lstsq``. Judge the
    result by the operator error ||T - M_sigma||, not by comparing the
    symbol against one you started with: a round trip through
    ``framemul`` reproduces the operator to machine precision while the
    symbol itself can differ substantially. LTFAT's Gabor test recovers
    the symbol exactly because for a Gabor system the map is injective;
    that does not carry over to general filterbanks.

    Cost is O(Lambda^2) memory and O(Lambda^3) time for the solve, plus
    L analysis calls and Lambda synthesis calls to build the frame
    matrices, where Lambda is the total number of coefficients.

    See Also
    --------
    MATH_REFERENCE.md 15a; LTFAT ``framemulappr``, ``gabmulappr``.
    """
    T = np.asarray(T)
    M = len(g_analysis)
    if method not in ("auto", "full", "diagonal"):
        raise ValueError(f"unknown method {method!r}")

    c0 = filterbank(np.zeros(L), g_analysis, a, L)
    Nm = [len(np.asarray(c0[m]).ravel()) for m in range(M)]
    Lam = int(sum(Nm))

    chosen = method
    if method == "auto":
        chosen = "full" if Lam <= max_gram else "diagonal"
        if chosen == "diagonal":
            warnings.warn(
                f"framemulappr: {Lam} coefficients exceeds max_gram={max_gram}, "
                f"falling back to the diagonal approximation. It is only valid "
                f"when the multiplier generators are near-orthogonal, which a "
                f"redundant frame does not satisfy; the symbol will be "
                f"mis-scaled. Pass method='full' to force the exact solve, or "
                f"raise max_gram.",
                RuntimeWarning, stacklevel=2,
            )

    if chosen == "full":
        A, Gamma, _ = _frame_matrices(g_analysis, g_synthesis, a, L)
        gram, rhs = _appr_system(A, Gamma, T, real=real)
        sol = np.real(np.linalg.lstsq(gram, rhs, rcond=rcond)[0])
        out, i = [], 0
        for n in Nm:
            out.append(sol[i:i + n])
            i += n
        return out

    # Diagonal fallback, one channel at a time so peak memory stays at
    # O(N_m * L) rather than O(Lambda * L).
    out = []
    for m in range(M):
        Am, Gm, _ = _frame_matrices(g_analysis, g_synthesis, a, L, channel=m)
        gram_m, rhs_m = _appr_system(Am, Gm, T, real=real)
        diag = np.real(np.diag(gram_m))
        safe = np.where(diag == 0, 1.0, diag)
        out.append(np.real(np.where(np.abs(diag) > 1e-30, rhs_m / safe, 0.0)))
    return out


# ---------------------------------------------------------------------------
# framemuleigs — eigenvalues
# ---------------------------------------------------------------------------

def framemuleigs(g_analysis: list[dict],
                 g_synthesis: list[dict],
                 a,
                 sigma: list[np.ndarray],
                 L: int,
                 K: int = 6,
                 *,
                 real: bool = True) -> np.ndarray:
    """Compute the K largest eigenvalues of a frame multiplier.

    For small L (≤ 200), uses direct eigendecomposition.
    For large L, uses iterative Arnoldi (scipy.sparse.linalg.eigsh).

    Parameters
    ----------
    g_analysis, g_synthesis : list of M filter dicts
        Analysis and synthesis frame filters.
    a : array-like
        Hop sizes.
    sigma : list of M ndarrays
        Time-frequency symbol.
    L : int
        Transform length.
    K : int
        Number of eigenvalues to compute (default 6).

    Returns
    -------
    eigenvalues : ndarray, shape (K,)
        K largest eigenvalues sorted by descending magnitude.
    """
    CROSSOVER = 200

    def _apply_mul(x):
        return framemul(x, g_analysis, g_synthesis, a, sigma, L, real=real)

    if L <= CROSSOVER:
        # Direct: build full matrix and use eig
        mat = np.zeros((L, L))
        for k in range(L):
            ek = np.zeros(L)
            ek[k] = 1.0
            mat[:, k] = _apply_mul(ek)

        # Symmetrise (should be symmetric for real symbol + same frames)
        mat = 0.5 * (mat + mat.T)  # type: ignore[assignment]
        eigvals = np.linalg.eigvalsh(mat)
        # Sort by descending magnitude
        idx = np.argsort(-np.abs(eigvals))
        return eigvals[idx[:K]]
    else:
        # Iterative: use Arnoldi via scipy
        op = LinearOperator((L, L), matvec=_apply_mul, dtype=float)
        # eigsh requires symmetric operator — true for real sigma + same frames
        eigvals, _ = eigsh(op, k=K, which='LM', tol=1e-9, maxiter=200)
        idx = np.argsort(-np.abs(eigvals))
        return eigvals[idx]  # type: ignore[no-any-return]
