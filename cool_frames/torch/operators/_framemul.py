"""
torch.operators._framemul
=========================
Frame multiplier operators for non-uniform filterbanks (PyTorch implementation).

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

import numpy as np
import torch

from .._dtypes import resolve
from ..filterbanks._core import filterbank, ifilterbank

# ---------------------------------------------------------------------------
# framemul — forward multiplier
# ---------------------------------------------------------------------------


def framemul(
    f: torch.Tensor,
    g_analysis: list[dict],
    g_synthesis: list[dict],
    a,
    sigma: list[torch.Tensor],
    L: int,
    *,
    real: bool = True,
) -> torch.Tensor:
    """Apply a frame multiplier to a signal.

    Computes M_sigma f = ifilterbank(sigma * filterbank(f, g_a, a, L),
                                      g_s, a, L).

    Parameters
    ----------
    f : Tensor, shape (L,)
        Input signal.
    g_analysis : list of M filter dicts
        Analysis frame filters.
    g_synthesis : list of M filter dicts
        Synthesis frame filters (dual, tight, or other).
    a : array-like
        Hop sizes, shape (M,) or (M, 2).
    sigma : list of M Tensors
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
    result : Tensor, shape (len(f) - delta,)
        M_sigma f, where delta depends on the hop sizes and filter bank
        structure. This matches the output length of ifilterbank.

    See Also
    --------
    framemuladj : adjoint operator
    framemulinv : inverse (PCG)
    MATH_REFERENCE.md §15a
    """
    # The caller's dtype wins.  Until v0.1.1 every entry point forced float32,
    # so a float64 call matched the NumPy backend only to ~1.8e-07 and
    # `torch.autograd.gradcheck` failed outright.
    f = torch.as_tensor(f)
    _rdtype, _cdtype = resolve(f)
    c = filterbank(f, g_analysis, a, L)
    c_masked = [c[m] * sigma[m] for m in range(len(c))]
    result = ifilterbank(c_masked, g_synthesis, a, L, real=real)
    return torch.real(result)


# ---------------------------------------------------------------------------
# framemuladj — adjoint
# ---------------------------------------------------------------------------


def framemuladj(
    f: torch.Tensor,
    g_analysis: list[dict],
    g_synthesis: list[dict],
    a,
    sigma: list[torch.Tensor],
    L: int,
    *,
    real: bool = True,
) -> torch.Tensor:
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
    result : Tensor, shape (len(f) - delta,)
        M*_sigma f, where delta depends on the hop sizes and filter bank
        structure. This matches the output length of ifilterbank.
    """
    sigma_conj = [torch.conj(s) for s in sigma]
    return framemul(f, g_synthesis, g_analysis, a, sigma_conj, L, real=real)


# ---------------------------------------------------------------------------
# framemulinv — inverse via PCG
# ---------------------------------------------------------------------------


def framemulinv(
    f: torch.Tensor,
    g_analysis: list[dict],
    g_synthesis: list[dict],
    a,
    sigma: list[torch.Tensor],
    L: int,
    *,
    real: bool = True,
    tol: float = 1e-9,
    maxit: int = 100,
) -> tuple[torch.Tensor, dict]:
    """Invert a frame multiplier via preconditioned conjugate gradient.

    Solves M_sigma x = f by applying PCG to the normal equation
    M*_sigma M_sigma x = M*_sigma f.

    Requires sigma[m][n] != 0 for all m, n (otherwise M is singular).

    Parameters
    ----------
    f : Tensor, shape (L,)
        Right-hand side (the output of framemul to invert).
    g_analysis, g_synthesis : list of M filter dicts
        Analysis and synthesis frame filters.
    a : array-like
        Hop sizes.
    sigma : list of M Tensors
        Time-frequency symbol (must be nonzero everywhere).
    L : int
        Transform length.
    tol : float
        Relative residual tolerance for convergence.
    maxit : int
        Maximum number of PCG iterations.

    Returns
    -------
    x : Tensor, shape (len(f),)
        Recovered signal such that M_sigma x ≈ f.
    info : dict
        Convergence info with keys 'relres' (final relative residual),
        'iter' (number of iterations), 'converged' (bool).
    """
    # The caller's dtype wins.  Until v0.1.1 every entry point forced float32,
    # so a float64 call matched the NumPy backend only to ~1.8e-07 and
    # `torch.autograd.gradcheck` failed outright.
    f = torch.as_tensor(f)
    _rdtype, _cdtype = resolve(f)
    len(f)

    def _apply_normal(x):
        """Apply M* M to x."""
        # framemul will return shorter output, so we need to pad or work with variable lengths
        # The framemulinv should work in the space defined by the coefficient representation
        # For consistency with numpy behavior, we initialize x with length Ls and
        # let framemul handle the dimension mismatch by working in coefficient space
        Mx = framemul(x, g_analysis, g_synthesis, a, sigma, L, real=real)
        return framemuladj(Mx, g_analysis, g_synthesis, a, sigma, L, real=real)

    # Right-hand side of normal equation
    rhs = framemuladj(f, g_analysis, g_synthesis, a, sigma, L, real=real)

    # Conjugate gradient - work with the actual output length of framemuladj
    x = torch.zeros(len(rhs), dtype=rhs.dtype, device=f.device)
    r = rhs - _apply_normal(x)
    p = r.clone()
    rsold = torch.dot(r, r)
    rhs_norm = torch.linalg.norm(rhs)
    if rhs_norm == 0:
        return x, {"relres": 0.0, "iter": 0, "converged": True}

    converged = False
    # `k` used to stay 0 because the loop bound the variable `_k`, so the
    # reported iteration count was always 1 regardless of the work done.
    # `k` is read after the loop for the iteration count, so it must be the
    # loop variable — but ruff cannot see that, hence the explicit noqa.
    k = 0
    for k in range(maxit):  # noqa: B007
        Ap = _apply_normal(p)
        pAp = torch.dot(p, Ap)
        if pAp <= 0:
            break
        alpha = rsold / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r, r)
        relres = torch.sqrt(rsnew) / rhs_norm
        if relres < tol:
            converged = True
            break
        beta = rsnew / rsold
        p = r + beta * p
        rsold = rsnew
    else:
        relres = torch.sqrt(rsold) / rhs_norm

    return x, {
        "relres": float(relres.detach().cpu().item()),
        "iter": k + 1,
        "converged": converged,
    }


# ---------------------------------------------------------------------------
# framemulappr — best HS approximation
# ---------------------------------------------------------------------------


def framemulappr(
    T: torch.Tensor,
    g_analysis: list[dict],
    g_synthesis: list[dict],
    a,
    L: int,
    *,
    real: bool = True,
    method: str = "auto",
    max_gram: int = 2000,
    rcond: float | None = None,
) -> list[torch.Tensor]:
    """Find the frame multiplier symbol that best approximates an operator.

    Computes the symbol sigma minimising the Hilbert-Schmidt norm
    ``||T - M_sigma||_HS``, following Balazs (2007).

    Parameters
    ----------
    T : Tensor, shape (L, L)
        The linear operator to approximate (as a matrix).
    g_analysis, g_synthesis : list of M filter dicts
        Analysis and synthesis frame filters.
    a : array-like
        Hop sizes.
    L : int
        Transform length.
    real, method, max_gram, rcond
        As in :func:`cool_frames.numpy.operators.framemulappr`.

    Returns
    -------
    sigma : list of M Tensors
        Symbol for each subband, on ``T``'s device and dtype.

    Notes
    -----
    This delegates to the NumPy implementation.  Fitting a symbol to a dense
    ``(L, L)`` operator is setup-time work — the same category as the dual
    windows, which this backend already computes with NumPy — and nothing
    downstream differentiates through it.

    .. versionchanged:: 0.1.1
       Was a separate, incorrect implementation.  It built its synthesis
       matrix by *analysing* with ``g_synthesis`` instead of measuring the
       synthesis atoms, only ever performed the diagonal approximation
       (NumPy's ``method='full'`` least-squares solve, ``max_gram`` and
       ``rcond`` were absent), and never read ``real`` — ``real=False`` output
       was bitwise identical to ``real=True``.  On an operator constructed as
       an exact multiplier, where a zero-error symbol provably exists, NumPy
       returned 1.05e-15 Hilbert-Schmidt error and this returned **1.033** —
       worse than returning a zero symbol.

    See Also
    --------
    MATH_REFERENCE.md §15a
    """
    from ...numpy.operators._framemul import framemulappr as _np_framemulappr

    T_t = torch.as_tensor(T)
    device = T_t.device
    rdtype, cdtype = resolve(T_t)

    sigma_np = _np_framemulappr(
        T_t.detach().cpu().numpy(),
        g_analysis,
        g_synthesis,
        a,
        L,
        real=real,
        method=method,
        max_gram=max_gram,
        rcond=rcond,
    )

    out = []
    for s in sigma_np:
        s = np.asarray(s)
        dt = cdtype if np.iscomplexobj(s) else rdtype
        out.append(torch.as_tensor(s, device=device).to(dt))
    return out


# ---------------------------------------------------------------------------
# framemuleigs — eigenvalues
# ---------------------------------------------------------------------------


def framemuleigs(
    g_analysis: list[dict],
    g_synthesis: list[dict],
    a,
    sigma: list[torch.Tensor],
    L: int,
    K: int = 6,
    *,
    real: bool = True,
) -> torch.Tensor:
    """Compute the K largest eigenvalues of a frame multiplier.

    For small L (≤ 200), uses direct eigendecomposition via torch.linalg.eigvalsh.
    For large L, uses iterative Arnoldi (scipy.sparse.linalg.eigsh) with a
    torch-backed LinearOperator.

    Parameters
    ----------
    g_analysis, g_synthesis : list of M filter dicts
        Analysis and synthesis frame filters.
    a : array-like
        Hop sizes.
    sigma : list of M Tensors
        Time-frequency symbol.
    L : int
        Transform length.
    K : int
        Number of eigenvalues to compute (default 6).

    Returns
    -------
    eigenvalues : Tensor, shape (K,)
        K largest eigenvalues sorted by descending magnitude.
    """
    CROSSOVER = 200

    # Precision follows the symbol, which is the only tensor argument here.
    _eig_rdtype, _ = resolve(*(sigma if isinstance(sigma, list) else [sigma]))

    def _apply_mul(x):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=_eig_rdtype)
        result = framemul(x, g_analysis, g_synthesis, a, sigma, L, real=real)
        if isinstance(result, torch.Tensor):
            return result.detach().cpu().numpy()
        return result

    if L <= CROSSOVER:
        # Direct: build full matrix and use torch.linalg.eigvalsh
        device = sigma[0].device if sigma else torch.device("cpu")
        mat = torch.zeros((L, L), dtype=_eig_rdtype, device=device)
        for k in range(L):
            ek = torch.zeros(L, dtype=_eig_rdtype, device=device)
            ek[k] = 1.0
            mat[:, k] = framemul(ek, g_analysis, g_synthesis, a, sigma, L, real=real)

        # Symmetrise (should be symmetric for real symbol + same frames)
        mat = 0.5 * (mat + mat.t())
        eigvals = torch.linalg.eigvalsh(mat)
        # Sort by descending magnitude
        idx = torch.argsort(-torch.abs(eigvals))
        return eigvals[idx[:K]]  # type: ignore[no-any-return]
    else:
        # Iterative: use Arnoldi via scipy with torch-backed LinearOperator
        from scipy.sparse.linalg import LinearOperator, eigsh

        op = LinearOperator((L, L), matvec=_apply_mul, dtype=float)
        # eigsh requires symmetric operator — true for real sigma + same frames
        eigvals, _ = eigsh(op, k=K, which="LM", tol=1e-9, maxiter=200)  # type: ignore[no-any-return]
        idx = np.argsort(-np.abs(eigvals))  # type: ignore[assignment]
        result = torch.as_tensor(eigvals[idx], dtype=_eig_rdtype)  # type: ignore[assignment]
        if sigma:
            result = result.to(sigma[0].device)
        return result
