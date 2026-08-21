"""
numpy/phaseret/_decolbfgs.py
==============================
Décorsière's L-BFGS phase retrieval adapted for filterbanks.

Port of ``phaseret/gabor/decolbfgs.m``.

The algorithm formulates phase retrieval as an unconstrained smooth
optimization problem.  Given target magnitudes ``s``, it minimizes:

    J(x) = sum_m sum_n | |c_m[n]|^p - s_m[n]^p |^2

where c = filterbank(x, g, a) and p is a compression parameter
(default 2/3).  The gradient is computed via the chain rule using
the filterbank analysis/synthesis operators.

The optimization uses L-BFGS from ``scipy.optimize.minimize``.

References: Décorsière et al., 2015.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..filterbanks._core import filterbank, ifilterbank
from ..filterbanks._frame import filterbankdual


def decolbfgs(
    s_list: list[np.ndarray],
    g: list[dict],
    a,
    *,
    L: int | None = None,
    Ls: int | None = None,
    real: bool = False,
    maxit: int = 100,
    tol: float = 1e-6,
    p: float = 2.0 / 3.0,
    startphase: Literal["input", "zero", "rand"] = "zero",
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, int]:
    """Décorsière's L-BFGS phase retrieval for filterbanks.

    Minimizes a smooth objective based on compressed magnitude
    differences using L-BFGS.

    Parameters
    ----------
    s_list : list of M arrays — target magnitudes
    g : list of M filter dicts
    a : hop sizes
    L : DFT length
    Ls : output signal length
    real : use real (single-sided) synthesis
    maxit : maximum L-BFGS iterations
    tol : convergence tolerance
    p : compression parameter for the objective (default: 2/3)
    startphase : ``'input'``, ``'zero'``, or ``'rand'``

    Returns
    -------
    c : list of M complex arrays — coefficients with reconstructed phase
    f : reconstructed signal
    relres : array of residuals at each iteration
    niter : number of iterations performed
    """
    try:
        from scipy.optimize import minimize as scipy_minimize
    except ImportError as err:
        raise ImportError("decolbfgs requires scipy. Install it with: pip install scipy") from err

    M = len(g)
    s_abs = [np.abs(np.asarray(s)).ravel() for s in s_list]

    from ..filterbanks._utils import normalise_a

    a_norm = normalise_a(a, M)

    N = [len(s) for s in s_abs]
    if L is None:
        afrac = a_norm[:, 0] / a_norm[:, 1]
        L = int(round(N[0] * afrac[0]))

    if real:
        gd = filterbankdual(g, a_norm, L)
    else:
        gd = filterbankdual(g, a_norm, L, real=False)

    # Precompute target magnitudes raised to power p
    s_p = [np.power(s + np.finfo(float).tiny, p) for s in s_abs]

    s_flat = np.concatenate(s_abs)
    norm_s = np.linalg.norm(s_flat)
    if norm_s == 0:
        norm_s = 1.0  # type: ignore[assignment]

    # Initial signal estimate
    if startphase == "zero":
        c0 = [s.copy().astype(complex) for s in s_abs]
    elif startphase == "rand":
        rng = np.random.default_rng()
        c0 = [s * np.exp(2j * np.pi * rng.random(len(s))) for s in s_abs]
    else:  # 'input'
        c0 = [np.asarray(s, dtype=complex).ravel().copy() for s in s_list]

    # Get initial signal via synthesis
    x0 = ifilterbank(c0, gd, a_norm, Ls=L, real=real)
    if real:
        x0 = np.real(x0)
    x0 = x0.ravel().astype(float) if real else np.concatenate([x0.real.ravel(), x0.imag.ravel()])

    relres_list = []

    def _objective_and_grad(x_flat):
        """Compute objective and gradient for L-BFGS."""
        if real:
            x = x_flat.copy()
        else:
            n = len(x_flat) // 2
            x = x_flat[:n] + 1j * x_flat[n:]

        # Analysis
        c = filterbank(x, g, a_norm, L=L)

        # Objective: sum |  |c_m[n]|^p - s_m[n]^p |^2
        obj = 0.0
        grad_c = []
        for m in range(M):
            cm = np.asarray(c[m]).ravel()
            cm_abs = np.abs(cm)
            cm_p = np.power(cm_abs + np.finfo(float).tiny, p)
            diff = cm_p - s_p[m]
            obj += np.sum(diff**2)

            # Gradient w.r.t. c_m:
            # d/d(c_m) |c_m|^p = p * |c_m|^(p-2) * conj(c_m)
            # d/d(c_m) (|c_m|^p - s^p)^2
            #   = 2 * (|c_m|^p - s^p) * p * |c_m|^(p-2) * conj(c_m)
            dcm = 2.0 * diff * p * np.power(cm_abs + np.finfo(float).tiny, p - 2.0) * np.conj(cm)
            grad_c.append(dcm)

        # Residual for tracking
        c_abs_flat = np.concatenate([np.abs(np.asarray(cm).ravel()) for cm in c])
        res = float(np.linalg.norm(c_abs_flat - s_flat) / norm_s)
        relres_list.append(res)

        # Gradient w.r.t. x via synthesis of grad_c with analysis window
        # grad_x = ifilterbank(grad_c, g, a)  (adjoint of filterbank)
        # For the filterbank adjoint: if c = filterbank(x, g, a),
        # then grad_x = ifilterbank(grad_c, g, a)
        grad_x = ifilterbank(grad_c, g, a_norm, Ls=L, real=real)

        if real:
            grad_x = np.real(grad_x).ravel()
        else:
            grad_x = np.concatenate([np.real(grad_x).ravel(), np.imag(grad_x).ravel()])

        return float(obj), grad_x.astype(float)

    # Run L-BFGS
    result = scipy_minimize(
        _objective_and_grad,
        x0.astype(float),
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": maxit,
            "ftol": tol * tol,
            "gtol": tol,
            "maxfun": maxit * 20,
        },
    )

    # Recover signal and coefficients
    if real:
        f = result.x.copy()
    else:
        n = len(result.x) // 2
        f = result.x[:n] + 1j * result.x[n:]

    c = filterbank(f, g, a_norm, L=L)

    # Enforce magnitude constraint on final output
    for m in range(M):
        cm = np.asarray(c[m]).ravel()
        phase = np.angle(cm)
        c[m] = s_abs[m] * np.exp(1j * phase)

    # Final synthesis
    f = ifilterbank(c, gd, a_norm, Ls=Ls or L, real=real)
    if real:
        f = np.real(f)

    niter = result.nit if hasattr(result, "nit") else len(relres_list)
    relres = np.array(relres_list) if relres_list else np.array([0.0])

    return c, f, relres, niter
