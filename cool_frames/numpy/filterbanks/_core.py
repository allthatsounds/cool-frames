"""
numpy/filterbanks/_core.py
=====================
Public analysis/synthesis API: filterbank, ifilterbank, ufilterbank.

MATLAB originals
----------------
  layer2/analysis_synthesis/filterbank.m
  layer2/analysis_synthesis/ifilterbank.m
  layer2/analysis_synthesis/ufilterbank.m
"""

from __future__ import annotations

import warnings

import numpy as np

from ..core._core import postpad
from ..filters._design import filterbanklength
from ._utils import comp_filterbank, comp_ifilterbank, normalise_a

# ---------------------------------------------------------------------------
# filterbank – non-uniform analysis
# ---------------------------------------------------------------------------


def filterbank(f, g: list[dict], a, L: int | None = None, stack: bool = False):
    """Apply a (non-uniform) filterbank to signal *f*.

    Parameters
    ----------
    f : array_like, shape (Ls,) or (Ls, W)
    g : list of M filter dicts (from blfilter / audfilters)
    a : hop sizes – int, (M,), or (M, 2) fractional
    L : DFT length.  Computed from Ls and a if omitted.
    stack : bool, default False
        If True, stack the per-channel coefficients into a single 2-D array
        instead of returning a ragged list.  This requires a *uniform*
        filterbank (every channel the same length, i.e. a scalar hop ``a``);
        a ``ValueError`` is raised otherwise.  Subsumes the former
        ``ufilterbank``.

    Returns
    -------
    c : list of M arrays  (``stack=False``, default)
        Each of shape (N_m,) for mono or (N_m, W) for multi-channel.
    c : ndarray  (``stack=True``)
        Shape (N, M) for mono or (N, M, W) for multi-channel.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbank
    >>> x = np.random.randn(8000)
    >>> g, a, fc, L, _info = audfilters(8000, len(x))
    >>> c = filterbank(x, g, a, L)
    >>> len(c) == len(g)  # M subbands
    True
    >>> all(cm.shape[0] > 0 for cm in c)  # each non-empty
    True
    """
    f = np.asarray(f)
    mono = f.ndim == 1
    if mono:
        f = f[:, np.newaxis]
    Ls, W = f.shape

    M = len(g)
    a_norm = normalise_a(a, M)

    if L is None:
        L = filterbanklength(Ls, a_norm)

    f_pad = postpad(f, L, axis=0)
    c = comp_filterbank(f_pad, g, a_norm)

    # Squeeze to 1-D for mono input
    if mono:
        c = [cm.ravel() if cm.shape[1] == 1 else cm for cm in c]

    if stack:
        arrs = [np.asarray(cm) for cm in c]
        lengths = {cm.shape[0] for cm in arrs}
        if len(lengths) != 1:
            raise ValueError(
                "filterbank(..., stack=True) needs a uniform filterbank "
                "(all channels the same length); got channel lengths "
                f"{sorted(lengths)}. Pass a scalar hop `a`, or leave stack=False."
            )
        if arrs[0].ndim == 1:
            return np.column_stack(arrs)  # (N, M)
        return np.stack(arrs, axis=1)  # (N, M, W)

    return c


# ---------------------------------------------------------------------------
# ifilterbank – non-uniform synthesis
# ---------------------------------------------------------------------------


def ifilterbank(
    c: list[np.ndarray], g: list[dict], a, Ls: int | None = None, real: bool = True
) -> np.ndarray:
    """Synthesise a signal from filterbank coefficients.

    Parameters
    ----------
    c    : list of M arrays (N_m,) or (N_m, W)
    g    : list of M filter dicts
    a    : hop sizes
    Ls   : output length.  Inferred from coefficients if omitted.
    real : default ``True`` --- the correct mode for **real audio**, where the
           synthesis filters (canonical dual/tight of a single-sided real
           filterbank) cover only the positive frequencies, so the one-sided
           spectrum is mirrored via ``2 * real(ifft(F))``.  Pass ``real=False``
           **for complex / two-sided frames**.  Equivalent to MATLAB
           ``ifilterbank(c, g, a, 'real')`` when ``True``.  The default matches
           ``filterbankdual``/``filterbanktight`` (also ``real=True``), so the
           common pipeline ``ifilterbank(filterbank(x,g,a,L), filterbankdual(g,a,L), a, L)``
           reconstructs exactly without passing any flag.  A convention mismatch
           (single-sided filters with ``real=False``, or two-sided with
           ``real=True``) is detected and warned about, since it otherwise
           reconstructs silently wrong.

    Returns
    -------
    f : (Ls,) or (Ls, W) reconstructed signal

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbank, ifilterbank, filterbankdual
    >>> x = np.random.randn(8000)
    >>> g, a, fc, L, _info = audfilters(8000, len(x))
    >>> c = filterbank(x, g, a, L)
    >>> x_recon = ifilterbank(c, filterbankdual(g, a, L), a, len(x))  # real=True default
    >>> x_recon.shape == x.shape
    True
    """
    M = len(c)
    a_norm = normalise_a(a, M)

    # Determine L from coefficients
    afrac = a_norm[:, 0] / a_norm[:, 1]
    c0 = c[0]
    N0 = c0.shape[0] if c0.ndim > 1 else len(c0)
    L = int(round(N0 * afrac[0]))

    # Ensure all subbands are 2-D
    c2d = [np.atleast_2d(cm).T if cm.ndim == 1 else cm for cm in c]

    F = comp_ifilterbank(c2d, g, a_norm, L)

    # Detect an obvious analysis/synthesis convention mismatch (cheap, on F).
    # A single-sided (real-audio) frame has energy only on the positive-frequency
    # half; a two-sided (complex) frame fills both halves. Warn when ``real`` does
    # not match the apparent frame -- the mismatch reconstructs silently wrong.
    if L >= 8:
        half = L // 2
        f_pos = float(np.linalg.norm(F[1:half]))
        f_neg = float(np.linalg.norm(F[half + 1 :]))
        # single-sided (real-audio) frames carry little negative-half energy
        # (empirically f_neg/f_pos ~ 0.1); two-sided (complex) frames carry
        # comparable energy on both halves (~1.0). Thresholds 0.3 / 0.7 separate
        # them with a wide margin, so the warning fires only on a clear mismatch.
        if real and f_neg > 0.7 * f_pos:
            warnings.warn(
                "ifilterbank(real=True) but the synthesis filters appear two-sided "
                "(comparable negative-frequency energy); folding will double-count. "
                "Pass real=False for complex/two-sided frames.",
                stacklevel=2,
            )
        elif (not real) and f_pos > 0.0 and f_neg < 0.3 * f_pos:
            warnings.warn(
                "ifilterbank(real=False) but the synthesis filters appear single-sided "
                "(little negative-frequency energy); this will not reconstruct a real "
                "signal. Pass real=True (the default) for real-audio frames.",
                stacklevel=2,
            )

    if real:
        # Single-sided filterbank: mirrors the one-sided spectrum
        out = 2.0 * np.fft.ifft(F, axis=0).real
    else:
        out = np.fft.ifft(F, axis=0)  # type: ignore[assignment]

    # Trim to Ls
    if Ls is not None and Ls <= L:
        out = out[:Ls]

    # Squeeze mono
    W = out.shape[1] if out.ndim > 1 else 1
    if W == 1:
        out = out.ravel()

    return out


# ---------------------------------------------------------------------------
# ifilterbankiter – iterative synthesis (frame algorithm)
# ---------------------------------------------------------------------------


def ifilterbankiter(
    c: list[np.ndarray],
    g: list[dict],
    a,
    Ls: int | None = None,
    tol: float = 1e-6,
    maxit: int = 100,
    alg: str = "cg",
    real: bool = False,
) -> tuple:
    """Iterative filterbank inversion via Conjugate Gradient.

    Solves  ``FF* x = F c``  where F is the analysis operator and F* is
    synthesis.  This is equivalent to MATLAB ``frsyniter``: given
    coefficients *c*, find the signal *x* whose analysis best matches *c*
    in the least-squares sense.

    Two algorithms are available:

    * ``'cg'``  — standard Conjugate Gradient (default).
    * ``'pcg'`` — Preconditioned CG using the diagonal of the frame
      operator (``filterbankresponse``) as preconditioner.  Falls back to
      plain CG if the diagonal cannot be computed.

    Parameters
    ----------
    c     : list of M coefficient arrays (from ``filterbank``)
    g     : list of M filter dicts
    a     : hop sizes
    Ls    : output signal length (default: inferred from coefficients)
    tol   : relative residual tolerance (default 1e-6)
    maxit : maximum CG iterations (default 100)
    alg   : ``'cg'`` or ``'pcg'``
    real  : if True, use real-filterbank dual/synthesis (single-sided
            filterbank covering positive frequencies only, as produced
            by ``audfilters``).  Equivalent to MATLAB ``ifilterbank(...,
            'real')``.

    Returns
    -------
    (xr, relres, niter) : tuple
        *xr* is the reconstructed signal, *relres* the final relative
        residual ``||FF*x - Fc|| / ||Fc||``, and *niter* the number of
        iterations performed.

    Notes
    -----
    When the painless dual frame is available the solution is exact in one
    step and CG is not needed.  This function detects that case and
    short-circuits accordingly.

    Port of MATLAB ``frsyniter.m`` (Søndergaard & Holighaus).
    """
    from ._frame import filterbankdual, filterbankresponse

    M = len(g)
    a_norm = normalise_a(a, M)
    afrac = a_norm[:, 0] / a_norm[:, 1]
    c0 = c[0]
    N0 = c0.shape[0] if c0.ndim > 1 else len(c0)
    L = int(round(N0 * afrac[0]))

    if Ls is None:
        Ls = L

    # Relative residual of the *analysis*:  || F(x) - c || / || c ||.
    # This is the honest convergence measure -- a candidate signal is a valid
    # reconstruction iff re-analysing it reproduces the coefficients.
    def _analysis_relres(x_vec):
        x2d = np.asarray(x_vec, dtype=complex).reshape(-1, 1)
        cf = comp_filterbank(x2d, g, a_norm)
        num = 0.0
        den = 0.0
        for cm_re, cm in zip(cf, c):
            cm_re = np.asarray(cm_re).ravel()
            cm = np.asarray(cm).ravel()
            num += float(np.sum(np.abs(cm_re - cm) ** 2))
            den += float(np.sum(np.abs(cm) ** 2))
        return np.sqrt(num / den) if den > 0 else 0.0

    # ------------------------------------------------------------------
    # Fast path: the closed-form (painless) dual is exact in ONE step --
    # but ONLY for painless frames.  ``filterbankdual`` always returns the
    # diagonal dual, which is *wrong* for non-painless banks (e.g. gabfilters
    # / waveletfilters whose filters are wider than L/a), so we must VALIDATE
    # it against the true analysis residual before trusting it.  If it does
    # not reconstruct, keep it as a warm start and fall through to CG.
    # ------------------------------------------------------------------
    x_warm = None
    try:
        if real:
            gd = filterbankdual(g, a, L)
        else:
            gd = filterbankdual(g, a, L, real=False)
        x_full = np.real(np.asarray(ifilterbank(c, gd, a, L, real=real))).astype(complex).ravel()
        rr = _analysis_relres(x_full)
        if rr <= tol:
            xr = np.real(x_full)
            if Ls is not None and Ls <= L:
                xr = xr[:Ls]
            return xr, rr, 1
        x_warm = x_full  # diagonal dual is only approximate -> warm start
    except Exception:
        x_warm = None

    # ------------------------------------------------------------------
    # CG / PCG on the normal equation  FF* x = F c  (= b)
    # ------------------------------------------------------------------
    # Right-hand side:  b = F* c  (synthesis with the analysis filters)
    b = np.asarray(ifilterbank(c, g, a, L, real=real), dtype=complex).ravel()
    norm_b = float(np.linalg.norm(b))
    if norm_b == 0.0:
        return np.zeros(Ls), 0.0, 0

    # Frame operator application:  A x = F*(F x)
    def _apply_frame_op(x_vec):
        x2d = x_vec.reshape(-1, 1)
        cf = comp_filterbank(x2d, g, a_norm)
        c2d = [np.atleast_2d(cm).T if cm.ndim == 1 else cm for cm in cf]
        F_out = comp_ifilterbank(c2d, g, a_norm, L)
        if real:
            return 2.0 * np.fft.ifft(F_out, axis=0).real.ravel().astype(complex)
        return np.fft.ifft(F_out, axis=0).ravel()

    # Preconditioner (diagonal of frame operator)
    precond = None
    if alg == "pcg":
        try:
            resp = filterbankresponse(g, a, L, real=real)
            resp_safe = np.where(np.abs(resp) < 1e-14, 1.0, resp)
            precond = 1.0 / resp_safe
        except Exception:
            precond = None

    # ------------------------------------------------------------------
    # Conjugate Gradient iteration
    # ------------------------------------------------------------------
    if x_warm is not None:
        x = x_warm.astype(complex).copy()  # warm start from the diagonal dual
        r = b - _apply_frame_op(x)
    else:
        x = np.zeros(L, dtype=complex)
        r = b.copy()  # r = b - A x = b  (x=0)
    if precond is not None:
        z = precond * r
    else:
        z = r.copy()
    p = z.copy()
    rz = np.vdot(r, z).real

    relres_list = []
    niter = 0

    for k in range(maxit):
        Ap = _apply_frame_op(p)
        pAp = np.vdot(p, Ap).real
        if abs(pAp) < 1e-30:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        relres_k = float(np.linalg.norm(r)) / norm_b
        relres_list.append(relres_k)
        niter = k + 1

        if relres_k < tol:
            break

        if precond is not None:
            z_new = precond * r
        else:
            z_new = r.copy()
        rz_new = np.vdot(r, z_new).real
        if abs(rz) < 1e-30:
            break
        beta = rz_new / rz
        p = z_new + beta * p
        rz = rz_new
        z = z_new

    # Report the honest analysis residual ||F(x)-c||/||c|| (not the
    # normal-equation residual, which can look small while the reconstruction
    # is still off for an ill-conditioned operator).
    relres_final = _analysis_relres(x)

    xr = np.real(x)
    if Ls is not None and Ls <= L:
        xr = xr[:Ls]

    return xr, relres_final, niter


# ---------------------------------------------------------------------------
# filterbankiter – iterative analysis (CG on FF*)
# ---------------------------------------------------------------------------


def filterbankiter(
    f,
    g: list[dict],
    a,
    L: int | None = None,
    tol: float = 1e-6,
    maxit: int = 100,
    alg: str = "cg",
    real: bool = False,
) -> tuple:
    """Iterative filterbank analysis via Conjugate Gradient.

    Renamed from ``filterbankanaiter`` (2026-06-12) for symmetry with the
    direct pair: ``filterbank``/``filterbankiter`` (analysis) mirror
    ``ifilterbank``/``ifilterbankiter`` (synthesis).

    Solves  ``FF* x = f``  then returns ``c = F x``, producing
    coefficients that give perfect reconstruction through the *synthesis*
    operator (without needing the dual frame).

    This is the filterbank-native equivalent of MATLAB ``franaiter``.

    Parameters
    ----------
    f     : array_like, shape (Ls,)
    g     : list of M filter dicts
    a     : hop sizes
    L     : DFT length (computed from Ls if omitted)
    tol   : relative residual tolerance (default 1e-6)
    maxit : maximum CG iterations (default 100)
    alg   : ``'cg'`` or ``'pcg'``
    real  : if True, use real-filterbank synthesis (``2*real(ifft(...))``)
            for single-sided filterbanks (e.g. from ``audfilters``).

    Returns
    -------
    (c, relres, niter) : tuple
        *c* is a list of coefficient arrays (like ``filterbank`` output),
        *relres* the final relative residual, and *niter* the iteration
        count.

    Notes
    -----
    The algorithm first recovers x via CG on ``FF* x = f``, then returns
    ``c = F x``.  Port of MATLAB ``franaiter.m`` (Søndergaard).
    """
    from ._frame import filterbankresponse

    f = np.asarray(f, dtype=np.float64)
    mono = f.ndim == 1
    if mono:
        f = f[:, np.newaxis]
    Ls = f.shape[0]

    M = len(g)
    a_norm = normalise_a(a, M)

    if L is None:
        L = filterbanklength(Ls, a_norm)

    f_pad = postpad(f, L, axis=0).ravel().astype(complex)
    norm_f = float(np.linalg.norm(f_pad))
    if norm_f == 0.0:
        c_out = filterbank(np.zeros(Ls), g, a, L)
        return c_out, 0.0, 0

    # Frame operator application:  A x = F*(F x)
    def _apply_frame_op(x_vec):
        x2d = x_vec.reshape(-1, 1)
        cf = comp_filterbank(x2d, g, a_norm)
        c2d = [np.atleast_2d(cm).T if cm.ndim == 1 else cm for cm in cf]
        F_out = comp_ifilterbank(c2d, g, a_norm, L)
        if real:
            return 2.0 * np.fft.ifft(F_out, axis=0).real.ravel().astype(complex)
        return np.fft.ifft(F_out, axis=0).ravel()

    # Preconditioner
    precond = None
    if alg == "pcg":
        try:
            resp = filterbankresponse(g, a, L, real=real)
            resp_safe = np.where(np.abs(resp) < 1e-14, 1.0, resp)
            precond = 1.0 / resp_safe
        except Exception:
            precond = None

    # CG on  FF* x = f
    b = f_pad
    x = np.zeros(L, dtype=complex)
    r = b.copy()
    z = (precond * r) if precond is not None else r.copy()
    p = z.copy()
    rz = np.vdot(r, z).real

    niter = 0
    relres_final = 1.0

    for k in range(maxit):
        Ap = _apply_frame_op(p)
        pAp = np.vdot(p, Ap).real
        if abs(pAp) < 1e-30:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap

        relres_k = float(np.linalg.norm(r)) / norm_f
        relres_final = relres_k
        niter = k + 1

        if relres_k < tol:
            break

        z_new = (precond * r) if precond is not None else r.copy()
        rz_new = np.vdot(r, z_new).real
        if abs(rz) < 1e-30:
            break
        beta = rz_new / rz
        p = z_new + beta * p
        rz = rz_new
        z = z_new

    # Compute coefficients c = F x
    x_real = np.real(x)[:Ls] if Ls <= L else np.real(x)
    c_out = filterbank(x_real, g, a, L)
    return c_out, relres_final, niter


# ---------------------------------------------------------------------------
# filterbanklengthcoef – infer L from coefficient arrays
# ---------------------------------------------------------------------------


def filterbanklengthcoef(coef, a) -> int:
    """Determine the filterbank length *L* from coefficient arrays.

    Parameters
    ----------
    coef : list of M arrays (each of shape ``(N_m, ...)``), or a single
           2-D array ``(N, M)`` for uniform filterbanks.
    a    : hop sizes – scalar, (M,) vector, or (M, 2) fractional.

    Returns
    -------
    L : int – the system length satisfying ``N_m = ceil(L * a_m(2) / a_m(1))``
        for every channel.

    Raises
    ------
    ValueError
        If the inferred *L* is inconsistent across channels.

    See Also
    --------
    filterbanklength : compute *L* from a signal length.

    .. note::

       MATLAB original: ``layer2/analysis_synthesis/filterbanklengthcoef.m``
    """
    if isinstance(coef, (list, tuple)):
        cl = np.array([c.shape[0] if hasattr(c, "shape") else len(c) for c in coef])
    else:
        coef = np.asarray(coef)
        M_coef = coef.shape[1] if coef.ndim > 1 else 1
        cl = np.full(M_coef, coef.shape[0])

    a = np.asarray(a)
    if a.ndim == 0:
        a = np.full(len(cl), int(a))
    if a.ndim == 1:
        a = a.ravel()
        if len(a) == 1:
            a = np.repeat(a, len(cl))
        L_arr = a * cl
    else:
        # Fractional: L = a(:,1) .* cl ./ a(:,2)
        L_arr = a[:, 0] * cl / a[:, 1]

    L_arr = L_arr.astype(float)
    if np.ptp(L_arr) > 0.5:
        raise ValueError(
            "Invalid set of coefficients. The product of the number of "
            "coefficients and the channel time shift must be the same for "
            "all channels."
        )

    return int(round(L_arr[0]))
