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


def _negative_frequency_ratio(g_ready: list, L: int) -> float:
    """Energy on the negative-frequency half, relative to the positive half.

    Used to detect an analysis/synthesis convention mismatch.  Measured on the
    *filters*, not on the reconstructed spectrum: until v0.1.1 the check used
    the latter, where a single-sided ERB bank scores 0.383 and two-sided banks
    0.28-1.0 — overlapping ranges, so the flagship ``audfilters`` bank slipped
    past the 0.3 threshold and reconstructed with 46 % error in silence.

    On the filters themselves the two families separate cleanly::

        single-sided (aud, cqt, gab, wavelet)               0.007 - 0.122
        two-sided (gab real=False, warped/wavelet complex)  0.590 - 1.000

    so a 0.3 threshold has a factor-of-five margin on both sides.
    """
    half = L // 2
    pos = neg = 0.0
    for gm in g_ready:
        H = gm.get("H")
        if H is None or len(np.asarray(H)) == 0:
            # A time-domain (FIR) filter is real, hence two-sided.
            return 1.0
        H = np.asarray(H)
        foff = int(gm.get("foff", 0) or 0)
        idx = (foff + np.arange(H.size)) % L
        mag2 = np.abs(H) ** 2
        pos += float(np.sum(mag2[(idx >= 1) & (idx < half)]))
        neg += float(np.sum(mag2[idx > half]))
    return neg / max(pos, 1e-30)


def filterbank_is_real(g: list[dict], a, L: int) -> bool:
    """Is this a single-sided (real-audio) filterbank?

    Returns ``True`` when the filters carry essentially all their energy on the
    non-negative frequencies — the convention produced by ``audfilters``,
    ``cqtfilters``, ``gabfilters`` and the real wavelet designs, and the one
    for which ``real=True`` is correct in ``ifilterbank``, ``filterbankdual``
    and the phase-retrieval routines.  Returns ``False`` for genuinely
    two-sided banks (complex wavelets, warped banks, ``gabfilters`` with
    ``real=False``), where folding the spectrum would double-count.

    This is the same measurement ``ifilterbank`` uses for its
    convention-mismatch warning; see :func:`_negative_frequency_ratio` for the
    separation between the two families and the factor-of-five threshold
    margin.

    Parameters
    ----------
    g : list of M filter dicts
    a : hop sizes (needed to prepare the filters at length ``L``)
    L : int — DFT length the filters are evaluated on.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbank_is_real
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> filterbank_is_real(g, a, L)
    True
    """
    from ._utils import prepare_filters as _prep

    a_norm = normalise_a(a, len(g))
    g_ready, _m_td, _m_fft, _m_fftbl = _prep(g, a_norm, int(L))
    return _negative_frequency_ratio(g_ready, int(L)) <= 0.3


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

    # Detect an analysis/synthesis convention mismatch, measured on the
    # synthesis *filters*.  See `_negative_frequency_ratio` for why not on F.
    from ._utils import prepare_filters as _prep

    g_ready, _m_td, _m_fft, _m_fftbl = _prep(g, a_norm, L)
    ratio = _negative_frequency_ratio(g_ready, L)
    if real and ratio > 0.3:
        warnings.warn(
            "ifilterbank(real=True) but the synthesis filters appear two-sided "
            f"(negative/positive frequency energy {ratio:.2f}); folding will "
            "double-count. Pass real=False for complex/two-sided frames.",
            stacklevel=2,
        )
    elif (not real) and ratio < 0.3:
        warnings.warn(
            "ifilterbank(real=False) but the synthesis filters appear single-sided "
            f"(negative/positive frequency energy {ratio:.2f}); this will not "
            "reconstruct a real signal. Pass real=True (the default) for "
            "real-audio frames.",
            stacklevel=2,
        )

    if real:
        # Single-sided filterbank: mirrors the one-sided spectrum
        out = 2.0 * np.fft.ifft(F, axis=0).real
    else:
        out = np.fft.ifft(F, axis=0)  # type: ignore[assignment]

    # Trim to Ls.
    #
    # ``Ls > L`` used to fall through this branch and return L samples with no
    # indication that the request had been ignored — the caller asked for a
    # length and silently got a different one, which then propagated into
    # whatever they did next (a shape mismatch several frames later, or worse,
    # a broadcast that quietly worked).  There is nothing to synthesise beyond
    # L: the coefficients only determine that many samples.  Say so.
    if Ls is not None and Ls > L:
        warnings.warn(
            f"ifilterbank: Ls={Ls} exceeds the transform length L={L}; the "
            f"coefficients determine only {L} samples, so {L} are returned. "
            f"Zero-pad the result yourself if you need {Ls}.",
            stacklevel=2,
        )
    elif Ls is not None:
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
    real: bool | None = None,
    dtype: np.dtype | type | None = None,
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

            Defaults to ``None``, meaning *derive it from the filters* via
            :func:`filterbank_is_real`.  The old ``False`` default reconstructed
            the package's flagship bank with 23 % error where the correct mode
            reaches 4.5e-16 — the same defect as ``filterbankiter``'s default,
            on the synthesis side.  Pass an explicit value to override.
    dtype : output dtype for the reconstructed signal.  ``None`` (the default)
            keeps the historical float64.  The NumPy backend is a float64
            reference implementation throughout — ``filterbank`` itself widens
            a float32 input to complex128 — so this is an opt-in narrowing for
            callers who want it, not a change of default.  Genuine dtype
            polymorphism lives in the torch backend.

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

    if real is None:
        real = filterbank_is_real(g, a_norm, L)

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
            if dtype is not None:
                xr = np.asarray(xr).astype(dtype, copy=False)
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
        return np.zeros(Ls, dtype=dtype or float), 0.0, 0

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
    # Diagonal (Jacobi) preconditioner.
    #
    # `filterbankresponse` is the frame-operator diagonal indexed by *DFT bin*,
    # while the CG vectors here live in the *time* domain.  Until v0.1.1 the
    # code multiplied the time-domain residual by that frequency-domain
    # diagonal, which is not a preconditioner at all — it is an arbitrary
    # window, and it consistently slowed convergence (9 -> 15 iterations on a
    # gabfilters bank where the correct preconditioner takes 4).
    #
    # The frame operator of a shift-invariant filterbank is diagonal in
    # frequency, so the preconditioner has to be applied there: transform,
    # divide, transform back.
    apply_precond = None
    if alg == "pcg":
        try:
            resp = filterbankresponse(g, a, L, real=real)
            resp_safe = np.where(np.abs(resp) < 1e-14, 1.0, resp)

            def apply_precond(v, _resp=resp_safe):
                return np.fft.ifft(np.fft.fft(v) / _resp)

        except Exception:
            apply_precond = None

    # ------------------------------------------------------------------
    # Conjugate Gradient iteration
    # ------------------------------------------------------------------
    if x_warm is not None:
        x = x_warm.astype(complex).copy()  # warm start from the diagonal dual
        r = b - _apply_frame_op(x)
    else:
        x = np.zeros(L, dtype=complex)
        r = b.copy()  # r = b - A x = b  (x=0)
    if apply_precond is not None:
        z = apply_precond(r)
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

        if apply_precond is not None:
            z_new = apply_precond(r)
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
    #
    # Measured on the value actually returned.  Until v0.1.1 this was computed
    # on the complex CG iterate `x` while `np.real(x)` was returned, so the
    # function could report a converged 3.2e-07 for a signal whose true
    # residual was 0.226.
    x_real = np.real(x)
    relres_final = _analysis_relres(x_real.astype(complex))

    xr = x_real[:Ls] if (Ls is not None and Ls <= L) else x_real
    if dtype is not None:
        xr = xr.astype(dtype, copy=False)

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
    real: bool | None = None,
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

            Defaults to ``None``, meaning *derive it from the filters* via
            :func:`filterbank_is_real`.  It used to default to ``False``, which
            made this the one member of the family whose default disagreed with
            its siblings — ``ifilterbank``, ``filterbankdual`` and
            ``filterbanktight`` all default to ``real=True`` — and, worse, the
            documented default diverged on the package's flagship bank: on
            ``audfilters(4000, 512)`` it ran the full 100 iterations to a
            relative residual of 58 and a round-trip error of 6.2e+06, where
            the correct mode converges in 9 iterations. Deriving it is better
            than flipping the default, which would only move the breakage onto
            two-sided banks. Pass an explicit value to override.

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

    if real is None:
        real = filterbank_is_real(g, a_norm, L)

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
    # Diagonal (Jacobi) preconditioner.
    #
    # `filterbankresponse` is the frame-operator diagonal indexed by *DFT bin*,
    # while the CG vectors here live in the *time* domain.  Until v0.1.1 the
    # code multiplied the time-domain residual by that frequency-domain
    # diagonal, which is not a preconditioner at all — it is an arbitrary
    # window, and it consistently slowed convergence (9 -> 15 iterations on a
    # gabfilters bank where the correct preconditioner takes 4).
    #
    # The frame operator of a shift-invariant filterbank is diagonal in
    # frequency, so the preconditioner has to be applied there: transform,
    # divide, transform back.
    apply_precond = None
    if alg == "pcg":
        try:
            resp = filterbankresponse(g, a, L, real=real)
            resp_safe = np.where(np.abs(resp) < 1e-14, 1.0, resp)

            def apply_precond(v, _resp=resp_safe):
                return np.fft.ifft(np.fft.fft(v) / _resp)

        except Exception:
            apply_precond = None

    # CG on  FF* x = f
    b = f_pad
    x = np.zeros(L, dtype=complex)
    r = b.copy()
    z = apply_precond(r) if apply_precond is not None else r.copy()
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

        z_new = apply_precond(r) if apply_precond is not None else r.copy()
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
