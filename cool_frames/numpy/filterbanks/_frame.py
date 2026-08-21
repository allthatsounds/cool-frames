"""
numpy/layer2/_frame.py
======================
Frame-theoretic functions: filterbankbounds, filterbankdual, filterbanktight,
filterbankscale, filterbankresponse, filterbankfreqz.

MATLAB originals
----------------
  layer2/frame/filterbankbounds.m
  layer2/frame/filterbankdual.m
  layer2/frame/filterbanktight.m
  layer2/frame/filterbankscale.m
  layer2/frame/filterbankresponse.m
  layer2/format/filterbankfreqz.m
"""
from __future__ import annotations

import numpy as np

from ..filters._filters import filter_freqresp
from ._utils import normalise_a

# ---------------------------------------------------------------------------
# filterbankfreqz – evaluate frequency responses
# ---------------------------------------------------------------------------

def filterbankfreqz(g: list[dict], a=None, L: int | None = None) -> np.ndarray:
    """Evaluate the frequency responses of all filters at *L* DFT bins.

    Parameters
    ----------
    g : list of M filter dicts
    a : ignored
        A filter's transfer function does not depend on the hop size, so this
        argument has never been read.  It is kept — and now explicitly
        optional — because the whole filterbank API takes ``(g, a, L)`` and
        removing it would break every call site.  Documenting it as a live
        parameter, as this did until v0.1.1, invited callers to believe passing
        a different ``a`` would change the result; it does not.
    L : DFT length

    Returns
    -------
    H : (L, M) complex array
        Frequency responses; each column is H_m(k) for k=0..L-1.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbankfreqz
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> H = filterbankfreqz(g, a, L)
    >>> H.shape
    (8000, 34)
    """
    if L is None:
        raise TypeError("filterbankfreqz: L is required")
    M = len(g)
    H = np.zeros((L, M), dtype=complex)
    for m, gm in enumerate(g):
        H_full, _ = filter_freqresp(gm, L)
        H[:, m]   = H_full
    return H


# ---------------------------------------------------------------------------
# filterbankresponse – total frame response
# ---------------------------------------------------------------------------

def filterbankresponse(g: list[dict], a, L: int,
                       real: bool = False) -> np.ndarray:
    """Compute the frame response (diagonal of the frame operator).

    ``resp[k] = sum_m |H_m(k)|^2 / a_m``

    Parameters
    ----------
    g    : list of M filter dicts
    a    : hop sizes
    L    : DFT length
    real : if True, treat the filterbank as real-valued (fold negative
           frequencies onto positives)

    Returns
    -------
    resp : (L,) real-valued array
        Pointwise frame response.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbankresponse
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> resp = filterbankresponse(g, a, L)
    >>> resp.shape
    (8000,)
    >>> np.all(resp > 0)
    True
    """
    M      = len(g)
    a_norm = normalise_a(a, M)
    afrac  = a_norm[:, 0] / a_norm[:, 1]

    H = filterbankfreqz(g, a_norm, L)
    # |H_m(k)|^2 / a_m, summed over m
    resp = np.real(H * H.conj()) @ (1.0 / afrac)

    if real:
        # MATLAB ``comp_filterbankresponse(g, a, L, 1)``:
        # ``gf = gf + involute(gf)`` where involute(f)[k] = conj(f[-k mod L]).
        # Since resp is real: involute(resp)[0] = resp[0],
        # involute(resp)[k] = resp[L-k] for k >= 1.
        resp_inv = np.empty_like(resp)
        resp_inv[0] = resp[0]
        resp_inv[1:] = resp[1:][::-1]
        return (resp + resp_inv).real  # type: ignore[no-any-return]

    return resp.real  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# filterbankbounds – frame lower and upper bounds
# ---------------------------------------------------------------------------

def filterbankbounds(g: list[dict], a, L: int, real: bool = True,
                     return_kappa: bool = False):
    """Return the frame bounds ``(A, B)`` for the filterbank.

    With ``return_kappa=True`` returns ``(A, B, kappa)`` where ``kappa = B/A``
    is the condition number (``inf`` if ``A == 0``). This is the single source
    of frame-quality numbers in cool_frames --- there is no separate ``diagnose``.

    ``A = min(resp)``, ``B = max(resp)`` where ``resp[k] = sum_m |H_m(k)|^2/a_m``.

    ``real`` (default ``True`` --- the right choice for real audio) uses the
    *folded* real frame response (``resp + involute(resp)``), whose extreme
    values are the painless-case frame bounds for real signals; ``real=False``
    uses the full two-sided complex response. Subsumes the former
    ``filterbankrealbounds``.

    **κ matches the SVD ground truth (2026-06-12).** The folded convention was
    validated against :func:`filterbankbounds_svd` (the exact eigenvalues of the
    frame operator): the condition number ``B/A`` returned here equals the SVD
    κ in every case tested --- e.g. a canonical tight ``audfilters`` frame reads
    κ=1, and a ``cqtfilters`` ``filterbanktight`` frame correctly reads κ=2 (it
    is genuinely *not* tight --- a known ``painlessfilterbank`` band-limited bug
    that the earlier un-folded formula masked). Valid only under the painless
    condition; for non-painless / aliased banks use :func:`filterbankbounds_svd`.
    Note the *absolute* level can differ from the SVD by a representation factor
    (folding double-counts two-sided ``realonly=0`` filters); the κ is exact, and
    :func:`filterbankbounds_svd` gives the exact absolute bounds.

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbankbounds, filterbanktight
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> A, B = filterbankbounds(g, a, L)
    >>> 0 < A <= B
    True
    >>> A, B, kappa = filterbankbounds(g, a, L, return_kappa=True)
    >>> abs(kappa - B / A) < 1e-9
    True
    >>> At, Bt = filterbankbounds(filterbanktight(g, a, L), a, L)
    >>> abs(Bt / At - 1.0) < 1e-6      # audfilters tight frame -> kappa = 1
    True
    """
    M = len(g)
    a_norm = normalise_a(a, M)
    resp = filterbankresponse(g, a_norm, L, real=real)   # fold when real
    A = float(np.min(resp))
    B = float(np.max(resp))
    if return_kappa:
        kappa = B / A if A > 0 else float("inf")
        return A, B, kappa
    return A, B


# ---------------------------------------------------------------------------
# filterbankbounds_svd – GENERAL frame bounds (any bank, painless or not)
# ---------------------------------------------------------------------------

def filterbankbounds_svd(g: list[dict], a, L: int,
                         real: bool = True) -> tuple[float, float]:
    r"""Exact frame bounds (A, B) via the eigenvalues of the frame operator.

    Unlike :func:`filterbankbounds` --- which uses the closed-form diagonal
    response and is therefore only valid under the *painless* condition (each
    filter band-limited to its hop) --- this computes the bounds for an
    **arbitrary** filterbank, including non-painless / aliased ones, from the
    actual frame operator.

    The analysis operator :math:`D` is materialised column by column
    (``D e_k = filterbank(e_k)``). The frame operator is :math:`S = D^{H}D` and
    its extreme eigenvalues are the frame bounds. For ``real=True`` the bounds
    are those of the operator restricted to real signals,
    :math:`\Re(D^{H}D)` (for real :math:`f`, :math:`\|Df\|^2 = f^{T}\Re(D^{H}D)f`,
    convention-independent of how each filter stores its negative half).
    For ``real=False`` the complex bounds (squared singular values of ``D``)
    are returned.

    .. note::
       This builds an ``L``-column operator and an ``L\times L`` Gram matrix,
       so it is :math:`O(L^2)` in memory and intended as ground truth for
       verification (and for non-painless banks), not as the hot path. Use
       :func:`filterbankbounds` for painless banks at scale.

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbankbounds_svd, filterbanktight
    >>> g, a, fc, L, _info = audfilters(2000, 256)
    >>> A, B = filterbankbounds_svd(filterbanktight(g, a, L), a, L)
    >>> abs(B / A - 1.0) < 1e-6     # a true tight frame has kappa = 1
    True
    """
    from ._core import filterbank
    M = len(g)
    a_norm = normalise_a(a, M)
    cols = []
    e = np.zeros(L)
    for k in range(L):
        e[k] = 1.0
        c = filterbank(e, g, a_norm, L)
        cols.append(np.concatenate([np.asarray(cm).ravel() for cm in c]))
        e[k] = 0.0
    D = np.asarray(cols).T               # (Ncoef, L): column k is D e_k
    gram = D.conj().T @ D                 # D^H D  (L, L)
    gram = np.real(gram) if real else gram
    gram = (gram + gram.conj().T) / 2.0   # symmetrise against round-off
    ev = np.linalg.eigvalsh(gram)
    # Clamp at zero.  A frame bound is a non-negative quantity by definition;
    # returning the raw eigenvalue extreme let rank-deficient banks report
    # A = -9.7e-16, which downstream turned into "condition numbers" like
    # -3.2e+15.  A genuinely zero lower bound means "not a frame", which is the
    # honest reading of a tiny negative eigenvalue.
    return float(max(ev[0], 0.0)), float(ev[-1])


# ---------------------------------------------------------------------------
# filterbankdual – canonical dual frame
# ---------------------------------------------------------------------------

def filterbankdual(g: list[dict], a, L: int, real: bool = True) -> list[dict]:
    """Return the canonical dual-frame filter bank (painless case).

    ``real`` (default ``True`` --- the right choice for real audio) folds the
    frame response for a single-sided real filterbank so that
    ``ifilterbank(..., real=True)`` reconstructs exactly; ``real=False`` gives
    the complex/two-sided dual. Subsumes the former ``filterbankrealdual``.
    Thin wrapper over :func:`painlessfilterbank`.

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbankdual
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> gd = filterbankdual(g, a, L)
    >>> len(gd) == len(g)
    True
    """
    return painlessfilterbank(g, a, L, 'dual', 1 if real else 0)


# ---------------------------------------------------------------------------
# filterbanktight – canonical tight frame
# ---------------------------------------------------------------------------

def filterbanktight(g: list[dict], a, L: int, real: bool = True) -> list[dict]:
    """Return the canonical tight-frame filter bank (painless case).

    ``real`` (default ``True`` --- the right choice for real audio) folds the
    response for a single-sided real filterbank; ``real=False`` gives the
    complex/two-sided tight frame. Subsumes the former ``filterbankrealtight``.
    Thin wrapper over :func:`painlessfilterbank`.

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbanktight
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> gt = filterbanktight(g, a, L)
    >>> len(gt) == len(g)
    True
    """
    return painlessfilterbank(g, a, L, 'tight', 1 if real else 0)


# ---------------------------------------------------------------------------
# filterbankscale – scale a set of filters
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# painlessfilterbank – standalone painless-case frame computation
# ---------------------------------------------------------------------------

def painlessfilterbank(
    g: list[dict],
    a,
    L: int,
    type_: str = "dual",
    do_real: int = 0,
) -> list[dict]:
    """Compute the painless-case canonical dual or tight frame filters.

    Standalone version of the computation inlined in :func:`filterbankdual`
    and :func:`filterbanktight`.  Matches MATLAB
    ``painlessfilterbank(g, a, L, type, do_real)``.

    For each filter *m*, the output is the band-limited quotient:

    - **dual**: ``Gd_m[j] = TF_m(foff + j) / resp(foff + j)``
    - **tight**: ``Gt_m[j] = TF_m(foff + j) / sqrt(resp(foff + j))``

    where ``TF_m`` is the full transfer function and ``resp`` is the
    diagonal of the frame operator (optionally folded for real signals).

    Parameters
    ----------
    g : list of M filter dicts
        Analysis filters.
    a : hop sizes
    L : DFT length
    type_ : ``"dual"`` or ``"tight"``
        Which canonical frame to compute.
    do_real : int
        If 1, use the real-signal response ``resp + involute(resp)``
        (matching MATLAB ``do_real=1``).

    Returns
    -------
    gout : list of M filter dicts
        Band-limited canonical frame filters with ``realonly=0``.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks._frame import painlessfilterbank
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> gd = painlessfilterbank(g, a, L, 'dual', 0)
    >>> len(gd) == len(g)
    True
    """
    from ..filters._filters import filter_freqresp
    from ._utils import prepare_filters

    M = len(g)
    a_norm = normalise_a(a, M)

    resp = filterbankresponse(g, a_norm, L, real=bool(do_real))
    resp_safe = np.where(resp < 1e-14, 1e-14, resp)

    g_ready, _, _, _ = prepare_filters(g, a_norm, L)

    # The diagonal construction is exact only under the painless condition
    # (each filter's frequency support at most L/a).  `filterbankwin` has
    # always computed `info['ispainless']` and nothing ever read it, so a
    # non-painless bank got a silently approximate "dual"/"tight" — on
    # `waveletfilters` the returned "tight" frame is rank-deficient and loses
    # 72 % of the signal while `filterbankbounds` prints kappa = 1.000000.
    _violations = []
    for m in range(M):
        gm = g_ready[m]
        H_m = gm.get("H")
        if H_m is None or len(np.asarray(H_m)) == 0:
            continue
        Nm = L / (a_norm[m, 0] / a_norm[m, 1])
        # Strictly greater, with one bin of slack: several designers emit
        # L/a + 1 bins by construction and are painless in practice.
        if len(np.asarray(H_m)) > Nm + 1:
            _violations.append(m)

    if _violations:
        import warnings as _warnings

        _warnings.warn(
            f"filterbank{type_}: {len(_violations)} of {M} channels exceed the "
            f"painless limit L/a (e.g. channel {_violations[0]}: "
            f"{len(np.asarray(g_ready[_violations[0]]['H']))} bins against "
            f"{L / (a_norm[_violations[0], 0] / a_norm[_violations[0], 1]):.0f}). "
            f"The diagonal {type_} is approximate for this bank; verify with "
            f"`filterbankbounds_svd`, or use `ifilterbankiter` for exact "
            f"reconstruction.",
            stacklevel=3,
        )

    if type_ == "dual":
        divisor = resp_safe
    elif type_ == "tight":
        divisor = np.sqrt(resp_safe)
    else:
        raise ValueError(f"type_ must be 'dual' or 'tight', got {type_!r}")

    gout = []
    for m in range(M):
        gm = g_ready[m]
        if "H" in gm and len(gm["H"]) > 0:
            foff_m = gm["foff"]
            LG = len(gm["H"])
            idx = np.mod(np.arange(foff_m, foff_m + LG), L)

            H_full_m, _ = filter_freqresp(gm, L)
            tf_at_support = H_full_m[idx]
            H_out_vals = tf_at_support / divisor[idx]

            gout.append({
                "H": H_out_vals,
                "foff": foff_m,
                "realonly": 0,
                "delay": 0,
                "fs": g[m].get("fs"),
            })
        else:
            # Time-domain (FIR) channel: `prepare_filters` leaves it as
            # {'h', 'offset'} with no 'H'.
            #
            # Until v0.1.1 this branch emitted the *zero filter*, so
            # `filterbankdual`/`filterbanktight` returned an all-zero bank for
            # any FIR filterbank and `ifilterbank` reconstructed exactly 0.0 —
            # silently, while the same bank still reported valid frame bounds.
            #
            # Computing a diagonal dual here instead would swap a visibly wrong
            # answer for a plausible-looking one: the painless construction is
            # valid only when each filter's *frequency* support is at most
            # L/a, and a time-limited (FIR) filter is full-band by
            # construction, so it never qualifies.  Refuse, and point at the
            # method that does work for an arbitrary frame.
            raise ValueError(
                f"filterbank{type_}: channel {m} is a time-domain (FIR) filter, "
                f"which has full frequency support and therefore never satisfies "
                f"the painless condition this diagonal construction requires. "
                f"There is no valid painless {type_} for such a bank.\n"
                f"Use the iterative inverse instead — `ifilterbankiter(c, g, a, L)` "
                f"reconstructs from any frame with positive lower bound — or design "
                f"the bank with band-limited filters (`blfilter`, `audfilters`, "
                f"`cqtfilters`, ...), for which the painless dual is exact."
            )
    return gout


# ---------------------------------------------------------------------------
# filterbankscale – scale a set of filters
# ---------------------------------------------------------------------------

def filterbankscale(g: list[dict], s) -> list[dict]:
    """Return a copy of the filter list with each filter scaled by *s[m]*.

    Parameters
    ----------
    g : list of M filter dicts
    s : float scalar or (M,) array of scale factors

    Returns
    -------
    gs : list of M filter dicts
        Scaled filters (copies).

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbankscale
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> gs = filterbankscale(g, 2.0)
    >>> len(gs) == len(g)
    True
    """
    M = len(g)
    s_arr = np.broadcast_to(np.asarray(s, dtype=float), (M,))

    gs = []
    for m, gm in enumerate(g):
        gm_new = dict(gm)
        sm     = float(s_arr[m])
        if "H" in gm:
            if callable(gm["H"]):
                H_old  = gm["H"]
                gm_new["H"] = (lambda L, _H=H_old, _s=sm: np.asarray(_H(L)) * _s)
            else:
                gm_new["H"] = np.asarray(gm["H"]) * sm
        elif "h" in gm:
            gm_new["h"] = np.asarray(gm["h"]) * sm
        gs.append(gm_new)
    return gs
