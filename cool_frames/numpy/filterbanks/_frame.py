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


def filterbankfreqz(
    g: list[dict], a=None, L: int | None = None, dtype: np.dtype | type = complex
) -> np.ndarray:
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
    (10368, 29)
    """
    if L is None:
        raise TypeError("filterbankfreqz: L is required")
    M = len(g)
    H = np.zeros((L, M), dtype=dtype)
    for m, gm in enumerate(g):
        H_full, _ = filter_freqresp(gm, L)
        H[:, m] = H_full
    return H


# ---------------------------------------------------------------------------
# filterbankresponse – total frame response
# ---------------------------------------------------------------------------


def filterbankresponse(
    g: list[dict], a, L: int, real: bool = False, dtype: np.dtype | type = float
) -> np.ndarray:
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
    >>> resp = filterbankresponse(g, a, L, real=True)
    >>> resp.shape
    (10368,)
    >>> bool(np.all(resp > 0))   # covered everywhere, so the frame is invertible
    True

    ``real=True`` is what makes that last claim true, and this example used to
    omit it: with the default ``real=False`` the negative-frequency half of a
    single-sided bank is simply empty, so 4376 of the 10368 bins are exactly
    zero and ``np.all(resp > 0)`` is ``False``.  The response is only
    everywhere-positive once the one-sided spectrum is folded.
    """
    M = len(g)
    a_norm = normalise_a(a, M)
    afrac = a_norm[:, 0] / a_norm[:, 1]

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
        return np.asarray((resp + resp_inv).real, dtype=dtype)

    return np.asarray(resp.real, dtype=dtype)


# ---------------------------------------------------------------------------
# filterbankbounds – frame lower and upper bounds
# ---------------------------------------------------------------------------


def filterbankbounds(g: list[dict], a, L: int, real: bool = True, return_kappa: bool = False):
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
    that the earlier un-folded formula masked). The closed form is valid only
    under the painless condition.  A *uniform* bank that is not painless
    (``gabfilters`` at its default lattice, a uniform FIR bank) gets exact
    bounds instead, from the polyphase blocks of its frame operator (LTFAT's
    ``filterbank(real)bounds``), in the same convention: the extreme
    eigenvalues of ``sum_m H_m H_m^H / a`` on each coset of bins, with the
    mirror image folded in for ``real=True``.  For a bank that is neither,
    use :func:`filterbankbounds_svd`.
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
    from ._utils import prepare_filters

    M = len(g)
    a_norm = normalise_a(a, M)
    a_uni = _uniform_hop(a_norm, L)
    if a_uni is not None and _nonpainless_channels(prepare_filters(g, a_norm, L)[0], a_norm, L):
        A, B = _uniform_bounds(g, a_uni, L, real)
    else:
        resp = filterbankresponse(g, a_norm, L, real=real)  # fold when real
        A = float(np.min(resp))
        B = float(np.max(resp))
    if return_kappa:
        kappa = B / A if A > 0 else float("inf")
        return A, B, kappa
    return A, B


# ---------------------------------------------------------------------------
# filterbankbounds_svd – GENERAL frame bounds (any bank, painless or not)
# ---------------------------------------------------------------------------


def filterbankbounds_svd(g: list[dict], a, L: int, real: bool = True) -> tuple[float, float]:
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
    D = np.asarray(cols).T  # (Ncoef, L): column k is D e_k
    gram = D.conj().T @ D  # D^H D  (L, L)
    gram = np.real(gram) if real else gram
    gram = (gram + gram.conj().T) / 2.0  # symmetrise against round-off
    ev = np.linalg.eigvalsh(gram)
    # Clamp at zero.  A frame bound is a non-negative quantity by definition;
    # returning the raw eigenvalue extreme let rank-deficient banks report
    # A = -9.7e-16, which downstream turned into "condition numbers" like
    # -3.2e+15.  A genuinely zero lower bound means "not a frame", which is the
    # honest reading of a tiny negative eigenvalue.
    return float(max(ev[0], 0.0)), float(ev[-1])


# ---------------------------------------------------------------------------
# Uniform, non-painless banks: the polyphase branch of LTFAT's
# filterbank(real)dual / filterbank(real)tight / filterbank(real)bounds
# ---------------------------------------------------------------------------
#
# A bank whose channels all share one integer hop ``a`` has a frame operator
# that the DFT splits into ``N = L/a`` independent ``a x a`` blocks, one per
# coset ``{w, w - N, w - 2N, ...}`` of frequency bins.  Painless banks make
# each block diagonal (the closed forms above); any other uniform bank --
# ``gabfilters`` at its default lattice, a uniform FIR bank -- still has an
# exact canonical dual and tight frame, obtained block by block.  This is
# LTFAT's own algorithm (``filterbankdual.m`` and its siblings, uniform
# branch); cool-frames had only ported the painless one, so ``gabfilters``
# reconstructed to 4.9e-4 and its closed-form bounds were off by 2.6e-4.

# Bins of a canonical dual or tight filter below this fraction of the
# filter's peak are dropped when its support is stored.  The dual of a
# non-painless bank is supported on all L bins in exact arithmetic, but it
# decays fast; at this level the stored support of the default gabfilters
# bank is about 4.7x the analysis support instead of L, and the round trip
# is unchanged (5.9e-16 against 5.7e-16 with every bin kept).
_COMPACT_REL = float(np.finfo(float).eps)


def _uniform_hop(a_norm: np.ndarray, L: int) -> int | None:
    """The common integer hop of a uniform bank, or ``None``."""
    num = np.asarray(a_norm[:, 0])
    den = np.asarray(a_norm[:, 1])
    if num.size == 0 or np.any(den != 1) or np.any(num != num[0]):
        return None
    a0 = int(num[0])
    if a0 <= 0 or L % a0 != 0:
        return None
    return a0


def _nonpainless_channels(g_ready: list[dict], a_norm: np.ndarray, L: int) -> list[int]:
    """Channels that violate the painless condition (FIR channels always do)."""
    from ..filters._painless import ALIAS_TOL, aliasing

    bad = []
    for m, gm in enumerate(g_ready):
        H_m = gm.get("H")
        if H_m is None:
            bad.append(m)  # time-domain FIR filter: full frequency support
            continue
        if len(np.asarray(H_m)) == 0:
            continue
        Nm = L / (a_norm[m, 0] / a_norm[m, 1])
        if aliasing(H_m, Nm) > ALIAS_TOL:
            bad.append(m)
    return bad


def _compact_support(col: np.ndarray) -> tuple[np.ndarray, int]:
    """Shortest circular run of bins holding every value above
    ``_COMPACT_REL`` of the peak, as ``(values, first_bin)``: the complement
    of the widest circular gap between those bins."""
    L = col.size
    mag = np.abs(col)
    peak = float(mag.max()) if L else 0.0
    if peak == 0.0:
        return np.zeros(1, dtype=col.dtype), 0
    kept = np.flatnonzero(mag > _COMPACT_REL * peak)
    gaps = np.diff(kept)
    wrap = int(kept[0] + L - kept[-1])
    i = int(np.argmax(gaps)) if gaps.size else -1
    if i < 0 or wrap >= gaps[i]:
        first, last = int(kept[0]), int(kept[-1])
    else:
        first, last = int(kept[i + 1]), int(kept[i] + L)
    n = last - first + 1
    if n >= L:
        return col.copy(), 0
    idx = (first + np.arange(n)) % L
    return col[idx].copy(), first


def _uniform_entries(g: list[dict], a_norm: np.ndarray, L: int):
    """The non-zero DFT bins of every channel, as flat ``(bins, values,
    channel)`` arrays; band-limited channels contribute only their stored
    support."""
    from ._utils import prepare_filters

    g_ready = prepare_filters(g, a_norm, L)[0]
    Ks, Vs, Cs = [], [], []
    for m, gm in enumerate(g_ready):
        H = gm.get("H")
        if H is not None and np.asarray(H).size <= L:
            V = np.asarray(H, dtype=complex).ravel()
            if V.size == 1 and V[0] == 0:
                continue  # zero-filter sentinel
            K = np.mod(np.arange(int(gm["foff"]), int(gm["foff"]) + V.size), L)
        else:
            full, _ = filter_freqresp(gm, L)
            K = np.flatnonzero(full)
            V = full[K]
        nz = V != 0
        Ks.append(K[nz].astype(np.int64))
        Vs.append(V[nz])
        Cs.append(np.full(int(nz.sum()), m, dtype=np.int64))
    if not Ks:
        z = np.zeros(0, dtype=np.int64)
        return z, np.zeros(0, dtype=complex), z
    return np.concatenate(Ks), np.concatenate(Vs), np.concatenate(Cs)


def _uniform_cosets(g: list[dict], a_norm: np.ndarray, a: int, L: int, real: bool):
    """Yield ``(w0, w1, gram, entries)`` for chunks of cosets of a uniform bank.

    Coset ``w`` holds the bins ``w - j N`` (mod ``L``), ``j = 0..a-1``
    (LTFAT's ordering), with ``N = L/a``.  ``gram`` is the ``(w1-w0, a, a)``
    stack of ``Ha Ha^H`` -- ``a`` times the frame operator's block --, plus
    ``Hb Hb^H`` with ``Hb = conj(G[-bins])`` for real signals (LTFAT's
    ``filterbankreal*``; the painless ``real=True`` response folds the same
    mirror images in).  Each channel has few non-zero bins per coset, so the
    blocks are accumulated from those alone.  ``entries`` are the chunk's
    ``(w, j, channel, value)`` of the channels themselves, sorted by
    ``(w, channel)``, for the product with the block inverses.
    """
    N = L // a
    K, V, C = _uniform_entries(g, a_norm, L)
    M = len(g)

    def coset(Kx):
        wx = Kx % N
        return wx, ((wx - Kx) % L) // N

    w, j = coset(K)
    if real:
        Km = (-K) % L
        wm, jm = coset(Km)
        gw = np.concatenate([w, wm])
        gj = np.concatenate([j, jm])
        gc = np.concatenate([C, C + M])
        gv = np.concatenate([V, np.conj(V)])
    else:
        gw, gj, gc, gv = w, j, C, V
    go = np.lexsort((gj, gc, gw))
    gw, gj, gc, gv = gw[go], gj[go], gc[go], gv[go]
    eo = np.lexsort((j, C, w))
    ew, ej, ec, ev = w[eo], j[eo], C[eo], V[eo]

    per_coset = max(1, int(np.ceil(gw.size / max(N, 1))))
    chunk = max(1, min(N, int(64e6 // (16 * max(a * a, per_coset * a)))))
    for w0 in range(0, N, chunk):
        w1 = min(N, w0 + chunk)
        nw = w1 - w0
        lo, hi = np.searchsorted(gw, [w0, w1])
        cw, cj, cv = gw[lo:hi] - w0, gj[lo:hi], gv[lo:hi]
        key = cw * (2 * M + 1) + gc[lo:hi]
        flat_re = np.zeros(nw * a * a)
        flat_im = np.zeros(nw * a * a)
        n = key.size
        o = 0
        while o < n:
            i1 = np.flatnonzero(key[: n - o] == key[o:])
            if i1.size == 0:
                break
            i2 = i1 + o
            prod = cv[i1] * np.conj(cv[i2])
            idx = cw[i1] * (a * a) + cj[i1] * a + cj[i2]
            flat_re += np.bincount(idx, weights=prod.real, minlength=nw * a * a)
            flat_im += np.bincount(idx, weights=prod.imag, minlength=nw * a * a)
            if o > 0:
                idx2 = cw[i1] * (a * a) + cj[i2] * a + cj[i1]
                flat_re += np.bincount(idx2, weights=prod.real, minlength=nw * a * a)
                flat_im -= np.bincount(idx2, weights=prod.imag, minlength=nw * a * a)
            o += 1
        gram = (flat_re + 1j * flat_im).reshape(nw, a, a)
        elo, ehi = np.searchsorted(ew, [w0, w1])
        yield w0, w1, gram, (ew[elo:ehi], ej[elo:ehi], ec[elo:ehi], ev[elo:ehi])


def _uniform_frame(g: list[dict], a: int, L: int, type_: str, real: bool) -> list[dict]:
    """Canonical dual or tight bank of a uniform bank, coset by coset.

    Per coset: dual ``a (Ha Ha^H [+ Hb Hb^H])^{-1} Ha``; tight
    ``sqrt(a) (...)^{-1/2} Ha``, as in LTFAT (its complex tight frame is
    written ``U V^H`` from the SVD of ``Ha``, the same matrix), with a
    pseudo-inverse where a block is singular or nearly so (eigenvalues below
    ``_PINV_REL`` of the operator's scale count as zero).  ``Ha`` is
    sparse, so each column of the product is a combination of the few block
    columns its channel touches.
    """
    from scipy.sparse import csr_matrix

    M = len(g)
    N = L // a
    a_norm = normalise_a(a, M)
    # Coset-major layout: out[m, w, j] is channel m at bin w - j N.
    out = np.zeros((M, N, a), dtype=complex)
    if type_ not in ("dual", "tight"):
        raise ValueError(f"type_ must be 'dual' or 'tight', got {type_!r}")
    # The frame operator's largest diagonal entry (its scale); eigenvalues
    # below _PINV_REL of it count as zero (see _block_pinv_power).
    K, V, _C = _uniform_entries(g, a_norm, L)
    diag = np.bincount(K, weights=np.abs(V) ** 2, minlength=L)
    if real:
        diag = diag + diag[(-np.arange(L)) % L]
    floor = _PINV_REL * float(diag.max()) if diag.size else 0.0
    for w0, w1, gram, (ew, ej, ec, ev) in _uniform_cosets(g, a_norm, a, L, real):
        if type_ == "dual":
            # LU inverse per block; a block that is singular, or has an
            # eigenvalue below the floor (its inverse has an entry above
            # 1/floor) or a condition number above 1e10, gets the
            # pseudo-inverse instead.
            try:
                Q = np.linalg.inv(gram)
                qmax = np.max(np.abs(Q), axis=(1, 2))
                bad = ~np.all(np.isfinite(Q), axis=(1, 2))
                bad |= qmax * np.max(np.abs(gram), axis=(1, 2)) > 1e10
                bad |= qmax * floor > 1.0
            except np.linalg.LinAlgError:
                Q = np.empty_like(gram)
                bad = np.ones(gram.shape[0], dtype=bool)
            if np.any(bad):
                Q[bad] = _block_pinv_power(gram[bad], -1.0, max(a, M), floor)
            Q *= a
        else:
            Q = _block_pinv_power(gram, -0.5, max(a, M), floor) * np.sqrt(a)
        # X_w = Q_w Ha_w with Ha_w sparse (a channel touches few bins of a
        # coset), computed transposed: X_w^T = Ha_w^T Q_w^T, one row per channel.
        bounds = np.searchsorted(ew, np.arange(w0, w1 + 1))
        for w in range(w0, w1):
            lo, hi = bounds[w - w0], bounds[w - w0 + 1]
            if lo == hi:
                continue
            HaT = csr_matrix((ev[lo:hi], (ec[lo:hi], ej[lo:hi])), shape=(M, a))
            out[:, w, :] = HaT @ Q[w - w0].T
    rows = ((np.arange(N)[:, None] - np.arange(a)[None, :] * N) % L).ravel()
    full = np.empty(L, dtype=complex)
    gout = []
    for m in range(M):
        full[rows] = out[m].ravel()
        H_m, first = _compact_support(full)
        gout.append(
            {
                "H": H_m,
                "foff": first,
                "realonly": 0,
                "delay": 0,
                "fs": g[m].get("fs"),
            }
        )
    return gout


# Eigenvalues of the frame operator below this fraction of its largest
# diagonal entry count as zero.  A frame whose lower bound is that small
# relative to its upper one has no numerically meaningful canonical dual:
# the inverse would amplify rounding by the same factor.  It matters for a
# bank that is not a frame, such as a single-sided bank with real=False,
# whose negative frequencies carry only tails of 1e-17 of the peak: kept,
# they made the dual's synthesis followed by the analysis wrong by 31 %.
_PINV_REL = 1e-10


def _block_pinv_power(gram: np.ndarray, power: float, dim: int, floor: float = 0.0) -> np.ndarray:
    """``gram^power`` on the range of each Hermitian block, zero on its
    null space: a pseudo-inverse for the dual (``power=-1``), its square
    root for the tight frame (``power=-1/2``).  Eigenvalues below MATLAB's
    ``pinv`` tolerance (``max(size) * norm * eps`` on the singular values of
    the analysis block, squared here) or below ``floor`` count as zero."""
    lam, V = np.linalg.eigh(gram)
    top = np.maximum(lam[:, -1:], 0.0)
    tol = np.maximum((dim * np.sqrt(top) * np.finfo(float).eps) ** 2, floor)
    keep = lam > tol
    inv = np.zeros_like(lam)
    inv[keep] = lam[keep] ** power
    return (V * inv[:, None, :]) @ np.conj(np.swapaxes(V, 1, 2))


def _uniform_bounds(g: list[dict], a: int, L: int, real: bool) -> tuple[float, float]:
    """Exact frame bounds of a uniform bank: the extreme eigenvalues of the
    coset blocks, divided by ``a`` (LTFAT's ``filterbank(real)bounds``)."""
    a_norm = normalise_a(a, len(g))
    A = np.inf
    B = 0.0
    for _w0, _w1, gram, _e in _uniform_cosets(g, a_norm, a, L, real):
        lam = np.linalg.eigvalsh(gram)
        A = min(A, float(lam[:, 0].min()))
        B = max(B, float(lam[:, -1].max()))
    return max(A, 0.0) / a, B / a


# The canonical dual / tight bank of a given bank, cached.  Iterative phase
# retrieval (gla, legla, the RTISI-LA family, decolbfgs) asks for the dual
# of the same bank on every call, and for a large non-painless bank that
# takes seconds.  Keyed by the *content* of the evaluated filters (not by
# object identity), so a filter changed in place, or a new bank with equal
# content, is looked up correctly; results are returned as copies.
_FRAME_CACHE: dict = {}
_FRAME_CACHE_SIZE = 4


def _frame_key(
    g_ready: list[dict], g: list[dict], a_norm: np.ndarray, L: int, type_: str, real: bool
) -> bytes:
    import hashlib

    h = hashlib.blake2b(digest_size=20)
    h.update(f"{type_}|{bool(real)}|{int(L)}|".encode())
    h.update(np.ascontiguousarray(a_norm, dtype=np.int64).tobytes())
    for gm, graw in zip(g_ready, g):
        for key in sorted(gm):
            v = gm[key]
            h.update(key.encode())
            if isinstance(v, np.ndarray):
                h.update(str(v.dtype).encode())
                h.update(str(v.shape).encode())
                h.update(np.ascontiguousarray(v).tobytes())
            elif isinstance(v, (int, float, complex, str, bool, np.number)) or v is None:
                h.update(repr(v).encode())
            else:
                h.update(repr(type(v)).encode())
        h.update(repr(graw.get("fs") if isinstance(graw, dict) else None).encode())
    return h.digest()


def _bank_digest(g: list[dict], a, L: int, tag: str = "") -> bytes:
    """Content digest of a filter bank at length ``L`` (callables evaluated),
    for caches of objects derived from it."""
    from ._utils import prepare_filters

    a_norm = normalise_a(a, len(g))
    g_ready = prepare_filters(g, a_norm, L)[0]
    return _frame_key(g_ready, g, a_norm, L, tag, False)


def _copy_bank(gout: list[dict]) -> list[dict]:
    return [
        {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in gm.items()}
        for gm in gout
    ]


def _canonical_frame(g: list[dict], a, L: int, type_: str, real: bool) -> list[dict]:
    """Canonical dual or tight bank: painless closed form, the exact uniform
    polyphase construction, or (non-uniform, non-painless) the diagonal
    approximation with a warning; cached by content."""
    from ._utils import prepare_filters

    M = len(g)
    a_norm = normalise_a(a, M)
    g_ready, _, _, _ = prepare_filters(g, a_norm, L)
    key = _frame_key(g_ready, g, a_norm, L, type_, real)
    hit = _FRAME_CACHE.get(key)
    if hit is not None:
        return _copy_bank(hit)

    a_uni = _uniform_hop(a_norm, L)
    if a_uni is not None and _nonpainless_channels(g_ready, a_norm, L):
        gout = _uniform_frame(g, a_uni, L, type_, real)
    else:
        gout = painlessfilterbank(g, a, L, type_, 1 if real else 0)

    if len(_FRAME_CACHE) >= _FRAME_CACHE_SIZE:
        _FRAME_CACHE.pop(next(iter(_FRAME_CACHE)))
    _FRAME_CACHE[key] = gout
    return _copy_bank(gout)


# ---------------------------------------------------------------------------
# filterbankdual – canonical dual frame
# ---------------------------------------------------------------------------


def filterbankdual(g: list[dict], a, L: int, real: bool = True) -> list[dict]:
    """Return the canonical dual-frame filter bank.

    ``real`` (default ``True`` --- the right choice for real audio) folds the
    frame response for a single-sided real filterbank so that
    ``ifilterbank(..., real=True)`` reconstructs exactly; ``real=False`` gives
    the complex/two-sided dual. Subsumes the former ``filterbankrealdual``.

    Painless banks get the closed form of :func:`painlessfilterbank`.  A
    *uniform* bank that is not painless (every channel the same integer hop,
    e.g. ``gabfilters`` at its default lattice, or a uniform FIR bank) gets
    its exact canonical dual from the polyphase construction of LTFAT's
    ``filterbankdual``: the frame operator splits into ``L/a`` blocks of size
    ``a x a``, each inverted exactly.  The dual's frequency responses are
    stored on the bins where they exceed ``eps`` of their peak.  Where the
    bank is not a frame -- e.g. a single-sided bank with ``real=False``,
    which sees no negative frequencies -- a pseudo-inverse is used, as
    LTFAT's ``pinv`` does, with eigenvalues below 1e-10 of the frame
    operator's scale counting as zero, so that synthesis followed by
    analysis is a projection to within 1e-6.  A bank that is neither
    painless nor uniform has no filterbank dual; the painless formula is
    then an approximation and a warning says so.

    Results are cached by the content of the filters, so iterative phase
    retrieval does not recompute the dual of the same bank on every call.

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbankdual
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> gd = filterbankdual(g, a, L)
    >>> len(gd) == len(g)
    True
    """
    return _canonical_frame(g, a, L, "dual", real)


# ---------------------------------------------------------------------------
# filterbanktight – canonical tight frame
# ---------------------------------------------------------------------------


def filterbanktight(g: list[dict], a, L: int, real: bool = True) -> list[dict]:
    """Return the canonical tight-frame filter bank.

    ``real`` (default ``True`` --- the right choice for real audio) folds the
    response for a single-sided real filterbank; ``real=False`` gives the
    complex/two-sided tight frame. Subsumes the former ``filterbankrealtight``.

    As for :func:`filterbankdual`: the closed form for painless banks, the
    exact polyphase construction (LTFAT's ``filterbanktight``) for uniform
    banks that are not painless, otherwise an approximation with a warning.
    Cached by the content of the filters.

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks import filterbanktight
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> gt = filterbanktight(g, a, L)
    >>> len(gt) == len(g)
    True
    """
    return _canonical_frame(g, a, L, "tight", real)


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
    from ..filters._painless import ALIAS_TOL, aliasing, nonzero_support

    _violations = []
    _supports = {}
    for m in range(M):
        gm = g_ready[m]
        H_m = gm.get("H")
        if H_m is None or len(np.asarray(H_m)) == 0:
            continue
        Nm = L / (a_norm[m, 0] / a_norm[m, 1])
        # Painless means no two bins N = L/a apart are both non-negligible
        # (``_painless.aliasing``), with no slack.  Several designers store
        # L/a + 1 bins whose end bins are exact zeros, and two-sided wavelet
        # banks store tails of 1e-11 past N; both alias onto nothing that
        # matters and are painless.  A support genuinely one bin wider is not,
        # and a one-bin allowance on the stored length used to let exactly
        # that through without a warning: cqtfilters(sampling='fractional')
        # had 999 non-zero Nyquist bins on N = 998 and reconstructed to
        # 1.4e-4 in silence.
        if aliasing(H_m, Nm) > ALIAS_TOL:
            _supports[m] = nonzero_support(H_m)
            _violations.append(m)

    if _violations:
        import warnings as _warnings

        _warnings.warn(
            f"filterbank{type_}: {len(_violations)} of {M} channels exceed the "
            f"painless limit L/a (e.g. channel {_violations[0]}: "
            f"{_supports[_violations[0]]} non-zero bins against "
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

            gout.append(
                {
                    "H": H_out_vals,
                    "foff": foff_m,
                    "realonly": 0,
                    "delay": 0,
                    "fs": g[m].get("fs"),
                }
            )
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
        sm = float(s_arr[m])
        if "H" in gm:
            if callable(gm["H"]):
                H_old = gm["H"]
                gm_new["H"] = lambda L, _H=H_old, _s=sm: np.asarray(_H(L)) * _s
            else:
                gm_new["H"] = np.asarray(gm["H"]) * sm
        elif "h" in gm:
            gm_new["h"] = np.asarray(gm["h"]) * sm
        gs.append(gm_new)
    return gs
