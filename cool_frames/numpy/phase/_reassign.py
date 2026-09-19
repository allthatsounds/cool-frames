"""
numpy/layer3/_reassign.py
=========================
Spectral reassignment and synchrosqueezing for non-uniform filterbanks.

MATLAB originals
----------------
  layer3/reassignment/comp_filterbankreassign.m
  (synchrosqueezing is a frequency-only reassignment variant)
"""

from __future__ import annotations

import numpy as np

from ..filterbanks._utils import normalise_a
from ..filters._design import filterbanklength
from ._phasegrad import filterbankphasegrad

# ---------------------------------------------------------------------------
# Centre frequencies for the reassignment grid
# ---------------------------------------------------------------------------


def _centre_frequencies(g, a_norm, L: int) -> np.ndarray:
    """Normalised centre frequency of every channel, in [0, 2) with 2 = fs.

    The circular mean of the DFT frequency weighted by ``|H_m|``, as LTFAT's
    ``cent_freqs`` computes it.  The arithmetic mean over [0, 2) that this
    replaces put any channel whose support wraps around 0 -- the DC
    complement, with bins near 0 *and* near L -- at about 1, i.e. at
    Nyquist, so its energy was reassigned across the whole band.

    The kernel expects the channels in ascending order with the wrap-around
    after the last one.  A DC complement whose centroid lands a hair below 0
    (just under 2 once wrapped) would put that wrap between channels 0 and 1
    and send its energy to the Nyquist channel, so such a channel is set to 0.

    Not ``_centerfreq.filter_center_frequencies``: that folds every centre
    into [0, fs/2], which suits phase retrieval on real banks, while a
    complex bank's channels here cover the whole of [0, 2).
    """
    from ..filterbanks._frame import filterbankfreqz

    H = np.asarray(filterbankfreqz(g, a_norm, L))
    k = np.arange(L) / L * 2.0
    z = np.exp(1j * np.pi * k) @ np.abs(H)
    fc = np.asarray(np.mod(np.angle(z) / np.pi, 2.0), dtype=float)
    if fc.size > 1 and fc[0] > fc[1] and 2.0 - fc[0] < fc[1]:
        fc[0] = 0.0
    return fc


def _length_from_coefficients(s, a_norm) -> int:
    """Transform length implied by the subband lengths and hops (L = N_m a_m)."""
    afrac = a_norm[:, 0] / a_norm[:, 1]
    lengths = np.array([np.asarray(sm).size for sm in s]) * afrac
    L = int(round(float(lengths[0])))
    if not np.allclose(lengths, L, rtol=0, atol=0.5):
        raise ValueError(
            "filterbankreassign: the subband lengths and hops do not agree on one "
            f"transform length (N_m * a_m ranges over {lengths.min():g}..{lengths.max():g}); "
            "pass the normalised centre frequencies instead of the filters"
        )
    return L


# ---------------------------------------------------------------------------
# comp_filterbankreassign – core reassignment kernel
# ---------------------------------------------------------------------------


#: Tests set this to run the reference loop instead of the vectorised search.
_FORCE_LOOP = False


def _reassign_targets(tgrad, fgrad, afrac, cfreq2, Lc):
    """Where the kernel sends every coefficient, computed for all at once.

    Returns ``(src, dst)`` -- flat source and destination positions (channels
    concatenated in order) -- listing the coefficients in the order the loop
    kernel visits them (channels ``M-1 .. 0``, time ascending), or ``None``
    when the centre frequencies are not in ascending order, where the loop's
    walk is not a nearest-neighbour search and only the loop reproduces it
    (equal neighbours are fine: a ``waveletfilters`` bank ending at Nyquist
    has two channels centred there).

    Each branch below is the closed form of one branch of the loop in
    :func:`comp_filterbankreassign` (kept there as the fall-back, and as the
    reference the tests compare against): the loop walks from the source
    channel towards ``cfreq + dev`` and stops at the first centre past it,
    then keeps whichever of the last two centres is closer; on an ascending
    grid that is a binary search plus one comparison, with the loop's
    wrap-around (``+-2``), its tie-breaking and its fall-backs kept.
    Identical to the loop, which is O(M) per coefficient in Python; on
    T8's Gabor bank (513 channels) that took 4 s.
    """
    M = len(Lc)
    c = np.asarray(cfreq2, dtype=float)
    if M < 2 or not np.all(np.diff(c) >= 0):
        return None
    order = range(M - 1, -1, -1)
    Lc_arr = np.asarray(Lc, dtype=np.int64)
    chan_pos = np.concatenate([[0], np.cumsum(Lc_arr)])
    tg_l = [np.asarray(tgrad[m], dtype=float).ravel() for m in range(M)]
    if any(tg_l[m].size < Lc[m] for m in range(M)):
        return None  # the loop raises IndexError here; let it

    def fitted(v, n):
        # The loop reads the first ``n`` values; a missing group delay falls
        # into its ``except IndexError`` and gives time index 0 (NaN here).
        v = np.asarray(v, dtype=float).ravel()[:n]
        return v if v.size == n else np.concatenate([v, np.full(n - v.size, np.nan)])

    mm = np.concatenate([np.full(Lc[m], m, dtype=np.int64) for m in order])
    jj = np.concatenate([np.arange(Lc[m], dtype=np.int64) for m in order])
    tg = np.concatenate([tg_l[m][: Lc[m]] for m in order])
    fg = np.concatenate([fitted(fgrad[m], Lc[m]) for m in order])

    cm = c[mm]
    dev = np.mod(tg - cm + 1.0, 2.0) - 1.0
    t = cm + dev
    idx = np.empty(mm.size, dtype=np.int64)

    # dev > 0: walk up from mm to the first centre >= t, wrapping with t - 2.
    up = dev > 0
    tu, mu = t[up], mm[up]
    r = np.empty(tu.size, dtype=np.int64)
    p = np.searchsorted(c, tu, "left")
    ok = p < M
    q, tq = p[ok], tu[ok]
    r[ok] = np.where(np.abs(c[q] - tq) < np.abs(c[q - 1] - tq), q, q - 1)
    w = ~ok
    tw, mw = tu[w], mu[w]
    p2 = np.searchsorted(c, tw - 2.0, "left")
    rw = np.empty(tw.size, dtype=np.int64)
    z = p2 == 0
    rw[z] = np.where(np.abs(c[0] - tw[z] + 2.0) < np.abs(c[M - 1] - tw[z]), 0, M - 1)
    mid = ~z & (p2 <= mw)
    q = p2[mid]
    rw[mid] = np.where(np.abs(c[q] - tw[mid] + 2.0) < np.abs(c[q - 1] - tw[mid] + 2.0), q, q - 1)
    rw[~z & (p2 > mw)] = mw[~z & (p2 > mw)]
    r[w] = rw
    idx[up] = r

    # dev <= 0 (and NaN): walk down from mm to the first centre <= t,
    # wrapping with t + 2.
    dn = ~up
    td, md = t[dn], mm[dn]
    r = np.empty(td.size, dtype=np.int64)
    p = np.minimum(np.searchsorted(c, td, "right") - 1, md)
    ok = p >= 0
    q, tq, start = p[ok], td[ok], p[ok] == md[ok]
    nxt = np.minimum(q + 1, M - 1)
    r[ok] = np.where(start, q, np.where(np.abs(c[q] - tq) < np.abs(c[nxt] - tq), q, q + 1))
    w = ~ok
    tw, mw = td[w], md[w]
    p2 = np.searchsorted(c, tw + 2.0, "right") - 1
    rw = np.empty(tw.size, dtype=np.int64)
    top = p2 == M - 1
    rw[top] = np.where(np.abs(c[M - 1] - tw[top] - 2.0) < np.abs(c[0] - tw[top]), M - 1, 0)
    mid = ~top & (p2 >= mw)
    q = p2[mid]
    rw[mid] = np.where(np.abs(c[q] - tw[mid] - 2.0) < np.abs(c[q + 1] - tw[mid] - 2.0), q, q + 1)
    rw[~top & (p2 < mw)] = mw[~top & (p2 < mw)]
    r[w] = rw
    idx[dn] = r
    idx = np.clip(idx, 0, M - 1)

    # Time index: round((fgrad + a_m n) / a_target) mod N_target (a
    # non-finite group delay goes to 0, as the loop's except clause does).
    x = (fg + afrac[mm] * jj) / afrac[idx]
    Lt = Lc_arr[idx]
    ft = np.zeros(x.size, dtype=np.int64)
    fin = np.isfinite(x)
    ft[fin] = np.mod(np.round(x[fin]), Lt[fin]).astype(np.int64)
    ft = np.minimum(ft, Lt - 1)

    return chan_pos[mm] + jj, chan_pos[idx] + ft


def comp_filterbankreassign(
    s: list[np.ndarray],
    tgrad: list[np.ndarray],
    fgrad: list[np.ndarray],
    a,
    cfreq: np.ndarray,
    return_repos: bool = False,
):
    """Port of ``comp_filterbankreassign.m``.

    Each coefficient in subband m, time index n is accumulated into the
    subband whose normalised centre frequency is closest to its
    instantaneous frequency ``tgrad[m][n]`` (wrap around 2), and placed at
    the time index closest to ``fgrad[m][n] + a[m]*n`` (mod N[target]).

    ``tgrad`` is the *absolute* normalised instantaneous frequency, as
    ``filterbankphasegrad`` and ``fbphasegradfrommag`` return it (2 = fs).
    LTFAT's kernel takes the deviation from the channel's centre frequency
    instead and adds ``cfreq[m]`` itself; this port used to do the same, on
    top of an already absolute ``tgrad``, so every coefficient was reassigned
    to about twice its frequency (on ``audfilters(16000, 8000)`` a 440 Hz
    tone landed in the 926 Hz channel).  The deviation is now formed here,
    wrapped to [-1, 1).

    Parameters
    ----------
    s       : list of M energy arrays (``|c[m]|^2`` or similar)
    tgrad   : list of M absolute instantaneous-frequency arrays (normalised)
    fgrad   : list of M group-delay arrays
    a       : hop sizes (M,) or (M,2)
    cfreq   : (M,) normalised centre frequencies in [0, 2)
    return_repos : if True also return a ``repos`` list

    Returns
    -------
    sr      : list of M reassigned arrays
    repos   : list of lists of source indices (only if return_repos=True)
    Lc      : list of M subband lengths
    """
    M = len(s)
    a_norm = normalise_a(a, M)
    afrac = a_norm[:, 0] / a_norm[:, 1]
    Lc = [len(np.asarray(s[m]).ravel()) for m in range(M)]

    # Wrap cfreq to [0, 2)
    cfreq2 = np.mod(cfreq, 2.0)

    moves = None if _FORCE_LOOP else _reassign_targets(tgrad, fgrad, afrac, cfreq2, Lc)
    if moves is not None:
        src, dst = moves
        total = int(sum(Lc))
        # The weights in the loop's order, so the sums are the loop's sums.
        w = np.concatenate([np.asarray(s[m]).ravel() for m in range(M - 1, -1, -1)])
        flat = np.bincount(dst, weights=w, minlength=total)
        pos = np.concatenate([[0], np.cumsum(Lc)])
        sr_v = [flat[pos[m] : pos[m + 1]] for m in range(M)]
        if return_repos:
            return sr_v, src[np.argsort(dst, kind="stable")], Lc
        return sr_v, Lc

    sr = [np.zeros(Lc[m]) for m in range(M)]

    repos: list[list[int]] | None
    if return_repos:
        chan_pos = np.zeros(M + 1, dtype=int)
        for m in range(M):
            chan_pos[m + 1] = chan_pos[m] + Lc[m]
        repos = [[] for _ in range(int(chan_pos[-1]))]
    else:
        repos = None

    for mm in range(M - 1, -1, -1):
        sm_arr = np.asarray(s[mm]).ravel()
        tg_arr = np.asarray(tgrad[mm]).ravel()
        fg_arr = np.asarray(fgrad[mm]).ravel()
        cfreqm = cfreq2[mm]
        am = afrac[mm]

        for jj in range(Lc[mm]):
            # Deviation of the (absolute) instantaneous frequency from this
            # channel's centre, wrapped to [-1, 1): the quantity LTFAT's
            # kernel expects in ``tgrad``.
            dev = float(np.mod(tg_arr[jj] - cfreqm + 1.0, 2.0) - 1.0)
            tgradmjj = cfreqm + dev
            oldtgrad = 10.0

            if dev > 0:
                pos = mm
                for ii in range(mm, M):
                    pos = ii
                    tmptgrad = cfreq2[ii] - tgradmjj
                    if tmptgrad >= 0:
                        tgradIdx = pos if abs(tmptgrad) < abs(oldtgrad) else pos - 1
                        break
                    oldtgrad = tmptgrad
                else:
                    # Wrapped around
                    for ii in range(0, mm + 1):
                        pos = ii
                        tmptgrad = cfreq2[ii] - tgradmjj + 2.0
                        if tmptgrad >= 0:
                            tgradIdx = pos if abs(tmptgrad) < abs(oldtgrad) else pos - 1
                            break
                        oldtgrad = tmptgrad
                    else:
                        tgradIdx = mm

                if tgradIdx < 0:
                    tgradIdx = M - 1
            else:
                pos = mm
                for ii in range(mm, -1, -1):
                    pos = ii
                    tmptgrad = cfreq2[ii] - tgradmjj
                    if tmptgrad <= 0:
                        tgradIdx = pos if abs(tmptgrad) < abs(oldtgrad) else pos + 1
                        break
                    oldtgrad = tmptgrad
                else:
                    for ii in range(M - 1, mm - 1, -1):
                        pos = ii
                        tmptgrad = cfreq2[ii] - tgradmjj - 2.0
                        if tmptgrad <= 0:
                            tgradIdx = pos if abs(tmptgrad) < abs(oldtgrad) else pos + 1
                            break
                        oldtgrad = tmptgrad
                    else:
                        tgradIdx = mm

                if tgradIdx >= M:
                    tgradIdx = 0

            # Clamp
            tgradIdx = max(0, min(M - 1, tgradIdx))
            at_idx = afrac[tgradIdx]
            Lt_idx = Lc[tgradIdx]

            # Compute fgradIdx with bounds checking
            try:
                fgradIdx_raw = int(np.mod(round((fg_arr[jj] + am * jj) / at_idx), Lt_idx))
            except (IndexError, ValueError):
                fgradIdx_raw = 0

            # Ensure fgradIdx is within bounds [0, Lt_idx)
            # Note: np.mod should never return Lt_idx, but floating point rounding can cause issues
            if fgradIdx_raw >= Lt_idx or fgradIdx_raw < 0:
                fgradIdx_raw = int(fgradIdx_raw % Lt_idx)
            fgradIdx = max(0, min(Lt_idx - 1, fgradIdx_raw))

            sr[tgradIdx][fgradIdx] += sm_arr[jj]

            if return_repos:
                assert repos is not None
                src_flat = int(chan_pos[mm]) + jj
                dst_flat = int(chan_pos[tgradIdx]) + fgradIdx
                repos[dst_flat].append(src_flat)

    if return_repos:
        # Flatten repos into a single concatenated array
        assert repos is not None
        repos_flat = np.concatenate(
            [np.asarray(r, dtype=int) if len(r) > 0 else np.array([], dtype=int) for r in repos]
        )
        return sr, repos_flat, Lc
    return sr, Lc


# ---------------------------------------------------------------------------
# filterbankreassign – public API
# ---------------------------------------------------------------------------


def filterbankreassign(
    f, g, a=None, L: int | None = None, fc: np.ndarray | None = None, return_repos: bool = False
):
    """Spectral reassignment of filterbank coefficients.

    Can be called in two ways:
    1. filterbankreassign(signal, g_filters, a_hops, L, fc, return_repos)
    2. filterbankreassign(magnitudes, tgrad, fgrad, a_hops, fc) – always returns (sr, repos, Lc)

    Parameters
    ----------
    f          : signal (Ls,) or list of magnitude arrays
    g          : list of M filter dicts, OR tgrad list (if f is magnitudes)
    a          : hop sizes (if g is filters) or fgrad (if g is tgrad)
    L          : DFT length or a_hops (if f is magnitudes with pre-computed gradients)
    fc         : (M,) normalised centre frequencies or filter cell
    return_repos : whether to return repositioning table (signal mode only)

    Returns
    -------
    When called with pre-computed magnitudes:
        sr    : list of M reassigned energy arrays
        repos : repositioning list
        Lc    : list of M subband lengths

    When called with signal:
        sr    : list of M reassigned energy arrays
        repos : repositioning list (only if return_repos=True)
        Lc    : list of M subband lengths (only if return_repos=True)
    """
    # Detect if f is a list (pre-computed magnitudes) vs signal array
    f_is_list = isinstance(f, (list, tuple))

    if f_is_list:
        # Pre-computed case: f=magnitudes, g=tgrad, a=fgrad, L=a_hops, fc=fc_or_g
        s = [np.asarray(fi) for fi in f]
        tgrad = [np.asarray(gi) for gi in g]
        fgrad = [np.asarray(ai) for ai in a]
        a_hops = L
        M = len(s)

        # fc can be frequencies or filter cell
        if isinstance(fc, (list, tuple)) and len(fc) > 0 and isinstance(fc[0], dict):
            # fc is the filter cell: take the centre frequencies from the
            # filters, at the transform length the subbands imply.  This used
            # to fall back to an even grid over [0, 2), which put the channels
            # of any non-uniform bank at the wrong frequencies.
            a_norm_pc = normalise_a(a_hops, M)
            fc_arr = _centre_frequencies(
                list(fc), a_norm_pc, _length_from_coefficients(s, a_norm_pc)
            )
        elif fc is None:
            # No fc provided; default to evenly spaced
            fc_arr = np.arange(M, dtype=float) / M * 2.0
        else:
            fc_arr = np.asarray(fc, dtype=float)

        # Pre-computed path ALWAYS returns (sr, repos, Lc)
        return_repos = True
    else:
        # Signal case: compute gradients
        f = np.asarray(f)
        M = len(g)
        if a is None:
            raise ValueError("filterbankreassign: a (hop sizes) is required when f is a signal")

        a_norm = normalise_a(a, M)

        if L is None:
            L = filterbanklength(len(f), a_norm)

        L_int: int = int(L)  # type: ignore[assignment]
        tgrad, fgrad, s, _c = filterbankphasegrad(f, g, a_norm, L_int)
        a_hops = a_norm  # type: ignore[assignment]

        # Compute normalised centre frequencies if not provided
        if fc is None:
            fc_arr = _centre_frequencies(g, a_norm, L_int)
        elif isinstance(fc, (list, tuple)) and len(fc) > 0 and isinstance(fc[0], dict):
            # fc is actually a filter cell g; compute from filters
            fc_arr = _centre_frequencies(list(fc), a_norm, L_int)
        else:
            fc_arr = np.asarray(fc)

    result = comp_filterbankreassign(s, tgrad, fgrad, a_hops, fc_arr, return_repos=return_repos)
    if return_repos:
        return result  # (sr, repos, Lc)
    else:
        return result[0]  # Just sr


# ---------------------------------------------------------------------------
# filterbanksynchrosqueeze – DEPRECATED, use filterbankreassign instead
# ---------------------------------------------------------------------------


def filterbanksynchrosqueeze(
    f, g, a=None, L: int | None = None, fc: np.ndarray | None = None, return_repos: bool = False
):
    """Synchrosqueezing (frequency-only reassignment).

    .. deprecated::
        ``filterbanksynchrosqueeze`` is deprecated and will be removed in a
        future release.  It is equivalent to :func:`filterbankreassign` with
        the group delay (``fgrad``) zeroed out, so coefficients move in
        frequency only.  Both functions accumulate energy via
        summation when multiple coefficients map to the same bin, so neither
        is truly invertible for signals with overlapping components.  Use
        :func:`filterbankreassign` directly for all reassignment tasks.

    Can be called in two ways:
    1. filterbanksynchrosqueeze(signal, g_filters, a_hops, L, fc, return_repos)
    2. filterbanksynchrosqueeze(coefficients, tgrad, fgrad, a_hops, fc, return_repos)

    Parameters
    ----------
    Same as :func:`filterbankreassign`.

    Returns
    -------
    sr    : list of M reassigned energy arrays
    repos : repositioning list (only if return_repos=True)
    Lc    : list of M subband lengths (only if return_repos=True)
    """
    import warnings

    warnings.warn(
        "filterbanksynchrosqueeze is deprecated and will be removed in a "
        "future release. Use filterbankreassign instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Detect if f is a list (pre-computed data) vs signal array
    f_is_list = isinstance(f, (list, tuple))

    if f_is_list:
        # Pre-computed case: f=coefficients, g=tgrad, a=fgrad, L=a_hops, fc=fc_or_g
        c = [np.asarray(fi) for fi in f]
        tgrad = [np.asarray(gi) for gi in g]
        fgrad = [np.asarray(ai) for ai in a]
        a_hops = L
        M = len(c)

        # fc can be frequencies or filter cell
        if isinstance(fc, (list, tuple)) and len(fc) > 0 and isinstance(fc[0], dict):
            # fc is the filter cell: centre frequencies from the filters (see
            # filterbankreassign)
            a_norm_pc = normalise_a(a_hops, M)
            fc_arr = _centre_frequencies(
                list(fc), a_norm_pc, _length_from_coefficients(c, a_norm_pc)
            )
        elif fc is None:
            # No fc provided; default to evenly spaced
            fc_arr = np.arange(M, dtype=float) / M * 2.0
        else:
            fc_arr = np.asarray(fc, dtype=float)

        # Compute energy
        s = [np.abs(np.asarray(ci)) ** 2 for ci in c]
        # Pre-computed path ALWAYS returns (sr, repos, Lc)
        return_repos = True
    else:
        # Signal case: compute gradients
        f = np.asarray(f)
        M = len(g)
        if a is None:
            raise ValueError(
                "filterbanksynchrosqueeze: a (hop sizes) is required when f is a signal"
            )

        a_norm = normalise_a(a, M)

        if L is None:
            L = filterbanklength(len(f), a_norm)

        L_int: int = int(L)  # type: ignore[assignment]
        tgrad, fgrad, s, c = filterbankphasegrad(f, g, a_norm, L_int)
        a_hops = a_norm  # type: ignore[assignment]

        if fc is None:
            fc_arr = _centre_frequencies(g, a_norm, L_int)
        elif isinstance(fc, (list, tuple)) and len(fc) > 0 and isinstance(fc[0], dict):
            fc_arr = _centre_frequencies(list(fc), a_norm, L_int)
        else:
            fc_arr = np.asarray(fc)

    # Synchrosqueezing is reassignment in frequency only: every coefficient
    # keeps its time position, so the group delay (``fgrad``, the time shift)
    # is zeroed and the instantaneous frequency (``tgrad``) is kept.  Until
    # this was corrected the *instantaneous frequency* was zeroed instead,
    # which -- with the kernel adding the centre frequency back -- kept every
    # coefficient in its own channel and moved it in time only.
    fgrad_zero = [np.zeros_like(fg) for fg in fgrad]

    result = comp_filterbankreassign(
        s, tgrad, fgrad_zero, a_hops, fc_arr, return_repos=return_repos
    )
    if return_repos:
        return result  # (sr, repos, Lc)
    else:
        return result[0]  # Just sr
