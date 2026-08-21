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
# comp_filterbankreassign – core reassignment kernel
# ---------------------------------------------------------------------------


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
    subband whose normalised centre frequency is closest to
    ``cfreq[m] + tgrad[m][n]`` (wrap around 2), and placed at the time
    index closest to ``fgrad[m][n] + a[m]*n`` (mod N[target]).

    Parameters
    ----------
    s       : list of M energy arrays (``|c[m]|^2`` or similar)
    tgrad   : list of M instantaneous-frequency arrays
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
            tgradmjj = tg_arr[jj] + cfreqm
            oldtgrad = 10.0

            if tg_arr[jj] > 0:
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
            # fc is actually a filter cell g; default to normalized [0, 1, 2, ..., M-1]
            # Without L we can't compute center frequencies from filter response
            fc_arr = np.arange(M, dtype=float) / M * 2.0
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
            from ..filterbanks._frame import filterbankfreqz

            H = filterbankfreqz(g, a_norm, L)
            # Centre = weighted mean frequency
            k = np.arange(L) / L * 2.0  # normalised [0, 2)
            fc_arr = np.array(
                [float(np.average(k, weights=np.abs(H[:, m]) ** 2 + 1e-30)) for m in range(M)]
            )
        elif isinstance(fc, (list, tuple)) and len(fc) > 0 and isinstance(fc[0], dict):
            # fc is actually a filter cell g; compute from filters
            from ..filterbanks._frame import filterbankfreqz

            H = filterbankfreqz(list(fc), a_norm, L)
            k = np.arange(L) / L * 2.0
            fc_arr = np.array(
                [float(np.average(k, weights=np.abs(H[:, m]) ** 2 + 1e-30)) for m in range(M)]
            )
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
        the time gradient zeroed out.  Both functions accumulate energy via
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
            # fc is actually a filter cell g; default to normalized [0, 1, 2, ..., M-1]
            fc_arr = np.arange(M, dtype=float) / M * 2.0
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
            from ..filterbanks._frame import filterbankfreqz

            H = filterbankfreqz(g, a_norm, L)
            k = np.arange(L) / L * 2.0
            fc_arr = np.array(
                [float(np.average(k, weights=np.abs(H[:, m]) ** 2 + 1e-30)) for m in range(M)]
            )
        else:
            fc_arr = np.asarray(fc)

    # Zero-out time gradient for synchrosqueeze
    tgrad_zero = [np.zeros_like(tg) for tg in tgrad]

    result = comp_filterbankreassign(
        s, tgrad_zero, fgrad, a_hops, fc_arr, return_repos=return_repos
    )
    if return_repos:
        return result  # (sr, repos, Lc)
    else:
        return result[0]  # Just sr
