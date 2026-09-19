"""Painless-condition helpers shared by the filter designers.

A band-limited channel is *painless* when its frequency support fits into
the channel's decimated length ``N = L / a``: then no two support bins alias
onto the same coefficient, the frame operator is diagonal, and the canonical
dual is the filter divided by that diagonal.  One bin too many breaks it --
the dual is then only approximate.

What decides it is not the support as such but whether two bins ``N`` apart
are *both* non-negligible: the frame operator's off-diagonal entries are
sums of ``H[k] conj(H[k + jN])``.  :func:`aliasing` measures exactly that.  A
filter whose stored response runs past ``N`` on tails of 1e-11 of its peak
(the wavelets of a two-sided ``waveletfilters`` bank do) is painless to
machine precision, while a Nyquist complement one bin too wide, both of
whose end bins are alive, is not (``cqtfilters(sampling='fractional')``
reconstructed to 1.4e-4).  Counting non-zero bins cannot tell the two apart.
"""

from __future__ import annotations

import numpy as np


def nonzero_support(H) -> int:
    """Number of bins from the first to the last non-zero value of ``H``.

    Exact zeros at either end (Hann windows end on zeros, for instance) do
    not count: they alias onto nothing.
    """
    h = np.abs(np.asarray(H))
    nz = np.flatnonzero(h > 0)
    return 0 if nz.size == 0 else int(nz[-1] - nz[0] + 1)


#: Largest ``|H[k] H[k + jN]| / max|H|^2`` still counted as painless: the
#: frame operator is then diagonal to about this relative accuracy.
ALIAS_TOL = 1e-15


def aliasing(H, N) -> float:
    """How far a channel is from painless at decimated length ``N``.

    The largest ``|H[k]| |H[k + jN]|`` over ``j >= 1``, relative to
    ``max|H|^2``, for the stored response ``H`` (contiguous bins).  0 when
    no two bins ``N`` apart are both non-zero; the channel is painless when
    this is at most :data:`ALIAS_TOL`.
    """
    h = np.abs(np.asarray(H)).ravel()
    n = int(round(float(N)))
    if n <= 0 or h.size <= n:
        return 0.0
    peak = float(h.max())
    if peak == 0.0:
        return 0.0
    worst = 0.0
    for shift in range(n, h.size, n):
        worst = max(worst, float(np.max(h[:-shift] * h[shift:])))
    return worst / (peak * peak)


def painless_length(H, n_min: int = 1) -> int:
    """Smallest decimated length ``N >= n_min`` at which ``H`` is painless.

    Searched upward from the support of the bins above ``sqrt(ALIAS_TOL)`` of
    the peak (when those bins are contiguous, as every designer's are, two
    of them lie ``N`` apart for any shorter ``N``); the non-zero support is
    always painless, so the search ends there at the latest.
    """
    h = np.abs(np.asarray(H)).ravel()
    full = nonzero_support(h)
    if full == 0:
        return max(int(n_min), 1)
    peak = float(h.max())
    nz = np.flatnonzero(h > np.sqrt(ALIAS_TOL) * peak)
    start = max(int(n_min), int(nz[-1] - nz[0] + 1) if nz.size else 1)
    for n in range(start, full):
        if aliasing(h, n) <= ALIAS_TOL:
            return n
    return max(full, int(n_min))


def _evaluate(H, L: int) -> np.ndarray:
    return np.asarray(H(L) if callable(H) else H)


def _scaled(H, k: float):
    if callable(H):
        return lambda L_, _H=H, _k=k: np.asarray(_H(L_)) * _k
    return np.asarray(H) * k


def fit_fractional_lengths(g: list[dict], a: np.ndarray, L: int) -> None:
    """Make every channel of a fractionally sampled bank painless, in place.

    Fractional sampling gives channel ``m`` the rational hop ``a[m] = L / N_m``
    with ``N_m = ceil(L / aprecise_m)``, computed from the bandwidth the
    designer *intended*.  The filter actually built can be a bin wider -- the
    DC and Nyquist complements are sized by a different rule than the one
    their ``N`` came from -- and then the bank is silently not painless:
    ``cqtfilters(..., sampling='fractional')`` reconstructed to 1.4e-4 because
    its Nyquist channel had 999 non-zero bins on ``N = 998``.

    Here each channel that is not painless at ``N_m`` (see :func:`aliasing`)
    gets the smallest ``N_m`` at which it is, and its filter is scaled by
    ``sqrt(N_old / N_new)`` so
    that its share of the frame response, ``|H|^2 N / L``, is unchanged --
    the bank's frame operator, bounds and dual stay what the designer
    intended.  Inner channels are fitted before the DC and Nyquist
    complements, whose closures read the inner filters and hops (``a`` is
    updated in place, so their views of it follow); the frame response they
    complement is invariant under the rescaling.

    Parameters
    ----------
    g : list of filter dicts (``'H'`` callable or array), modified in place
    a : (M, 2) integer array ``[L, N_m]``, modified in place
    L : transform length
    """
    a = np.asarray(a)
    if a.ndim != 2:
        return
    M = len(g)
    order = list(range(1, M - 1)) + [0, M - 1] if M > 2 else list(range(M))
    for m in order:
        gm = g[m]
        if gm is None or "H" not in gm:
            continue
        Hm = _evaluate(gm["H"], L)
        n_old = int(a[m, 1])
        if aliasing(Hm, n_old) > ALIAS_TOL:
            n_new = painless_length(Hm, n_old + 1)
            gm["H"] = _scaled(gm["H"], float(np.sqrt(n_old / n_new)))
            a[m, 1] = n_new
