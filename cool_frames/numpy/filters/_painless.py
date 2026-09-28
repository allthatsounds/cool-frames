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

import math

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


def repair_painless_hops(g: list[dict], a, L: int, order=None) -> int:
    r"""Lower any channel's hop until it meets its own painless limit.

    The designers choose each hop from the bandwidth they *intend* a channel
    to have; the filter actually built can be wider.  The DC and Nyquist
    complements are sized by a different rule than the one their hop came
    from, a prototype is ``round(L * fsupp / fs)`` bins made odd, and a
    ``min_win`` floor can add bins the hop never saw.  Measured on the banks
    returned with default parameters, 1197 of the 11,112 ``audfilters``,
    ``greenwoodfilters`` and ``cqtfilters`` configurations of the W03
    admissibility sweep had a channel over its limit.  720 were
    ``audfilters(scale='greenwood')`` with the wrong defaults (G9); in the
    477 ``greenwoodfilters`` and ``cqtfilters`` banks it was most often a
    complement one bin too wide (``cqtfilters(8000, 512, fmin=20, bins=1,
    Qvar=0.5)``: 5 bins on ``N = 4``), and they reconstructed with the
    canonical dual to a median of 3.8e-3 (up to 0.18) instead of 1e-16,
    silently.  This is the
    repair, applied to the bank that is actually returned; it was
    ``waveletfilters``' own until 2026-09-28.

    Lowering a hop never invalidates the painless inequality for any other
    channel, so this is a safe local repair.  Two constraints shape it:

    * ``L`` must stay a whole number of hops, so an integer hop is lowered to
      the largest **divisor of L** that is within the limit rather than to the
      limit itself.  (Rational ``[L, N]`` hops are divisor-free: raising ``N``
      is enough.)
    * ``g[m]["H"]`` was scaled by ``sqrt(a_m)``, the package's per-channel
      energy convention, so changing the hop without rescaling would leave the
      channel with the gain of a hop it no longer has.  The response is
      rescaled by ``sqrt(a_new / a_old)``, which leaves the channel's share of
      the frame response, ``|H|^2 / a``, and so the bounds and the dual, as
      the designer intended.

    ``order`` is the order in which channels are visited (default: the inner
    channels, then the DC and the Nyquist complement, whose closures may read
    the inner filters and hops -- ``a`` is updated in place).  Returns the
    number of channels repaired.
    """
    a_arr = np.asarray(a)
    M = len(g)
    if order is None:
        order = list(range(1, M - 1)) + [0, M - 1] if M > 2 else list(range(M))
    fixed = 0
    for m in order:
        gm = g[m]
        if gm is None:
            continue
        H = gm.get("H")
        if H is None:
            continue
        Hm = np.asarray(H(L) if callable(H) else H).ravel()
        if not nonzero_support(Hm):
            continue
        if a_arr.ndim == 2:
            a_old = float(a_arr[m, 0]) / float(a_arr[m, 1])
            if aliasing(Hm, L / a_old) <= ALIAS_TOL:
                continue
            N_new = painless_length(Hm, int(math.ceil(L / a_old)) + 1)
            a_arr[m, 0] = int(L)
            a_arr[m, 1] = N_new
            a_new_m = float(L) / float(N_new)
        else:
            a_old = float(a_arr[m])
            if aliasing(Hm, L / a_old) <= ALIAS_TOL:
                continue
            d = int(a_old) - 1
            while d > 1 and (L % d or aliasing(Hm, L // d) > ALIAS_TOL):
                d -= 1
            if d < 1:
                continue
            a_arr[m] = d
            a_new_m = float(d)
        s = math.sqrt(a_new_m / a_old)
        if callable(H):
            # Keep it lazy: some channels build their response from L, and
            # freezing it here would pin the filter to this one length.
            gm["H"] = (lambda fn, sc: lambda Lq: np.asarray(fn(Lq)) * sc)(H, s)
        else:
            gm["H"] = Hm * s
        fixed += 1
    return fixed


def repair_uniform_hop(g: list[dict], a, L: int) -> int:
    """:func:`repair_painless_hops` for a bank whose point is one hop.

    Lowers the common hop, for every channel, to the largest divisor of ``L``
    at which every channel is painless, and rescales every response by
    ``sqrt(a_new / a_old)``.  Returns the new hop's reduction (0 if none).
    """
    a_arr = np.asarray(a)
    if a_arr.ndim != 1 or a_arr.size == 0:
        return 0
    a_old = int(a_arr[0])
    resp = []
    for gm in g:
        H = None if gm is None else gm.get("H")
        resp.append(None if H is None else np.asarray(H(L) if callable(H) else H).ravel())

    def ok(d):
        return all(h is None or not nonzero_support(h) or aliasing(h, L // d) <= ALIAS_TOL
                   for h in resp)

    if ok(a_old):
        return 0
    d = a_old - 1
    while d > 1 and (L % d or not ok(d)):
        d -= 1
    s = math.sqrt(d / a_old)
    for gm in g:
        if gm is None or gm.get("H") is None:
            continue
        H = gm["H"]
        gm["H"] = ((lambda fn, sc: lambda Lq: np.asarray(fn(Lq)) * sc)(H, s)
                   if callable(H) else np.asarray(H) * s)
    a_arr[:] = d
    return a_old - d
