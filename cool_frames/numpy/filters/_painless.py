"""Painless-condition helpers shared by the filter designers.

A band-limited channel is *painless* when its frequency support fits into
the channel's decimated length ``N = L / a``: then no two support bins alias
onto the same coefficient, the frame operator is diagonal, and the canonical
dual is the filter divided by that diagonal.  One bin too many breaks it --
the dual is then only approximate.
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

    Here each channel whose non-zero support exceeds ``N_m`` gets
    ``N_m = support``, and its filter is scaled by ``sqrt(N_old / N_new)`` so
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
        support = nonzero_support(_evaluate(gm["H"], L))
        n_old = int(a[m, 1])
        if support > n_old:
            gm["H"] = _scaled(gm["H"], float(np.sqrt(n_old / support)))
            a[m, 1] = support
