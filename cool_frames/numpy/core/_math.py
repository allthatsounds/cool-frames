"""
numpy.core._math
=================
Signal-processing math utilities ported from LTFAT.

Previously in the separate ``ltfat_core`` package; vendored here to
remove the external dependency.

Functions:
    involute     — involution  f[l] → conj(f[−l mod L])
    modcent      — centred modulo
    fftindex     — DFT frequency bin indices
    floor23      — largest integer ≤ n with only factors {2, 3}
    postpad      — zero-pad or truncate along an axis
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# involute
# ---------------------------------------------------------------------------


def involute(f: np.ndarray) -> np.ndarray:
    """Return the involution of *f* along its first axis.

    ``finv[l] = conj(f[-l mod L])``
    """
    f = np.asarray(f)
    out = np.empty_like(f)
    out[0] = f[0].conj()
    out[1:] = f[1:][::-1].conj()
    return out


# ---------------------------------------------------------------------------
# modcent
# ---------------------------------------------------------------------------


def modcent(x: np.ndarray | float, m: float) -> np.ndarray | float:
    """Centred modulo: result in (−m/2, m/2].

    Equivalent to MATLAB ``modcent(x, m)``.
    """
    return np.mod(x + m / 2, m) - m / 2


# ---------------------------------------------------------------------------
# fftindex
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# floor23
# ---------------------------------------------------------------------------


def floor23(n: np.ndarray | Sequence | int | float) -> np.ndarray | int:
    """Return the largest integer ≤ *n* with only prime factors 2 and 3.

    Vectorised: accepts scalars, lists, or ndarrays.
    """
    scalar = np.ndim(n) == 0
    n_arr = np.atleast_1d(np.asarray(n, dtype=float))
    out = np.zeros_like(n_arr, dtype=int)

    maxval = 2**20
    table = sorted(
        2**i * 3**j
        for i in range(21)
        for j in range(int(math.log(maxval, 3)) + 1)
        if 2**i * 3**j <= maxval
    )
    table = np.array(table)  # type: ignore[assignment]

    for ii, val in enumerate(n_arr.flat):
        n2reduce = 0
        v = int(val)
        if v <= 0:
            out.flat[ii] = 0
            continue
        if v > maxval:
            n2reduce = math.ceil(math.log2(v / maxval))
            v = v >> n2reduce
        idx = np.searchsorted(table, v, side="right") - 1
        result = int(table[idx]) << n2reduce
        out.flat[ii] = result

    return int(out.flat[0]) if scalar else out.reshape(n_arr.shape)


# ---------------------------------------------------------------------------
# postpad
# ---------------------------------------------------------------------------


def postpad(x: np.ndarray, n: int, axis: int = 0) -> np.ndarray:
    """Zero-pad (or truncate) *x* to length *n* along *axis*."""
    cur = x.shape[axis]
    if cur == n:
        return x
    if cur > n:
        slc = [slice(None)] * x.ndim
        slc[axis] = slice(0, n)
        return x[tuple(slc)]
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, n - cur)
    return np.pad(x, pad_width)
