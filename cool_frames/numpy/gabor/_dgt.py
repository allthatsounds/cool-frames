"""Discrete Gabor transform and Gabor frame windows, in LTFAT's conventions.

``dgt`` computes LTFAT's DGT on a rectangular lattice with its default
frequency-invariant phase convention,

    c[m, n] = sum_l f[l] * conj(g[l - a*n]) * exp(-2j*pi*m*l/M),

for ``m = 0..M-1`` and ``n = 0..N-1`` with ``N = L/a``.  ``idgt`` is its
adjoint, ``f = sum_{m,n} c[m, n] * g_{m,n}``.  ``gabdual``, ``gabtight``,
``gabframebounds`` and ``gabframediag`` refer to the frame operator
``S f = sum_{m,n} <f, g_{m,n}> g_{m,n}`` of that system, as LTFAT's do, so
``idgt(dgt(f, g, a, M), gabdual(g, a, M, L), a)`` returns ``f``.

A window shorter than the transform length is extended with ``middlepad``
(LTFAT's ``fir2long``).  Every window goes through the long-window
factorisation of Søndergaard (``_walnut.py``).

Not ported from LTFAT: the filter-bank algorithm for short windows
(``comp_dgt_fb``, a speed optimisation), non-rectangular lattices (``lt``),
the time-invariant phase convention, window specifications by name (such as
``'gauss'``; pass an array, e.g. from ``cool_frames.filters.pgauss``), and
complex windows in the frame-window functions.
"""

from __future__ import annotations

from math import lcm

import numpy as np

from ..core import middlepad, postpad
from ._factorised import (
    _gabdual_normalised,
    _gabframebounds,
    _gabframediag,
    _gabtight_normalised,
)
from ._walnut import _dgt_long, _idgt_long

__all__ = [
    "dgt",
    "dgtlength",
    "dgtreal",
    "gabdual",
    "gabframebounds",
    "gabframediag",
    "gabtight",
    "idgt",
    "idgtreal",
]

# A lower frame bound this small relative to the upper one means the system
# is not a frame at this length: the dual would be Inf or garbage.
_NOT_A_FRAME_RTOL = 1e-12


# ---------------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------------
def _check_lattice(a: int, M: int) -> tuple[int, int]:
    if int(a) != a or int(M) != M or a < 1 or M < 1:
        raise ValueError(f"a and M must be positive integers, got a={a!r}, M={M!r}")
    return int(a), int(M)


def _check_length(L: int, a: int, M: int) -> int:
    if int(L) != L or L < 1:
        raise ValueError(f"L must be a positive integer, got {L!r}")
    L = int(L)
    if L % a or L % M:
        raise ValueError(
            f"L={L} is not a valid transform length for a={a}, M={M}: it must be "
            f"a multiple of lcm(a, M) = {lcm(a, M)}; dgtlength({L}, {a}, {M}) = "
            f"{dgtlength(L, a, M)}"
        )
    return L


def _as_window(g: np.ndarray, L: int) -> np.ndarray:
    """LTFAT's ``fir2long``: zero-extend a window to length L around index 0."""
    g = np.asarray(g)
    if g.ndim != 1:
        raise ValueError(f"the window must be one-dimensional, got shape {g.shape}")
    if g.shape[0] > L:
        raise ValueError(f"the window (length {g.shape[0]}) is longer than L={L}")
    return middlepad(g, L) if g.shape[0] < L else g


def _real_window(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g)
    if np.iscomplexobj(g):
        if np.any(g.imag != 0):
            raise ValueError("complex windows are not supported here; pass a real window")
        g = g.real
    return np.asarray(g, dtype=np.float64)


def _frame_length(g: np.ndarray, a: int, M: int, L: int | None) -> tuple[np.ndarray, int, bool]:
    """Resolve the transform length for the frame-window functions.

    Returns the window extended to that length, the length, and whether the
    result may be cropped back to the window's own support afterwards (the
    painless case with no L given, where the dual and tight windows have
    the support of g).
    """
    gl = g.shape[0]
    if L is None:
        if gl <= M:
            Lw = dgtlength(gl, a, M)
            return _as_window(g, Lw), Lw, True
        if gl % a or gl % M:
            raise ValueError(
                f"the window (length {gl}) is longer than M={M}, so the transform "
                f"length is needed: pass L (a multiple of lcm(a, M) = {lcm(a, M)})"
            )
        return g, gl, False
    L = _check_length(L, a, M)
    return _as_window(g, L), L, False


def _check_fir_crop(g: np.ndarray, what: str) -> None:
    """Refuse to return a FIR-length result that cannot represent the answer.

    ``middlepad`` (LTFAT's ``fir2long``) splits the middle sample of an
    even-length window between times +gl/2 and -gl/2.  The dual or tight
    window weights those two halves differently, so cutting it back to
    ``gl`` samples is only exact when that sample is zero, as it is for the
    windows ``firwin`` designs.
    """
    gl = g.shape[0]
    if gl % 2 == 0 and g[gl // 2] != 0:
        raise ValueError(
            f"cannot return the {what} window at the window's own length {gl}: an "
            f"even-length window with a non-zero middle sample (index {gl // 2}) has "
            f"no exact FIR {what}. Pass L to get it at a transform length, or use a "
            f"window whose middle sample is zero (e.g. from cool_frames.filters.firwin)"
        )


def _require_frame(g: np.ndarray, a: int, M: int, L: int) -> None:
    A, B = _gabframebounds(g, a, M, L)
    if not B > 0 or A <= _NOT_A_FRAME_RTOL * B:
        raise ValueError(
            f"(g, a={a}, M={M}) is not a frame at L={L} (frame bounds A={A:.3g}, "
            f"B={B:.3g}); there is no canonical dual or tight window"
        )


# ---------------------------------------------------------------------------
# Transform length
# ---------------------------------------------------------------------------
def dgtlength(Ls: int, a: int, M: int) -> int:
    """Smallest valid DGT length that is at least ``Ls``.

    A rectangular-lattice DGT needs ``L`` to be a multiple of ``lcm(a, M)``;
    this returns the smallest such ``L >= Ls`` (LTFAT's ``dgtlength``).
    """
    a, M = _check_lattice(a, M)
    if int(Ls) != Ls or Ls < 0:
        raise ValueError(f"Ls must be a non-negative integer, got {Ls!r}")
    step = lcm(a, M)
    return max(1, -(-int(Ls) // step)) * step


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def dgt(
    f: np.ndarray,
    g: np.ndarray,
    a: int,
    M: int,
    L: int | None = None,
) -> np.ndarray:
    """Discrete Gabor transform (LTFAT ``dgt``, rectangular lattice).

    Parameters
    ----------
    f : ndarray, shape (Ls,) or (Ls, W)
        Signal, or W signals as columns.  Real or complex.
    g : ndarray, shape (gl,)
        Window, real or complex, with ``gl <= L``; a shorter window is
        extended with ``middlepad``.
    a : int
        Time shift (hop size).
    M : int
        Number of frequency channels.
    L : int, optional
        Transform length, a multiple of ``lcm(a, M)``.  The signal is
        zero-padded or truncated to it.  Default:
        ``dgtlength(max(Ls, gl), a, M)``.

    Returns
    -------
    c : ndarray, shape (M, N) or (M, N, W), complex
        ``c[m, n] = sum_l f[l] * conj(g[l - a*n]) * exp(-2j*pi*m*l/M)``,
        with ``N = L/a``.
    """
    a, M = _check_lattice(a, M)
    f = np.asarray(f)
    if f.ndim not in (1, 2):
        raise ValueError(f"f must be one- or two-dimensional, got shape {f.shape}")
    g = np.asarray(g)
    L = dgtlength(max(f.shape[0], g.shape[0]), a, M) if L is None else _check_length(L, a, M)
    f = postpad(f, L)
    c: np.ndarray = np.asarray(_dgt_long(f, _as_window(g, L), a, M)) * np.sqrt(M)
    return c


def idgt(
    c: np.ndarray,
    g: np.ndarray,
    a: int,
    Ls: int | None = None,
) -> np.ndarray:
    """Inverse discrete Gabor transform (LTFAT ``idgt``): the adjoint of ``dgt``.

    ``idgt(dgt(f, g, a, M), gabdual(g, a, M, L), a)`` reconstructs ``f``.

    Parameters
    ----------
    c : ndarray, shape (M, N) or (M, N, W)
        Coefficients.
    g : ndarray, shape (gl,)
        Synthesis window, ``gl <= L = N*a``.
    a : int
        Time shift.
    Ls : int, optional
        Truncate (or zero-pad) the output to this length.  Default: ``L``.

    Returns
    -------
    f : ndarray, shape (Ls,) or (Ls, W), complex
        ``f = sum_{m,n} c[m, n] * g[l - a*n] * exp(2j*pi*m*l/M)``.
    """
    c = np.asarray(c)
    if c.ndim not in (2, 3):
        raise ValueError(f"c must have shape (M, N) or (M, N, W), got {c.shape}")
    M, N = c.shape[0], c.shape[1]
    a, M = _check_lattice(a, M)
    L = _check_length(N * a, a, M)
    f = _idgt_long(c, _as_window(g, L), L, a, M) * np.sqrt(M)
    return f if Ls is None else postpad(f, int(Ls))


def dgtreal(
    f: np.ndarray,
    g: np.ndarray,
    a: int,
    M: int,
    L: int | None = None,
) -> np.ndarray:
    """DGT of a real signal with a real window, non-negative frequencies only.

    LTFAT ``dgtreal``: the first ``M // 2 + 1`` channels of ``dgt``; the rest
    are their complex conjugates.

    Returns
    -------
    c : ndarray, shape (M // 2 + 1, N) or (M // 2 + 1, N, W), complex
    """
    f = np.asarray(f)
    if np.iscomplexobj(f):
        raise ValueError("dgtreal needs a real signal; use dgt for complex input")
    return dgt(f, _real_window(g), a, M, L)[: int(M) // 2 + 1]


def idgtreal(
    c: np.ndarray,
    g: np.ndarray,
    a: int,
    M: int,
    Ls: int | None = None,
) -> np.ndarray:
    """Inverse of ``dgtreal`` (LTFAT ``idgtreal``); returns a real signal.

    ``M`` is required because it cannot be recovered from ``M // 2 + 1``.
    """
    a, M = _check_lattice(a, M)
    c = np.asarray(c)
    M2 = M // 2 + 1
    if c.ndim not in (2, 3) or c.shape[0] != M2:
        raise ValueError(f"c must have M // 2 + 1 = {M2} rows for M={M}, got shape {c.shape}")
    full = np.empty((M, *c.shape[1:]), dtype=complex)
    full[:M2] = c
    m = np.arange(1, (M + 1) // 2)
    full[M - m] = np.conj(c[m])
    return np.real(idgt(full, _real_window(g), a, Ls))


# ---------------------------------------------------------------------------
# Frame windows and bounds
# ---------------------------------------------------------------------------
def gabdual(
    g: np.ndarray,
    a: int,
    M: int,
    L: int | None = None,
) -> np.ndarray:
    """Canonical dual window ``S^{-1} g`` (LTFAT ``gabdual``).

    Parameters
    ----------
    g : ndarray, shape (gl,)
        Real window.
    a, M : int
        Lattice.
    L : int, optional
        Transform length (a multiple of ``lcm(a, M)``, at least ``gl``).  The
        dual is returned at this length.  If omitted, ``g`` must either fit
        the painless case ``gl <= M`` -- the dual then has the support of
        ``g`` and is returned at length ``gl`` (for an even ``gl`` this needs
        a zero middle sample, see Raises) -- or already have a valid
        transform length.

    Raises
    ------
    ValueError
        If the system is not a frame, so no dual exists; or if ``L`` is omitted
        for an even-length window whose middle sample is non-zero, whose dual
        has no exact representation at length ``gl``.
    """
    a, M = _check_lattice(a, M)
    g = _real_window(g)
    gw, Lw, crop = _frame_length(g, a, M, L)
    if crop:
        _check_fir_crop(g, "dual")
    _require_frame(gw, a, M, Lw)
    gd = np.real_if_close(_gabdual_normalised(gw, a, M, Lw), tol=1e6) / M
    return middlepad(gd, g.shape[0]) if crop else gd


def gabtight(
    g: np.ndarray,
    a: int,
    M: int,
    L: int | None = None,
) -> np.ndarray:
    """Canonical tight window ``S^{-1/2} g`` (LTFAT ``gabtight``).

    Same arguments and length rules as ``gabdual``.  With the result ``gt``,
    ``idgt(dgt(f, gt, a, M), gt, a)`` reconstructs ``f``.
    """
    a, M = _check_lattice(a, M)
    g = _real_window(g)
    gw, Lw, crop = _frame_length(g, a, M, L)
    if crop:
        _check_fir_crop(g, "tight")
    _require_frame(gw, a, M, Lw)
    gt = np.real_if_close(_gabtight_normalised(gw, a, M, Lw), tol=1e6) / np.sqrt(M)
    return middlepad(gt, g.shape[0]) if crop else gt


def gabframebounds(
    g: np.ndarray,
    a: int,
    M: int,
    L: int | None = None,
) -> tuple[float, float]:
    """Frame bounds ``(A, B)`` of the Gabor system (LTFAT ``gabframebounds``).

    ``A`` and ``B`` are the smallest and largest eigenvalues of the frame
    operator ``S``; ``B / A`` is its condition number.  Length rules as in
    ``gabdual``.
    """
    a, M = _check_lattice(a, M)
    g = _real_window(g)
    gw, Lw, _ = _frame_length(g, a, M, L)
    A, B = _gabframebounds(gw, a, M, Lw)
    return float(A), float(B)


def gabframediag(
    g: np.ndarray,
    a: int,
    M: int,
    L: int,
) -> np.ndarray:
    """Diagonal of the frame operator, ``M * sum_n |g[l - a*n]|**2`` (LTFAT ``gabframediag``).

    In the painless case (window support at most ``M``) this is the whole
    frame operator.  Returns an array of length ``L``.
    """
    a, M = _check_lattice(a, M)
    L = _check_length(L, a, M)
    return _gabframediag(_as_window(_real_window(g), L), a, M, L)
