"""
numpy.core._fourier
====================
Basic Fourier utilities ported from the LTFAT ``fourier/`` folder.

Previously in the separate ``ltfat_core`` package; vendored here to
remove the external dependency.

Functions:
    pgauss           — periodic, sampled Gaussian window
    middlepad        — symmetric zero-extension or truncation

All functions follow LTFAT normalisation conventions.

References
----------
  [1] P. L. Søndergaard, "LTFAT — The Large Time-Frequency Analysis
      Toolbox," 2008-2023.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# middlepad — symmetric zero-extend / cut
# ---------------------------------------------------------------------------


def middlepad(
    f: np.ndarray,
    L: int,
    *,
    centering: str = "wp",
) -> np.ndarray:
    """Symmetrically zero-extend or truncate a function.

    Operates on 1-D arrays (or the first axis of 2-D arrays).
    Preserves whole-point even symmetry: if *f* is WPE then the
    output is also WPE at the new length.

    Parameters
    ----------
    f : ndarray, shape (Ls,) or (Ls, W)
        Input signal(s).
    L : int
        Target length.
    centering : ``"wp"`` or ``"hp"``
        Whole-point even (default) or half-point even centering.

    Returns
    -------
    h : ndarray, shape (L,) or (L, W)
        Zero-extended or truncated signal.

    Notes
    -----
    When extending an even-length signal, the Nyquist sample is split
    in half across the two boundary positions (matching LTFAT convention).
    When cutting to an even length, the two boundary samples are averaged.
    """
    if L < 1:
        raise ValueError("L must be >= 1")

    squeeze = False
    if f.ndim == 1:
        f = f[:, np.newaxis]
        squeeze = True

    Ls, W = f.shape

    if Ls == L:
        out = f.copy()
    elif centering == "wp":
        out = _middlepad_wp(f, L, Ls, W)
    elif centering == "hp":
        out = _middlepad_hp(f, L, Ls, W)
    else:
        raise ValueError(f"centering must be 'wp' or 'hp', got {centering!r}")

    return out[:, 0] if squeeze else out


def _middlepad_wp(
    f: np.ndarray,
    L: int,
    Ls: int,
    W: int,
) -> np.ndarray:
    """Whole-point even middlepad."""
    out = np.zeros((L, W), dtype=f.dtype)

    if Ls == 1:
        out[0, :] = f[0, :]
        return out

    if Ls > L:
        # --- Cut ---
        if L % 2 == 0:
            out[: L // 2, :] = f[: L // 2, :]
            out[L // 2, :] = (f[L // 2, :] + f[Ls - L // 2, :]) / 2
            out[L // 2 + 1 :, :] = f[Ls - L // 2 + 1 :, :]
        else:
            half = (L + 1) // 2
            out[:half, :] = f[:half, :]
            out[half:, :] = f[Ls - (L - half) :, :]
    else:
        # --- Extend ---
        d = L - Ls
        if Ls % 2 == 0:
            half = Ls // 2
            out[:half, :] = f[:half, :]
            out[half, :] = f[half, :] / 2
            out[half + d, :] = f[half, :] / 2
            out[half + d + 1 :, :] = f[half + 1 :, :]
        else:
            half = (Ls + 1) // 2
            out[:half, :] = f[:half, :]
            out[half + d :, :] = f[half:, :]

    return out


def _middlepad_hp(
    f: np.ndarray,
    L: int,
    Ls: int,
    W: int,
) -> np.ndarray:
    """Half-point even middlepad."""
    out = np.zeros((L, W), dtype=f.dtype)

    if Ls == 1:
        out[0, :] = f[0, :]
        return out

    if Ls > L:
        # --- Cut ---
        if L % 2 == 0:
            half = L // 2
            out[:half, :] = f[:half, :]
            out[half:, :] = f[Ls - half :, :]
        else:
            half = (L - 1) // 2
            out[:half, :] = f[:half, :]
            out[half, :] = (f[half, :] + f[Ls - half - 1, :]) / 2
            out[half + 1 :, :] = f[Ls - half :, :]
    else:
        # --- Extend ---
        d = L - Ls
        if Ls % 2 == 0:
            half = Ls // 2
            out[:half, :] = f[:half, :]
            out[half + d :, :] = f[half:, :]
        else:
            half = (Ls - 1) // 2
            out[:half, :] = f[:half, :]
            out[half, :] = f[half, :] / 2
            out[half + d, :] = f[half, :] / 2
            out[half + d + 1 :, :] = f[half + 1 :, :]

    return out
