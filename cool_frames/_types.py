"""
_types.py
=========
Shared dtype utilities and type aliases used by both the NumPy and PyTorch
backends.
"""
from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# Canonical type aliases for the filterbank data model
# ---------------------------------------------------------------------------

class FilterDict(TypedDict, total=False):
    """Canonical filter descriptor used throughout the library.

    A filter is stored in one of two representations:

    *Frequency-domain (band-limited):*
        H        – callable(L) -> ndarray, or ndarray (transfer function)
        foff     – callable(L) -> int, or int (frequency offset in DFT bins)
        realonly  – 0 or 1 (1 → real-valued signal filter, mirror implied)
        delay    – int (filter delay in samples)
        fs       – float or None (sampling rate, metadata only)

    *Time-domain (FIR):*
        h        – ndarray (impulse response)
        offset   – int (causal skip / delay)
        delay    – int (same as offset)
        fs       – float or None
    """
    # Frequency-domain keys
    H: np.ndarray | Callable
    foff: int | Callable
    realonly: int
    delay: int
    fs: float | None
    # Time-domain keys
    h: np.ndarray
    offset: int


# Type alias for a list of filterbank coefficients (one array per channel)
CoeffList = list[np.ndarray]

# Type alias for hop-size arrays: 1-D integer or (M, 2) rational
HopArray = np.ndarray


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

def real_dtype(dtype) -> np.dtype:
    """Return the real-valued counterpart of *dtype* (float32 or float64)."""
    d = np.dtype(dtype)
    if d.kind == 'c':
        return np.dtype('float32') if d.itemsize <= 8 else np.dtype('float64')
    return np.dtype('float32') if d.itemsize <= 4 else np.dtype('float64')


def complex_dtype(dtype) -> np.dtype:
    """Return the complex-valued counterpart of *dtype* (complex64 or complex128)."""
    d = np.dtype(dtype)
    if d.kind == 'c':
        return d  # type: ignore[no-any-return]
    return np.dtype('complex64') if d.itemsize <= 4 else np.dtype('complex128')


def promote_dtype(*arrays) -> np.dtype:
    """Return the common complex dtype for a sequence of arrays/dtypes."""
    dtypes = [np.dtype(a) if not isinstance(a, np.ndarray) else a.dtype
              for a in arrays]
    out = np.result_type(*dtypes)
    return complex_dtype(out)


# ---------------------------------------------------------------------------
# Phase convention enum
# ---------------------------------------------------------------------------

class PhaseConvention(IntEnum):
    """Phase convention used by phase-gradient routines."""
    FREQINV  = 0   # frequency-invariant (LTFAT default)
    TIMEINV  = 1   # time-invariant
    LOCAL    = 2   # local phase
