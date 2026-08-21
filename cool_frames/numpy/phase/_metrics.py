"""
numpy/phase/_metrics.py
===========================
Spectral convergence metrics for phase retrieval evaluation.

Port of ``phaseret/gabor/magnitudeerr.m`` and ``magnitudeerrdb.m``.
"""

from __future__ import annotations

import math

import numpy as np


def magnitudeerr(target, reconstructed) -> float:
    r"""Spectral convergence (Frobenius-norm relative error).

    Parameters
    ----------
    target, reconstructed : array_like or list of arrays
        Magnitude spectrograms or lists of per-channel coefficient arrays.
        If lists, they are concatenated before comparison.

    Returns
    -------
    E : float
        Relative Frobenius-norm error:  \|\|target\| - \|reconstructed\|\| / \|\|target\|\|

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.phase import magnitudeerr
    >>> target = np.array([1.0, 2.0, 3.0])
    >>> reconstructed = np.array([1.0, 2.1, 2.9])
    >>> err = magnitudeerr(target, reconstructed)
    >>> 0 <= err < 1
    True
    """
    if isinstance(target, (list, tuple)):
        target = np.concatenate([np.asarray(t).ravel() for t in target])
        reconstructed = np.concatenate([np.asarray(r).ravel() for r in reconstructed])
    else:
        target = np.asarray(target).ravel()
        reconstructed = np.asarray(reconstructed).ravel()

    t_abs = np.abs(target)
    r_abs = np.abs(reconstructed)

    norm_t = np.linalg.norm(t_abs)
    if norm_t == 0:
        return 0.0

    return float(np.linalg.norm(t_abs - r_abs) / norm_t)


def magnitudeerrdb(target, reconstructed) -> float:
    """Spectral convergence in decibels.

    Parameters
    ----------
    target, reconstructed : same as :func:`magnitudeerr`

    Returns
    -------
    Edb : float
        ``20 * log10(magnitudeerr(target, reconstructed))`` in dB.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.phase import magnitudeerrdb
    >>> target = np.array([1.0, 2.0, 3.0])
    >>> reconstructed = np.array([1.0, 2.0, 3.0])
    >>> err_db = magnitudeerrdb(target, reconstructed)
    >>> err_db == -np.inf  # perfect reconstruction
    True
    """
    e = magnitudeerr(target, reconstructed)
    if e <= 0:
        return -math.inf
    return 20.0 * math.log10(e)
