"""
torch/phase/_metrics.py
========================
Spectral convergence metrics for phase retrieval evaluation (PyTorch).

Native torch port of the numpy ``_metrics.py``.
"""

from __future__ import annotations

import math

import torch


def magnitudeerr(target, reconstructed) -> float:
    r"""Spectral convergence (Frobenius-norm relative error).

    Parameters
    ----------
    target, reconstructed : list of tensors or tensor
        Magnitude spectrograms or lists of per-channel coefficient tensors.

    Returns
    -------
    E : float
        Relative Frobenius-norm error: \|\|target\| - \|reconstructed\|\| / \|\|target\|\|

    Examples
    --------
    >>> import torch
    >>> from cool_frames.torch.phase import magnitudeerr
    >>> # Compare two lists of magnitude tensors
    >>> target = [torch.abs(torch.randn(64, dtype=torch.complex64)) for _ in range(3)]
    >>> recon = [t + 0.01 * torch.randn_like(t) for t in target]
    >>> err = magnitudeerr(target, recon)
    >>> 0 <= err < 1
    True
    """
    if isinstance(target, (list, tuple)):
        t_flat = torch.cat([torch.abs(t.flatten()) for t in target])
        r_flat = torch.cat([torch.abs(r.flatten()) for r in reconstructed])
    else:
        t_flat = torch.abs(target.flatten())
        r_flat = torch.abs(reconstructed.flatten())

    norm_t = torch.linalg.norm(t_flat).item()
    if norm_t == 0:
        return 0.0

    return float(torch.linalg.norm(t_flat - r_flat).item() / norm_t)


def magnitudeerrdb(target, reconstructed) -> float:
    """Spectral convergence in decibels.

    Parameters
    ----------
    target, reconstructed : same as :func:`magnitudeerr`

    Returns
    -------
    Edb : float
        ``20 * log10(magnitudeerr(target, reconstructed))``

    Examples
    --------
    >>> import torch
    >>> from cool_frames.torch.phase import magnitudeerrdb
    >>> target = [torch.abs(torch.randn(64, dtype=torch.complex64)) for _ in range(3)]
    >>> recon = [t + 0.01 * torch.randn_like(t) for t in target]
    >>> errdb = magnitudeerrdb(target, recon)
    >>> errdb < 0  # error is between 0 and 1, so dB is negative
    True
    """
    e = magnitudeerr(target, reconstructed)
    if e <= 0:
        return -math.inf
    return 20.0 * math.log10(e)
