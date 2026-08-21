"""
torch/sigproc/_sigproc.py
==========================
Coefficient-domain sparsity primitives (PyTorch).

Native torch port of ``thresh`` from ``cool_frames.numpy.sigproc``.

Differentiable, and works on tensors directly.
"""

from __future__ import annotations

import torch


def thresh(
    x: torch.Tensor,
    lam: torch.Tensor | float,
    mode: str = "hard",
) -> tuple[torch.Tensor, int]:
    """Coefficient thresholding.

    Parameters
    ----------
    x : tensor
        Input coefficients (real or complex).
    lam : float or tensor
        Threshold value.
    mode : str
        ``'hard'``, ``'soft'``, or ``'wiener'``.

    Returns
    -------
    xo : tensor
        Thresholded coefficients.
    N : int
        Number of non-zero coefficients after thresholding.

    Examples
    --------
    >>> x = torch.tensor([0.5, 1.5, 0.2, -2.0])
    >>> threshed, count = thresh(x, 0.8, mode='hard')
    >>> count  # Coefficients >= 0.8
    2
    >>> threshed  # Hard threshold  # doctest: +SKIP
    tensor([ 0., 1.5,  0., -2.])
    """
    if isinstance(lam, (int, float)):
        lam = torch.tensor(lam, dtype=x.real.dtype, device=x.device)

    mode = mode.lower()
    if mode == "hard":
        mask = torch.abs(x) >= lam
        xo = x * mask
    elif mode == "soft":
        ax = torch.abs(x)
        shrunk = torch.clamp(ax - lam, min=0.0)
        # Preserve phase for complex, sign for real
        if x.is_complex():
            xo = shrunk * torch.exp(1j * torch.angle(x))
        else:
            xo = shrunk * torch.sign(x)
    elif mode == "wiener":
        ax = torch.abs(x)
        safe_ax = torch.where(ax > 0, ax, torch.ones_like(ax))
        ratio = lam / safe_ax
        ratio = torch.where(ax > 0, ratio, torch.zeros_like(ratio))
        gain = torch.clamp(1.0 - ratio**2, min=0.0)
        xo = x * gain
    else:
        raise ValueError(f"Unknown thresholding mode: {mode!r}")

    N = int(torch.count_nonzero(xo).item())
    return xo, N
