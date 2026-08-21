"""
torch/phase/_phasegrad.py
=========================
Phase-gradient computation for non-uniform filterbanks (PyTorch).

Ports ``comp_filterbankphasegrad`` and ``filterbankphasegrad`` from the
NumPy backend as native torch operations, enabling gradient flow through
the phase-gradient computation.
"""

from __future__ import annotations

import torch

from ...numpy.filterbanks._utils import normalise_a
from ...numpy.filters._design import filterbanklength
from ...numpy.phase._phasegrad import comp_phasegradfilters
from ..filterbanks._core import filterbank

# ---------------------------------------------------------------------------
# comp_filterbankphasegrad — core torch computation
# ---------------------------------------------------------------------------


def comp_filterbankphasegrad(
    c: list[torch.Tensor],
    ch: list[torch.Tensor],
    cd: list[torch.Tensor],
    L: int,
    minlvl: float = 1e-6,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Compute phase gradients from filterbank coefficients.

    All operations are differentiable.

    Parameters
    ----------
    c, ch, cd : lists of M complex tensors
    L : signal length
    minlvl : relative floor for the spectrogram denominator

    Returns
    -------
    tgrad : list of M real tensors  (normalised inst. frequency, clipped to [-2,2])
    fgrad : list of M real tensors  (group delay)
    s     : list of M non-negative real tensors  (spectrogram)

    Examples
    --------
    >>> import torch
    >>> # Typically called from filterbankphasegrad, but shown here
    >>> # Create three coefficient lists from filterbank and derivatives
    >>> c = [torch.randn(64, dtype=torch.complex64) for _ in range(3)]
    >>> ch = [torch.randn(64, dtype=torch.complex64) for _ in range(3)]
    >>> cd = [torch.randn(64, dtype=torch.complex64) for _ in range(3)]
    >>> tgrad, fgrad, s = comp_filterbankphasegrad(c, ch, cd, L=1024)
    >>> len(tgrad) == len(c)
    True
    >>> tgrad[0].shape[0]
    64
    """
    M = len(c)

    # Global maximum for the floor
    all_abs_sq = torch.cat([torch.abs(cm).ravel() ** 2 for cm in c])
    lvl = minlvl * torch.max(all_abs_sq).item() if all_abs_sq.numel() > 0 else minlvl

    tgrad_out: list[torch.Tensor] = []
    fgrad_out: list[torch.Tensor] = []
    s_out: list[torch.Tensor] = []

    for m in range(M):
        cm = c[m]
        chm = ch[m]
        cdm = cd[m]

        # True spectrogram |c|^2; floor only the gradient denominator so the
        # returned s is exact (matches the numpy backend; avoids reporting the
        # floor as a constant in low-energy bins).
        sm = torch.abs(cm) ** 2
        sm_den = torch.clamp(sm, min=lvl)

        tg = (cdm * cm.conj()).real / sm_den / L * 2
        # Clip to [-2, 2] as in MATLAB — zero out values outside
        tg = tg * (torch.abs(tg) <= 2).to(tg.dtype)

        fg = (chm * cm.conj()).imag / sm_den

        tgrad_out.append(tg.real if tg.is_complex() else tg)
        fgrad_out.append(fg.real if fg.is_complex() else fg)
        s_out.append(sm.real if sm.is_complex() else sm)

    return tgrad_out, fgrad_out, s_out


# ---------------------------------------------------------------------------
# filterbankphasegrad — public API
# ---------------------------------------------------------------------------


def filterbankphasegrad(
    f: torch.Tensor,
    g: list[dict],
    a,
    L: int | None = None,
    minlvl: float = 1e-6,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Compute phase gradients for a filterbank (torch).

    Parameters
    ----------
    f : signal tensor, shape ``(Ls,)`` or ``(Ls, W)``
    g : list of M filter dicts
    a : hop sizes
    L : DFT length (computed from Ls and a if omitted)
    minlvl : relative spectrogram floor

    Returns
    -------
    tgrad : list of M real tensors  (normalised instantaneous frequency)
    fgrad : list of M real tensors  (group delay in samples)
    s     : list of M non-negative real tensors  (spectrogram)
    c     : list of M complex tensors  (filterbank coefficients)

    Examples
    --------
    >>> import torch
    >>> from cool_frames.torch.phase import filterbankphasegrad
    >>> from cool_frames.torch.filters import audfilters
    >>> x = torch.randn(8000)
    >>> g = audfilters(16000)
    >>> tgrad, fgrad, s, c = filterbankphasegrad(x, g, a=64)
    >>> len(tgrad) == len(g)
    True
    >>> tgrad[0].dtype == torch.float64
    True
    """
    M = len(g)
    a_norm = normalise_a(a, M)

    if L is None:
        L = filterbanklength(f.shape[0], a_norm)

    # Build derivative filters (numpy, setup-time)
    ch_filt, cd_filt = comp_phasegradfilters(g, a_norm, L)

    # Run all three filterbanks (torch, differentiable)
    c = filterbank(f, g, a_norm, L=L)
    ch_c = filterbank(f, ch_filt, a_norm, L=L)
    cd_c = filterbank(f, cd_filt, a_norm, L=L)

    tgrad, fgrad, s = comp_filterbankphasegrad(c, ch_c, cd_c, L, minlvl)
    return tgrad, fgrad, s, c
