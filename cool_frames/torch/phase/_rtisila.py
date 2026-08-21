"""
torch/phase/_rtisila.py
=======================
Real-Time Iterative Spectrogram Inversion with Look-Ahead (RTISILA)
adapted for filterbanks (PyTorch).

Port of ``phaseret/gabor/rtisila.m`` and numpy implementation.

RTISILA processes frames one at a time (causally), with a configurable
look-ahead buffer.  At each frame it performs multiple analysis-synthesis
iterations on a small window of frames to refine the phase estimate.

For filterbanks, the "frame" concept is adapted to work with the
multi-rate event schedule (same as in ``rtpghifb_nonuniform``).
"""

from __future__ import annotations

from typing import Literal

import torch

from .._dtypes import resolve
from ..filterbanks._core import filterbank, ifilterbank
from ..filterbanks._frame import filterbankdual


def rtisila(
    s_list: list[torch.Tensor],
    g: list[dict],
    a,
    *,
    L: int | None = None,
    Ls: int | None = None,
    real: bool = False,
    maxit: int = 5,
    lookahead: int | None = None,
    startphase: Literal["zero", "rand"] = "zero",
    seed: int | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, float, int]:
    """RTISILA for filterbanks (PyTorch).

    Real-time iterative spectrogram inversion with look-ahead [rtisila-gnann]_
    and improved signal modeling [rtisila-zhu]_.

    Processes frames left-to-right with a look-ahead buffer, performing
    ``maxit`` iterations per frame.  This is an offline simulation of
    the real-time algorithm.

    Parameters
    ----------
    s_list : list of M tensors — target magnitudes
    g : list of M filter dicts
    a : hop sizes
    L : DFT length
    Ls : output signal length
    real : use real (single-sided) synthesis
    maxit : iterations per frame
    lookahead : number of look-ahead frames (default: 2)
    startphase : initial phase strategy

    Returns
    -------
    c : list of M complex tensors
    f : reconstructed signal tensor
    relres : final residual
    niter : total iterations (maxit × N)

    References
    ----------
    .. [rtisila-gnann] D. Gnann and M. Spiertz, "Improving PGHI-based phase reconstruction by using a
           stronger signal model," DAFx-19, 2019.
    .. [rtisila-zhu] N. Zhu, K. Müller, and H. Liebig, "Real-time iterative spectrogram inversion with
           look ahead," IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 29,
           pp. 2601–2609, 2021.
    """
    # Determine device and dtype from inputs
    device = s_list[0].device if isinstance(s_list[0], torch.Tensor) else torch.device("cpu")
    # The caller's dtype wins; see cool_frames/torch/_dtypes.py.
    dtype, cdtype = resolve(*(s_list if isinstance(s_list, list) else [s_list]))

    M = len(g)
    s_abs = [torch.abs(s.to(dtype=dtype, device=device)).flatten() for s in s_list]

    # Setup-time: compute dual window (numpy)
    from ...numpy.filterbanks._utils import normalise_a

    a_norm = normalise_a(a, M)

    N = [len(s) for s in s_abs]
    if L is None:
        afrac = a_norm[:, 0] / a_norm[:, 1]
        L = int(round(N[0] * afrac[0]))

    gd = filterbankdual(g, a_norm, L, real=real)

    if lookahead is None:
        lookahead = 2

    # Generator for `startphase='rand'`.  `None` means torch's global RNG,
    # i.e. unseeded; an explicit `seed` makes a random start reproducible.
    _rng = None
    if seed is not None:
        _rng = torch.Generator(device=device)
        _rng.manual_seed(int(seed))

    # Initialise coefficients
    if startphase == "rand":
        c = [
            s
            * torch.exp(
                2j * torch.pi * torch.rand(len(s), generator=_rng, device=device, dtype=dtype)
            )
            for s in s_abs
        ]
    else:  # "zero"
        c = [s.clone().to(dtype=cdtype) for s in s_abs]

    # For uniform filterbanks, process frame-by-frame.
    # For non-uniform, we process by time step.
    a_int = a_norm[:, 0].astype(int)

    # Build time-sorted events
    events = []
    for m in range(M):
        for n in range(N[m]):
            events.append((n * a_int[m], m, n))
    events.sort(key=lambda x: (x[0], x[1]))

    # Group by time
    time_groups = []
    i = 0
    while i < len(events):
        t = events[i][0]
        group = []
        while i < len(events) and events[i][0] == t:
            group.append(events[i])
            i += 1
        time_groups.append((t, group))

    n_groups = len(time_groups)

    # Process each time group with look-ahead
    for gi in range(n_groups):
        la_end = min(gi + lookahead + 1, n_groups)

        for _it in range(maxit):
            # Synthesise from current coefficients
            f_iter = ifilterbank(c, gd, a_norm, Ls=L, real=real)

            # Re-analyse
            f_input = f_iter.real if (real and f_iter.is_complex()) else f_iter
            c_new = filterbank(f_input, g, a_norm, L=L)

            # Phase update: only for frames in [gi, la_end)
            for gj in range(gi, la_end):
                _, group = time_groups[gj]
                for _t, m, n in group:
                    cn = c_new[m].flatten()
                    phase = torch.angle(cn[n])
                    c[m][n] = s_abs[m][n] * torch.exp(1j * phase)

    # Final synthesis
    f = ifilterbank(c, gd, a_norm, Ls=Ls or L, real=real)
    if real and f.is_complex():
        f = f.real

    # Compute final residual
    s_flat = torch.cat(s_abs)
    c_flat = torch.cat([torch.abs(cm.flatten()) for cm in c])
    norm_s = torch.linalg.norm(s_flat).item()
    relres = float(torch.linalg.norm(c_flat - s_flat).item() / norm_s) if norm_s > 0 else 0.0

    return c, f, relres, maxit * n_groups
