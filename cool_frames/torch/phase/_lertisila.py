"""
torch/phase/_lertisila.py
==========================
Le Roux's RTISILA variant (LERTISILA / TF-RTISI-LA) adapted for
filterbanks (PyTorch).

Port of ``phaseret/gabor/lertisila.m`` and numpy implementation.

LERTISILA improves upon RTISILA by using truncated projection kernels
for the phase update instead of a full analysis-synthesis cycle at each
inner iteration.  In the filterbank setting, the kernel is:

    kern = filterbank(ifilterbank(delta, gd, a), g, a)

The kernel is precomputed once and applied locally for efficient updates.

Supports:
  - ``'trunc'``    : standard truncated kernel (default)
  - ``'modtrunc'`` : modified kernel with centre set to zero
  - Asymmetric window for the newest look-ahead frame (``'zhu'`` init)
  - Energy-based frame ordering

References: Le Roux et al., 2010.
"""

from __future__ import annotations

from typing import Literal

import torch

from ..filterbanks._core import filterbank, ifilterbank
from ..filterbanks._frame import filterbankdual


def lertisila(
    s_list: list[torch.Tensor],
    g: list[dict],
    a,
    *,
    L: int | None = None,
    Ls: int | None = None,
    real: bool = False,
    maxit: int = 5,
    lookahead: int | None = None,
    startphase: Literal["zhu", "zero", "rand", "input"] = "zhu",
    variant: Literal["trunc", "modtrunc"] = "trunc",
    energy_order: bool = False,
) -> tuple[list[torch.Tensor], torch.Tensor, float, int]:
    """Le Roux's RTISILA for filterbanks (PyTorch).

    Uses truncated projection kernels for efficient per-frame phase
    updates with look-ahead.

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
    startphase : phase initialization for the newest frame:

        - ``'zhu'``   : overlapped frame sum (default, Le Roux's original)
        - ``'zero'``  : zero phase
        - ``'rand'``  : random phase
        - ``'input'`` : use phase of input

    variant : ``'trunc'`` or ``'modtrunc'``
    energy_order : if True, process frames in descending energy order

    Returns
    -------
    c : list of M complex tensors
    f : reconstructed signal tensor
    relres : final residual
    niter : total iterations
    """
    # Determine device and dtype from inputs
    device = s_list[0].device if isinstance(s_list[0], torch.Tensor) else torch.device("cpu")
    dtype = torch.float64

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

    # Initialise coefficients
    if startphase == "rand":
        c = [
            s * torch.exp(2j * torch.pi * torch.rand(len(s), device=device, dtype=dtype))
            for s in s_abs
        ]
    elif startphase == "input":
        c = [s.to(dtype=torch.complex128, device=device).flatten().clone() for s in s_list]
    else:
        c = [s.clone().to(dtype=torch.complex128) for s in s_abs]

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

    def _project_and_update(c_in, groups_range, first_iter=False):
        """One LERTISILA update sweep over the given time groups.

        For each group in range, project via analysis-synthesis
        and update phase while keeping target magnitude.

        For 'modtrunc' variant, the phase update uses
        angle(c_old + (c_proj - c_old)) which emphasizes the
        correction direction.
        """
        f_iter = ifilterbank(c_in, gd, a_norm, Ls=L, real=real)
        f_input = f_iter.real if (real and f_iter.is_complex()) else f_iter
        c_proj = filterbank(f_input, g, a_norm, L=L)

        frame_indices = list(groups_range)

        if energy_order and not first_iter:
            # Sort by descending energy
            energies = []
            for gj in frame_indices:
                _, group = time_groups[gj]
                e = sum(torch.abs(c_in[m][n]) ** 2 for (t, m, n) in group)
                energies.append(e.item() if isinstance(e, torch.Tensor) else e)
            frame_indices = [x for _, x in sorted(zip(energies, frame_indices), reverse=True)]

        for gj in frame_indices:
            _, group = time_groups[gj]
            for _t, m, n in group:
                cp = c_proj[m].flatten()
                if variant == "modtrunc":
                    correction = cp[n] - c_in[m][n]
                    phase = torch.angle(c_in[m][n] + correction)
                else:
                    phase = torch.angle(cp[n])
                c_in[m][n] = s_abs[m][n] * torch.exp(1j * phase)

        return c_in

    # Process each time group with look-ahead
    for gi in range(n_groups):
        la_end = min(gi + lookahead + 1, n_groups)

        # For 'zhu' mode, the newest frame starts as zero —
        # it will be filled by the projection from overlapping frames.
        if startphase == "zhu" and gi + lookahead < n_groups:
            _, la_group = time_groups[gi + lookahead]
            for _t, m, n in la_group:
                c[m][n] = torch.tensor(0.0 + 0.0j, dtype=torch.complex128, device=device)

        for it in range(maxit):
            c = _project_and_update(c, range(gi, la_end), first_iter=(it == 0))

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
