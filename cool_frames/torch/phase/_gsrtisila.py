"""
torch/phase/_gsrtisila.py
==========================
Gnann and Spiertz's Real-Time Iterative Spectrogram Inversion with
Look-Ahead (GSRTISILA) adapted for filterbanks (PyTorch).

Port of ``phaseret/gabor/gsrtisila.m`` and numpy implementation.

GSRTISILA extends RTISILA with configurable phase initialization
strategies for the newest look-ahead frame.  In addition to the
standard zero-phase initialization, it supports:

  - ``'input'``  : use the phase of the input coefficients
  - ``'unwrap'`` : phase-vocoder-style phase unwrapping
  - ``'spsi'``   : Single-Pass Spectrogram Inversion

For filterbanks, the "frame" concept is adapted to work with the
multi-rate event schedule.

References: Gnann & Spiertz, 2008/2010.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch

from .._dtypes import resolve
from ..filterbanks._core import filterbank, ifilterbank
from ..filterbanks._frame import filterbankdual


def gsrtisila(
    s_list: list[torch.Tensor],
    g: list[dict],
    a,
    *,
    L: int | None = None,
    Ls: int | None = None,
    real: bool = False,
    maxit: int = 5,
    lookahead: int | None = None,
    startphase: Literal["zero", "input", "unwrap", "spsi"] = "zero",
    unwrappar: float = 0.3,
) -> tuple[list[torch.Tensor], torch.Tensor, float, int]:
    """GSRTISILA for filterbanks (PyTorch).

    Gnann and Spiertz's variant of RTISILA, with configurable phase
    initialization for the newest look-ahead frame.

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
    startphase : phase initialization strategy for new frames:

        - ``'zero'``   : zero phase (default)
        - ``'input'``  : use phase from input ``s_list``
        - ``'unwrap'`` : phase-vocoder unwrapping
        - ``'spsi'``   : single-pass spectrogram inversion

    unwrappar : blending parameter for unwrap mode (default: 0.3)

    Returns
    -------
    c : list of M complex tensors
    f : reconstructed signal tensor
    relres : final residual
    niter : total iterations
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

    # Initialise coefficients
    if startphase == "input":
        c = [s.to(dtype=cdtype, device=device).flatten().clone() for s in s_list]
    else:
        c = [s.clone().to(dtype=cdtype) for s in s_abs]

    a_int = a_norm[:, 0].astype(int)

    # Build time-sorted events
    events = []
    for m in range(M):
        for n in range(N[m]):
            events.append((n * a_int[m], m, n))
    events.sort(key=lambda x: (x[0], x[1]))

    # Group by time
    time_groups: list[tuple[int, list[tuple[int, int, int]]]] = []
    i = 0
    while i < len(events):
        t = events[i][0]
        group: list[tuple[int, int, int]] = []
        while i < len(events) and events[i][0] == t:
            group.append(events[i])
            i += 1
        time_groups.append((t, group))

    n_groups: int = len(time_groups)

    # SPSI pre-initialization
    # Normalised centre frequencies, recovered from the filters themselves.
    # (Before v0.1.1 both branches below used `m / M`, a linear ramp reaching
    # ~0.96 cycles/sample — nearly twice Nyquist — irrespective of the actual
    # filter layout.)
    if startphase in ("spsi", "unwrap"):
        from ...numpy.phase._centerfreq import filter_center_frequencies

        fc_norm = filter_center_frequencies(g, L)

    if startphase == "spsi":
        from ._spsi import spsi

        # Convert s_abs to numpy for spsi (which returns numpy arrays)
        s_abs_np = [s.cpu().numpy() for s in s_abs]
        # fs=1.0: fc_norm is already in cycles per sample.
        c_spsi, _ = spsi(s_abs_np, a_int, fc_norm, 1.0)  # type: ignore[arg-type]
        # Convert back to torch
        c = [
            torch.from_numpy(np.asarray(ci, dtype=np.complex128))
            .to(device=device, dtype=cdtype)
            .flatten()
            for ci in c_spsi
        ]

    # Phase accumulator for unwrap mode
    if startphase == "unwrap":
        omega = torch.tensor(
            [2.0 * np.pi * a_int[m] * fc_norm[m] for m in range(M)], dtype=dtype, device=device
        )

    # Process each time group with look-ahead
    for gi in range(n_groups):
        la_end = min(gi + lookahead + 1, n_groups)

        # Initialise phase for the newest look-ahead frame
        if gi + lookahead < n_groups and startphase == "unwrap":
            _, la_group = time_groups[min(gi + lookahead, n_groups - 1)]
            for _t, m, n in la_group:
                if n >= 2:
                    phase_prev2 = torch.angle(c[m][n - 2])
                    phase_prev1 = torch.angle(c[m][n - 1])
                    # Phase vocoder unwrapping
                    om = omega[m]
                    delta = phase_prev1 - phase_prev2 - om
                    delta -= 2.0 * np.pi * torch.round(delta / (2.0 * np.pi))
                    phase_new = phase_prev1 + om + delta
                    c[m][n] = unwrappar * s_abs[m][n] * torch.exp(1j * phase_new)

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
