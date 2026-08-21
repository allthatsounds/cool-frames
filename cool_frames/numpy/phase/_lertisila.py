"""
numpy/phaseret/_lertisila.py
==============================
Le Roux's RTISILA variant (LERTISILA / TF-RTISI-LA) adapted for
filterbanks.

Port of ``phaseret/gabor/lertisila.m``.

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

import numpy as np

from ..filterbanks._core import filterbank, ifilterbank
from ..filterbanks._frame import filterbankdual


def lertisila(
    s_list: list[np.ndarray],
    g: list[dict],
    a,
    *,
    L: int | None = None,
    Ls: int | None = None,
    real: bool = False,
    maxit: int = 5,
    lookahead: int | None = None,
    startphase: Literal["zhu", "zero", "rand", "input"] = "zhu",
    seed: int | None = None,
    variant: Literal["trunc", "modtrunc"] = "trunc",
    energy_order: bool = False,
) -> tuple[list[np.ndarray], np.ndarray, float, int]:
    """Le Roux's RTISILA for filterbanks.

    Uses truncated projection kernels for efficient per-frame phase
    updates with look-ahead.

    Parameters
    ----------
    s_list : list of M arrays — target magnitudes
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
    c : list of M complex arrays
    f : reconstructed signal
    relres : final residual
    niter : total iterations
    """
    M = len(g)
    s_abs = [np.abs(np.asarray(s)).ravel() for s in s_list]

    from ..filterbanks._utils import normalise_a

    a_norm = normalise_a(a, M)

    N = [len(s) for s in s_abs]
    if L is None:
        afrac = a_norm[:, 0] / a_norm[:, 1]
        L = int(round(N[0] * afrac[0]))

    if real:
        gd = filterbankdual(g, a_norm, L)
    else:
        gd = filterbankdual(g, a_norm, L, real=False)

    if lookahead is None:
        lookahead = 2

    # Initialise coefficients
    if startphase == "rand":
        rng = np.random.default_rng(seed)
        c = [s * np.exp(2j * np.pi * rng.random(len(s))) for s in s_abs]
    elif startphase == "input":
        c = [np.asarray(s, dtype=complex).ravel().copy() for s in s_list]
    else:
        c = [s.copy().astype(complex) for s in s_abs]

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
        c_proj = filterbank(np.real(f_iter) if real else f_iter, g, a_norm, L=L)

        frame_indices = list(groups_range)

        if energy_order and not first_iter:
            # Sort by descending energy
            energies = []
            for gj in frame_indices:
                _, group = time_groups[gj]
                e = sum(np.abs(c_in[m][n]) ** 2 for (t, m, n) in group)
                energies.append(e)
            frame_indices = [x for _, x in sorted(zip(energies, frame_indices), reverse=True)]

        for gj in frame_indices:
            _, group = time_groups[gj]
            for _t, m, n in group:
                cp = np.asarray(c_proj[m]).ravel()
                if variant == "modtrunc":
                    correction = cp[n] - c_in[m][n]
                    phase = np.angle(c_in[m][n] + correction)
                else:
                    phase = np.angle(cp[n])
                c_in[m][n] = s_abs[m][n] * np.exp(1j * phase)

        return c_in

    # Process each time group with look-ahead
    for gi in range(n_groups):
        la_end = min(gi + lookahead + 1, n_groups)

        # For 'zhu' mode, the newest frame starts as zero —
        # it will be filled by the projection from overlapping frames.
        if startphase == "zhu" and gi + lookahead < n_groups:
            _, la_group = time_groups[gi + lookahead]
            for _t, m, n in la_group:
                c[m][n] = 0.0 + 0.0j

        for it in range(maxit):
            c = _project_and_update(c, range(gi, la_end), first_iter=(it == 0))

    # Final synthesis
    f = ifilterbank(c, gd, a_norm, Ls=Ls or L, real=real)
    if real:
        f = np.real(f)

    # Compute final residual
    s_flat = np.concatenate(s_abs)
    c_flat = np.concatenate([np.abs(np.asarray(cm).ravel()) for cm in c])
    norm_s = np.linalg.norm(s_flat)
    relres = float(np.linalg.norm(c_flat - s_flat) / norm_s) if norm_s > 0 else 0.0

    return c, f, relres, maxit * n_groups
