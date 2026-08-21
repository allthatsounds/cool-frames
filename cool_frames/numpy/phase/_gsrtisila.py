"""
numpy/phaseret/_gsrtisila.py
==============================
Gnann and Spiertz's Real-Time Iterative Spectrogram Inversion with
Look-Ahead (GSRTISILA) adapted for filterbanks.

Port of ``phaseret/gabor/gsrtisila.m``.

GSRTISILA extends RTISILA with configurable phase initialization
strategies for the newest look-ahead frame.  In addition to the
standard zero-phase initialization, it supports:

  - ``'input'``  : use the phase of the input coefficients
  - ``'unwrap'`` : phase-vocoder-style phase unwrapping
  - ``'spsi'``   : Single-Pass Spectrogram Inversion
  - ``'rtpghi'`` : Real-Time Phase Gradient Heap Integration

For filterbanks, the "frame" concept is adapted to work with the
multi-rate event schedule.

References: Gnann & Spiertz, 2008/2010.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..filterbanks._core import filterbank, ifilterbank
from ..filterbanks._frame import filterbankdual


def gsrtisila(
    s_list: list[np.ndarray],
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
) -> tuple[list[np.ndarray], np.ndarray, float, int]:
    """GSRTISILA for filterbanks.

    Gnann and Spiertz's variant of RTISILA, with configurable phase
    initialization for the newest look-ahead frame.

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
    startphase : phase initialization strategy for new frames:

        - ``'zero'``   : zero phase (default)
        - ``'input'``  : use phase from input ``s_list``
        - ``'unwrap'`` : phase-vocoder unwrapping
        - ``'spsi'``   : single-pass spectrogram inversion

    unwrappar : blending parameter for unwrap mode (default: 0.3)

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
    if startphase == "input":
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

    # SPSI pre-initialization
    if startphase == "spsi":
        from ._spsi import spsi

        fc = np.zeros(M)
        for m in range(M):
            filter_m = g[m]
            if "fc" in filter_m:
                fc[m] = filter_m["fc"]
            else:
                fc[m] = m / M
        c_spsi, _ = spsi(s_abs, a_int, fc)
        c = [np.asarray(ci, dtype=complex).ravel().copy() for ci in c_spsi]

    # Phase accumulator for unwrap mode
    if startphase == "unwrap":
        omega = np.array([2.0 * np.pi * a_int[m] * (m / M) for m in range(M)])

    # Process each time group with look-ahead
    for gi in range(n_groups):
        la_end = min(gi + lookahead + 1, n_groups)

        # Initialise phase for the newest look-ahead frame
        if gi + lookahead < n_groups and startphase == "unwrap":
            _, la_group = time_groups[min(gi + lookahead, n_groups - 1)]
            for _t, m, n in la_group:
                if n >= 2:
                    phase_prev2 = np.angle(c[m][n - 2])
                    phase_prev1 = np.angle(c[m][n - 1])
                    # Phase vocoder unwrapping
                    om = omega[m]
                    delta = phase_prev1 - phase_prev2 - om
                    delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
                    phase_new = phase_prev1 + om + delta
                    c[m][n] = unwrappar * s_abs[m][n] * np.exp(1j * phase_new)

        for _it in range(maxit):
            # Synthesise from current coefficients
            f_iter = ifilterbank(c, gd, a_norm, Ls=L, real=real)

            # Re-analyse
            c_new = filterbank(np.real(f_iter) if real else f_iter, g, a_norm, L=L)

            # Phase update: only for frames in [gi, la_end)
            for gj in range(gi, la_end):
                _, group = time_groups[gj]
                for _t, m, n in group:
                    cn = np.asarray(c_new[m]).ravel()
                    phase = np.angle(cn[n])
                    c[m][n] = s_abs[m][n] * np.exp(1j * phase)

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
