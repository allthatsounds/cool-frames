"""
numpy/phaseret/_rtisila.py
============================
Real-Time Iterative Spectrogram Inversion with Look-Ahead (RTISILA)
adapted for filterbanks.

Port of ``phaseret/gabor/rtisila.m``.

RTISILA processes frames one at a time (causally), with a configurable
look-ahead buffer.  At each frame it performs multiple analysis-synthesis
iterations on a small window of frames to refine the phase estimate.

For filterbanks, the "frame" concept is adapted to work with the
multi-rate event schedule (same as in ``rtpghifb_nonuniform``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from ..filterbanks._core import filterbank, ifilterbank
from ..filterbanks._frame import filterbankdual


def rtisila(
    s_list: list[np.ndarray],
    g: list[dict],
    a,
    *,
    L: int | None = None,
    Ls: int | None = None,
    real: bool = False,
    maxit: int = 5,
    lookahead: int | None = None,
    startphase: Literal["zero", "rand"] = "zero",
) -> tuple[list[np.ndarray], np.ndarray, float, int]:
    """RTISILA for filterbanks.

    Real-time iterative spectrogram inversion with look-ahead [rtisila-gnann]_
    and improved signal modeling [rtisila-zhu]_.

    Processes frames left-to-right with a look-ahead buffer, performing
    ``maxit`` iterations per frame.  This is an offline simulation of
    the real-time algorithm.

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
    startphase : initial phase strategy

    Returns
    -------
    c : list of M complex arrays
    f : reconstructed signal
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
        rng = np.random.default_rng()
        c = [s * np.exp(2j * np.pi * rng.random(len(s))) for s in s_abs]
    else:
        c = [s.copy().astype(complex) for s in s_abs]

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
