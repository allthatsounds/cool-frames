"""
numpy/phaseret/_spsi.py
========================
Single-Pass Spectrogram Inversion (SPSI) for filterbanks.

Port of ``phaseret/gabor/spsi.m`` / ``comp_spsireal.m``.

The original algorithm works in the Gabor domain and uses spectral peaks
to estimate instantaneous frequency.  This filterbank adaptation operates
on per-channel magnitude arrays with non-uniform hop sizes:

    For each time step, peaks are identified across channels.  For each
    peak, the instantaneous frequency is estimated via parabolic
    interpolation of log-magnitudes, and the phase is propagated from the
    previous frame using a phase accumulator.  Non-peak channels inherit
    the phase of the nearest peak.
"""

from __future__ import annotations

import numpy as np


def spsi(
    s_list: list[np.ndarray],
    a: np.ndarray,
    fc: np.ndarray,
    *,
    startphase: np.ndarray | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Single-Pass Spectrogram Inversion for filterbanks.

    Single-pass phase reconstruction with local gradient estimation [spsi-beauregard]_.

    Parameters
    ----------
    s_list : list of M arrays, each (N_m,)
        Magnitude coefficients per channel.
    a : (M,) int — hop sizes per channel
    fc : (M,) float — normalised centre frequencies
    startphase : (M,) optional initial phase per channel

    Returns
    -------
    c_list : list of M complex arrays
    phase_list : list of M float arrays

    References
    ----------
    .. [spsi-beauregard] T. Beauregard, M. Harish, and L. Wyse, "Single pass spectrogram inversion using
           local gradient estimation," IEEE ICSC, 2015.
    """
    a = np.asarray(a, dtype=int).ravel()
    fc = np.asarray(fc, dtype=float).ravel()
    M = len(a)

    N = np.array([len(s) for s in s_list], dtype=int)

    if startphase is None:
        m_phase = np.zeros(M)
    else:
        m_phase = np.asarray(startphase, dtype=float).ravel().copy()

    # Build event schedule (time-sorted)
    events = []
    for m in range(M):
        for n in range(N[m]):
            events.append((n * a[m], m, n))
    events.sort(key=lambda x: (x[0], x[1]))

    phase_out = [np.zeros(N[m]) for m in range(M)]

    # Process events by time step
    i = 0
    while i < len(events):
        t_now = events[i][0]
        batch = []
        while i < len(events) and events[i][0] == t_now:
            batch.append(events[i])
            i += 1

        # Get magnitudes for channels present at this time
        current_mags = np.full(M, -1.0)
        current_frame = {}
        for _t, m, n in batch:
            current_mags[m] = abs(s_list[m][n])
            current_frame[m] = n

        # Only consider channels that have a frame at this time
        active = sorted(current_frame.keys())
        if len(active) < 2:
            # Single channel — just accumulate phase
            for m in active:
                n = current_frame[m]
                instf = fc[m]
                m_phase[m] = m_phase[m] + 2.0 * np.pi * a[m] * instf
                phase_out[m][n] = m_phase[m]
            continue

        # Get magnitudes for active channels
        sabs = np.array([current_mags[m] for m in active])
        log_sabs = np.log(sabs + np.finfo(float).tiny)

        # Find peaks: channels where magnitude > both neighbours
        peaks = []
        for j in range(1, len(active) - 1):
            if sabs[j] > sabs[j - 1] and sabs[j] > sabs[j + 1]:
                # Parabolic interpolation in log-magnitude
                alpha_val = log_sabs[j - 1]
                beta_val = log_sabs[j]
                gamma_val = log_sabs[j + 1]
                denom = alpha_val - 2 * beta_val + gamma_val
                if denom != 0:
                    p = 0.5 * (alpha_val - gamma_val) / denom
                else:
                    p = 0.0

                # Instantaneous frequency: interpolated fc
                m_idx = active[j]
                if j + 1 < len(active) and p > 0:
                    m_next = active[j + 1]
                    instf = fc[m_idx] + p * (fc[m_next] - fc[m_idx])
                elif j > 0 and p < 0:
                    m_prev = active[j - 1]
                    instf = fc[m_idx] + p * (fc[m_idx] - fc[m_prev])
                else:
                    instf = fc[m_idx]

                peaks.append((j, instf, p))

        if not peaks:
            # No peaks found — use centre frequency for all
            for m in active:
                n = current_frame[m]
                m_phase[m] = m_phase[m] + 2.0 * np.pi * a[m] * fc[m]
                phase_out[m][n] = m_phase[m]
            continue

        # Assign each active channel to nearest peak
        peak_js = [pk[0] for pk in peaks]
        peak_phases = []
        for j, instf, _p in peaks:
            m_idx = active[j]
            peak_phase = m_phase[m_idx] + 2.0 * np.pi * a[m_idx] * instf
            peak_phases.append(peak_phase)

        # Propagate: assign each channel the phase of the nearest peak
        assigned = np.zeros(len(active), dtype=bool)
        channel_phase = np.zeros(len(active))

        # First assign peak channels
        for k, (j, _instf, p) in enumerate(peaks):
            channel_phase[j] = peak_phases[k]
            assigned[j] = True

            # Propagate around peak
            if p > 0 and j + 1 < len(active):
                channel_phase[j + 1] = peak_phases[k]
                assigned[j + 1] = True
                # Go up from peak
                for jj in range(j + 2, len(active)):
                    if sabs[jj] < sabs[jj - 1]:
                        channel_phase[jj] = peak_phases[k]
                        assigned[jj] = True
                    else:
                        break
                # Go down from peak
                for jj in range(j - 1, -1, -1):
                    if not assigned[jj] and sabs[jj] < sabs[jj + 1]:
                        channel_phase[jj] = peak_phases[k]
                        assigned[jj] = True
                    else:
                        break
            elif p < 0 and j > 0:
                channel_phase[j - 1] = peak_phases[k]
                assigned[j - 1] = True
                # Go down from peak
                for jj in range(j - 2, -1, -1):
                    if not assigned[jj] and sabs[jj] < sabs[jj + 1]:
                        channel_phase[jj] = peak_phases[k]
                        assigned[jj] = True
                    else:
                        break
                # Go up from peak
                for jj in range(j + 1, len(active)):
                    if not assigned[jj] and sabs[jj] < sabs[jj - 1]:
                        channel_phase[jj] = peak_phases[k]
                        assigned[jj] = True
                    else:
                        break
            else:
                # Propagate both directions from peak
                for jj in range(j - 1, -1, -1):
                    if not assigned[jj] and sabs[jj] < sabs[jj + 1]:
                        channel_phase[jj] = peak_phases[k]
                        assigned[jj] = True
                    else:
                        break
                for jj in range(j + 1, len(active)):
                    if not assigned[jj] and sabs[jj] < sabs[jj - 1]:
                        channel_phase[jj] = peak_phases[k]
                        assigned[jj] = True
                    else:
                        break

        # Unassigned channels: use nearest peak
        for j in range(len(active)):
            if not assigned[j]:
                dists = [abs(j - pj) for pj in peak_js]
                nearest = int(np.argmin(dists))
                channel_phase[j] = peak_phases[nearest]

        # Write back
        for j, m in enumerate(active):
            m_phase[m] = channel_phase[j]
            n = current_frame[m]
            phase_out[m][n] = m_phase[m]

    # Build complex output
    c_list = []
    for m in range(M):
        c_list.append(np.abs(np.asarray(s_list[m])) * np.exp(1j * phase_out[m]))

    return c_list, phase_out
