"""
torch/phase/_spsi.py
====================
Single-Pass Spectrogram Inversion (SPSI) for filterbanks (PyTorch).

Port of ``numpy/phase/_spsi.py`` to PyTorch. The algorithm works in the
filterbank domain and uses spectral peaks to estimate instantaneous frequency.

This implementation operates on per-channel magnitude arrays with non-uniform
hop sizes:

    For each time step, peaks are identified across channels. For each peak,
    the instantaneous frequency is estimated via parabolic interpolation of
    log-magnitudes, and the phase is propagated from the previous frame using
    a phase accumulator. Non-peak channels inherit the phase of the nearest peak.
"""

from __future__ import annotations

import torch

from .._dtypes import resolve


def spsi(
    s_list: list[torch.Tensor],
    a: torch.Tensor | list | tuple,
    fc: torch.Tensor | list | tuple,
    fs: float,
    *,
    startphase: torch.Tensor | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Single-Pass Spectrogram Inversion for filterbanks (PyTorch).

    Single-pass phase reconstruction with local gradient estimation [spsi-beauregard]_.

    Parameters
    ----------
    s_list : list of M tensors, each (N_m,)
        Magnitude coefficients per channel.
    a : (M,) tensor or array-like — hop sizes per channel
    fc : (M,) tensor or array-like — centre frequencies **in Hz**, exactly as
        returned by ``audfilters`` and the other filterbank constructors.
    fs : float — sampling rate in Hz.  Pass ``fs=1.0`` if ``fc`` is already
        normalised (cycles per sample).  See the NumPy implementation's notes
        for why this argument exists.
    startphase : (M,) tensor, optional
        Initial phase per channel. If None, starts at zero.

    Returns
    -------
    c_list : list of M complex tensors
        Complex coefficients with reconstructed phase.
    phase_list : list of M float tensors
        Phase values per channel.

    References
    ----------
    .. [spsi-beauregard] T. Beauregard, M. Harish, and L. Wyse, "Single pass spectrogram inversion using
           local gradient estimation," IEEE ICSC, 2015.
    """
    # Determine device and dtype from inputs
    device = s_list[0].device if isinstance(s_list[0], torch.Tensor) else torch.device("cpu")
    # The caller's dtype wins; see cool_frames/torch/_dtypes.py.
    dtype, cdtype = resolve(*(s_list if isinstance(s_list, list) else [s_list]))

    # Convert inputs to tensors and normalize
    a = torch.as_tensor(a, dtype=torch.int32, device=device).reshape(-1)
    fc = torch.as_tensor(fc, dtype=dtype, device=device).reshape(-1)

    fs = float(fs)
    if not (fs > 0) or fs != fs or fs == float("inf"):
        raise ValueError(f"spsi: fs must be a positive sampling rate, got {fs!r}")
    if fc.numel() and float(torch.max(torch.abs(fc))) > 0.5 * fs * (1.0 + 1e-9):
        raise ValueError(
            f"spsi: max|fc| = {float(torch.max(torch.abs(fc))):.4g} exceeds the Nyquist "
            f"frequency fs/2 = {0.5 * fs:.4g}. `fc` is expected in Hz and `fs` in Hz; if "
            f"`fc` is already normalised (cycles per sample), pass fs=1.0."
        )
    fc = fc / fs

    M = len(a)

    # Get frame counts per channel
    N = torch.tensor([len(s) for s in s_list], dtype=torch.int32, device=device)

    # Initialize phase
    if startphase is None:
        m_phase = torch.zeros(M, dtype=dtype, device=device)
    else:
        m_phase = torch.as_tensor(startphase, dtype=dtype, device=device).reshape(-1).clone()

    # Build event schedule (time-sorted)
    events = []
    for m in range(M):
        for n in range(int(N[m].item())):
            events.append((int(n * a[m].item()), m, n))
    events.sort(key=lambda x: (x[0], x[1]))

    phase_out = [torch.zeros(int(N[m].item()), dtype=dtype, device=device) for m in range(M)]

    # Process events by time step
    i = 0
    while i < len(events):
        t_now = events[i][0]
        batch = []
        while i < len(events) and events[i][0] == t_now:
            batch.append(events[i])
            i += 1

        # Get magnitudes for channels present at this time
        current_mags = torch.full((M,), float("-inf"), dtype=dtype, device=device)
        current_frame = {}
        for _t, m, n in batch:
            current_mags[m] = torch.abs(s_list[m][n]).to(dtype=dtype)
            current_frame[m] = n

        # Only consider channels that have a frame at this time
        active = sorted(current_frame.keys())
        if len(active) < 2:
            # Single channel — just accumulate phase
            for m in active:
                n = current_frame[m]
                instf = fc[m]
                m_phase[m] = m_phase[m] + 2.0 * torch.pi * a[m] * instf
                phase_out[m][n] = m_phase[m]
            continue

        # Get magnitudes for active channels
        sabs = torch.tensor([current_mags[m].item() for m in active], dtype=dtype, device=device)
        eps = torch.finfo(dtype).tiny
        log_sabs = torch.log(sabs + eps)

        # Find peaks: channels where magnitude > both neighbours
        peaks = []
        for j in range(1, len(active) - 1):
            if sabs[j] > sabs[j - 1] and sabs[j] > sabs[j + 1]:
                # Parabolic interpolation in log-magnitude
                alpha_val = log_sabs[j - 1].item()
                beta_val = log_sabs[j].item()
                gamma_val = log_sabs[j + 1].item()
                denom = alpha_val - 2 * beta_val + gamma_val
                if abs(denom) > 1e-12:
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
                m_phase[m] = m_phase[m] + 2.0 * torch.pi * a[m] * fc[m]
                phase_out[m][n] = m_phase[m]
            continue

        # Assign each active channel to nearest peak
        peak_js = [pk[0] for pk in peaks]
        peak_phases = []
        for j, instf, _p in peaks:
            m_idx = active[j]
            peak_phase = m_phase[m_idx] + 2.0 * torch.pi * a[m_idx] * instf
            peak_phases.append(peak_phase)

        # Propagate: assign each channel the phase of the nearest peak
        assigned = torch.zeros(len(active), dtype=torch.bool, device=device)
        channel_phase = torch.zeros(len(active), dtype=dtype, device=device)

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
                nearest = int(torch.tensor(dists, dtype=dtype, device=device).argmin().item())
                channel_phase[j] = peak_phases[nearest]

        # Write back
        for j, m in enumerate(active):
            m_phase[m] = channel_phase[j]
            n = current_frame[m]
            phase_out[m][n] = m_phase[m]

    # Build complex output
    c_list = []
    for m in range(M):
        s_abs = torch.abs(s_list[m].to(dtype=dtype, device=device))
        c_list.append(s_abs * torch.exp(1j * phase_out[m].to(dtype=cdtype)))

    return c_list, phase_out
