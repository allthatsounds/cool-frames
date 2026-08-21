"""phase._ridge — Ridge extraction from phase gradients (PyTorch).

Extracts connected ridges (IF trajectories / sinusoidal partials) from
filterbank phase gradients and magnitudes. Each ridge tracks a single
signal component through the time-frequency plane.

The algorithm:
1. At each time frame, find magnitude peaks across channels.
2. Link peaks across adjacent frames using IF continuity
   (greedy nearest-frequency matching).
3. Prune ridges shorter than a minimum duration.

This provides the input for ``ridges_to_symbol`` in the operators module,
which constructs frame-multiplier symbols centered on the extracted
ridges — enabling automatic, reassignment-guided denoising.

Note: Ridge extraction is inherently non-differentiable (sequential peak tracking),
but we use torch tensors for all storage and computation where possible.

See MATH_REFERENCE.md §15a (frame multipliers).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Ridge:
    """A single extracted IF ridge / sinusoidal partial.

    Attributes
    ----------
    channel_indices : torch.Tensor of int64
        Channel index at each time step along the ridge.
    time_indices : torch.Tensor of int64
        Coefficient time-frame index at each step.
    inst_freq : torch.Tensor of float32
        Instantaneous frequency (normalised, from tgrad) at each step.
    amplitude : torch.Tensor of float32
        Magnitude envelope along the ridge.
    bandwidth : torch.Tensor of float32 or None
        Local bandwidth estimate at each step (from ``ff`` second
        derivative, if available). None if not computed.
    onset : int
        First time index of the ridge.
    offset : int
        Last time index of the ridge.
    """

    channel_indices: torch.Tensor
    time_indices: torch.Tensor
    inst_freq: torch.Tensor
    amplitude: torch.Tensor
    bandwidth: torch.Tensor | None = None
    onset: int = 0
    offset: int = 0

    def duration_hops(self) -> int:
        """Number of time frames the ridge spans."""
        return self.offset - self.onset + 1

    def mean_channel(self) -> float:
        """Mean channel index (useful for sorting by frequency)."""
        return float(torch.mean(self.channel_indices.float()).item())


def segment_ridge(
    ridge: Ridge,
    *,
    max_segment_hops: int = 30,
    chirp_threshold: float | None = None,
    bw_change_threshold: float | None = None,
    tt: list[torch.Tensor] | None = None,
    a: np.ndarray | None = None,
    a_min: int | None = None,
    min_segment_hops: int = 3,
) -> list[Ridge]:
    """Split a ridge into short segments at stationarity boundaries.

    A segment boundary is placed where any of these conditions hold:

    - The segment exceeds ``max_segment_hops``.
    - The local chirp rate (from ``tt``) exceeds ``chirp_threshold``
      (rapid frequency sweep → the multiplier shape should change).
    - The bandwidth changes by more than ``bw_change_threshold``
      between adjacent points.

    The segments overlap by one point at each boundary to ensure
    smooth overlap-add in the symbol construction.

    Parameters
    ----------
    ridge : Ridge
        The ridge to segment.
    max_segment_hops : int
        Maximum length of a single segment (default 30 frames).
    chirp_threshold : float or None
        If set, split where ``|chirp_rate|`` exceeds this value. The chirp
        rate is read from the ``tt`` second derivative at the ridge
        points. Requires ``tt``, ``a``, and ``a_min``.
    bw_change_threshold : float or None
        If set, split where the bandwidth changes by more than this
        factor between adjacent points (requires ridge.bandwidth).
    tt : list of M torch.Tensor or None
        Second time-derivative (chirp rate) from ``filterbankphasederiv``.
    a : ndarray or None
        Hop sizes per channel.
    a_min : int or None
        Minimum hop size (for time-index mapping).
    min_segment_hops : int
        Minimum segment length; shorter segments are merged with
        their neighbour.

    Returns
    -------
    segments : list of Ridge
        Sub-ridges, each a contiguous piece of the original ridge.
    """
    K = len(ridge.time_indices)
    if min_segment_hops >= K:
        return [ridge]

    # --- Detect split points ---
    splits: list[int] = []  # indices into the ridge arrays

    for k in range(1, K):
        should_split = False

        # Max length
        if (k - (splits[-1] + 1 if splits else 0)) >= max_segment_hops:
            should_split = True

        # Chirp rate
        if chirp_threshold is not None and tt is not None and a is not None and a_min is not None:
            m = int(ridge.channel_indices[k].item())
            t_common = int(ridge.time_indices[k].item())
            n_m = int(round(t_common * a_min / a[m]))
            if n_m < len(tt[m]):
                cr = abs(float(tt[m][n_m].item()))
                if cr > chirp_threshold:
                    should_split = True

        # Bandwidth change
        if bw_change_threshold is not None and ridge.bandwidth is not None:
            bw_prev = float(ridge.bandwidth[k - 1].item())
            bw_curr = float(ridge.bandwidth[k].item())
            if bw_prev > 0:
                ratio = bw_curr / bw_prev
                if ratio > (1 + bw_change_threshold) or ratio < 1 / (1 + bw_change_threshold):
                    should_split = True

        if should_split:
            splits.append(k)

    # --- Build segments from split points ---
    if not splits:
        return [ridge]

    segments = []
    boundaries = [0, *splits, K]
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start < min_segment_hops:
            # Too short — will be merged below
            continue

        seg = Ridge(
            channel_indices=ridge.channel_indices[start:end].clone(),
            time_indices=ridge.time_indices[start:end].clone(),
            inst_freq=ridge.inst_freq[start:end].clone(),
            amplitude=ridge.amplitude[start:end].clone(),
            bandwidth=(
                ridge.bandwidth[start:end].clone() if ridge.bandwidth is not None else None
            ),
            onset=int(ridge.time_indices[start].item()),
            offset=int(ridge.time_indices[end - 1].item()),
        )
        segments.append(seg)

    # If all segments were too short, return the original
    if not segments:
        return [ridge]

    return segments


def segment_ridges(
    ridges: list[Ridge],
    **kwargs,
) -> list[Ridge]:
    """Segment all ridges in a list.

    Parameters
    ----------
    ridges : list of Ridge
        Ridges to segment.
    **kwargs
        Passed to ``segment_ridge``.

    Returns
    -------
    segments : list of Ridge
        All segments from all ridges, sorted by onset.
    """
    segments = []
    for ridge in ridges:
        segments.extend(segment_ridge(ridge, **kwargs))
    segments.sort(key=lambda r: (r.onset, r.mean_channel()))
    return segments


def extract_ridges(
    tgrad: list[torch.Tensor],
    s: list[torch.Tensor],
    fc: torch.Tensor | np.ndarray,
    a: np.ndarray,
    L: int,
    *,
    mag_threshold_db: float = -40.0,
    min_ridge_hops: int = 3,
    max_gap: int = 1,
    freq_tolerance: float | None = None,
    ff: list[torch.Tensor] | None = None,
) -> list[Ridge]:
    """Extract ridges from filterbank phase gradients.

    Parameters
    ----------
    tgrad : list of M torch.Tensor
        Normalised instantaneous frequency per channel (from
        ``filterbankphasegrad``). Values in [-2, 2]; actual frequency
        is ``fc[m] + tgrad[m] * fs / 2`` (but we use the normalised
        form for channel-local IF deviation).
    s : list of M torch.Tensor
        Spectrogram (squared magnitude) per channel.
    fc : torch.Tensor or ndarray, shape (M,)
        Centre frequencies in Hz (or normalised).
    a : ndarray, shape (M,)
        Hop sizes per channel.
    L : int
        Transform length.
    mag_threshold_db : float
        Peaks below this dB threshold (relative to global max) are
        ignored. Default -40 dB.
    min_ridge_hops : int
        Minimum ridge length in time frames. Shorter ridges are pruned.
    max_gap : int
        Maximum number of consecutive missing frames before a ridge is
        terminated. Default 1 (allow single-frame gaps).
    freq_tolerance : float or None
        Maximum normalised frequency deviation for linking peaks across
        frames. If None, uses 2/M (two channel spacing) as default.
    ff : list of M torch.Tensor or None
        Second frequency derivative (group delay dispersion, from
        ``filterbankphasederiv``). If provided, used to estimate local
        bandwidth on each ridge point.

    Returns
    -------
    ridges : list of Ridge
        Extracted ridges, sorted by onset time then by mean channel.
    """
    M = len(tgrad)
    device = tgrad[0].device
    dtype_float = torch.float32

    fc = torch.as_tensor(fc, dtype=dtype_float, device=device)
    a = np.asarray(a)

    if freq_tolerance is None:
        freq_tolerance = 2.0 / M  # normalised IF units

    # --- Step 0: Build uniform-time representation ---
    # Each channel has different time resolution (N_m = L / a_m).
    # We work in "channel time frames" per channel, so peaks at frame n
    # in channel m correspond to physical time t = n * a[m] / L.
    # For linking, we resample to a common time grid.
    N = [len(tgrad[m]) for m in range(M)]
    # Common time grid: finest resolution
    a_min = int(np.min(a))
    N_common = L // a_min

    # --- Step 1: Find peaks at each common time frame ---
    # Compute global magnitude threshold
    all_s_np = np.concatenate([torch.abs(s[m]).detach().cpu().numpy().ravel() for m in range(M)])
    s_max = float(np.max(all_s_np)) if len(all_s_np) > 0 else 1.0
    mag_floor = s_max * 10.0 ** (mag_threshold_db / 10.0)

    # For each common time index, gather (channel, magnitude, tgrad) tuples
    # A "peak" is a channel that is a local maximum in magnitude among
    # its neighbours (m-1, m, m+1) at the same time.
    peaks_by_time: list[list[tuple[int, float, float]]] = []

    for t_common in range(N_common):
        frame_peaks = []
        for m in range(M):
            # Map common time to channel-local time index
            n_m = int(round(t_common * a_min / a[m]))
            if n_m >= N[m]:
                continue

            s_val = float(torch.abs(s[m][n_m]).item())
            if s_val < mag_floor:
                continue

            # Check if local max in frequency direction
            is_peak = True
            for dm in (-1, 1):
                m2 = m + dm
                if 0 <= m2 < M:
                    n_m2 = int(round(t_common * a_min / a[m2]))
                    if n_m2 < N[m2] and float(torch.abs(s[m2][n_m2]).item()) > s_val:
                        is_peak = False
                        break

            if is_peak:
                tg_val = float(tgrad[m][n_m].item())
                frame_peaks.append((m, s_val, tg_val))

        peaks_by_time.append(frame_peaks)

    # --- Step 2: Link peaks across time (greedy nearest-IF matching) ---
    active_ridges: list[_ActiveRidge] = []
    finished_ridges: list[_ActiveRidge] = []

    for t_common, peaks in enumerate(peaks_by_time):
        used_peaks = set()
        used_ridges = set()

        # Sort active ridges by amplitude (strongest first for priority)
        ridge_order = sorted(
            range(len(active_ridges)),
            key=lambda i: -active_ridges[i].last_amp,
        )

        # For each active ridge, find the best matching peak
        for ri in ridge_order:
            ar = active_ridges[ri]
            best_dist = freq_tolerance
            best_pi = -1

            for pi, (m, _s_val, tg_val) in enumerate(peaks):
                if pi in used_peaks:
                    continue
                # Distance: channel distance + IF deviation
                ch_dist = abs(m - ar.last_channel) / max(M, 1)
                if_dist = abs(tg_val - ar.last_tgrad)
                dist = ch_dist + if_dist
                if dist < best_dist:
                    best_dist = dist
                    best_pi = pi

            if best_pi >= 0:
                m, s_val, tg_val = peaks[best_pi]
                ar.append(t_common, m, s_val, tg_val)
                used_peaks.add(best_pi)
                used_ridges.add(ri)

        # Check for ridge gaps / termination
        for ri in range(len(active_ridges) - 1, -1, -1):
            ar = active_ridges[ri]
            gap = t_common - ar.last_time
            if gap > max_gap and ri not in used_ridges:
                finished_ridges.append(active_ridges.pop(ri))

        # Start new ridges from unmatched peaks
        for pi, (m, s_val, tg_val) in enumerate(peaks):
            if pi not in used_peaks:
                ar = _ActiveRidge()
                ar.append(t_common, m, s_val, tg_val)
                active_ridges.append(ar)

    # Flush remaining active ridges
    finished_ridges.extend(active_ridges)

    # --- Step 3: Convert to Ridge objects, prune short ones ---
    ridges = []
    for ar in finished_ridges:
        if len(ar.times) < min_ridge_hops:
            continue

        ch_idx = torch.tensor(ar.channels, dtype=torch.int64, device=device)
        t_idx = torch.tensor(ar.times, dtype=torch.int64, device=device)
        inst_f = torch.tensor(ar.tgrads, dtype=dtype_float, device=device)
        amp = torch.sqrt(
            torch.tensor(ar.amps, dtype=dtype_float, device=device)
        )  # s is |c|², take sqrt

        # Bandwidth from ff if available
        bw = None
        if ff is not None:
            bw_vals = []
            for k in range(len(t_idx)):
                m = int(ch_idx[k].item())
                n_m = int(round(int(t_idx[k].item()) * a_min / a[m]))
                if n_m < len(ff[m]):
                    # ff is d²φ/df² — local bandwidth ∝ 1/sqrt(|ff|)
                    ff_val = abs(float(ff[m][n_m].item()))
                    bw_vals.append(1.0 / max(np.sqrt(ff_val), 1e-10))
                else:
                    bw_vals.append(0.0)
            bw = torch.tensor(bw_vals, dtype=dtype_float, device=device)

        ridge = Ridge(
            channel_indices=ch_idx,
            time_indices=t_idx,
            inst_freq=inst_f,
            amplitude=amp,
            bandwidth=bw,
            onset=int(t_idx[0].item()),
            offset=int(t_idx[-1].item()),
        )
        ridges.append(ridge)

    # Sort by onset, then by mean channel (frequency)
    ridges.sort(key=lambda r: (r.onset, r.mean_channel()))
    return ridges


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


class _ActiveRidge:
    """Mutable accumulator for a ridge being built."""

    __slots__ = (
        "amps",
        "channels",
        "last_amp",
        "last_channel",
        "last_tgrad",
        "last_time",
        "tgrads",
        "times",
    )

    def __init__(self):
        self.times: list[int] = []
        self.channels: list[int] = []
        self.amps: list[float] = []
        self.tgrads: list[float] = []
        self.last_time = -1
        self.last_channel = 0
        self.last_amp = 0.0
        self.last_tgrad = 0.0

    def append(self, t: int, m: int, amp: float, tg: float) -> None:
        self.times.append(t)
        self.channels.append(m)
        self.amps.append(amp)
        self.tgrads.append(tg)
        self.last_time = t
        self.last_channel = m
        self.last_amp = amp
        self.last_tgrad = tg
