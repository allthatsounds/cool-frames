"""numpy/core/_onset.py
======================
Shared, low-cost transient-onset detector via spectral flux (Duxbury et al.
2002). Used by the filterbank adviser
(``cool_frames.numpy.diagnostics.recommend_filterbank``) to measure onset density.
"""

from __future__ import annotations

import numpy as np


def _detect_onsets(
    f: np.ndarray,
    fs: float,
    onset_threshold: float = 0.3,
    hop_stft: int | None = None,
) -> np.ndarray:
    """Detect transient onsets via spectral flux.

    Computes a quick low-cost spectrogram using short-window FFT, then detects
    onset times as peaks in spectral flux above a threshold.

    Parameters
    ----------
    f : array_like, shape (Ls,)
        Input signal.
    fs : float
        Sampling rate (Hz).
    onset_threshold : float, optional
        Threshold for onset detection as fraction of max flux (default: 0.3).
        Larger values → fewer detections.
    hop_stft : int, optional
        Hop size for STFT (default: ~10 ms worth of samples).

    Returns
    -------
    onset_indices : (num_onsets,) int array
        Sample indices where onsets occur.

    Notes
    -----
    Spectral flux is computed as::

        flux[n] = sum(max(0, |X[n]| - |X[n-1]|))

    (half-wave rectified magnitude increase per time frame).
    """
    f = np.asarray(f, dtype=np.float64)
    Ls = len(f)

    if hop_stft is None:
        hop_stft = max(1, int(0.01 * fs))  # ~10 ms

    # Window length: aim for ~50 ms
    win_len = max(256, int(0.05 * fs))
    if win_len > Ls:
        win_len = Ls // 2

    win = np.hanning(win_len)

    # Compute STFT magnitude
    num_frames = (Ls - win_len) // hop_stft + 1
    if num_frames < 2:
        return np.array([], dtype=np.int64)

    stft_mag = np.zeros((win_len // 2 + 1, num_frames))
    for n in range(num_frames):
        start = n * hop_stft
        end = start + win_len
        if end > Ls:
            break
        frame = f[start:end] * win
        spec = np.fft.rfft(frame)
        stft_mag[:, n] = np.abs(spec)

    # Spectral flux: half-wave rectified magnitude increase
    flux = np.zeros(num_frames)
    for n in range(1, num_frames):
        flux[n] = np.sum(np.maximum(0, stft_mag[:, n] - stft_mag[:, n - 1]))

    # Normalize flux to [0, 1]
    flux_max = np.max(flux)
    if flux_max > 0:
        flux = flux / flux_max  # type: ignore[assignment]

    # Find peaks above threshold
    threshold = onset_threshold
    onset_frames = np.where(flux > threshold)[0]

    # Suppress nearby peaks (within 50 ms) and keep only local maxima
    if len(onset_frames) > 0:
        suppress_dist = max(1, int(0.05 * fs / hop_stft))
        filtered_onsets: list[int] = []
        for frame_idx in onset_frames:
            # Check if this is a local maximum
            is_local_max = True
            for other_idx in onset_frames:
                if (
                    other_idx != frame_idx
                    and abs(other_idx - frame_idx) <= suppress_dist
                    and flux[other_idx] > flux[frame_idx]
                ):
                    is_local_max = False
                    break
            if is_local_max and (
                not filtered_onsets or frame_idx - filtered_onsets[-1] > suppress_dist
            ):
                filtered_onsets.append(frame_idx)
        onset_frames = np.array(filtered_onsets, dtype=np.int64)

    # Convert frame indices back to sample indices
    onset_indices = onset_frames * hop_stft
    onset_indices = np.clip(onset_indices, 0, Ls - 1).astype(np.int64)

    return onset_indices
