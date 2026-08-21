"""
recommend_filterbank — signal-driven filterbank recommendation
==============================================================

Analyses an audio signal and recommends the most appropriate filterbank
designer from the cool_frames toolkit based on measured signal properties.

The recommendation considers:

* **Harmonicity**: ratio of harmonic vs. inharmonic energy → cqtfilters
  for strongly harmonic content (music, speech), audfilters for noise-like.
* **Bandwidth occupancy**: how much of the spectrum is active → narrowband
  signals benefit from CQT's log spacing, broadband from auditory scales.
* **Temporal dynamics**: onset rate, spectral flux statistics →
  high transient density suggests timeadaptivefilters.
* **Stationarity**: variance of spectral centroid over time → stationary
  signals favour longer windows (lower redundancy).

The function returns a structured recommendation with the designer name,
suggested parameters, and the analysis metrics that drove the decision.

Design note
-----------
``waveletfilters`` and ``warpedfilters`` are *methodology-driven* choices
(e.g. geophysics convention, custom warped frequency axis) and are not
recommended automatically. ``polezerofilters`` is for IIR/real-time
applications and likewise needs explicit user intent.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


@dataclasses.dataclass
class FilterbankRecommendation:
    """Result of ``recommend_filterbank``.

    The attributes are documented on the fields themselves rather than in an
    ``Attributes`` section: autodoc picks up dataclass fields *and* the section,
    and describing them twice makes Sphinx emit a "duplicate object description"
    warning for every one of them.
    """

    #: Recommended filter design function name
    #: (e.g. ``"audfilters"``, ``"cqtfilters"``, ``"timeadaptivefilters"``).
    designer: str

    #: Suggested keyword arguments for the designer.
    params: dict[str, Any]

    #: Human-readable explanation of why this designer was chosen.
    rationale: str

    #: Signal analysis metrics that informed the recommendation.
    metrics: dict[str, float | None]

    #: Other viable designers with brief reasoning.
    alternatives: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Signal analysis helpers
# ---------------------------------------------------------------------------


def _spectral_centroid(mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Per-frame spectral centroid (Hz)."""
    power = mag ** 2
    total = power.sum(axis=0) + 1e-30
    return (freqs[:, None] * power).sum(axis=0) / total  # type: ignore[no-any-return]


def _spectral_flux(mag: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux."""
    diff = np.diff(mag, axis=1)
    return np.sum(np.maximum(0, diff), axis=0)  # type: ignore[no-any-return]


def _energy_frequency_range(
    f: np.ndarray, fs: float, threshold_db: float = -40.0,
) -> tuple[float, float]:
    """Estimate the frequency range containing significant energy.

    Parameters
    ----------
    f : (Ls,) array
        Input signal.
    fs : float
        Sampling rate (Hz).
    threshold_db : float
        Energy threshold below peak (dB) for defining the active band.

    Returns
    -------
    fmin_est, fmax_est : float
        Lower and upper frequency bounds of the active band (Hz).
    """
    win_len = min(len(f), 8192)
    centre = len(f) // 2
    segment = f[max(0, centre - win_len // 2) : max(0, centre - win_len // 2) + win_len]
    spec = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(win_len, 1.0 / fs)

    spec_db = 20 * np.log10(spec / (spec.max() + 1e-30) + 1e-30)
    active = spec_db > threshold_db

    if not np.any(active):
        return 20.0, fs / 2

    active_freqs = freqs[active]
    return float(max(20.0, active_freqs[0])), float(min(fs / 2, active_freqs[-1]))


def _estimate_f0(f: np.ndarray, fs: float) -> float | None:
    """Estimate fundamental frequency via autocorrelation.

    Returns None if signal is not clearly harmonic.
    """
    win_len = min(len(f), max(2048, int(0.05 * fs)))
    centre = len(f) // 2
    start = max(0, centre - win_len // 2)
    segment = f[start : start + win_len].copy()
    segment = segment - segment.mean()
    norm = np.dot(segment, segment)
    if norm < 1e-30:
        return None

    n_fft = 2 * win_len
    F = np.fft.rfft(segment, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:win_len]
    acf = acf / (norm + 1e-30)

    # Search lags corresponding to 40 Hz – 2000 Hz
    min_lag = max(1, int(fs / 2000))
    max_lag = min(win_len // 2, int(fs / 40))
    if min_lag >= max_lag:
        return None

    region = acf[min_lag:max_lag]
    if np.max(region) < 0.4:  # not clearly periodic
        return None

    # Find the first prominent peak (not the global max, which can be
    # a sub-period alias for pure tones).  A peak is where the ACF goes
    # from rising to falling and exceeds the threshold.
    threshold = 0.4
    peak_lag = None
    for i in range(1, len(region) - 1):
        if region[i] > region[i - 1] and region[i] >= region[i + 1]:
            if region[i] >= threshold:
                peak_lag = min_lag + i
                break

    if peak_lag is None:
        # No prominent peak found — check if global max exceeds threshold
        best_idx = int(np.argmax(region))
        if region[best_idx] >= threshold:
            peak_lag = min_lag + best_idx
        else:
            return None

    return float(fs / peak_lag)  # type: ignore[operator]


def _harmonicity_ratio(f: np.ndarray, fs: float) -> float:
    """Estimate harmonicity as normalized autocorrelation peak in pitch range.

    A strongly harmonic signal has high autocorrelation peaks at pitch lag.
    Returns a value in [0, 1] where 1 = perfectly periodic.
    """
    # Use a ~50 ms window from the signal's centre
    win_len = min(len(f), max(2048, int(0.05 * fs)))
    centre = len(f) // 2
    start = max(0, centre - win_len // 2)
    segment = f[start : start + win_len].copy()

    # Remove DC and normalize
    segment = segment - segment.mean()
    norm = np.dot(segment, segment)
    if norm < 1e-30:
        return 0.0

    # Compute full autocorrelation via FFT (much faster)
    n_fft = 2 * win_len
    F = np.fft.rfft(segment, n=n_fft)
    acf = np.fft.irfft(F * np.conj(F), n=n_fft)[:win_len]
    acf = acf / (norm + 1e-30)  # Normalize so acf[0] ≈ 1

    # Search for peaks between 1 ms and 25 ms (40 Hz – 1000 Hz)
    min_lag = max(1, int(0.001 * fs))
    max_lag = min(win_len // 2, int(0.025 * fs))
    if min_lag >= max_lag:
        return 0.0

    peak = np.max(acf[min_lag:max_lag])
    return float(np.clip(peak, 0, 1))


def _onset_rate(f: np.ndarray, fs: float) -> float:
    """Estimate onset rate (onsets/second) via spectral flux peaks."""
    from cool_frames.numpy.core._onset import _detect_onsets

    onsets = _detect_onsets(f, fs, onset_threshold=0.5)
    duration = len(f) / fs
    if duration < 0.01:
        return 0.0
    return len(onsets) / duration


def _bandwidth_occupancy(f: np.ndarray, fs: float) -> float:
    """Fraction of frequency bins containing significant energy.

    Returns value in [0, 1] where 1 = energy spread across full spectrum.
    """
    win_len = min(len(f), 4096)
    centre = len(f) // 2
    segment = f[max(0, centre - win_len // 2) : max(0, centre - win_len // 2) + win_len]
    spec = np.abs(np.fft.rfft(segment))
    spec_db = 20 * np.log10(spec / (spec.max() + 1e-30) + 1e-30)
    # Count bins within 40 dB of peak
    active = np.sum(spec_db > -40)
    return float(active / len(spec_db))


def _stationarity(f: np.ndarray, fs: float) -> float:
    """Measure stationarity as inverse coefficient of variation of spectral centroid.

    Returns value in [0, 1] where 1 = perfectly stationary.
    """
    hop = max(1, int(0.01 * fs))
    win_len = max(256, int(0.03 * fs))
    if len(f) < win_len + hop:
        return 1.0

    num_frames = (len(f) - win_len) // hop
    if num_frames < 3:
        return 1.0

    freqs = np.fft.rfftfreq(win_len, 1.0 / fs)
    centroids = np.zeros(num_frames)
    for n in range(num_frames):
        start = n * hop
        frame = f[start : start + win_len]
        mag = np.abs(np.fft.rfft(frame))
        power = mag ** 2
        total = power.sum() + 1e-30
        centroids[n] = (freqs * power).sum() / total

    mean_c = centroids.mean()
    std_c = centroids.std()
    if mean_c < 1e-10:
        return 1.0
    cv = std_c / mean_c  # coefficient of variation
    # Map CV to [0, 1]: CV=0 → 1.0, CV≥2 → 0.0
    return float(np.clip(1.0 - cv / 2.0, 0, 1))


# ---------------------------------------------------------------------------
# Main recommendation function
# ---------------------------------------------------------------------------


def recommend_filterbank(
    f: np.ndarray,
    fs: float,
    *,
    purpose: str = "analysis",
) -> FilterbankRecommendation:
    """Analyse a signal and recommend the best filterbank designer.

    Parameters
    ----------
    f : array_like, shape (Ls,)
        Input audio signal.
    fs : float
        Sampling rate (Hz).
    purpose : str, optional
        Intended use case. One of:

        - ``"analysis"`` (default): visualization, feature extraction
        - ``"modification"``: audio effects, denoising, EQ
        - ``"phase_retrieval"``: magnitude-to-audio reconstruction
        - ``"real_time"``: streaming / low-latency applications

    Returns
    -------
    rec : FilterbankRecommendation
        Structured recommendation with designer name, suggested parameters,
        rationale, signal metrics, and alternative choices.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.diagnostics import recommend_filterbank
    >>> fs = 16000
    >>> t = np.arange(fs * 2) / fs
    >>> x = np.sin(2 * np.pi * 440 * t)  # pure tone
    >>> rec = recommend_filterbank(x, fs)
    >>> print(rec.designer)
    cqtfilters
    >>> print(rec.rationale)
    ...

    Notes
    -----
    This function does **not** recommend ``waveletfilters``,
    ``warpedfilters``, or ``polezerofilters``, as these are
    methodology-driven or application-specific choices that require
    explicit user intent. See the filterbank decision tree documentation
    for guidance on when to use these designers.
    """
    f = np.asarray(f, dtype=np.float64)
    Ls = len(f)
    fs = float(fs)

    # ------------------------------------------------------------------
    # 1. Compute signal metrics
    # ------------------------------------------------------------------
    harmonicity = _harmonicity_ratio(f, fs)
    onset_rate = _onset_rate(f, fs)
    bw_occupancy = _bandwidth_occupancy(f, fs)
    stationarity_val = _stationarity(f, fs)
    fmin_est, fmax_est = _energy_frequency_range(f, fs)
    f0_est = _estimate_f0(f, fs)

    metrics = {
        "harmonicity": round(harmonicity, 3),
        "onset_rate_per_sec": round(onset_rate, 2),
        "bandwidth_occupancy": round(bw_occupancy, 3),
        "stationarity": round(stationarity_val, 3),
        "duration_sec": round(Ls / fs, 3),
        "fs": fs,
        "fmin_hz": round(fmin_est, 1),
        "fmax_hz": round(fmax_est, 1),
        "f0_hz": round(f0_est, 1) if f0_est is not None else None,
    }

    # ------------------------------------------------------------------
    # 2. Decision logic
    # ------------------------------------------------------------------
    alternatives = []

    # Strongly harmonic → CQT (log-frequency for pitch)
    is_harmonic = harmonicity > 0.4

    # High transient density: only if NOT harmonic and NOT stationary
    # (avoids false positives from beating in chords or broadband noise)
    is_highly_transient = (
        onset_rate > 8.0
        and not is_harmonic
        and stationarity_val < 0.8
    )

    # Non-stationary → needs adaptive or higher redundancy
    is_nonstationary = stationarity_val < 0.5

    # ------------------------------------------------------------------
    # 3. Signal-driven fmin / fmax
    # ------------------------------------------------------------------
    # Snap estimated fmin down and fmax up to "nice" values so the user
    # gets round numbers rather than e.g. 73.2 Hz.  Clamp to audible range.
    _nice_fmin = max(20.0, float(10 * int(fmin_est / 10)))
    _nice_fmax = min(fs / 2, float(100 * int(np.ceil(fmax_est / 100))))

    # CQT bins-per-octave from f0 if available: choose bins so that the
    # harmonic spacing is resolved (at least 2 bins between consecutive
    # harmonics at the fundamental).
    _cqt_bins = 48  # default fine resolution
    if f0_est is not None and f0_est > 0:
        # Interval between consecutive harmonics at f0 is
        # log2(2/1)=1 octave at f0, but log2((n+1)/n) ≈ 1/(n·ln2)
        # octaves for the n-th harmonic.  At harmonic n=4 the spacing
        # is ~0.32 octaves. To have ≥2 bins in that interval:
        # bins_per_oct ≥ 2 / 0.32 ≈ 6.  For n=8: ≥ 2/(1/(8·ln2)) ≈ 11.
        # We pick enough resolution to separate up to the 8th harmonic.
        # Until v0.1.1 this line read
        #     max(12, min(96, ceil(8 * log(2) * 2)))
        # which contains no `f0` at all and evaluates to 12 for every input —
        # so detecting an f0 made the resolution *coarser* than the no-f0
        # default of 48, and the `f0_est > 500` branch below could never change
        # anything (min(12, 24) == 12).
        #
        # Resolving harmonics up to the n-th needs bins_per_octave >= 2*n*ln2
        # (the spacing between harmonics n and n+1 is ~1/(n*ln2) octaves, and
        # we want two bins across it).  Ask for the 8th harmonic, but only as
        # far as the octave actually reaches: a high f0 has fewer harmonics
        # below Nyquist and needs less resolution.
        _n_harm = 8.0
        _cqt_bins = int(np.ceil(2.0 * _n_harm * np.log(2.0)))
        _cqt_bins = max(12, min(96, _cqt_bins))
        # A very low f0 packs its harmonics closer in log-frequency, so give it
        # more resolution; a high f0 needs less.
        if f0_est < 100:
            _cqt_bins = max(_cqt_bins, 48)
        elif f0_est > 500:
            _cqt_bins = min(_cqt_bins, 24)

    # ------------------------------------------------------------------
    # 4. Decision logic
    # ------------------------------------------------------------------

    # Purpose-specific adjustments
    if purpose == "real_time":
        # Real-time: avoid time-adaptive (needs full signal), prefer audfilters
        designer = "audfilters"
        params: dict[str, Any] = {
            "scale": "erb",
            "fmin": _nice_fmin,
            "fmax": _nice_fmax,
        }
        rationale = (
            "For real-time applications, audfilters with ERB scale provides "
            "the best balance of frequency resolution and computational "
            f"efficiency with fixed-size frames. Signal energy spans "
            f"{_nice_fmin:.0f}–{_nice_fmax:.0f} Hz."
        )
        if is_harmonic:
            alternatives.append({
                "designer": "cqtfilters",
                "reason": f"Signal is harmonic (harmonicity={harmonicity:.2f}), "
                          "but CQT's variable hop complicates streaming.",
            })
        alternatives.append({
            "designer": "polezerofilters",
            "reason": "IIR filters are naturally suited to sample-by-sample "
                      "real-time processing with minimal latency.",
        })

    elif is_highly_transient and purpose != "phase_retrieval":
        designer = "timeadaptivefilters"
        params = {
            "scale": "erb",
            "onset_threshold": 0.3,
            "transient_bwmul": 2.0,
        }
        if _nice_fmin > 30:
            params["fmin"] = _nice_fmin
        if _nice_fmax < fs / 2 - 100:
            params["fmax"] = _nice_fmax
        rationale = (
            f"High onset rate ({onset_rate:.1f}/s) indicates frequent transients. "
            "Time-adaptive filterbank varies resolution across time: wider filters "
            "at transients for temporal precision, narrower at sustained regions "
            "for frequency detail."
        )
        alternatives.append({
            "designer": "audfilters",
            "reason": "Standard auditory filterbank with higher redundancy "
                      "can also handle transients reasonably well.",
        })

    elif is_harmonic:
        # Use f0 to set fmin: one octave below f0 or the estimated fmin
        _cqt_fmin = _nice_fmin
        if f0_est is not None:
            _cqt_fmin = max(20.0, float(10 * int(f0_est / 2 / 10)))

        if bw_occupancy < 0.3:
            # Narrowband harmonic → CQT with fine resolution
            designer = "cqtfilters"
            params = {
                "fmin": _cqt_fmin,
                "fmax": _nice_fmax,
                "bins": max(_cqt_bins, 36),
            }
            f0_note = ""
            if f0_est is not None:
                f0_note = f" Estimated f0={f0_est:.0f} Hz."
            rationale = (
                f"Strongly harmonic signal (harmonicity={harmonicity:.2f}) with "
                f"narrow bandwidth ({bw_occupancy:.0%} occupied).{f0_note} CQT "
                f"with {params['bins']} bins/octave over "
                f"{params['fmin']:.0f}–{params['fmax']:.0f} Hz provides "
                "logarithmic resolution ideal for resolving harmonics."
            )
        else:
            # Broadband harmonic → CQT with standard bins
            designer = "cqtfilters"
            params = {
                "fmin": _cqt_fmin,
                "fmax": _nice_fmax,
                "bins": min(_cqt_bins, 24),
            }
            rationale = (
                f"Harmonic signal (harmonicity={harmonicity:.2f}) with broad "
                f"bandwidth ({bw_occupancy:.0%}). CQT with {params['bins']} "
                f"bins/octave over {params['fmin']:.0f}–{params['fmax']:.0f} Hz "
                "balances pitch resolution with computational cost."
            )
        alternatives.append({
            "designer": "audfilters",
            "reason": "ERB-spaced filters also work well for harmonic content "
                      "and offer tighter frame bounds.",
        })

    else:
        # Default: audfilters with ERB or Bark scale
        designer = "audfilters"
        scale = "erb"
        if bw_occupancy > 0.6:
            scale = "bark"
            rationale = (
                f"Broadband signal ({bw_occupancy:.0%} bandwidth occupied) "
                "with low harmonicity. Bark scale provides perceptually "
                "uniform frequency resolution suited for noise-like or "
                "environmental sounds."
            )
        else:
            rationale = (
                f"Signal with moderate bandwidth ({bw_occupancy:.0%}) and "
                f"low harmonicity ({harmonicity:.2f}). ERB scale provides "
                "the most general-purpose auditory frequency resolution."
            )
        params = {
            "scale": scale,
            "fmin": _nice_fmin,
            "fmax": _nice_fmax,
        }
        rationale += (
            f" Frequency range narrowed to {_nice_fmin:.0f}–{_nice_fmax:.0f} Hz "
            "based on energy analysis."
        )
        alternatives.append({
            "designer": "cqtfilters",
            "reason": "CQT can be useful if subsequent analysis is pitch-based.",
        })

    # Add stationarity-based parameter adjustments
    if is_nonstationary and designer in ("audfilters", "cqtfilters"):
        params["redmul"] = 2.0
        rationale += (
            f" Non-stationary content (stationarity={stationarity_val:.2f}) → "
            "redundancy increased to 2× for better temporal tracking."
        )

    # Purpose-specific parameter tweaks
    if purpose == "modification":
        if "redmul" not in params:
            params["redmul"] = 2.0
        rationale += (
            " For modification tasks, higher redundancy improves "
            "robustness to coefficient manipulation."
        )

    elif purpose == "phase_retrieval":
        params["redmul"] = params.get("redmul", 4.0)
        rationale += (
            " Phase retrieval benefits from high redundancy (4×) "
            "for faster convergence and lower artifacts."
        )
        if designer == "timeadaptivefilters":
            # Phase retrieval doesn't work well with segmented approach
            designer = "audfilters"
            params = {
                "scale": "erb",
                "redmul": 4.0,
                "fmin": _nice_fmin,
                "fmax": _nice_fmax,
            }
            rationale = (
                "Phase retrieval requires a single consistent filterbank. "
                "Audfilters with high redundancy (4×) provides the best "
                f"frame properties for iterative phase reconstruction "
                f"({_nice_fmin:.0f}–{_nice_fmax:.0f} Hz)."
            )

    # Always add timeadaptive as an alternative if not already chosen
    if designer != "timeadaptivefilters" and onset_rate > 3.0:
        alternatives.append({
            "designer": "timeadaptivefilters",
            "reason": f"Moderate onset rate ({onset_rate:.1f}/s) could "
                      "benefit from time-varying resolution.",
        })

    return FilterbankRecommendation(
        designer=designer,
        params=params,
        rationale=rationale,
        metrics=metrics,
        alternatives=alternatives,
    )


__all__ = ["FilterbankRecommendation", "recommend_filterbank"]
