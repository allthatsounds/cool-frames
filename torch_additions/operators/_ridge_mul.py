"""operators._ridge_mul — Fit frame-multiplier symbols to IF ridges (PyTorch).

Given ridges extracted from ``extract_ridges`` (in phase._ridge), this
module constructs frame-multiplier symbols that isolate specific signal
components.  The symbol is ≈ 1 near the ridge and smoothly tapers to 0
away from it, enabling targeted denoising or component extraction.

The full pipeline is:

    analyse → phase gradients → extract ridges → fit multiplier →
    apply → synthesise

See MATH_REFERENCE.md §15a (frame multipliers) and PORTING_STATUS.md
§ MulAcLab Port.
"""

from __future__ import annotations

import numpy as np
import torch
from cool_frames.torch.filterbanks._core import filterbank, ifilterbank
from cool_frames.torch.filterbanks._frame import filterbankdual
from cool_frames.torch.phase._phasederiv import filterbankphasederiv
from cool_frames.torch.phase._phasegrad import filterbankphasegrad

from ..phase._ridge import Ridge, extract_ridges, segment_ridges


def ridges_to_symbol(
    ridges: list[Ridge],
    coeff_lengths: list[int],
    a: np.ndarray,
    L: int,
    *,
    bandwidth: float | None = None,
    taper_width: int = 3,
    combine: str = "max",
) -> list[torch.Tensor]:
    """Construct a frame-multiplier symbol from extracted ridges.

    For each ridge, creates a per-channel Gaussian-like mask centered
    on the ridge's channel trajectory, then combines multiple ridge
    masks into a single symbol.

    Parameters
    ----------
    ridges : list of Ridge
        Ridges to include in the symbol.
    coeff_lengths : list of M ints
        Number of coefficients per channel.
    a : ndarray, shape (M,)
        Hop sizes per channel.
    L : int
        Transform length.
    bandwidth : float or None
        Fixed half-bandwidth in channels. If None, uses the per-point
        bandwidth from the ridge (if available) or defaults to 2
        channels.
    taper_width : int
        Width of the smooth onset/offset taper in time frames.
    combine : str
        How to combine multiple ridges:
        ``'max'``  — take the maximum across ridges (union-like)
        ``'sum'``  — sum (can exceed 1, useful for additive decomposition)
        ``'product'`` — multiply (intersection-like)

    Returns
    -------
    sigma : list of M Tensors
        Symbol in coefficient format. Values in [0, 1] for 'max' mode.
    """
    M = len(coeff_lengths)
    a = np.asarray(a)
    a_min = int(np.min(a))

    if combine == "product":
        sigma = [torch.ones(coeff_lengths[m], dtype=torch.float32) for m in range(M)]
    else:
        sigma = [torch.zeros(coeff_lengths[m], dtype=torch.float32) for m in range(M)]

    for ridge in ridges:
        ridge_sigma = _single_ridge_symbol(
            ridge,
            coeff_lengths,
            a,
            L,
            a_min,
            bandwidth=bandwidth,
            taper_width=taper_width,
        )

        if combine == "max":
            sigma = [
                torch.clamp(torch.maximum(sigma[m], ridge_sigma[m]), min=0.0) for m in range(M)
            ]
        elif combine == "sum":
            sigma = [sigma[m] + ridge_sigma[m] for m in range(M)]
        elif combine == "product":
            sigma = [sigma[m] * ridge_sigma[m] for m in range(M)]
        else:
            raise ValueError(f"Unknown combine mode: {combine!r}")

    return sigma


def _single_ridge_symbol(
    ridge: Ridge,
    coeff_lengths: list[int],
    a: np.ndarray,
    L: int,
    a_min: int,
    *,
    bandwidth: float | None = None,
    taper_width: int = 3,
) -> list[torch.Tensor]:
    """Build a symbol for a single ridge.

    For each point on the ridge at common time t:
      - The ridge is at channel m_ridge(t)
      - The symbol has a Gaussian profile across channels centered at m_ridge(t)
        with half-width = bandwidth
      - Onset/offset are smoothly tapered

    Returns list of M per-channel symbol arrays as torch tensors.
    """
    M = len(coeff_lengths)
    # Use numpy arrays for mutation, then convert to tensors at the end
    sigma_np = [np.zeros(coeff_lengths[m], dtype=np.float32) for m in range(M)]

    if len(ridge.time_indices) == 0:
        return [torch.from_numpy(s) for s in sigma_np]

    default_bw = 2.0

    # Build a lookup: common_time → (channel, bandwidth)
    ridge_points = {}
    for k in range(len(ridge.time_indices)):
        t = int(ridge.time_indices[k])
        m_r = int(ridge.channel_indices[k])
        if bandwidth is not None:
            bw = bandwidth
        elif ridge.bandwidth is not None:
            bw = float(ridge.bandwidth[k])
            bw = max(bw, 0.5)  # floor
        else:
            bw = default_bw
        ridge_points[t] = (m_r, bw)

    # Onset/offset taper
    t_onset = ridge.onset
    t_offset = ridge.offset
    t_offset - t_onset + 1

    for t_common, (m_ridge, bw) in ridge_points.items():
        # Time taper: smooth onset/offset
        t_rel_onset = t_common - t_onset
        t_rel_offset = t_offset - t_common
        time_weight = 1.0
        if taper_width > 0:
            if t_rel_onset < taper_width:
                time_weight *= (t_rel_onset + 1) / (taper_width + 1)
            if t_rel_offset < taper_width:
                time_weight *= (t_rel_offset + 1) / (taper_width + 1)

        # For each channel, compute the Gaussian weight based on
        # distance from the ridge channel
        for m in range(M):
            # Map common time to channel-local time index
            n_m = int(round(t_common * a_min / a[m]))
            if n_m >= coeff_lengths[m]:
                continue

            # Channel distance
            ch_dist = abs(m - m_ridge)

            # Gaussian profile: exp(-0.5 * (ch_dist / bw)^2)
            if bw > 0:
                weight = float(np.exp(-0.5 * (ch_dist / bw) ** 2) * time_weight)
            else:
                weight = (1.0 if ch_dist == 0 else 0.0) * time_weight

            sigma_np[m][n_m] = max(sigma_np[m][n_m], weight)

    return [torch.from_numpy(s) for s in sigma_np]


def fit_ridge_multiplier(
    f: torch.Tensor,
    g: list[dict],
    a: np.ndarray,
    L: int,
    fc: np.ndarray,
    *,
    mag_threshold_db: float = -40.0,
    min_ridge_hops: int = 3,
    bandwidth: float | None = None,
    taper_width: int = 3,
    use_second_order: bool = False,
    max_ridges: int | None = None,
    segment: bool = False,
    max_segment_hops: int = 30,
    chirp_threshold: float | None = None,
    bw_change_threshold: float | None = None,
    real: bool = True,
) -> tuple[list[torch.Tensor], list[Ridge], dict]:
    """Full pipeline: analyse → extract ridges → (segment) → build symbol.

    This is the high-level convenience function that runs the complete
    reassignment-guided multiplier fitting:

    1. Compute phase gradients (and optionally second-order derivatives)
    2. Extract ridges from the TF representation
    3. Optionally segment ridges at stationarity boundaries
    4. Construct a frame-multiplier symbol centered on the ridges

    Parameters
    ----------
    f : Tensor
        Input signal.
    g : list of M filter dicts
        Analysis filters.
    a : ndarray
        Hop sizes.
    L : int
        Transform length.
    fc : ndarray
        Centre frequencies (Hz).
    mag_threshold_db : float
        Magnitude threshold for peak detection (dB below max).
    min_ridge_hops : int
        Minimum ridge length in time frames.
    bandwidth : float or None
        Fixed bandwidth in channels; None = auto from second derivatives.
    taper_width : int
        Onset/offset taper width.
    use_second_order : bool
        If True, compute second-order phase derivatives and use the
        ``ff`` (group delay dispersion) to estimate per-point bandwidth
        and (if ``segment=True``) the ``tt`` (chirp rate) to detect
        stationarity boundaries.
    max_ridges : int or None
        If set, keep only the N strongest ridges (by mean amplitude).
    segment : bool
        If True, segment each ridge into short pieces at stationarity
        boundaries (chirp rate jumps, bandwidth changes). Each segment
        gets its own locally-adapted multiplier. This is the "many
        small multipliers" mode for non-stationary signals.
    max_segment_hops : int
        Maximum segment length when ``segment=True``.
    chirp_threshold : float or None
        Split ridges where ``|chirp rate|`` exceeds this (requires
        ``use_second_order=True``).
    bw_change_threshold : float or None
        Split ridges where bandwidth changes by more than this factor.
    real : bool
        Whether the filterbank is real-valued (auditory).

    Returns
    -------
    sigma : list of M Tensors
        Frame-multiplier symbol (pass to ``framemul`` or
        ``MulaclabSession.apply_symbol``).
    ridges : list of Ridge
        The extracted ridges (or segments, if ``segment=True``).
    info : dict
        Diagnostic information (num_ridges, num_segments, etc.).
    """
    f = torch.as_tensor(f, dtype=torch.float32)
    a = np.asarray(a)
    fc = np.asarray(fc)

    # Step 1: Phase gradients
    tgrad, _fgrad, s, c = filterbankphasegrad(f, g, a, L)

    # Optional: second-order for bandwidth and chirp detection
    ff = None
    tt = None
    if use_second_order:
        derivs_to_get = ["ff"]
        if segment and chirp_threshold is not None:
            derivs_to_get.append("tt")
        derivs_dict, _ = filterbankphasederiv(f, g, a, derivs=tuple(derivs_to_get), L=L)
        ff = derivs_dict.get("ff")
        tt = derivs_dict.get("tt")

    # Step 2: Extract ridges
    ridges = extract_ridges(
        tgrad,
        s,
        fc,
        a,
        L,
        mag_threshold_db=mag_threshold_db,
        min_ridge_hops=min_ridge_hops,
        ff=ff,
    )

    # Optional: keep only the strongest ridges
    if max_ridges is not None and len(ridges) > max_ridges:
        ridges.sort(key=lambda r: -float(np.mean(r.amplitude)))
        ridges = ridges[:max_ridges]

    num_ridges_before = len(ridges)

    # Step 3: Optionally segment ridges at stationarity boundaries
    if segment:
        a_min = int(np.min(a))
        ridges = segment_ridges(
            ridges,
            max_segment_hops=max_segment_hops,
            chirp_threshold=chirp_threshold,
            bw_change_threshold=bw_change_threshold,
            tt=tt,
            a=a,
            a_min=a_min,
            min_segment_hops=max(min_ridge_hops, 3),
        )

    # Step 4: Build symbol (return as torch tensors)
    coeff_lengths = [len(c[m]) for m in range(len(c))]
    sigma = ridges_to_symbol(
        ridges,
        coeff_lengths,
        a,
        L,
        bandwidth=bandwidth,
        taper_width=taper_width,
    )

    info = {
        "num_ridges": num_ridges_before,
        "num_segments": len(ridges) if segment else num_ridges_before,
        "segmented": segment,
        "num_channels": len(g),
        "transform_length": L,
        "signal_length": len(f),
    }

    return sigma, ridges, info


def denoise_by_ridges(
    f: torch.Tensor,
    g: list[dict],
    a: np.ndarray,
    L: int,
    fc: np.ndarray,
    g_dual: list[dict] | None = None,
    **kwargs,
) -> tuple[torch.Tensor, list[Ridge], dict]:
    """Denoise a signal by extracting ridges and applying the multiplier.

    Full pipeline:
        analyse → extract ridges → fit symbol → apply multiplier → synthesise

    Parameters
    ----------
    f : Tensor
        Noisy input signal.
    g : list of M filter dicts
        Analysis filters.
    a : ndarray
        Hop sizes.
    L : int
        Transform length.
    fc : ndarray
        Centre frequencies.
    g_dual : list of M filter dicts or None
        Dual (synthesis) filters. If None, computed via filterbankdual(..., real=True).
    **kwargs
        Passed to ``fit_ridge_multiplier``.

    Returns
    -------
    f_denoised : Tensor
        Denoised signal.
    ridges : list of Ridge
        Extracted ridges.
    info : dict
        Diagnostic information.
    """
    a = np.asarray(a)
    sigma, ridges, info = fit_ridge_multiplier(f, g, a, L, fc, **kwargs)

    # Analyse
    c = filterbank(f, g, a, L)

    # Apply multiplier (element-wise)
    c_mod = [c[m] * sigma[m] for m in range(len(c))]

    # Synthesise
    if g_dual is None:
        g_dual = filterbankdual(g, a, L, real=True)

    Ls = len(f)
    f_denoised = ifilterbank(c_mod, g_dual, a, L, real=True)
    f_denoised = torch.real(f_denoised)[:Ls]

    info["snr_improvement_db"] = None  # caller can compute if clean ref available

    return f_denoised, ridges, info
