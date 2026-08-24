"""
numpy/filters/_greenwoodfilters.py
===================================
Filterbank design based on the Greenwood frequency–position function.

The Greenwood function maps cochlear position *x* ∈ [0, 1] to
frequency::

    f(x) = A · (10^(α·x) − k)

Different species have different (A, α, k) parameters (Greenwood 1990,
Table I).  This module provides a filterbank design that spaces centre
frequencies uniformly on the cochlear position axis, producing a
frequency spacing that matches the tonotopic organisation of the
target species' cochlea.

For species with lower hearing ranges (e.g. elephants), the Greenwood
function naturally concentrates filters at low frequencies where the
cochlea has the highest density of hair cells—making it ideal for
signals like infrasonic rumbles.

Public API
----------
  greenwoodfilters(fs, Ls, ...)
    -> (g, a, fc, L)

  frame_bounds_greenwood(g, a, L)
    -> (A, B, kappa)

  partial_tighten_greenwood(g, a, L, alpha)
    -> g_alpha

References
----------
* D. D. Greenwood, "A cochlear frequency–position function for
  several species—29 years later," *J. Acoust. Soc. Am.*, vol. 87,
  no. 6, pp. 2592-2605, 1990.
"""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from ..core._core import filterbanklength, floor23
from ._audscale import GREENWOOD_DEFAULTS, audfiltbw
from ._cqtfilters import _make_direct_filter, _normalise_a_local
from ._edge_filters import build_complement_highpass, build_complement_lowpass
from ._firwin import hann_winbw


# ---------------------------------------------------------------------------
# Greenwood-specific frequency helpers (parametric, not using global state)
# ---------------------------------------------------------------------------

def _greenwood_freq(x: np.ndarray, A: float, alpha: float, k: float) -> np.ndarray:
    """Greenwood function: cochlear position x → frequency (Hz)."""
    return A * (10.0 ** (alpha * x) - k)  # type: ignore[return-value]


def _greenwood_pos(f: np.ndarray, A: float, alpha: float, k: float) -> np.ndarray:
    """Inverse Greenwood: frequency (Hz) → cochlear position x."""
    return np.log10(f / A + k) / alpha  # type: ignore[return-value]


def _greenwood_bw(fc: np.ndarray, A: float, alpha: float, k: float) -> np.ndarray:
    """Bandwidth (df/dx) at centre frequency fc.

    The derivative of the Greenwood function is:
        df/dx = A · α · ln(10) · 10^(α·x) = α · ln(10) · (f + A·k)
    """
    return alpha * np.log(10.0) * (fc + A * k)  # type: ignore[no-any-return]


def _greenwood_space(
    fmin: float, fmax: float, n: int,
    A: float, alpha: float, k: float,
) -> np.ndarray:
    """Return *n* frequencies uniformly spaced on the Greenwood cochlear axis."""
    x_min = _greenwood_pos(np.asarray(fmin), A, alpha, k)
    x_max = _greenwood_pos(np.asarray(fmax), A, alpha, k)
    x_pts = np.linspace(float(x_min), float(x_max), n)
    return _greenwood_freq(x_pts, A, alpha, k)


# ---------------------------------------------------------------------------
# greenwoodfilters
# ---------------------------------------------------------------------------

def greenwoodfilters(
    fs:       float,
    Ls:       int,
    *,
    A:        float | None = None,
    alpha:    float | None = None,
    k:        float | None = None,
    sampling: str   = "regsampling",
    fmin:     float | None = None,
    fmax:     float | None = None,
    M:        int   | None = None,
    spacing:  float | None = None,
    bwmul:    float = 1.0,
    redmul:   float = 1.0,
    min_win:  int   = 4,
    window:   str   = "hann",
    norm:     str   = "energy",
    hop_ms:   float | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray, int, dict]:
    """Construct a filterbank using the Greenwood frequency–position function.

    Centre frequencies are uniformly spaced on the cochlear position
    axis (x ∈ [0, 1]), producing a frequency layout that mirrors the
    tonotopic organisation of the target species' cochlea.

    The Greenwood function is::

        f(x) = A · (10^(α·x) − k)

    Parameters
    ----------
    fs       : float  – sampling rate (Hz)
    Ls       : int    – signal length (samples)
    A        : float  – Greenwood frequency scaling (Hz).
                        Default: 165.4 (human).  Elephant: ~200.
    alpha    : float  – Greenwood curvature parameter.
                        Default: 2.1 (human).  Elephant: ~1.4.
    k        : float  – low-frequency correction.
                        Default: 0.88 (human).  Elephant: ~0.85.
    sampling : str    – 'regsampling' (default) | 'uniform' | 'fractional'
    fmin     : float  – minimum centre frequency (Hz).
                        Default: one spacing step above f(0) on the
                        Greenwood axis.
    fmax     : float  – maximum centre frequency (Hz).
                        Default: fs/2.
    M        : int    – number of inner channels (overrides spacing).
    spacing  : float  – spacing in cochlear position units (default: 0.02,
                        i.e. 50 channels across the full cochlea).
    bwmul    : float  – bandwidth multiplier (default: 1.0).
    redmul   : float  – redundancy multiplier (default: 1.0).
    min_win  : int    – minimum filter window length in DFT bins (default: 4).
    window   : str    – prototype window shape (default: 'hann').
    norm     : str    – filter normalisation (default: 'energy').

    Returns
    -------
    g   : list of M+2 filter dicts (DC + inner + Nyquist)
    a   : (M+2,) integer hop-size array   [or (M+2,2) for fractional]
    fc  : (M+2,) centre-frequency array (Hz)
    L   : int – filterbank frame length

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import greenwoodfilters
    >>> # Human cochlea defaults
    >>> g, a, fc, L, _info = greenwoodfilters(16000, 32000)
    >>> float(fc[0]), float(fc[-1])
    (0.0, 8000.0)
    >>> # Elephant cochlea
    >>> g, a, fc, L, _info = greenwoodfilters(16000, 32000,
    ...     A=200.0, alpha=1.4, k=0.85, fmin=5, fmax=500)
    >>> len(g) >= 4  # DC + inner + Nyquist
    True

    References
    ----------
    * D. D. Greenwood, "A cochlear frequency–position function for
      several species—29 years later," *J. Acoust. Soc. Am.*, vol. 87,
      no. 6, pp. 2592-2605, 1990.
    """
    fs = float(fs)
    Ls = int(Ls)

    # hop_ms implies a single common hop -> uniform sampling.
    if hop_ms is not None and sampling != "uniform":
        if sampling != "regsampling":
            warnings.warn(
                f"greenwoodfilters: hop_ms overrides sampling={sampling!r} -> 'uniform'.",
                stacklevel=2)
        sampling = "uniform"

    # Greenwood parameters (fall back to global defaults)
    gw_A     = A     if A     is not None else GREENWOOD_DEFAULTS["A"]
    gw_alpha = alpha if alpha is not None else GREENWOOD_DEFAULTS["alpha"]
    gw_k     = k     if k     is not None else GREENWOOD_DEFAULTS["k"]

    # Default spacing: 0.02 cochlear-position units ≈ 50 channels across
    # the full cochlea.  For comparison, the human cochlea has ~3500 inner
    # hair cells across ~35 mm, so 0.02 ≈ one channel per 0.7 mm.
    if spacing is None:
        spacing = 0.02

    # Default frequency range
    if fmax is None:
        fmax = fs / 2.0
    if fmin is None:
        # One spacing step above f(0) = A · (1 − k) on the Greenwood axis
        f_base = gw_A * (1.0 - gw_k)
        x_base = _greenwood_pos(np.asarray(max(f_base, 1e-3)), gw_A, gw_alpha, gw_k)
        fmin = float(_greenwood_freq(np.asarray(float(x_base) + spacing), gw_A, gw_alpha, gw_k))
    fmin = float(fmin)
    fmax = min(float(fmax), fs / 2.0)

    if fmin <= 0:
        raise ValueError(
            f"greenwoodfilters: fmin ({fmin:.2f}) must be positive. "
            f"The Greenwood function f(0) = A·(1−k) = {gw_A * (1 - gw_k):.2f} Hz. "
            f"Choose fmin above this value."
        )
    if fmin >= fmax:
        raise ValueError(
            f"greenwoodfilters: fmin ({fmin}) must be < fmax ({fmax})."
        )

    # Cochlear positions of fmin and fmax
    x_min = float(_greenwood_pos(np.asarray(fmin), gw_A, gw_alpha, gw_k))
    x_max = float(_greenwood_pos(np.asarray(fmax), gw_A, gw_alpha, gw_k))

    if M is not None:
        # M overrides spacing
        spacing = (x_max - x_min) / max(M - 1, 1)

    # Number of inner channels
    inner_n = int(math.floor((x_max - x_min) / spacing)) + 1

    # Trim so all channels lie strictly below Nyquist
    while inner_n > 1:
        x_last = x_min + (inner_n - 1) * spacing
        f_last = float(_greenwood_freq(np.asarray(x_last), gw_A, gw_alpha, gw_k))
        if f_last < fs / 2.0:
            break
        inner_n -= 1
    fmax = float(_greenwood_freq(
        np.asarray(x_min + (inner_n - 1) * spacing), gw_A, gw_alpha, gw_k
    ))

    if inner_n < 1:
        raise ValueError(
            "greenwoodfilters: no valid channels for the given parameters."
        )

    # Centre frequencies (inner + DC + Nyquist)
    fc_inner = _greenwood_space(fmin, fmax, inner_n, gw_A, gw_alpha, gw_k)
    fc_arr = np.concatenate([[0.0], fc_inner, [fs / 2.0]])
    M2 = len(fc_arr)
    ind = np.arange(1, M2 - 1)   # inner channel indices

    # Bandwidth: Greenwood derivative scaled by bwmul and Hann window bandwidth
    winbw = hann_winbw()   # ≈ 0.375
    fsupp = np.zeros(M2)
    fsupp[ind] = (
        _greenwood_bw(fc_arr[ind], gw_A, gw_alpha, gw_k)
        * spacing / winbw * bwmul
    )

    # Enforce minimum bandwidth
    fsuppmin = min_win / Ls * fs
    fsupp[ind] = np.maximum(fsupp[ind], fsuppmin)

    # Precise (continuous) hop sizes
    aprecise = np.ones(M2)
    aprecise[ind] = fs / fsupp[ind] / redmul  # type: ignore[assignment]
    aprecise = np.maximum(aprecise, 1.0)  # type: ignore[assignment]

    # Edge filter hop sizes (from bandwidth geometry)
    fc_in1 = float(fc_arr[1])
    fc_inK = float(fc_arr[-2])
    nf = fs / 2.0
    # Lowpass: prototype extends ~4 spacings above first inner channel
    x_in1 = float(_greenwood_pos(np.asarray(fc_in1), gw_A, gw_alpha, gw_k))
    fpe_lp = float(_greenwood_freq(
        np.asarray(min(x_in1 + 4.0 * spacing, x_max)), gw_A, gw_alpha, gw_k
    ))
    fsupp_lp = 2.0 * fpe_lp
    # Highpass: prototype extends ~4 spacings below last inner channel
    x_inK = float(_greenwood_pos(np.asarray(fc_inK), gw_A, gw_alpha, gw_k))
    fpe_hp = float(_greenwood_freq(
        np.asarray(max(x_inK - 4.0 * spacing, x_min)), gw_A, gw_alpha, gw_k
    ))
    fsupp_hp = 2.0 * (nf - fpe_hp)

    aprecise[0] = fs / max(fsupp_lp, fsuppmin) / redmul  # type: ignore[assignment]
    aprecise[-1] = fs / max(fsupp_hp, fsuppmin) / redmul  # type: ignore[assignment]
    aprecise = np.maximum(aprecise, 1.0)  # type: ignore[assignment]

    # Integer / fractional hop sizes
    if sampling == "regsampling":
        a = floor23(aprecise)  # type: ignore[union-attr]
        a = a.astype(int)  # type: ignore[union-attr]
        L = filterbanklength(Ls, a)
        while 2 * Ls < L and not np.all(a == a[0]):
            maxa = np.max(a)
            a[a == maxa] = np.max(a[a != maxa]) if np.any(a != maxa) else maxa
            L = filterbanklength(Ls, a)

    elif sampling == "uniform":
        a_painless = max(1, int(math.floor(np.min(aprecise))))
        if hop_ms is not None:
            a_scalar = max(1, int(round(hop_ms / 1000.0 * fs)))
            if a_scalar > a_painless:
                warnings.warn(
                    f"greenwoodfilters: uniform hop {a_scalar} (hop_ms={hop_ms}) exceeds "
                    f"the painless limit ({a_painless}); the wide high-frequency filters "
                    f"are undersampled. The bank stays invertible via ifilterbankiter but "
                    f"is ill-conditioned (expect many CG iterations) -- increase M for "
                    f"better conditioning, or use gabfilters for a well-conditioned "
                    f"uniform-hop (STFT) analysis.",
                    stacklevel=2)
        else:
            a_scalar = a_painless
        a = np.full(M2, a_scalar, dtype=int)
        L = filterbanklength(Ls, a)

    elif sampling in ("fractional", "fractionaluniform"):
        L = Ls
        N = np.ceil(Ls / aprecise).astype(int)
        a = np.column_stack([np.full(M2, Ls, dtype=int), N])

    else:
        raise ValueError(f"greenwoodfilters: unknown sampling mode {sampling!r}")

    # Scaling factors
    if a.ndim == 1:
        a_num = a
        a_den = np.ones_like(a)
    else:
        a_num = a[:, 0]
        a_den = a[:, 1]

    scal = np.sqrt(a_num / a_den)
    scal[0] /= math.sqrt(2.0)
    scal[-1] /= math.sqrt(2.0)

    # Build inner filter descriptors
    g_list: list[dict[str, Any] | None] = [None] * M2

    for m in ind:
        g_list[m] = _make_direct_filter(
            fc_arr[m], fsupp[m], fs,
            scal=float(scal[m]), min_win=min_win, norm=norm,
            winname=window,
        )

    # Edge filters: complement construction
    a_inner = a[1:-1] if a.ndim == 1 else a[1:-1, :]
    g_inner = [g for g in g_list if g is not None]

    # DC lowpass — prototype taper from Greenwood spacing
    fps_lp = float(_greenwood_freq(
        np.asarray(min(x_in1 + 3.0 * spacing, x_max)), gw_A, gw_alpha, gw_k
    ))
    ratio_lp = max(0.0, 2.0 * (fpe_lp - fps_lp) / fsupp_lp) if fsupp_lp > 0 else 0.0

    g_list[0] = build_complement_lowpass(  # type: ignore[arg-type]
        g_inner, a_inner, fc_in1, fs,
        scal=float(scal[0]),
        fsupp_lp=fsupp_lp,
        taper_ratio=ratio_lp,
        min_win=min_win,
    )

    # Nyquist highpass
    fps_hp = float(_greenwood_freq(
        np.asarray(max(x_inK - 3.0 * spacing, x_min)), gw_A, gw_alpha, gw_k
    ))
    ratio_hp = max(0.0, 2.0 * (fps_hp - fpe_hp) / fsupp_hp) if fsupp_hp > 0 else 0.0

    g_list[M2 - 1] = build_complement_highpass(  # type: ignore[arg-type]
        g_inner, a_inner, fc_inK, fs,
        scal=float(scal[M2 - 1]),
        fsupp_hp=fsupp_hp,
        taper_ratio=ratio_hp,
        min_win=min_win,
    )

    from ..diagnostics.admissibility import check_admissible

    admissible = check_admissible(
        fc_arr[1:-1], fsupp[1:-1], fs=fs, L=int(L),
        fsupp_dc=fsupp_lp, fsupp_nyq=fsupp_hp,
        min_win=min_win, window=window, designer="greenwoodfilters")

    from ._tfr import tfr_from_bandwidth

    # The DC and Nyquist complements carry their bandwidth in `fsupp_lp` /
    # `fsupp_hp`, not in `fsupp` -- audfilters and greenwoodfilters store 0
    # there.  Feeding the raw array to the rule gives tfr = nan on exactly
    # those two channels, and `sqrt(info["tfr"])` then poisons every
    # coefficient of the magnitude path.
    _bw = np.asarray(fsupp, dtype=float).copy()
    _bw[0] = float(fsupp_lp)
    _bw[-1] = float(fsupp_hp)

    info = {"fc": fc_arr, "a": a, "L": int(L), "designer": "greenwoodfilters",
            "fsupp": fsupp, "fsupp_inner": fsupp[1:-1],
            "fsupp_dc": float(fsupp_lp), "fsupp_nyq": float(fsupp_hp),
            "tfr": tfr_from_bandwidth(_bw, fs, int(L)),
            "tfr_source": "LTFAT rule (no LTFAT export for this designer)",
            "admissible": admissible}
    return g_list, a, fc_arr, int(L), info  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Frame-theory helpers
# ---------------------------------------------------------------------------

