"""
numpy/layer1/_design.py
=======================
High-level filter-bank design: filterbanklength, audfilters, and helpers.

MATLAB originals
----------------
  filterbanklength.m
  layer1/filter_design/audfilters.m
  layer0/math_utils/floor23.m

Design note — blfilter fftshift convention
-------------------------------------------
The shared ``blfilter`` helper stores the Hann window as
``np.fft.fftshift(h)`` (peak at index 0) and computes::

    foff = round(L/2 * fc_norm) - win_len // 2

which places the peak at ``fc_bin - win_len//2`` — half a window away
from the intended centre frequency.  For even-length windows, the
centre-frequency bin receives the Hann zero-crossing, creating dead DFT
bins and frame lower bound A = 0 in sparse configurations.

*All* filters in ``audfilters`` are therefore built with
``_make_direct_filter`` (from ``_cqtfilters``), which stores the
un-shifted Hann window (peak at index ``n // 2``) and sets
``foff = fc_bin - win_bins // 2`` so the peak maps exactly to
``fc_bin = round(L * fc_hz / fs)``.  Odd window lengths are enforced.
This is fully compatible with ``filter_freqresp`` and all layer-2 functions.

Mel-scale default spacing
--------------------------
The MATLAB ``audfilters.m`` uses ``spacing = 100`` and ``bwmul = 100``
for the 'mel' / 'mel1000' scales (the mel axis spans ~3000 units, so
spacing = 1 gives thousands of channels).  This module matches that
convention via scale-dependent defaults.

Edge-filter design (DC / Nyquist complement approach)
------------------------------------------------------
The DC and Nyquist edge filters use the MATLAB ``audlowpassfilter`` /
``audhighpassfilter`` **complement** strategy:

    H_edge(k) = P(k) · sqrt( S_max − S_inner(k) )

where:
  * ``S_inner(k)`` is the real-filterbank frame response from the inner
    channels only (``filterbankresponse(g_inner, a_inner, L, real=True)``).
  * ``S_max = max(S_inner)`` is the passband level.
  * ``P(k)`` is a Hann-taper prototype centred at 0 Hz (lowpass) or
    fs/2 Hz (highpass) whose bandwidth and plateau are determined by
    3-/4-spacing offsets on the auditory scale — matching MATLAB exactly.
  * The FIR coefficients are recovered via ``long2fir`` + fftshift:
    ``h = fftshift( real(ifft(P · Hinv))[:Lw] ) · scal``.

This construction fills exactly the gap left by the inner channels,
unlike the earlier fixed-bandwidth Hann approach (which used an
arbitrary 4-spacing constant and produced edge filters 4–150× wider
than needed).
"""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from ..core._core import filterbanklength, floor23
from ._audscale import audfiltbw, audspace, audtofreq, freqtoaud
from ._cqtfilters import _make_direct_filter, _normalise_a_local
from ._edge_filters import (
    build_complement_highpass,
    build_complement_lowpass,
    edge_params_from_geometry,
)
from ._firwin import hann_winbw

# ---------------------------------------------------------------------------
# Scale-dependent parameter defaults (mirrors MATLAB audfilters.m logic)
# ---------------------------------------------------------------------------

_MEL_SCALES = {"mel", "mel1000"}
_DEFAULT_SPACING = {s: 100.0 for s in _MEL_SCALES}    # mel units
_DEFAULT_BWMUL   = {s: 100.0 for s in _MEL_SCALES}


def _scale_default(scale: str, param: str, value):
    """Return scale-appropriate default parameter values.

    Parameters
    ----------
    scale : str
        Auditory scale ('erb', 'bark', 'mel', etc.).
    param : str
        Parameter name: 'spacing' or 'bwmul'.
    value : float or None
        If not None, return as-is. Otherwise use scale default.

    Returns
    -------
    float
        The value (if provided) or scale-dependent default.

    Raises
    ------
    KeyError
        If param is not recognized.
    """
    if value is not None:
        return value
    if param == "spacing":
        return _DEFAULT_SPACING.get(scale, 1.0)
    if param == "bwmul":
        return _DEFAULT_BWMUL.get(scale, 1.0)
    raise KeyError(f"Unknown param {param!r}")


# ---------------------------------------------------------------------------
# audfilters
# ---------------------------------------------------------------------------

def audfilters(fs: float, Ls: int, *,
               scale:    str   = "erb",
               sampling: str   = "regsampling",
               fmin:     float | None = None,
               fmax:     float | None = None,
               M:        int   | None = None,
               spacing:  float | None = None,
               bwmul:    float | None = None,
               redmul:   float = 1.0,
               min_win:  int   = 4,
               window:   str   = "hann",
               norm:     str   = "energy",
               hop_ms:   float | None = None) -> tuple[list[dict], np.ndarray, np.ndarray, int, dict]:
    """Construct an auditory filterbank.

    Auditory filterbank using gammatone filters based on auditory psychology
    (Moore 2003), efficient approximations (Patterson et al. 1992), and
    frequency analysis (Hohmann 2002).  See *References* below.

    .. note:: **Parameter-order convention.**
       ``audfilters``, ``cqtfilters``, ``greenwoodfilters`` and (since the
       2026-06 redesign) ``gabfilters`` take ``fs`` (sampling rate) as the first
       positional argument and return ``fc`` in Hz.  Only ``waveletfilters``
       still takes ``Ls`` (signal length) first, following the MATLAB LTFAT
       convention (its scales are dimensionless).

    Parameters
    ----------
    fs       : float  – sampling rate (Hz)
    Ls       : int    – signal length (samples)
    scale    : str    – auditory scale: 'erb' (default), 'bark', 'mel', ...
    sampling : str    – 'regsampling' (default) | 'uniform' | 'fractional'
    fmin     : float  – minimum centre frequency (default: one spacing step
                        above 0 Hz on the selected scale)
    fmax     : float  – maximum centre frequency (default: fs/2)
    M        : int    – total number of *inner* channels (overrides spacing)
    spacing  : float  – spacing in auditory scale units
                        (default: 1 for ERB/bark; 100 for mel/mel1000)
    bwmul    : float  – bandwidth multiplier
                        (default: 1 for ERB/bark; 100 for mel/mel1000)
    redmul   : float  – redundancy multiplier (default: 1)
    min_win  : int    – minimum filter window length in DFT bins (default: 4)
    window   : str    – prototype window shape (default: 'hann')
    norm     : str    – filter normalisation (default: 'energy')
    hop_ms   : float  – uniform hop (time step between frames) in milliseconds.
                        When given, the bank is sampled with a single common hop
                        of ``round(hop_ms/1000*fs)`` samples for every channel
                        (i.e. ``sampling='uniform'``), yielding the regular
                        ``[channels x frames]`` layout used by spectrogram-style
                        ML pipelines. Hops coarser than the painless limit give a
                        non-painless frame (invert with ``ifilterbankiter``).

    Returns
    -------
    g   : list of M+2 filter dicts (DC + inner + Nyquist)
    a   : (M+2,) integer hop-size array   [or (M+2,2) for fractional]
    fc  : (M+2,) centre-frequency array (Hz)
    L   : int – filterbank frame length

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> g, a, fc, L, _info = audfilters(16000, 32000)
    >>> len(g)  # DC + inner + Nyquist
    35
    >>> a.shape[0] == len(g)
    True
    >>> float(fc[0]), float(fc[-1])  # DC and Nyquist
    (0.0, 8000.0)

    References
    ----------
    * B. C. J. Moore, *An Introduction to the Psychology of Hearing*,
      5th ed. Academic Press, 2003.
    * R. D. Patterson et al., "An efficient auditory filterbank
      based on the gammatone function," APU Report 2341, 1992.
    * V. Hohmann, "Frequency analysis and synthesis using a Gammatone
      filterbank," *Acta Acustica*, vol. 88, pp. 433-442, 2002.
    """
    fs  = float(fs)
    Ls  = int(Ls)

    # hop_ms is a uniform (single-hop) request: it only makes sense when every
    # channel shares one hop, so it implies sampling='uniform'.
    if hop_ms is not None and sampling != "uniform":
        if sampling != "regsampling":
            warnings.warn(
                f"audfilters: hop_ms overrides sampling={sampling!r} -> 'uniform'.",
                stacklevel=2)
        sampling = "uniform"

    # Apply scale-dependent defaults (matches MATLAB audfilters.m)
    spacing = _scale_default(scale, "spacing", spacing)
    bwmul   = _scale_default(scale, "bwmul",   bwmul)

    # Default frequency range
    if fmax is None:
        fmax = fs / 2.0
    if fmin is None:
        fmin = float(audtofreq(spacing, scale))
    fmin = float(fmin)
    fmax = min(float(fmax), fs / 2.0)

    if M is not None:
        # M overrides spacing
        aud_min = freqtoaud(fmin, scale)
        aud_max = freqtoaud(fmax, scale)
        spacing = (aud_max - aud_min) / max(M - 1, 1)

    # Number of inner channels
    aud_min  = freqtoaud(fmin, scale)
    aud_max  = freqtoaud(fmax, scale)
    inner_n  = int(math.floor((aud_max - aud_min) / spacing)) + 1

    # Trim so all channels lie strictly below Nyquist
    count = 0
    _fmax = float(audtofreq(aud_min + (inner_n - 1) * spacing, scale))
    while _fmax >= fs / 2.0 and inner_n > 1:
        count  += 1
        inner_n -= 1
        _fmax   = float(audtofreq(aud_min + (inner_n - 1) * spacing, scale))
    fmax = _fmax

    if inner_n < 1:
        raise ValueError("audfilters: no valid channels for the given parameters.")

    # Centre frequencies (inner + DC + Nyquist)
    fc_inner = audspace(fmin, fmax, inner_n, scale)
    fc = np.concatenate([[0.0], fc_inner, [fs / 2.0]])
    M2  = len(fc)
    ind = np.arange(1, M2 - 1)   # inner channel indices

    # ERB bandwidth of the prototype window
    winbw = hann_winbw()   # ≈ 0.375 for Hann

    # Filter support in Hz for inner channels
    fsupp = np.zeros(M2)
    fsupp[ind] = audfiltbw(fc[ind], scale) / winbw * bwmul

    # Enforce minimum bandwidth
    fsuppmin  = min_win / Ls * fs
    fsupp[ind] = np.maximum(fsupp[ind], fsuppmin)

    # Precise (continuous) hop sizes
    aprecise = np.ones(M2)
    aprecise[ind] = fs / fsupp[ind] / redmul  # type: ignore[assignment]
    aprecise = np.maximum(aprecise, 1.0)  # type: ignore[assignment]

    # ── Edge filter hop sizes (match MATLAB convention) ───────────────────
    # MATLAB audfilters computes meaningful hop sizes for DC and Nyquist
    # filters based on their bandwidth, rather than leaving them at 1.
    fc_in1   = float(fc[1])
    fc_inK   = float(fc[-2])
    nf       = fs / 2.0
    fpe_lp_  = float(audtofreq(freqtoaud(fc_in1, scale) + 4.0 * spacing, scale))
    fsupp_lp_ = 2.0 * fpe_lp_
    fpe_hp_  = float(audtofreq(freqtoaud(fc_inK, scale) - 4.0 * spacing, scale))
    fsupp_hp_ = 2.0 * (nf - fpe_hp_)

    aprecise[0]    = fs / max(fsupp_lp_, fsuppmin) / redmul  # type: ignore[assignment]
    aprecise[-1]   = fs / max(fsupp_hp_, fsuppmin) / redmul  # type: ignore[assignment]
    aprecise = np.maximum(aprecise, 1.0)  # type: ignore[assignment]

    # ── Integer / fractional hop sizes ────────────────────────────────────
    if sampling == "regsampling":
        a = floor23(aprecise)  # type: ignore[union-attr]
        a = a.astype(int)  # type: ignore[union-attr]
        L = filterbanklength(Ls, a)
        while 2 * Ls < L and not np.all(a == a[0]):
            maxa = np.max(a)
            a[a == maxa] = np.max(a[a != maxa]) if np.any(a != maxa) else maxa
            L = filterbanklength(Ls, a)

    elif sampling == "uniform":
        a_painless = int(math.floor(np.min(aprecise)))
        if hop_ms is not None:
            a_scalar = max(1, int(round(hop_ms / 1000.0 * fs)))
            if a_scalar > a_painless:
                warnings.warn(
                    f"audfilters: uniform hop {a_scalar} (hop_ms={hop_ms}) exceeds the "
                    f"painless limit ({a_painless}); the wide high-frequency filters are "
                    f"undersampled. The bank stays invertible via ifilterbankiter but is "
                    f"ill-conditioned (expect many CG iterations) -- increase M for better "
                    f"conditioning, or use gabfilters for a well-conditioned uniform-hop "
                    f"(STFT) analysis.",
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
        raise ValueError(f"audfilters: unknown sampling mode {sampling!r}")

    # Scaling factors: sqrt(a_m) for energy normalisation
    if a.ndim == 1:
        a_num = a
        a_den = np.ones_like(a)
    else:
        a_num = a[:, 0]
        a_den = a[:, 1]

    scal = np.sqrt(a_num / a_den)
    # Edge channels scaled by 1/sqrt(2) for real filterbank
    scal[0]  /= math.sqrt(2.0)
    scal[-1] /= math.sqrt(2.0)

    # ── Build filter descriptors ───────────────────────────────────────────
    # Use _make_direct_filter (no fftshift — peak at fc_bin).
    g_list: list[dict[str, Any] | None] = [None] * M2

    for m in ind:
        g_list[m] = _make_direct_filter(
            fc[m], fsupp[m], fs,
            scal=float(scal[m]), min_win=min_win, norm=norm,
            winname=window,
        )

    # Edge filters: complement construction via shared builder, with prototype
    # bandwidth/taper from the shared geometry heuristic --- IDENTICAL to
    # cqtfilters (and hop/wavelet) so no designer differs in DC/Nyquist
    # construction (unified 2026-06-12; was auditory-scale-spacing-specific).
    a_inner = a[1:-1] if a.ndim == 1 else a[1:-1, :]
    g_inner = [g for g in g_list if g is not None]   # inner channels only

    # DC lowpass
    fsupp_lp, ratio_lp = edge_params_from_geometry(
        float(fc[1]), float(fsupp[1]), fs, target="dc")
    g_list[0] = build_complement_lowpass(  # type: ignore[arg-type]
        g_inner, a_inner, float(fc[1]), fs,
        scal=float(scal[0]),
        fsupp_lp=fsupp_lp,
        taper_ratio=ratio_lp,
        min_win=min_win,
    )

    # Nyquist highpass
    fsupp_hp, ratio_hp = edge_params_from_geometry(
        float(fc[-2]), float(fsupp[-2]), fs, target="nyquist")
    g_list[M2 - 1] = build_complement_highpass(  # type: ignore[arg-type]
        g_inner, a_inner, float(fc[-2]), fs,
        scal=float(scal[M2 - 1]),
        fsupp_hp=fsupp_hp,
        taper_ratio=ratio_hp,
        min_win=min_win,
    )

    # Announce a non-frame geometry here, where the parameters were chosen,
    # rather than letting it surface later as an all-zero dual.
    from ..diagnostics.admissibility import check_admissible

    admissible = check_admissible(
        fc[1:-1], fsupp[1:-1], fs=fs, L=int(L),
        fsupp_dc=fsupp_lp, fsupp_nyq=fsupp_hp,
        min_win=min_win, window=window, designer="audfilters")

    from ._tfr import tfr_from_bandwidth

    # The DC and Nyquist complements carry their bandwidth in `fsupp_lp` /
    # `fsupp_hp`, not in `fsupp` -- audfilters and greenwoodfilters store 0
    # there.  Feeding the raw array to the rule gives tfr = nan on exactly
    # those two channels, and `sqrt(info["tfr"])` then poisons every
    # coefficient of the magnitude path.
    _bw = np.asarray(fsupp, dtype=float).copy()
    _bw[0] = float(fsupp_lp)
    _bw[-1] = float(fsupp_hp)

    info = {"fc": fc, "a": a, "L": int(L), "scale": scale, "designer": "audfilters",
            "fsupp": fsupp, "fsupp_inner": fsupp[1:-1],
            "fsupp_dc": float(fsupp_lp), "fsupp_nyq": float(fsupp_hp),
            "tfr": tfr_from_bandwidth(_bw, fs, int(L)),
            "tfr_source": "LTFAT rule (matches info.tfr(L) to 4.5e-05)",
            "admissible": admissible}
    return g_list, a, fc, int(L), info  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Frame-theory helpers (match the cqtfilters convention)
# ---------------------------------------------------------------------------

def partial_tighten(g: list, a, L: int, alpha: float) -> list[dict]:
    """Partially tighten a filterbank toward a tight frame (any designer).

    Each filter is modified as::

        G_alpha_m(k) = G_m(k) / S(k)^(alpha / 2)

    where ``S(k)`` is the full frame response.

    * ``alpha = 0``  → identity
    * ``alpha = 1``  → tight frame (canonical tight)

    Parameters
    ----------
    g     : list of filter dicts
    a     : hop sizes
    L     : DFT length
    alpha : tightening exponent in [0, 1]

    Returns
    -------
    g_alpha : list of dict with precomputed H_full arrays

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters, partial_tighten
    >>> g, a, fc, L, _info = audfilters(16000, 32000)
    >>> g_tight = partial_tighten(g, a, L, 1.0)
    >>> len(g_tight) == len(g)
    True
    """
    from ..filterbanks._frame import filterbankresponse
    from ._filters import filter_freqresp

    M      = len(g)
    a_norm = _normalise_a_local(a, M)

    # Folded (real-signal) response, matching filterbanktight / painlessfilterbank
    # (do_real=1). alpha=1 then yields the canonical *real* tight frame (kappa=1);
    # the un-folded response left alpha=1 at kappa=2 for single-sided banks.
    # See research/companion/kappa_reconciliation_diagnosis.md.
    resp      = filterbankresponse(g, a_norm, L, real=True)     # (L,) real, folded
    resp_safe = np.where(resp < 1e-30, 1e-30, resp)
    denom     = resp_safe ** (alpha / 2.0)

    g_alpha = []
    for m in range(M):
        H_full, _ = filter_freqresp(g[m], L)
        H_a = H_full / denom
        _H  = H_a.copy()
        g_alpha.append({
            "H":        lambda _L, _h=_H: _h,
            "foff":     0,
            "realonly": 0,
            "delay":    0,
            "fs":       g[m].get("fs"),
        })

    return g_alpha


# ---------------------------------------------------------------------------
# Edge-filter helpers — complement approach
# ---------------------------------------------------------------------------

def _zero_filter(fs: float) -> dict:
    """Trivial filter that contributes nothing."""
    return {
        "H":        lambda L: np.array([0.0]),
        "foff":     0,
        "realonly": 0,
        "delay":    0,
        "fs":       fs,
    }


