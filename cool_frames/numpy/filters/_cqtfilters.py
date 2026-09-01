"""
numpy/layer1/_cqtfilters.py
============================
CQT filterbank design — Python port of LTFAT ``cqtfilters.m``.

MATLAB original
---------------
  layer1/filter_design/cqtfilters.m
  (Holighaus, Velasco; modified by Prusa, 2013-2014)

Design note — blfilter fftshift convention
-------------------------------------------
The shared ``blfilter`` helper in ``_filters.py`` stores the window as
``np.fft.fftshift(h)`` (peak at index 0) and computes::

    foff = round(L/2 * fc_norm) - win_len // 2

which places the peak at bin ``fc_bin - win_len//2`` instead of at
``fc_bin``.  For even-length windows, the centre-frequency bin receives
the zero-crossing of the Hann taper, so some DFT bins have zero coverage
(frame lower bound A = 0).

To avoid this, ``_make_direct_filter`` stores the **un-shifted** Hann
window (peak at index ``win_bins // 2``) and sets::

    foff = round(L * fc_hz / fs) - win_bins // 2

so that the centre index of the window maps exactly to ``fc_bin``.  We
always use odd window lengths so the centre is unambiguous.  This
representation is fully compatible with ``filter_freqresp`` and the
layer-2 frame-theory functions (``filterbankbounds``, ``filterbanktight``,
etc.).

Public API
----------
  cqtfilters(fs, fmin, fmax, bins, Ls, ...)
    -> (g, a, fc, L)

  frame_bounds_cqt(g, a, L)
    -> (A, B, kappa)

  partial_tighten_cqt(g, a, L, alpha)
    -> g_alpha  (list of dicts with precomputed H_full arrays)
"""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from ..core._core import filterbanklength, floor23
from ._edge_filters import (
    build_complement_highpass,
    build_complement_lowpass,
    edge_params_from_geometry,
)
from ._firwin import firwin as _firwin
from ._firwin import firwin_taper

# ---------------------------------------------------------------------------
# Hann window helper (no fftshift – peak at index n//2)
# ---------------------------------------------------------------------------

def _hann_window(n: int) -> np.ndarray:
    """Return a length-*n* Hann window, peak at index ``n // 2``."""
    if n == 1:
        return np.ones(1)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))


# ---------------------------------------------------------------------------
# Direct filter constructor (bypasses blfilter fftshift convention)
# ---------------------------------------------------------------------------

def _make_direct_filter(
    fc_hz:    float,
    fsupp_hz: float,
    fs:       float,
    scal:     float = 1.0,
    min_win:  int   = 4,
    norm:     str   = "energy",
    taper_ratio: float | None = None,
    winname:  str   = "hann",
    realonly: int   = 0,
) -> dict:
    """Build a filter dict compatible with ``filter_freqresp``.

    The Hann (or Hann-taper) window is stored un-shifted, with::

        foff(L) = round(L * fc_hz / fs) - win_bins // 2

    so the window peak sits exactly at ``fc_bin = round(L * fc_hz / fs)``.
    Window lengths are always made odd to guarantee an unambiguous centre
    sample.

    Parameters
    ----------
    fc_hz       : centre frequency in Hz
    fsupp_hz    : filter support (bandwidth) in Hz
    fs          : sampling rate in Hz
    scal        : amplitude scaling factor
    min_win     : minimum window length in DFT bins
    norm        : 'energy' | '1' | 'inf'
    taper_ratio : if not None, use a Hann-taper window with this taper ratio
                  (used for the DC / Nyquist edge filters)
    realonly    : 0 (complex-valued output) or 1 (real-valued output)
    """
    fc_norm  = fc_hz  / fs          # normalised in [0, 1)
    bw_norm  = fsupp_hz / fs        # normalised bandwidth

    def _win(win_bins: int) -> np.ndarray:
        if taper_ratio is not None:
            # Build a *centered* taper (peak at win_bins // 2) so that
            # foff = fc_bin − win_bins // 2 places the peak at fc_bin.
            # firwin_taper uses WPE convention (peak at index 0) which is
            # incompatible with the centered foff calculation below.
            n = np.arange(win_bins, dtype=float)
            x = np.abs(n - win_bins // 2) / win_bins   # 0 at centre, ~0.5 at edges
            plat = (1.0 - taper_ratio) / 2.0
            g = np.ones(win_bins)
            mask = x > plat
            if np.any(mask) and taper_ratio > 0:
                xt = np.minimum((x[mask] - plat) / taper_ratio, 1.0)
                g[mask] = 0.5 + 0.5 * np.cos(np.pi * xt)  # type: ignore[assignment]
            g = np.maximum(g, 0.0)  # type: ignore[assignment]
            mx = float(np.max(np.abs(g)))
            return g / mx if mx > 0 else g  # type: ignore[assignment]
        if winname == "hann":
            return _hann_window(win_bins)
        return _firwin(winname, win_bins, norm="inf")

    def _scale(h: np.ndarray, L: int) -> np.ndarray:
        if norm in ("energy", "2"):
            return h * scal * math.sqrt(float(L))
        elif norm in ("1", "area"):
            return h * scal * float(L)
        else:  # "inf" / "peak"
            return h * scal

    def H(L: int) -> np.ndarray:
        raw = max(min_win, round(L * bw_norm))
        win_bins = raw if raw % 2 == 1 else raw + 1   # ensure odd
        h = _win(win_bins)
        return _scale(h, L)

    def foff(L: int) -> int:
        raw = max(min_win, round(L * bw_norm))
        win_bins = raw if raw % 2 == 1 else raw + 1
        fc_bin = int(round(L * fc_norm))
        return fc_bin - win_bins // 2

    return {
        "H":        H,
        "foff":     foff,
        "realonly": realonly,
        "delay":    0,
        "fs":       fs,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def cqtfilters(
    fs:       float,
    Ls:       int,
    *,
    fmin:     float = 50.0,
    fmax:     float | None = None,
    bins=12,                       # int or array-like of ints per octave
    sampling: str   = "regsampling",
    Qvar:     float = 1.0,
    min_win:  int   = 4,
    window:   str   = "hann",
    redmul:   float = 1.0,
    norm:     str   = "energy",
    hop_ms:   float | None = None,
) -> tuple[list[dict], np.ndarray, np.ndarray, int, dict]:
    """Construct a constant-Q filterbank.

    Constant-Q spectral transform (Brown 1991) with efficient toolbox
    implementation (Schoerkhuber & Klapuri 2010).  See *References* below.

    Parameters
    ----------
    fs       : float         – sampling rate (Hz)
    Ls       : int           – signal length (samples)
    fmin     : float         – lowest CQT centre frequency (Hz, default 50)
    fmax     : float or None – highest CQT centre frequency (Hz); clipped to
                               the Nyquist frequency if larger. ``None``
                               (default) uses ``fs/2``.
    bins     : int or 1-D array_like
                             – bins per octave (default 12, i.e. semitones).
                               Scalar → same value for every octave.  A 1-D
                               array may specify a different count for each
                               octave.
    sampling : str           – 'regsampling' (default) | 'uniform' |
                               'fractional' | 'fractionaluniform'
    Qvar     : float         – bandwidth scaling factor (1.0 = standard CQT).
                               Values > 1 widen bands (lower Q, less
                               constant-Q).
    min_win  : int           – minimum window length in DFT bins (default 4)
    window   : str           – prototype window shape (default 'hann');
                               currently only 'hann' is used internally
    redmul   : float         – redundancy multiplier; values > 1 increase
                               overlap, values < 1 may break the frame
                               property
    norm     : str           – filter normalisation ('energy', '1', 'inf')

    Returns
    -------
    g   : list[dict]  – M+2 filter descriptors (M CQT + DC + Nyquist),
                        each a dict with keys 'H', 'foff', 'realonly',
                        'delay', 'fs' compatible with layer-2 functions
    a   : ndarray     – (M+2,) integer hop sizes  [or (M+2, 2) for
                        fractional sampling]
    fc  : ndarray     – (M+2,) centre frequencies in Hz
                        (0 = DC, fs/2 = Nyquist)
    L   : int         – smallest admissible transform length >= Ls

    Raises
    ------
    ValueError  – if fmin >= fmax, or if the computed bandwidths exceed fs

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import cqtfilters
    >>> g, a, fc, L, _info = cqtfilters(8000, 1024, fmin=100, fmax=4000, bins=12)
    >>> len(g)  # DC + CQT + Nyquist
    66
    >>> float(fc[0]), float(fc[-1])  # DC and Nyquist frequencies
    (0.0, 4000.0)
    >>> a.shape[0] == len(g)
    True

    References
    ----------
    * J. C. Brown, "Calculation of a constant Q spectral transform,"
      *J. Acoust. Soc. Am.*, vol. 89, no. 1, pp. 425-434, 1991.
    * C. Schoerkhuber and A. Klapuri, "Constant-Q transform toolbox
      for music processing," SMC, 2010.
    """
    fs   = float(fs)
    fmin = float(fmin)
    fmax = float(fs / 2.0 if fmax is None else fmax)
    Ls   = int(Ls)

    # hop_ms implies a single common hop -> uniform sampling.
    if hop_ms is not None and sampling != "uniform":
        if sampling != "regsampling":
            warnings.warn(
                f"cqtfilters: hop_ms overrides sampling={sampling!r} -> 'uniform'.",
                stacklevel=2)
        sampling = "uniform"

    if fmin >= fmax:
        raise ValueError(
            f"cqtfilters: fmin ({fmin}) must be strictly less than fmax ({fmax})."
        )
    if fmin <= 0:
        raise ValueError(f"cqtfilters: fmin ({fmin}) must be positive.")
    if fs <= 0:
        raise ValueError(f"cqtfilters: fs ({fs}) must be positive.")
    if Ls < 1:
        raise ValueError(f"cqtfilters: Ls ({Ls}) must be a positive integer.")

    nf = fs / 2.0  # Nyquist frequency

    # Clip fmax to Nyquist
    fmax = min(fmax, nf)

    # ── 1. Expand bins to one entry per octave ────────────────────────────────
    n_oct = int(math.ceil(math.log2(fmax / fmin))) + 1
    bins_arr = _expand_bins(bins, n_oct)

    # ── 2. CQT centre frequencies ─────────────────────────────────────────────
    fc_inner = _make_cqt_centers(fmin, bins_arr)
    fc_inner = _trim_centers(fc_inner, fmax, nf)

    M  = len(fc_inner)  # number of inner CQT filters
    fc = np.concatenate([[0.0], fc_inner, [nf]])
    M2 = M + 2          # total filter count

    # ── 3. Bandwidths (filter support in Hz) ──────────────────────────────────
    fsupp = _compute_bandwidths(fc, M, bins_arr, fmin, nf)

    # Apply Qvar scaling to CQT inner channels (not DC / Nyquist)
    fsupp[1 : M2 - 1] *= Qvar

    # Enforce minimum bandwidth
    fsuppmin = min_win / Ls * fs
    fsupp = np.maximum(fsupp, fsuppmin)

    # ── 4. Precise (continuous) hop sizes ────────────────────────────────────
    aprecise = fs / fsupp / redmul
    aprecise = np.maximum(aprecise, 1.0)

    if np.any(fsupp > fs):
        raise ValueError(
            "cqtfilters: bandwidth of one or more filters exceeds fs. "
            "Check fmin, fmax, bins, and Qvar."
        )

    # ── 5. Integer / fractional hop sizes ────────────────────────────────────
    a, L = _compute_hop_sizes(aprecise, bins_arr, M, M2, Ls, sampling,
                              hop_ms=hop_ms, fs=fs)

    # ── 6. Filter scaling ─────────────────────────────────────────────────────
    if a.ndim == 1:
        a_num = a.astype(float)
        a_den = np.ones(M2, dtype=float)
    else:  # fractional: (M2, 2)
        a_num = a[:, 0].astype(float)  # type: ignore[assignment]
        a_den = a[:, 1].astype(float)  # type: ignore[assignment]

    scal = np.sqrt(a_num / a_den)
    # Real filterbank: edge channels scaled by 1/sqrt(2) because they appear
    # in both halves of the spectrum.
    scal[0]  /= math.sqrt(2.0)
    scal[-1] /= math.sqrt(2.0)

    # ── 7. Build inner filter descriptors ─────────────────────────────────────
    g: list[dict[str, Any] | None] = [None] * M2
    for m in range(1, M2 - 1):
        g[m] = _make_direct_filter(
            fc[m], fsupp[m], fs,
            scal=float(scal[m]), min_win=min_win, norm=norm,
            winname=window,
            realonly=1,  # Inner filters are real-only (single-sided)
        )

    # ── 8. Edge filters via complement construction ────────────────────────
    a_inner = a[1:-1] if a.ndim == 1 else a[1:-1, :]
    g_inner = g[1:-1]

    fsupp_dc, ratio_dc = edge_params_from_geometry(
        fc[1], fsupp[1], fs, target="dc")
    fsupp_nyq, ratio_nyq = edge_params_from_geometry(
        fc[-2], fsupp[-2], fs, target="nyquist")

    g[0] = build_complement_lowpass(
        g_inner, a_inner, float(fc[1]), fs,  # type: ignore[arg-type]
        scal=float(scal[0]),
        fsupp_lp=fsupp_dc,
        taper_ratio=ratio_dc,
        min_win=min_win,
    )
    g[M2 - 1] = build_complement_highpass(
        g_inner, a_inner, float(fc[-2]), fs,  # type: ignore[arg-type]
        scal=float(scal[-1]),
        fsupp_hp=fsupp_nyq,
        taper_ratio=ratio_nyq,
        min_win=min_win,
    )

    from ..diagnostics.admissibility import check_admissible

    admissible = check_admissible(
        fc[1:-1], fsupp[1:-1], fs=fs, L=int(L),
        fsupp_dc=fsupp_dc, fsupp_nyq=fsupp_nyq,
        min_win=min_win, window=window, designer="cqtfilters")

    from ._tfr import tfr_from_bandwidth

    # The DC and Nyquist complements carry their bandwidth in `fsupp_dc` /
    # `fsupp_nyq`, not in `fsupp` -- audfilters and greenwoodfilters store 0
    # there.  Feeding the raw array to the rule gives tfr = nan on exactly
    # those two channels, and `sqrt(info["tfr"])` then poisons every
    # coefficient of the magnitude path.
    _bw = np.asarray(fsupp, dtype=float).copy()
    _bw[0] = float(fsupp_dc)
    _bw[-1] = float(fsupp_nyq)

    info = {"fc": fc, "a": a, "L": int(L), "designer": "cqtfilters",
            "fsupp": fsupp, "fsupp_inner": fsupp[1:-1],
            "fsupp_dc": float(fsupp_dc), "fsupp_nyq": float(fsupp_nyq),
            "tfr": tfr_from_bandwidth(_bw, fs, int(L)),
            "tfr_source": "LTFAT rule (matches info.tfr(L) to 2.1e-05)",
            "admissible": admissible}
    return g, a, fc, int(L), info  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Frame-theory helpers for Paper 3
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Small normalise-a helper (avoids circular import)
# ---------------------------------------------------------------------------

def _normalise_a_local(a, M: int) -> np.ndarray:
    """Return a (M, 2) integer array of [numerator, denominator] hop sizes."""
    a = np.asarray(a)
    if a.ndim == 0:
        a = np.full((M, 2), [int(a), 1], dtype=int)
    elif a.ndim == 1:
        a = np.column_stack([a.astype(int), np.ones(M, dtype=int)])
    # else already (M, 2)
    return a.astype(int)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expand_bins(bins, n_oct: int) -> np.ndarray:
    """Return a 1-D int array of bins-per-octave with length n_oct."""
    bins_arr = np.atleast_1d(np.asarray(bins, dtype=int)).ravel()
    if len(bins_arr) == 1:
        bins_arr = np.full(n_oct, int(bins_arr[0]), dtype=int)
    elif len(bins_arr) < n_oct:
        bins_arr = np.concatenate([
            bins_arr,
            np.full(n_oct - len(bins_arr), max(int(bins_arr[-1]), 1), dtype=int)
        ])
    bins_arr = np.where(bins_arr <= 0, 1, bins_arr)  # type: ignore[assignment]
    return bins_arr[:n_oct]


def _make_cqt_centers(fmin: float, bins_arr: np.ndarray) -> np.ndarray:
    """Generate geometric CQT centre frequencies for each octave."""
    centers = []
    for kk, b in enumerate(bins_arr):
        exp_start = kk * b
        exp_end   = (kk + 1) * b
        exponents = np.arange(exp_start, exp_end)
        centers.append(fmin * 2.0 ** (exponents / float(b)))
    return np.concatenate(centers)


def _trim_centers(fc_inner: np.ndarray, fmax: float, nf: float) -> np.ndarray:
    """Keep fc < fmax, plus the first fc >= fmax if it's < nf."""
    idx = np.searchsorted(fc_inner, fmax, side="left")
    if idx < len(fc_inner) and fc_inner[idx] < nf:
        idx += 1
    return fc_inner[:idx]


def _compute_bandwidths(
    fc: np.ndarray, M: int, bins_arr: np.ndarray, fmin: float, nf: float
) -> np.ndarray:
    """
    Compute per-filter frequency support (bandwidth) in Hz.

    Indexing (0-based):
      fsupp[0]      – DC filter:  2 * fmin
      fsupp[1]      – first CQT: fc[1] * (2^(1/bins[0]) - 2^(-1/bins[0]))
      fsupp[2..M-1] – inner CQT: fc[k+1] - fc[k-1]
      fsupp[M]      – last  CQT: fc[M]  * (2^(1/bins[-1]) - 2^(-1/bins[-1]))
      fsupp[M+1]    – Nyquist:   2 * (nf - fc[M])
    """
    M2 = M + 2
    fsupp = np.zeros(M2)

    fsupp[0] = 2.0 * fmin

    b0 = int(bins_arr[0])
    fsupp[1] = fc[1] * (2.0 ** (1.0 / b0) - 2.0 ** (-1.0 / b0))

    for k in range(2, M):
        fsupp[k] = fc[k + 1] - fc[k - 1]

    b_last = int(bins_arr[-1])
    fsupp[M] = fc[M] * (2.0 ** (1.0 / b_last) - 2.0 ** (-1.0 / b_last))

    fsupp[M + 1] = 2.0 * (nf - fc[M])

    return fsupp


def _compute_hop_sizes(
    aprecise: np.ndarray,
    bins_arr: np.ndarray,
    M: int,
    M2: int,
    Ls: int,
    sampling: str,
    hop_ms: float | None = None,
    fs: float = 2.0,
) -> tuple[np.ndarray, int]:
    if sampling == "regsampling":
        a = _regsampling_hops(aprecise, bins_arr, M, M2)
        L = filterbanklength(Ls, a)

        n_iter = 0
        while 2 * Ls < L and not np.all(a == a[0]) and n_iter < 50:
            maxa = np.max(a)
            candidates = a[a < maxa]
            if len(candidates) == 0:
                break
            a[a == maxa] = np.max(candidates)
            L = filterbanklength(Ls, a)
            n_iter += 1

    elif sampling == "uniform":
        a_painless = max(1, int(math.floor(float(np.min(aprecise)))))
        if hop_ms is not None:
            a_val = max(1, int(round(hop_ms / 1000.0 * fs)))
            if a_val > a_painless:
                warnings.warn(
                    f"cqtfilters: uniform hop {a_val} (hop_ms={hop_ms}) exceeds the "
                    f"painless limit ({a_painless}); the wide high-frequency filters are "
                    f"undersampled. The bank stays invertible via ifilterbankiter but is "
                    f"ill-conditioned (expect many CG iterations) -- increase the bin "
                    f"count, or use gabfilters for a well-conditioned uniform-hop (STFT) "
                    f"analysis.",
                    stacklevel=2)
        else:
            a_val = a_painless
        a = np.full(M2, a_val, dtype=int)
        L = filterbanklength(Ls, a)

    elif sampling == "fractional":
        L = Ls
        N = np.ceil(Ls / aprecise).astype(int)
        N = np.maximum(N, 1)
        a = np.column_stack([np.full(M2, Ls, dtype=int), N])

    elif sampling == "fractionaluniform":
        L = Ls
        min_aprecise = float(np.min(aprecise[1:-1]))
        aprecise_fu = aprecise.copy()
        aprecise_fu[1:-1] = min_aprecise
        N = np.ceil(Ls / aprecise_fu).astype(int)
        N = np.maximum(N, 1)
        a = np.column_stack([np.full(M2, Ls, dtype=int), N])

    else:
        raise ValueError(
            f"cqtfilters: unknown sampling mode {sampling!r}. "
            "Choose from 'regsampling', 'uniform', 'fractional', "
            "'fractionaluniform'."
        )

    return a, int(L)


def _regsampling_hops(
    aprecise: np.ndarray, bins_arr: np.ndarray, M: int, M2: int
) -> np.ndarray:
    a = np.empty(M2, dtype=int)

    a[0]    = int(floor23(aprecise[0]))
    a[M2-1] = int(floor23(aprecise[M2-1]))

    start = 1
    for b in bins_arr:
        end = min(start + int(b), M2 - 1)
        if start >= M2 - 1:
            break
        oct_min = float(np.min(aprecise[start:end]))
        a_oct   = max(int(floor23(oct_min)), 1)
        a[start:end] = a_oct
        start = end

    return a


