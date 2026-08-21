"""
numpy/layer1/_freqwavelet.py
============================
Frequency-domain wavelet constructor.

Port of: layer1/filter_constructors/freqwavelet.m

Returns wavelets in three formats:
  - 'full'          : (L, M) array
  - 'econ'          : list of truncated arrays
  - 'asfreqfilter'  : list of filter-descriptor dicts (H, foff, realonly, delay)
"""
from __future__ import annotations

import numpy as np

from ._wavelet import wavelet_generator_func

# ---------------------------------------------------------------------------
# Internal: convert continuous support → sample indices
# ---------------------------------------------------------------------------

def _fsuppL(fsupp_col, fs, L, idx=None):
    """Convert continuous support vector to sample indices.

    Parameters
    ----------
    fsupp_col : (5,) or (5, M) array – continuous support values
    fs : float
    L : int
    idx : list of int indices (0-based) to return, or None for all 5

    Returns
    -------
    fsuppL : array of ints
    """
    fsupp_col = np.atleast_2d(fsupp_col)
    if fsupp_col.shape[0] != 5:
        fsupp_col = fsupp_col.T
    M = fsupp_col.shape[1]

    all5 = np.zeros((5, M), dtype=int)
    all5[0, :] = np.ceil(fsupp_col[0, :] / fs * L).astype(int)
    all5[1, :] = np.ceil(fsupp_col[1, :] / fs * L).astype(int)
    all5[2, :] = np.round(fsupp_col[2, :] / fs * L).astype(int)
    all5[3, :] = np.floor(fsupp_col[3, :] / fs * L).astype(int)
    all5[4, :] = np.floor(fsupp_col[4, :] / fs * L).astype(int)

    if idx is None:
        return all5
    return all5[idx, :]


# ---------------------------------------------------------------------------
# Normalisation helper (setnorm equivalent)
# ---------------------------------------------------------------------------

def _setnorm(g: np.ndarray, norm: str) -> np.ndarray:
    """Apply normalisation to a 1-D or 2-D array (column-wise)."""
    norm = norm.lower() if norm else "null"
    if norm in ("null", "none", ""):
        return g
    elif norm in ("inf", "peak"):
        mx = np.max(np.abs(g), axis=0, keepdims=True)
        mx = np.where(mx > 0, mx, 1.0)
        return g / mx  # type: ignore[no-any-return]
    elif norm in ("1", "area"):
        s = np.sum(np.abs(g), axis=0, keepdims=True)
        s = np.where(s > 0, s, 1.0)
        return g / s  # type: ignore[no-any-return]
    elif norm in ("2", "energy"):
        s = np.sqrt(np.sum(np.abs(g)**2, axis=0, keepdims=True))
        s = np.where(s > 0, s, 1.0)
        return g / s  # type: ignore[no-any-return]
    else:
        return g


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def freqwavelet(
    name,
    L: int,
    scale=1.0,
    *,
    output_format: str = "full",
    basefc: float = 0.1,
    fs: float = 2.0,
    scal=None,
    delay=0,
    bwthr: float = 10**(-3/10),
    efsuppthr: float = 1e-5,
    norm: str = "null",
    freqrange: str = "positive",
) -> tuple:
    """Frequency-domain wavelet constructor.

    Parameters
    ----------
    name : str or list
        Wavelet name with optional parameters, e.g. ``'cauchy'``,
        ``['cauchy', 300]``, ``['fbsp', 4, 3]``.
    L : int
        Transform length.
    scale : float or array
        Wavelet scale(s). Values > 1 → wider wavelet; < 1 → narrower.
        Centre frequency = basefc / scale.
    output_format : str
        ``'full'``, ``'econ'``, or ``'asfreqfilter'``.
    basefc : float
        Normalised centre frequency of the mother wavelet (scale=1).
    fs : float
        Sampling rate (normalised; default 2).
    scal : float or array or None
        Per-channel scaling factors.
    delay : int or array
        Per-channel delays.
    bwthr : float
        Bandwidth threshold.
    efsuppthr : float
        Effective-support threshold for truncation.
    norm : str
        Normalisation flag (``'null'``, ``'inf'``, ``'1'``, ``'2'``).
    freqrange : str
        ``'positive'``, ``'negative'``, or ``'analytic'``.

    Returns
    -------
    H : ndarray or list of dicts
        Wavelet(s) in the requested format.
    info : dict
        Metadata: fc, foff, fsupp, basefc, scale, dilation, bw, tfr,
        aprecise, a_natural, cauchyAlpha.
    """
    # Ensure name is a list
    if isinstance(name, str):
        name = [name]
    else:
        name = list(name)

    # Handle MATLAB-style calling: freqwavelet("name", L, "peak")
    # If scale is a string, treat it as a norm flag
    if isinstance(scale, str):
        norm = scale
        scale = 1.0

    # Scale as 1-D array
    scale = np.atleast_1d(np.asarray(scale, dtype=float)).ravel()
    M = len(scale)

    negative = (freqrange == "negative")

    # Default scal
    if scal is None:
        scal = scale.copy()
    else:
        scal = np.atleast_1d(np.asarray(scal, dtype=float)).ravel()
    if len(scal) != M:
        raise ValueError("scal must have the same number of entries as scale")

    # Delay
    if np.ndim(delay) == 0:
        delay = np.full(M, int(delay))
    else:
        delay = np.atleast_1d(np.asarray(delay, dtype=int)).ravel()

    # Validate scales
    if freqrange == "positive":
        if np.any(scale <= 0) or np.any(basefc / scale > 1):
            raise ValueError("positive: scale must be positive and >= basefc")
    elif freqrange == "negative":
        if np.any(scale >= 0) or np.any(basefc / scale < -1):
            raise ValueError("negative: scale must be negative and <= -basefc")
    elif freqrange == "analytic":
        if np.any(scale <= 0) or np.any(basefc / scale > 2):
            raise ValueError("analytic: scale must be positive and >= basefc/2")

    if efsuppthr < 0:
        raise ValueError("efsuppthr must be >= 0")
    if bwthr < 0:
        raise ValueError("bwthr must be >= 0")
    if bwthr < efsuppthr:
        raise ValueError("efsuppthr must be <= bwthr")

    # Generate the wavelet prototype
    fun, fsupp_, peakpos, cauchy_alpha = wavelet_generator_func(
        name, negative=negative, efsuppthr=efsuppthr, bwthr=bwthr
    )

    # Compute basedil (base dilation)
    basedil = peakpos / basefc
    alpha_step = fs / L

    # Compute continuous support for each scale
    fsupp = np.zeros((5, M))
    fsupp[4, :] = fs
    if efsuppthr > 0:
        fsupp[0, :] = np.maximum(0, (fsupp_[0] / basedil) / scale)
        fsupp[4, :] = np.minimum(fs, (fsupp_[4] / basedil) / scale)
    fsupp[1, :] = np.maximum(0, (fsupp_[1] / basedil) / scale)
    fsupp[2, :] = (fsupp_[2] / basedil) / scale
    fsupp[3, :] = np.minimum(fs, (fsupp_[3] / basedil) / scale)

    if negative:
        fsupp = fsupp[::-1, :]  # type: ignore[assignment]

    fsuppL_all = _fsuppL(fsupp, fs, L)

    # Build output
    if output_format == "full":
        if not negative:
            y = np.arange(L).reshape(-1, 1) * basedil * alpha_step * scale.reshape(1, -1)
        else:
            idx = np.concatenate(([0], np.arange(L-1, 0, -1)))
            y = idx.reshape(-1, 1) * basedil * alpha_step * scale.reshape(1, -1)
        H_raw = fun(y)
        H = np.abs(scal).reshape(1, -1) * _setnorm(H_raw, norm)

    elif output_format == "econ":
        H = []
        for ii in range(M):
            y_ii = np.arange(fsuppL_all[0, ii], fsuppL_all[4, ii]) * basedil * alpha_step * abs(scale[ii])
            h = abs(scal[ii]) * _setnorm(fun(y_ii), norm)
            H.append(h)

    elif output_format == "asfreqfilter":
        H = []
        for m_idx in range(M):
            fsupp_m = fsupp[:, m_idx]
            s_m = abs(scal[m_idx])
            sc_m = abs(scale[m_idx])
            d_m = int(delay[m_idx])

            def _make_H(fsupp_m=fsupp_m, s_m=s_m, sc_m=sc_m):
                def H_func(L_):
                    idx_lo = _fsuppL(fsupp_m.reshape(5, 1), fs, L_, [0])[0, 0]
                    idx_hi = _fsuppL(fsupp_m.reshape(5, 1), fs, L_, [4])[0, 0]
                    y_ = np.arange(idx_lo, idx_hi) * basedil * sc_m * fs / L_
                    return s_m * _setnorm(fun(y_), norm)
                return H_func

            def _make_foff(fsupp_m=fsupp_m):
                def foff_func(L_):
                    return int(_fsuppL(fsupp_m.reshape(5, 1), fs, L_, [0])[0, 0])
                return foff_func

            g = {
                "H": _make_H(),
                "foff": _make_foff(),
                "realonly": 0,
                "delay": d_m,
            }
            H.append(g)
    else:
        raise ValueError(f"Unknown output_format {output_format!r}")

    # Build info struct
    info = {
        "fc": fsupp[2, :].copy(),
        "basefc": basefc,
        "foff": fsuppL_all[0, :].copy(),
        "fsupp": (fsuppL_all[4, :] - fsuppL_all[0, :] + 1).copy(),
        "scale": scale.copy(),
        "dilation": basedil * scale,
        "bw": fsupp[3, :] - fsupp[1, :],
        "tfr": np.zeros(M),
        "aprecise": np.ones(M),
        "a_natural": np.column_stack([np.full(M, L), np.full(M, L)]),
        "cauchyAlpha": cauchy_alpha,
    }

    # Compute aprecise and a_natural
    bw_in_samples = info["bw"] / alpha_step
    bw_in_samples = np.maximum(bw_in_samples, 1.0)
    info["aprecise"] = L / bw_in_samples
    info["a_natural"] = np.column_stack([
        np.full(M, L, dtype=int),
        np.ceil(bw_in_samples).astype(int)
    ])

    # TFR
    fc_arr = info["fc"]
    nonzero = fc_arr != 0
    if cauchy_alpha is not None:
        info["tfr"][nonzero] = (cauchy_alpha - 1) / (np.pi * fc_arr[nonzero]**2 * L)

    info["fsupp"] = np.maximum(info["fsupp"], 0)

    # Unwrap single-element cell
    if M == 1 and isinstance(H, list):
        H = H[0]

    # For single-scale full output, squeeze the column dimension
    if M == 1 and output_format == "full" and hasattr(H, 'ndim') and H.ndim == 2 and H.shape[1] == 1:
        H = H.ravel()

    return H, info
