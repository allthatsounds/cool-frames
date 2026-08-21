"""
numpy/layer1/_filters.py
========================
Filter-descriptor constructors: blfilter, firfilter, freqfilter,
biquadfilter, comp_biquad.

Each constructor returns a Python dict with the same fields as the
MATLAB struct:

  {
    'H'       : callable(L) -> ndarray   OR  ndarray (full-length)
    'foff'    : callable(L) -> int       OR  int
    'realonly': bool
    'delay'   : int
    'fs'      : float | None
  }

biquadfilter additionally stores pole parameters (r, theta, rho, phi)
for the IIR resonator and its ML-friendly unconstrained parametrisation.

MATLAB originals: layer1/filter_constructors/{blfilter,firfilter,freqfilter,biquadfilter}.m
                  layer0/math_utils/comp_biquad.m
"""
from __future__ import annotations

import math

import numpy as np

from ..core._core import involute, modcent, postpad
from ._firwin import firwin, firwin_taper

# ---------------------------------------------------------------------------
# Helper: Parse MATLAB-style varargs for filter functions
# ---------------------------------------------------------------------------

def _parse_filter_varargs(args):
    """Parse MATLAB-style varargs for blfilter, freqfilter, and firfilter.

    Accepts positional flag/key-value pairs:
        blfilter(name, fsupp, fc, "energy")
        blfilter(name, fsupp, fc, "peak")
        blfilter(name, fsupp, fc, "scal", 3.0)
        blfilter(name, fsupp, fc, "delay", 5)
        blfilter(name, fsupp, fc, "fs", 44100)
        blfilter(name, fsupp, fc, "real")
        blfilter(name, fsupp, fc, "causal")

    Returns a dict of keyword arguments.
    """
    flags = {"energy", "peak", "inf", "1", "2", "area", "real", "complex", "causal"}
    kv_keys = {"fs", "delay", "scal", "norm", "min_win"}

    kw: dict = {}
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, str):
            low = a.lower()
            if low in kv_keys:
                if i + 1 >= len(args):
                    raise ValueError(f"filter: '{a}' requires a value")
                kw[low] = args[i + 1]
                i += 2
            elif low in flags:
                if low in ("real", "complex"):
                    kw["realonly"] = (low == "real")
                elif low == "causal":
                    kw["causal"] = True
                else:
                    kw["norm"] = low
                i += 1
            else:
                raise ValueError(f"filter: unknown flag '{a}'")
        else:
            raise ValueError(f"filter: unexpected positional arg {a!r}")
    return kw


# ---------------------------------------------------------------------------
# blfilter – band-limited filter
# ---------------------------------------------------------------------------

def blfilter(winname, fsupp: float, fc: float = 0.0, *args,
             fs: float | None = None,
             norm: str = "energy",
             delay: int = 0,
             scal: float = 1.0,
             min_win: int = 1,
             pedantic: bool = False,
             realonly: bool = False) -> dict:
    """Construct a band-limited filter descriptor.

    Parameters
    ----------
    winname : str or list
        Window name accepted by :func:`firwin`.  Can also be
        ``['hann', 'taper', ratio]`` to request a tapering window.
    fsupp   : float
        Frequency support measured in Hz (if *fs* given) or in
        normalised units [0, 2].
    fc      : float
        Centre frequency (same units as *fsupp*).
    fs      : float, optional
        Sampling rate.  Converts *fsupp* and *fc* to normalised units.
    norm    : str
        Filter normalisation (``'energy'``, ``'1'``, ``'inf'``).
    delay   : int
        Desired filter delay in samples.
    scal    : float
        Additional scaling factor.
    min_win : int
        Minimum window length in samples.
    pedantic : bool
        If True, subsample window position is adjusted.
    realonly : bool
        If True, mark filter as real-valued (Hermitian symmetry).

    Returns
    -------
    g : dict  with keys ``'H'``, ``'foff'``, ``'realonly'``, ``'delay'``,
        ``'fs'``.

    Examples
    --------
    >>> from cool_frames.numpy.filters.lowlevel import blfilter
    >>> g = blfilter('hann', 1000, 0.5, fs=44100)
    >>> callable(g['H'])
    True
    >>> H = g['H'](8192)         # compact (support-length) frequency response
    >>> H.ndim == 1 and H.shape[0] <= 8192
    True
    """
    # Merge positional varargs (MATLAB-style flags) into keyword args
    if args:
        parsed = _parse_filter_varargs(args)
        if "fs" in parsed:
            fs = parsed["fs"]
        if "norm" in parsed:
            norm = parsed["norm"]
        if "delay" in parsed:
            delay = int(parsed["delay"])
        if "scal" in parsed:
            scal = float(parsed["scal"])
        if "min_win" in parsed:
            min_win = int(parsed["min_win"])
        if "pedantic" in parsed:
            pedantic = parsed["pedantic"]
        if "realonly" in parsed:
            realonly = parsed["realonly"]

    # Normalise to [0, 2] units
    if fs is not None:
        fsupp_norm = fsupp / fs * 2.0
        fc_norm    = fc    / fs * 2.0
    else:
        fsupp_norm = float(fsupp)
        fc_norm    = float(fc)

    fc_norm = float(modcent(fc_norm, 2.0))

    # Resolve window name(s)
    if isinstance(winname, (list, tuple)):
        wnames = list(winname)
    else:
        wnames = [str(winname)]

    def _win(win_len: int) -> np.ndarray:
        """Evaluate the window function at length win_len.

        Supports both simple window names and tapering windows.
        """
        if len(wnames) >= 3 and wnames[1].lower() == "taper":
            ratio = float(wnames[2])
            return firwin_taper(wnames[0], wnames[1], ratio, win_len, norm="inf")
        return firwin(wnames[0], win_len, norm="inf")

    def _apply_scal_norm(h: np.ndarray, L: int) -> np.ndarray:
        """Apply scaling and normalization to a filter response.

        Parameters
        ----------
        h : np.ndarray
            Filter response (typically from _win).
        L : int
            DFT length (used for energy normalization).

        Returns
        -------
        np.ndarray
            Scaled and normalized filter response.
        """
        if norm in ("energy", "2"):
            # Energy norm: (1/L) * sum|H|^2 = 1  =>  ||H|| = sqrt(L)
            h_norm = np.linalg.norm(h)
            if h_norm > 0:
                return h * scal * math.sqrt(L) / h_norm
            return h
        elif norm in ("1", "area"):
            h_sum = np.sum(np.abs(h))
            if h_sum > 0:
                return h * scal * float(L) / h_sum  # type: ignore[no-any-return]
            return h
        else:  # inf / peak
            mx = np.max(np.abs(h))
            if mx > 0:
                return h * scal / mx  # type: ignore[no-any-return]
            return h

    def _fc_offset(L: int) -> float:
        """Compute subsample offset for centre frequency adjustment.

        Returns the fractional error in centre-frequency alignment, used for
        pedantic mode to apply linear-phase correction.
        """
        if pedantic:
            return L / 2 * fc_norm - round(L / 2 * fc_norm)
        return 0.0

    def H(L: int) -> np.ndarray:
        win_len = max(min_win, round(L / 2 * fsupp_norm))
        h = _win(win_len)
        shift = _fc_offset(L)
        if shift != 0:
            # Sub-sample shift via linear phase
            n = np.arange(len(h)) - len(h) // 2
            h = h * np.exp(-1j * 2 * np.pi * n * shift / len(h))
        return _apply_scal_norm(h, L)

    def foff(L: int) -> int:
        win_len = max(min_win, round(L / 2 * fsupp_norm))
        # Compute the window to find the peak position
        h = _win(win_len)
        peak_idx = int(np.argmax(np.abs(h)))
        # Place the peak at the desired center frequency bin
        return int(round(L / 2 * fc_norm)) - peak_idx

    return {
        "H":        H,
        "foff":     foff,
        "realonly": 1 if realonly else 0,
        "delay":    int(delay),
        "fs":       fs,
    }


# ---------------------------------------------------------------------------
# firfilter – FIR (time-domain) filter
# ---------------------------------------------------------------------------

def firfilter(winname, M=None, fc=0.0, *args, delay=0, fs=None, norm="energy",
              scal=1.0, realonly=False, causal=False) -> dict:
    r"""Construct a time-domain FIR filter descriptor.

    Can be called in two modes:

    Mode 1 (new): Generate a windowed FIR filter
        firfilter(winname, M, fc, ...)
        winname : str - window name (passed to firwin)
        M : int - window length
        fc : float - centre frequency (default 0)
        Additional args: MATLAB-style flags/key-value pairs

    Mode 2 (legacy): Use pre-computed impulse response
        firfilter(h_array, ...)
        h_array : ndarray - impulse response

    Parameters
    ----------
    winname : str or ndarray
        Window name for Mode 1, or impulse response array for Mode 2
    M : int, optional
        Window length (Mode 1 only). If None and winname is an array,
        falls back to Mode 2 (legacy impulse response mode).
    fc : float or str
        Centre frequency (Mode 1 only, default 0). If a string, treated
        as a MATLAB-style flag and moved to ``*args`` for parsing.
    \*args
        MATLAB-style flags/key-value pairs (Mode 1 only).
        Supported flags: 'energy', 'peak', 'real', 'causal'
        Supported key-value pairs: 'delay', 'fs', 'scal', 'norm'
    delay : int
        Desired delay in samples.
    fs : float, optional
        Sampling rate.
    norm : str
        Filter normalisation ('energy', 'peak', 'inf', '1', '2', 'area').
    scal : float
        Additional scaling factor.
    realonly : bool
        If True, mark filter as real-valued.
    causal : bool
        If True, set offset to 0 (causal); else offset = -M//2.

    Returns
    -------
    g : dict with keys ``'h'``, ``'offset'``, ``'delay'``, ``'realonly'``, ``'fs'``.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters.lowlevel import firfilter
    >>> g = firfilter('hann', 128, fc=0.25)
    >>> g['h'].shape
    (128,)
    >>> g['offset']
    -64
    """
    # If fc is a string, it's actually a flag that should be in args
    if isinstance(fc, str):
        args = (fc,) + args
        fc = 0.0

    # Merge positional varargs (MATLAB-style flags) into keyword args
    if args:
        parsed = _parse_filter_varargs(args)
        if "delay" in parsed:
            delay = int(parsed["delay"])
        if "fs" in parsed:
            fs = parsed["fs"]
        if "norm" in parsed:
            norm = parsed["norm"]
        if "scal" in parsed:
            scal = float(parsed["scal"])
        if "realonly" in parsed:
            realonly = parsed["realonly"]
        if "causal" in parsed:
            causal = parsed["causal"]

    # Mode 2: Legacy impulse response mode
    if M is None and not isinstance(winname, str):
        h_arr = np.asarray(winname, dtype=float)
        return {
            "h":        h_arr,
            "offset":   int(delay),
            "delay":    int(delay),
            "realonly": 1 if realonly else 0,
            "fs":       fs,
        }

    # Mode 1: Generate windowed FIR filter
    if M is None:
        raise ValueError("firfilter: M (window length) required for named windows")

    M = int(M)

    # Convert fc from LTFAT convention [0, 2] to per-sample frequency [0, 1]
    # LTFAT: fc ∈ [0, 2], where 2 = Nyquist = fs/2
    # Per-sample: fc_sample ∈ [0, 1], where 1 = Nyquist
    fc_sample = fc / 2.0

    h = firwin(str(winname), M, norm="inf")  # Get unnormalized window

    # Apply centre frequency shift if fc != 0
    if fc_sample != 0.0:
        n = np.arange(M) - M / 2
        h = h * np.exp(1j * 2.0 * np.pi * fc_sample * n)

    # Apply normalisation
    if norm.lower() in ("energy", "2"):
        # Energy normalisation: sum(h^2) == 1
        h_energy = np.sum(np.abs(h) ** 2)
        if h_energy > 0:
            h = h / np.sqrt(h_energy)
    elif norm.lower() in ("peak", "inf"):
        # Peak normalisation: max|h| == 1
        h_max = np.max(np.abs(h))
        if h_max > 0:
            h = h / h_max
    elif norm.lower() in ("1", "area"):
        # Area normalisation: sum|h| == 1
        h_sum = np.sum(np.abs(h))
        if h_sum > 0:
            h = h / h_sum

    # Apply scaling
    h = h * float(scal)

    # Determine offset
    if causal:
        offset = 0
    else:
        offset = -M // 2

    return {
        "h":        h,
        "offset":   int(offset),
        "delay":    int(delay),
        "realonly": 1 if realonly else 0,
        "fs":       fs,
    }


# ---------------------------------------------------------------------------
# freqfilter – frequency-domain filter specified as full frequency response
# ---------------------------------------------------------------------------

def freqfilter(winname, fsupp: float, fc: float = 0.0, *args,
               fs: float | None = None,
               norm: str = "energy",
               delay: int = 0,
               scal: float = 1.0,
               min_win: int = 1,
               pedantic: bool = False,
               bwtruncmul: float = np.inf) -> dict:
    """Frequency-domain filter (full-length transfer function).

    Delegates to :func:`blfilter` for now (band-limited approximation).

    Parameters
    ----------
    winname : str
        Window name.
    fsupp : float
        Frequency support in Hz (if fs given) or normalized units.
    fc : float
        Centre frequency.
    *args
        MATLAB-style flags/key-value pairs.
    fs : float, optional
        Sampling rate in Hz.
    norm : str
        Normalization type.
    delay : int
        Filter delay in samples.
    scal : float
        Scaling factor.
    min_win : int
        Minimum window length.
    pedantic : bool
        Subsample adjustment flag.
    bwtruncmul : float
        Bandwidth truncation multiplier (unused).

    Returns
    -------
    dict
        Filter descriptor.

    Examples
    --------
    >>> from cool_frames.numpy.filters.lowlevel import freqfilter
    >>> g = freqfilter('hann', 500, 1000, fs=16000)
    >>> 'H' in g
    True
    """
    # Merge positional varargs (MATLAB-style flags) into keyword args
    realonly = False
    if args:
        parsed = _parse_filter_varargs(args)
        if "fs" in parsed:
            fs = parsed["fs"]
        if "norm" in parsed:
            norm = parsed["norm"]
        if "delay" in parsed:
            delay = int(parsed["delay"])
        if "scal" in parsed:
            scal = float(parsed["scal"])
        if "min_win" in parsed:
            min_win = int(parsed["min_win"])
        if "pedantic" in parsed:
            pedantic = parsed["pedantic"]
        if "realonly" in parsed:
            realonly = parsed["realonly"]

    return blfilter(winname, fsupp, fc, fs=fs, norm=norm, delay=delay,
                    scal=scal, min_win=min_win, pedantic=pedantic, realonly=realonly)


# ---------------------------------------------------------------------------
# Helper: evaluate a filter's full-length frequency response
# ---------------------------------------------------------------------------

def filter_freqresp(g: dict, L: int) -> tuple[np.ndarray, int]:
    """Return ``(H_full, foff_int)`` for filter *g* at length *L*.

    ``H_full`` is a *dense* (L,) complex array with the filter's
    frequency-response samples placed at their correct DFT bins.

    Parameters
    ----------
    g : dict
        Filter descriptor (from blfilter, firfilter, etc.).
    L : int
        DFT length.

    Returns
    -------
    H_full : (L,) complex array
        Full-length frequency response.
    foff_v : int
        Frequency offset of the filter support.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters.lowlevel import blfilter
    >>> from cool_frames.numpy.filters import filter_freqresp
    >>> g = blfilter('hann', 500, 1000, fs=8000)
    >>> H, foff = filter_freqresp(g, 2048)
    >>> H.shape
    (2048,)
    >>> isinstance(foff, (int, np.integer))
    True
    """
    H_full = np.zeros(L, dtype=complex)

    if "H" in g:
        H_callable = g["H"]
        foff_callable = g["foff"]

        if callable(H_callable):
            H_vals = np.asarray(H_callable(L), dtype=complex)
        else:
            H_vals = np.asarray(H_callable, dtype=complex)

        if callable(foff_callable):
            foff_v = int(foff_callable(L))
        else:
            foff_v = int(foff_callable)

        # Handle scalar sentinel (zero filter)
        if H_vals.ndim == 0 or (H_vals.size == 1 and H_vals.flat[0] == 0):
            return H_full, 0

        n_h = len(H_vals)
        idx = np.mod(np.arange(foff_v, foff_v + n_h), L)
        H_full[idx] += H_vals

        # NOTE (2026-06-13): no ``realonly`` conjugate-mirror is applied here.
        # cool_frames's analysis/synthesis kernels (``comp_filterbank_fftbl`` and its
        # inverse) deliberately ignore the ``realonly`` flag -- they keep the
        # coefficients complex and let ``ifilterbank(..., real=True)`` mirror the
        # one-sided spectrum via ``2*real(ifft)`` (machine-precision round-trip;
        # the MATLAB ``(H+involute(H))/2`` averaging discarded the imaginary part
        # and lost ~6 dB). ``filter_freqresp`` must report the SAME transfer
        # function the transform actually applies, so it returns the stored
        # single-sided response without the mirror. Applying the mirror here
        # (the old behaviour) made the frame response / bounds / painless
        # tight-frame construction inconsistent with the transform, which left
        # a canonical tight CQT frame reading kappa=2 instead of 1.
        return H_full, foff_v

    elif "h" in g:
        # FIR filter – apply offset via circular shift
        h = np.asarray(g["h"])
        offset = g.get("offset", 0)
        H_full = np.fft.fft(np.roll(postpad(h, L), offset))  # type: ignore[assignment]
        return H_full, 0

    return H_full, 0


# ---------------------------------------------------------------------------
# comp_biquad – full-length DFT response of a second-order IIR resonator
# ---------------------------------------------------------------------------

def comp_biquad(r: float, theta: float, L: int,
                norm: str = "energy") -> np.ndarray:
    """Full-length DFT response of a second-order allpole resonator.

    Computes ``H(k) = 1 / D(z⁻¹)`` for ``k = 0, …, L-1``, where
    ``D(z⁻¹) = 1 − 2r cos(θ) z⁻¹ + r² z⁻²``, giving conjugate poles
    at ``z = r exp(±iθ)``.

    Parameters
    ----------
    r : float
        Pole radius, ``0 < r < 1`` (``r ≥ 1`` is unstable).
    theta : float
        Pole angle in radians, ``0 ≤ θ ≤ π``.
    L : int
        Transform length; output has length *L*.
    norm : str
        Normalisation string:

        - ``'energy'`` — ``(1/L) Σ|H|² = 1``, i.e. ``‖H‖₂ = √L``
        - ``'inf'`` / ``'peak'`` — ``max|H| = 1``  (consistent with ‖·‖∞)
        - ``'1'`` — ``(1/L) Σ|H| = 1``
        - ``'none'`` — raw (unnormalised allpole response)

    Returns
    -------
    H : ndarray, shape (L,), complex
    """
    omega = 2.0 * np.pi * np.arange(L) / L
    z_inv = np.exp(-1j * omega)

    a1 = -2.0 * r * math.cos(theta)
    a2 = r * r
    D = 1.0 + a1 * z_inv + a2 * (z_inv * z_inv)

    H_raw = 1.0 / D

    if norm in ("energy", "2"):
        H_raw *= math.sqrt(L) / np.linalg.norm(H_raw)
    elif norm in ("inf", "peak"):
        H_raw /= np.max(np.abs(H_raw))
    elif norm in ("1", "area"):
        H_raw *= L / np.sum(np.abs(H_raw))
    # else ("none" or anything else): no normalisation

    return H_raw  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# comp_transferfunction – evaluate filter's full-length transfer function
# ---------------------------------------------------------------------------

def comp_transferfunction(g: dict, L: int) -> np.ndarray:
    """Evaluate a filter's full-length transfer function at DFT length L.

    Evaluates the full-length L-point frequency response of a filter struct.

    Parameters
    ----------
    g : dict
        Filter descriptor (from blfilter, firfilter, freqfilter, biquadfilter, etc.)
    L : int
        DFT length

    Returns
    -------
    H : (L,) complex array
        Transfer function samples
    """
    H, foff = filter_freqresp(g, L)

    # Apply delay parameter (phase shift in frequency domain)
    if "delay" in g and g.get("delay", 0) != 0:
        delay = int(g["delay"])
        k = np.arange(L)
        H = H * np.exp(-2j * np.pi * delay * k / L)

    return H


# ---------------------------------------------------------------------------
# comp_filterbank_pre – sanitize and evaluate filter parameters
# ---------------------------------------------------------------------------

def comp_filterbank_pre(g_cell: list[dict], a, L: int, crossover: int = 0) -> list[dict]:
    """Return sanitized filterbank with all parameters evaluated.

    Evaluates all callable parameters of the filters (H, foff) that depend on L.
    Converts filters to numeric form suitable for FFT-based processing.

    Parameters
    ----------
    g_cell : list of dict
        Filter descriptors (from blfilter, firfilter, etc.)
    a : int or array_like
        Hop size(s) – integer or (M, 2) fractional
    L : int
        Signal/DFT length
    crossover : int, optional
        Threshold for separating time-domain from frequency-domain filters.
        If 0 (default), use all filters in frequency domain (after evaluation).

    Returns
    -------
    g_out : list of dict
        Sanitized filter descriptors with:
        - All callable 'H' evaluated to numeric arrays
        - All callable 'foff' evaluated to integers
        - 'delay' parameter applied via phase modulation
        - 'fc' parameter applied via modulation (if present)
        - 'realonly' normalized (converted to time or frequency domain)
    """
    g_out = [dict(g) for g in g_cell]  # Make shallow copies
    M = len(g_out)
    a_arr = np.atleast_2d(np.asarray(a))
    if a_arr.shape[0] == 1 and a_arr.shape[1] == 1:
        a_arr = np.column_stack([np.full(M, a_arr.flat[0]), np.ones(M)])
    elif a_arr.ndim == 1 or a_arr.shape[1] == 1:
        a_arr = a_arr.ravel()
        if len(a_arr) == 1:
            a_arr = np.column_stack([np.full(M, a_arr[0]), np.ones(M)])
        else:
            a_arr = np.column_stack([a_arr[:M], np.ones(M)])

    # Check for fractional subsampling
    is_fractional = not np.allclose(a_arr[:, 1], 1.0)

    # Identify time-domain vs frequency-domain filters
    m_time = []
    for m in range(M):
        if "h" in g_out[m]:
            h_len = len(np.asarray(g_out[m]["h"]))
            if h_len <= crossover:
                m_time.append(m)

    # Process time-domain filters
    for m in m_time:
        # Handle .fc parameter (frequency modulation)
        if "fc" in g_out[m] and g_out[m].get("fc", 0) != 0:
            h = np.asarray(g_out[m]["h"])
            offset = g_out[m].get("offset", 0)
            l = (np.arange(offset, offset + len(h)) / float(L))
            fc_val = g_out[m]["fc"]
            g_out[m]["h"] = h * np.exp(2j * np.pi * np.round(fc_val * L / 2.0) * l)
            g_out[m]["fc"] = 0

        # Convert realonly filters to time domain
        if g_out[m].get("realonly", 0):
            g_out[m]["h"] = np.real(np.asarray(g_out[m]["h"]))
            g_out[m]["realonly"] = 0

        # Normalize offset to [0, len(h)-1] range
        h_len = len(np.asarray(g_out[m]["h"]))
        offset = g_out[m].get("offset", 0)

        if offset > 0:
            # Prepend zeros
            g_out[m]["h"] = np.concatenate(
                [np.zeros(offset, dtype=np.asarray(g_out[m]["h"]).dtype),
                 np.asarray(g_out[m]["h"])]
            )
            g_out[m]["offset"] = 0
        elif offset < -(h_len - 1):
            # Append zeros
            g_out[m]["h"] = np.concatenate(
                [np.asarray(g_out[m]["h"]),
                 np.zeros(-offset - h_len + 1, dtype=np.asarray(g_out[m]["h"]).dtype)]
            )
            g_out[m]["offset"] = -(len(np.asarray(g_out[m]["h"])) - 1)

    # Process frequency-domain filters
    m_freq = [m for m in range(M) if m not in m_time]

    for m in m_freq:
        # Convert time-domain h to frequency domain if needed
        if "h" in g_out[m]:
            h = np.asarray(g_out[m]["h"])
            offset = g_out[m].get("offset", 0)
            tmpg = np.roll(postpad(h, L), offset)

            # Apply fc modulation in frequency domain
            if "fc" in g_out[m] and g_out[m].get("fc", 0) != 0:
                l = np.arange(L) / float(L)
                fc_val = g_out[m]["fc"]
                tmpg = tmpg * np.exp(2j * np.pi * np.round(fc_val * L / 2.0) * l)
                del g_out[m]["fc"]

            g_out[m]["H"] = np.fft.fft(tmpg)
            g_out[m]["foff"] = 0
            g_out[m]["L"] = L
            del g_out[m]["h"]
            if "offset" in g_out[m]:
                del g_out[m]["offset"]

        elif "H" in g_out[m]:
            # Evaluate H if it's callable
            H_val = g_out[m]["H"]
            if callable(H_val):
                g_out[m]["H"] = np.asarray(H_val(L), dtype=complex)
                g_out[m]["L"] = L
            elif isinstance(H_val, np.ndarray):
                # Already numeric; ensure it's stored with L
                if "L" not in g_out[m]:
                    g_out[m]["L"] = L

            # Evaluate foff if it's callable
            if "foff" in g_out[m]:
                foff_val = g_out[m]["foff"]
                if callable(foff_val):
                    g_out[m]["foff"] = int(foff_val(L))
            else:
                g_out[m]["foff"] = 0

            # Apply delay parameter (phase modulation)
            if "delay" in g_out[m] and g_out[m].get("delay", 0) != 0:
                delay = g_out[m]["delay"]
                foff = g_out[m].get("foff", 0)
                H_len = len(np.asarray(g_out[m]["H"]))
                lrange = np.mod(np.arange(foff, foff + H_len), L) / float(L)
                g_out[m]["H"] = (np.asarray(g_out[m]["H"]) *
                                 np.exp(-2j * np.pi * np.round(delay) * lrange))
                g_out[m]["delay"] = 0

            # Handle full-length H with non-fractional sampling
            H_len = len(np.asarray(g_out[m]["H"]))
            if H_len == L and not is_fractional:
                foff = g_out[m].get("foff", 0)
                if foff != 0:
                    # Apply frequency offset
                    g_out[m]["H"] = np.roll(np.asarray(g_out[m]["H"]), foff)
                    g_out[m]["foff"] = 0

                # Apply realonly symmetry
                if g_out[m].get("realonly", 0):
                    g_out[m]["H"] = (np.asarray(g_out[m]["H"]) + involute(np.asarray(g_out[m]["H"]))) / 2.0
                    g_out[m]["realonly"] = 0

    return g_out


# ---------------------------------------------------------------------------
# biquadfilter – IIR biquad resonator filter descriptor
# ---------------------------------------------------------------------------

def _parse_biquad_varargs(args):
    """Parse MATLAB-style varargs for biquadfilter.

    Accepts positional flag/key-value pairs after (fc, bw):
        biquadfilter(fc, bw, "peak")
        biquadfilter(fc, bw, "fs", 44100)
        biquadfilter(fc, bw, "rho", 2.0, "phi", 0.5)

    Returns a dict of keyword arguments.
    """
    flags = {"energy", "peak", "inf", "1", "2", "area", "real", "complex"}
    kv_keys = {"fs", "delay", "scal", "rho", "phi"}

    kw: dict = {}
    i = 0
    while i < len(args):
        a = args[i]
        if isinstance(a, str):
            low = a.lower()
            if low in kv_keys:
                if i + 1 >= len(args):
                    raise ValueError(f"biquadfilter: '{a}' requires a value")
                kw[low] = args[i + 1]
                i += 2
            elif low in flags:
                if low in ("real", "complex"):
                    kw["realonly"] = (low == "real")
                else:
                    kw["norm"] = low
                i += 1
            else:
                raise ValueError(f"biquadfilter: unknown flag '{a}'")
        else:
            raise ValueError(f"biquadfilter: unexpected positional arg {a!r}")
    return kw


def biquadfilter(fc, bw, *args,
                 fs: float | None = None,
                 norm: str = "energy",
                 delay: int = 0,
                 scal: float = 1.0,
                 realonly: bool = False,
                 rho: float | None = None,
                 phi: float | None = None) -> dict | list[dict]:
    """Construct a second-order IIR (biquad) resonator filter descriptor.

    Builds an allpole resonator with conjugate poles at
    ``z = r exp(±iθ)``.  The centre frequency and bandwidth determine
    the pole parameters via ``θ = π|fc|`` and ``r = 1 − π bw / 2``.

    For machine-learning use, the pole parameters are stored in
    unconstrained form: ``rho = logit(r)`` and ``phi = logit(θ/π)``.
    Gradient descent on ``(rho, phi)`` always produces stable filters
    because ``r = sigmoid(rho) ∈ (0, 1)`` and ``θ = π sigmoid(phi) ∈
    (0, π)`` by construction.

    Parameters
    ----------
    fc : float or array_like
        Centre frequency in normalised units [0, 2] (where 1 = Nyquist).
        If *fs* is given, in Hz.
    bw : float or array_like
        Approximate −3 dB bandwidth (same units as *fc*).
    *args
        MATLAB-style flags and key-value pairs.  Supported flags:
        ``'energy'``, ``'peak'``, ``'inf'``, ``'1'``, ``'real'``,
        ``'complex'``.  Supported key-value pairs: ``'fs'``, ``'delay'``,
        ``'scal'``, ``'rho'``, ``'phi'``.
    fs : float, optional
        Sampling rate.  Converts *fc* and *bw* to normalised units.
    norm : str
        Normalisation: ``'energy'`` (default), ``'inf'``/``'peak'``,
        ``'1'``, ``'none'`` (raw).
    delay : int
        Desired filter delay in samples.
    scal : float
        Amplitude scaling factor.
    realonly : bool
        If *True*, mark filter as real-valued (Hermitian symmetry).
    rho : float, optional
        Override pole radius via unconstrained ML parameter:
        ``r = sigmoid(rho)``.
    phi : float, optional
        Override pole angle via unconstrained ML parameter:
        ``θ = π sigmoid(phi)``.

    Returns
    -------
    g : dict or list[dict]
        Filter descriptor(s) with keys ``'H'``, ``'foff'``, ``'realonly'``,
        ``'delay'``, ``'fs'``, ``'fc'``, ``'bw'``, ``'r'``, ``'theta'``,
        ``'rho'``, ``'phi'``.
        Scalar inputs return a single dict; vector inputs return a list.
    """
    # Merge positional varargs (MATLAB-style flags) into keyword args
    if args:
        parsed = _parse_biquad_varargs(args)
        if "fs" in parsed:
            fs = parsed["fs"]
        if "norm" in parsed:
            norm = parsed["norm"]
        if "delay" in parsed:
            delay = int(parsed["delay"])
        if "scal" in parsed:
            scal = float(parsed["scal"])
        if "realonly" in parsed:
            realonly = parsed["realonly"]
        if "rho" in parsed:
            rho = float(parsed["rho"])
        if "phi" in parsed:
            phi = float(parsed["phi"])

    fc_arr = np.atleast_1d(np.asarray(fc, dtype=float))
    bw_arr = np.atleast_1d(np.asarray(bw, dtype=float))

    # Broadcast scalar inputs
    N = max(len(fc_arr), len(bw_arr))
    if len(fc_arr) == 1 and N > 1:
        fc_arr = np.full(N, fc_arr[0])
    if len(bw_arr) == 1 and N > 1:
        bw_arr = np.full(N, bw_arr[0])

    # Convert Hz → normalised
    if fs is not None:
        fc_arr = fc_arr / fs * 2.0
        bw_arr = bw_arr / fs * 2.0

    # Centre-wrap
    fc_arr = np.asarray([float(modcent(f, 2.0)) for f in fc_arr])

    gout: list[dict] = []

    for ii in range(N):
        fc_ii = fc_arr[ii]
        bw_ii = bw_arr[ii]

        # Pole angle
        if phi is not None:
            phi_ii = float(phi)
            theta_ii = math.pi / (1.0 + math.exp(-phi_ii))
        else:
            theta_ii = math.pi * abs(fc_ii)
            frac = theta_ii / math.pi
            if 0 < frac < 1:
                phi_ii = math.log(frac) - math.log(1.0 - frac)
            else:
                phi_ii = 0.0

        # Pole radius
        if rho is not None:
            rho_ii = float(rho)
            r_ii = 1.0 / (1.0 + math.exp(-rho_ii))
        else:
            r_ii = max(0.0, min(1.0 - 1e-6, 1.0 - math.pi * bw_ii / 2.0))
            if 0 < r_ii < 1:
                rho_ii = math.log(r_ii) - math.log(1.0 - r_ii)
            else:
                rho_ii = 0.0

        # Build closures with captured values
        r_c, theta_c, scal_c, norm_c = r_ii, theta_ii, scal, norm

        def _make_H(r_v, th_v, sc, nm):
            """Create a closure for the biquad frequency response callable.

            Captures pole parameters (r, theta) and normalization settings.
            """
            def H(L: int) -> np.ndarray:
                return comp_biquad(r_v, th_v, L, nm) * sc  # type: ignore[no-any-return]
            return H

        def _make_foff():
            """Create a closure for the frequency offset callable.

            Biquad filters are full-length responses with foff=0.
            """
            def foff(L: int) -> int:
                return 0
            return foff

        g_dict: dict = {
            "H":        _make_H(r_c, theta_c, scal_c, norm_c),
            "foff":     _make_foff(),
            "realonly":  1 if realonly else 0,
            "delay":     int(delay),
            "fs":        fs,
            # Physical parameters
            "fc":        fc_ii,
            "bw":        bw_ii,
            "r":         r_ii,
            "theta":     theta_ii,
            # ML parameters
            "rho":       rho_ii,
            "phi":       phi_ii,
        }
        gout.append(g_dict)

    if N == 1 and np.ndim(fc) == 0:
        return gout[0]
    return gout
