"""
numpy/filters/_audscale.py
==========================
Auditory frequency-scale conversions and bandwidth functions.

MATLAB originals
----------------
  freqtoaud.m  – frequency → auditory units
  audtofreq.m  – auditory units → frequency (Hz)
  audspace.m   – linearly-spaced auditory grid
  audfiltbw.m  – ERB-type bandwidth at a centre frequency
  gammatonefir.m – gammatone FIR filter coefficients
"""
from __future__ import annotations

import math
from types import MappingProxyType
from typing import Mapping

import numpy as np

# Supported scale names (aliases accepted)
_SCALE_ALIASES: dict[str, str] = {
    "erb":     "erb",
    "erb83":   "erb83",
    "bark":    "bark",
    "mel":     "mel",
    "mel1000": "mel1000",
    "greenwood": "greenwood",
    # general (non-auditory) scales (added 2026-06-12)
    "linear": "linear", "lin": "linear", "hz": "linear",
    "log": "log", "octave": "log", "oct": "log",
    "semitone": "semitone", "midi": "semitone",
    "third-octave": "third-octave", "third_octave": "third-octave",
    "thirdoct": "third-octave", "tertz": "third-octave",
}

# ---------------------------------------------------------------------------
# Greenwood function parameters
# ---------------------------------------------------------------------------
# The Greenwood frequency-position function is  f(x) = A * (10^(alpha*x) - k).
#
# GREENWOOD_DEFAULTS are the HUMAN cochlea parameters (Greenwood 1990) and are
# IMMUTABLE (a read-only mapping). There is no global setter: to use a different
# species, pass the parameters per call to ``greenwoodfilters(A=..., alpha=...,
# k=...)`` (a pure function of its arguments, unlike the old mutable global).
#
# Common species parameters (Greenwood 1990, Table I):
#     Species      A        alpha    k
#     Human        165.4    2.1      0.88   <- default
#     Cat          456.0    0.8      0.80
#     Guinea pig   350.0    0.85     0.80
#     Chinchilla   163.5    0.85     0.10
#     Elephant     200.0    1.4      0.85
GREENWOOD_DEFAULTS: Mapping[str, float] = MappingProxyType({
    "A": 165.4,
    "alpha": 2.1,
    "k": 0.88,
})


def _check_scale(scale: str) -> str:
    """Validate and normalize an auditory scale name.

    Parameters
    ----------
    scale : str
        Auditory scale name (case-insensitive).

    Returns
    -------
    str
        Normalized scale name from _SCALE_ALIASES.

    Raises
    ------
    ValueError
        If scale is not recognized.
    """
    key = scale.lower()
    if key not in _SCALE_ALIASES:
        raise ValueError(f"Unknown auditory scale: {scale!r}. "
                         f"Supported: {list(_SCALE_ALIASES)}")
    return _SCALE_ALIASES[key]


# ---------------------------------------------------------------------------
# freqtoaud
# ---------------------------------------------------------------------------

def freqtoaud(f, scale: str = "erb"):
    """Convert frequency (Hz) to auditory units.

    Parameters
    ----------
    f     : float or array_like
        Frequency in Hz.
    scale : {'erb', 'erb83', 'bark', 'mel', 'mel1000', 'greenwood'}
        Auditory scale. Default: 'erb' (Glasberg & Moore 1990).

    Returns
    -------
    aud : same shape as *f*
        Frequency in auditory units.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import freqtoaud
    >>> round(float(freqtoaud(1000.0)), 3)  # 1 kHz on the ERB-rate scale
    15.572
    >>> np.round(freqtoaud([100, 1000, 10000], scale='erb'), 3)
    array([ 3.359, 15.572, 35.205])
    """
    scale = _check_scale(scale)
    f = np.asarray(f, dtype=float)

    if scale == "erb":
        # Glasberg & Moore (1990) ERB scale
        return 9.2645 * np.sign(f) * np.log(1.0 + np.abs(f) * 0.00437)

    elif scale == "erb83":
        # Moore & Glasberg (1983) ERB scale
        return (1000.0 / (24.673 * 4.368)) * np.log(1.0 + np.abs(f) / 228.833)

    elif scale == "bark":
        # Zwicker & Terhardt (1980)
        return (26.81 / (1.0 + 1960.0 / np.abs(f).clip(1e-6))) - 0.53

    elif scale == "mel":
        return 2595.0 * np.log10(1.0 + np.abs(f) / 700.0)

    elif scale == "mel1000":
        return 1000.0 / np.log(2.0) * np.log(1.0 + np.abs(f) / 1000.0)

    elif scale == "greenwood":
        # Greenwood inverse: f → x = log10((f/A) + k) / α
        gw = GREENWOOD_DEFAULTS
        return np.log10(np.abs(f) / gw["A"] + gw["k"]) / gw["alpha"]  # type: ignore[no-any-return]

    elif scale == "linear":
        return f  # Hz (identity)

    elif scale == "log":
        # octaves relative to 1 Hz
        return np.log2(np.abs(f).clip(1e-12))  # type: ignore[no-any-return]

    elif scale == "semitone":
        # MIDI note number (A4 = 440 Hz = 69)
        return 69.0 + 12.0 * np.log2(np.abs(f).clip(1e-12) / 440.0)  # type: ignore[no-any-return]

    elif scale == "third-octave":
        # 1/3-octave band index relative to 1 kHz (IEC 61260)
        return 3.0 * np.log2(np.abs(f).clip(1e-12) / 1000.0)  # type: ignore[no-any-return]

    raise ValueError(scale)  # unreachable


# ---------------------------------------------------------------------------
# audtofreq
# ---------------------------------------------------------------------------

def audtofreq(aud, scale: str = "erb"):
    """Convert auditory units back to frequency (Hz).

    Parameters
    ----------
    aud   : float or array_like
        Frequency in auditory units.
    scale : {'erb', 'erb83', 'bark', 'mel', 'mel1000'}
        Auditory scale. Default: 'erb' (Glasberg & Moore 1990).

    Returns
    -------
    f : same shape as *aud*
        Frequency in Hz.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audtofreq
    >>> round(float(audtofreq(15.572)), 1)  # back to ~1 kHz
    1000.0
    >>> np.round(audtofreq([1, 5, 10], scale='mel'), 3)
    array([0.621, 3.113, 6.239])
    """
    scale = _check_scale(scale)
    aud = np.asarray(aud, dtype=float)

    if scale == "erb":
        return (1.0 / 0.00437) * np.sign(aud) * (np.expm1(np.abs(aud) / 9.2645))

    elif scale == "erb83":
        return 228.833 * (np.exp(np.abs(aud) * 24.673 * 4.368 / 1000.0) - 1.0)

    elif scale == "bark":
        return 1960.0 / (26.81 / (aud + 0.53) - 1.0)

    elif scale == "mel":
        return 700.0 * (10.0 ** (np.abs(aud) / 2595.0) - 1.0)

    elif scale == "mel1000":
        return 1000.0 * (np.exp(np.abs(aud) * np.log(2.0) / 1000.0) - 1.0)

    elif scale == "greenwood":
        # Inverse Greenwood: x → f = A * (10^(α*x) - k)
        gw = GREENWOOD_DEFAULTS
        return gw["A"] * (10.0 ** (gw["alpha"] * np.abs(aud)) - gw["k"])  # type: ignore[no-any-return]

    elif scale == "linear":
        return aud

    elif scale == "log":
        return 2.0 ** aud  # type: ignore[no-any-return]

    elif scale == "semitone":
        return 440.0 * 2.0 ** ((aud - 69.0) / 12.0)  # type: ignore[no-any-return]

    elif scale == "third-octave":
        return 1000.0 * 2.0 ** (aud / 3.0)  # type: ignore[no-any-return]

    raise ValueError(scale)


# ---------------------------------------------------------------------------
# freqtoerb and erbtofreq – convenience aliases
# ---------------------------------------------------------------------------

def freqtoerb(f):
    """Convert frequency (Hz) to ERB-rate scale.

    Convenience alias for ``freqtoaud(f, 'erb')``.

    Parameters
    ----------
    f : float or array_like
        Frequency in Hz.

    Returns
    -------
    erb : same shape as *f*
        ERB-rate scale value(s).

    Examples
    --------
    >>> from cool_frames.numpy.filters import freqtoerb
    >>> round(float(freqtoerb(1000.0)), 3)
    15.572
    """
    return freqtoaud(f, "erb")


def erbtofreq(erb):
    """Convert ERB-rate scale to frequency (Hz).

    Convenience alias for ``audtofreq(erb, 'erb')``.

    Parameters
    ----------
    erb : float or array_like
        ERB-rate scale value(s).

    Returns
    -------
    f : same shape as *erb*
        Frequency in Hz.

    Examples
    --------
    >>> from cool_frames.numpy.filters import erbtofreq
    >>> round(float(erbtofreq(15.572)), 1)
    1000.0
    """
    return audtofreq(erb, "erb")


# ---------------------------------------------------------------------------
# audspace
# ---------------------------------------------------------------------------

def audspace(fmin: float, fmax: float, n: int, scale: str = "erb") -> np.ndarray:
    """Return *n* frequencies linearly spaced on the *scale* between
    *fmin* and *fmax* (in Hz).

    Parameters
    ----------
    fmin : float
        Minimum frequency in Hz.
    fmax : float
        Maximum frequency in Hz.
    n : int
        Number of frequencies.
    scale : {'erb', 'erb83', 'bark', 'mel', 'mel1000'}
        Auditory scale. Default: 'erb'.

    Returns
    -------
    np.ndarray
        Array of *n* frequencies in Hz.

    Examples
    --------
    >>> from cool_frames.numpy.filters import audspace
    >>> freqs = audspace(50, 16000, 128)
    >>> len(freqs)
    128
    >>> round(float(freqs[0]), 6), round(float(freqs[-1]), 6)
    (50.0, 16000.0)
    """
    aud_min = freqtoaud(fmin, scale)
    aud_max = freqtoaud(fmax, scale)
    aud_pts = np.linspace(aud_min, aud_max, n)
    return audtofreq(aud_pts, scale)  # type: ignore[no-any-return]


def audspacebw(fmin: float, fmax: float, bw: float = 1.0,
               hitme=None, scale: str = "erb") -> np.ndarray:
    """Frequencies spaced ``bw`` auditory units apart between *fmin* and *fmax*.

    Port of LTFAT ``audspacebw`` (the engine behind ``erbspacebw``). Unlike
    :func:`audspace` (which fits *n* points to the endpoints), this steps by a
    fixed bandwidth ``bw`` on the auditory *scale*.

    ``hitme`` (a frequency in Hz, e.g. the model ``basef``) anchors the grid so
    one centre lands exactly on it: points are laid ``bw`` apart on either side
    of ``freqtoaud(hitme)``. With ``hitme=None`` the grid is centred in the
    ``[fmin, fmax]`` auditory range (LTFAT's default).

    Returns the centre frequencies in Hz.
    """
    if hitme is None:
        a_lo = freqtoaud(fmin, scale)
        a_hi = freqtoaud(fmax, scale)
        audrange = a_hi - a_lo
        n = int(np.floor(audrange / bw))
        remainder = audrange - n * bw
        audpoints = a_lo + np.arange(0, n + 1) * bw + remainder / 2.0
    else:
        a_lo = freqtoaud(fmin, scale)
        a_hi = freqtoaud(fmax, scale)
        a_mid = freqtoaud(hitme, scale)
        nlow = int(np.floor((a_mid - a_lo) / bw))
        nhigh = int(np.floor((a_hi - a_mid) / bw))
        audpoints = np.arange(-nlow, nhigh + 1) * bw + a_mid
    return np.asarray(audtofreq(audpoints, scale), dtype=float)


def erbspacebw(fmin: float, fmax: float, bw: float = 1.0, hitme=None) -> np.ndarray:
    """ERB-scale ``audspacebw`` — LTFAT ``erbspacebw(fmin,fmax,bw,hitme)``.

    Used by AMT ``auditoryfilterbank`` (dau1996/1997/jepsen2008/osses2021): the
    gammatone centre frequencies are ``bw``-ERB apart, anchored at the model
    ``basef`` (``hitme``).
    """
    return audspacebw(fmin, fmax, bw, hitme, scale="erb")


# ---------------------------------------------------------------------------
# audfiltbw
# ---------------------------------------------------------------------------

def audfiltbw(fc, scale: str = "erb") -> np.ndarray | float:
    """Return the ERB-type bandwidth of an auditory filter at centre
    frequency *fc* (Hz).

    Parameters
    ----------
    fc    : float or array_like
        Centre frequency in Hz.
    scale : {'erb', 'erb83', 'bark', 'mel', 'mel1000'}
        Auditory scale. Default: 'erb' (ANSI S1.11 approximation).

    Returns
    -------
    bw : same shape as *fc*
        Bandwidth in Hz.

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfiltbw
    >>> round(float(audfiltbw(1000.0)), 3)  # ERB bandwidth at 1 kHz
    132.633
    >>> np.round(audfiltbw([100, 1000, 10000], scale='erb'), 3)
    array([  35.493,  132.633, 1104.031])
    """
    scale = _check_scale(scale)
    fc = np.asarray(fc, dtype=float)

    if scale == "erb":
        # ANSI S1.11 approximation
        return 24.7 + fc / 9.265  # type: ignore[no-any-return]

    elif scale == "erb83":
        return 6.23e-6 * fc**2 + 93.39e-3 * fc + 28.52  # type: ignore[no-any-return]

    elif scale == "bark":
        return 25.0 + 75.0 * (1.0 + 1.4e-6 * fc**2) ** 0.69  # type: ignore[no-any-return]

    elif scale in ("mel", "mel1000"):
        # Approximate: use frequency spacing on scale
        delta_aud = 1.0  # one mel/bark unit
        return (  # type: ignore[no-any-return]
            audtofreq(freqtoaud(fc, scale) + delta_aud / 2, scale) -
            audtofreq(freqtoaud(fc, scale) - delta_aud / 2, scale)
        )

    elif scale == "greenwood":
        # Bandwidth = df/dx for unit spacing on the cochlear position axis.
        # df/dx = A · α · ln(10) · 10^(α·x) = α · ln(10) · (f + A·k)
        gw = GREENWOOD_DEFAULTS
        return gw["alpha"] * np.log(10.0) * (fc + gw["A"] * gw["k"])  # type: ignore[no-any-return]

    elif scale == "linear":
        return 0.0 * fc + 1.0  # constant 1 Hz/unit (broadcast to fc shape)

    elif scale == "log":
        return np.log(2.0) * np.abs(fc)  # constant-Q (~69% per octave)

    elif scale == "semitone":
        return (np.log(2.0) / 12.0) * np.abs(fc)  # ~5.8% (one semitone)

    elif scale == "third-octave":
        return (np.log(2.0) / 3.0) * np.abs(fc)  # ~23.1% (IEC 1/3-octave)

    raise ValueError(scale)


# ---------------------------------------------------------------------------
# gammatonefir – time-domain gammatone FIR filter
# ---------------------------------------------------------------------------

def gammatonefir(fc, fs, n=None, *, betamul=1.0183,
                 real=True, peakphase=False,
                 norm="null"):
    """Compute gammatone FIR filter coefficients.

    Port of LTFAT ``gammatonefir.m`` (Peter L. Søndergaard).

    The impulse response is

        g(t) = a * t^3 * cos(2π fc t) * exp(-2π β t)

    (4th-order gammatone with cosine carrier).

    Parameters
    ----------
    fc : float or array_like
        Centre frequency/frequencies in Hz.
    fs : float
        Sampling rate in Hz.
    n : int or None
        Maximum filter length.  ``None`` → 5000 (MATLAB default).
    betamul : float
        Bandwidth multiplier.  The bandwidth of each filter is
        ``betamul * audfiltbw(fc)``.  Default 1.0183 (Glasberg & Moore
        1990).
    real : bool
        If ``True`` (default), return real-valued (cosine-modulated)
        filters.  If ``False``, return complex (analytic) filters.
    peakphase : bool
        If ``True``, shift phase so that the peak of the envelope has
        zero phase.  Default ``False`` (causal phase).
    norm : str
        Normalisation passed to :func:`setnorm`.  Default ``'null'``
        (no normalisation; the natural scaling constant is used).

    Returns
    -------
    list of dict
        Each dict has keys ``'h'`` (ndarray, the impulse response),
        ``'offset'`` (int, sample offset for causal alignment), and
        ``'realonly'`` (int, 0).  This matches the LTFAT filter struct
        convention used by :func:`filterbank`.

    Examples
    --------
    >>> from cool_frames.numpy.filters import gammatonefir
    >>> filters = gammatonefir([500, 1000], fs=16000, n=2048)
    >>> len(filters)
    2
    >>> filters[0]['h'].shape
    (1120,)
    """
    fc = np.atleast_1d(np.asarray(fc, dtype=float))
    if np.any(fc < 0) or np.any(fc > fs / 2):
        raise ValueError("fc must be in [0, fs/2].")
    if fs <= 0:
        raise ValueError("fs must be positive.")
    if n is None:
        n = 5000

    nchannels = len(fc)
    ourbeta = betamul * audfiltbw(fc)

    filters = []
    for ii in range(nchannels):
        delay = 3.0 / (2.0 * np.pi * ourbeta[ii])

        # Scaling constant so that the filter has unit gain at fc
        scalconst = 2.0 * (2.0 * np.pi * ourbeta[ii]) ** 4 / math.factorial(3) / fs

        nfirst = int(np.ceil(fs * delay))

        if nfirst > n // 2:
            raise ValueError(
                f"Filter length {n} is too short for fc={fc[ii]:.1f} Hz. "
                f"Need at least {nfirst * 2} samples."
            )

        nlast = n // 2

        # Time axis: pre-peak samples then post-peak samples
        t_pre = np.arange(nfirst) / fs - nfirst / fs + delay
        t_post = np.arange(nlast) / fs + delay
        t = np.concatenate([t_pre, t_post])

        # g(t) = a * t^3 * carrier * exp(-2π β t)
        envelope = scalconst * t ** 3 * np.exp(-2.0 * np.pi * ourbeta[ii] * t)
        if real:
            bwork = envelope * np.cos(2.0 * np.pi * fc[ii] * t)
        else:
            bwork = envelope * np.exp(2j * np.pi * fc[ii] * t)

        if peakphase:
            bwork = bwork * np.exp(-2j * np.pi * fc[ii] * delay)
            if real:
                bwork = np.real(bwork)

        # Normalise if requested
        if norm.lower() not in ("null", "none", ""):
            from ..core._core import setnorm as _setnorm
            bwork, _ = _setnorm(bwork, norm)

        filters.append({
            "h": bwork,
            "offset": -nfirst,
            "realonly": 0,
        })

    return filters
