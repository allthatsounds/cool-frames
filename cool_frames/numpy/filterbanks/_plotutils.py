"""
_plotutils.py
=============
FFT visualization utility: plotfft.

Public API
----------
    plotfft(coef, fs=None, dynrange=None, ax=None, *, real=False, L=None) -> Axes
        Plot DFT magnitude spectrum.

        ``real=False`` (default): two-sided complex DFT coefficients (length L).
        ``real=True``: single-sided real-FFT coefficients (length L//2+1). Pass
        ``L`` = original signal length; if omitted it is inferred as
        ``2*(len(coef)-1)`` (assumes even length). This subsumes the former
        ``plotfftreal``.

Parameters
----------
coef : ndarray
    FFT coefficients (complex; one-sided when ``real=True``).
fs : float, optional
    Sampling rate in Hz.  If given, x-axis is in Hz; otherwise normalised.
dynrange : float, optional
    Dynamic range in dB.  Clips magnitude to [max(dB) - dynrange, max(dB)].
ax : matplotlib.axes.Axes, optional
    Target axes.  If None, uses current axes.
real : bool, keyword-only
    Treat ``coef`` as single-sided real-FFT output.
L : int, keyword-only
    Original signal length (real=True only).

Returns
-------
ax : matplotlib.axes.Axes
"""

from __future__ import annotations

import numpy as np

__all__ = ["plotfft"]


def plotfft(coef, fs=None, dynrange=None, ax=None, *, real=False, L=None):
    """Plot a DFT magnitude spectrum (two-sided, or single-sided if ``real``).

    See the module docstring for the parameter description. ``real=True``
    subsumes the former ``plotfftreal``.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotfft requires matplotlib.  Install it with:\n"
            "    pip install matplotlib"
        ) from exc

    coef = np.asarray(coef)
    mag = np.abs(coef)

    if dynrange is not None:
        eps_val = np.finfo(float).tiny
        mag_db = 20.0 * np.log10(mag + eps_val)
        max_val = np.max(mag_db)
        mag = np.maximum(mag_db, max_val - float(dynrange))

    if real:
        if L is None:
            L = 2 * (len(coef) - 1)          # infer even original length
        L = int(L)
        freq = np.fft.rfftfreq(L, d=1.0 / float(fs)) if fs is not None \
            else np.fft.rfftfreq(L)
        xlabel = "Frequency (Hz)" if fs is not None else "Normalized Frequency"
    else:
        N = len(coef)
        freq = np.fft.fftfreq(N, d=1.0 / float(fs)) if fs is not None \
            else np.fft.fftfreq(N)
        xlabel = "Frequency (Hz)" if fs is not None else "Bin"

    if ax is None:
        ax = plt.gca()

    ax.plot(freq, mag)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Magnitude (dB)" if dynrange is not None else "Magnitude")
    ax.grid(True, alpha=0.3)
    return ax
