"""
numpy/filters/_freqwin.py
=========================
Frequency-domain window and magnitude response functions.

Contains frequency-domain window functions (Gauss, Roex, Butterworth, Gammatone)
and magnitude response computation utilities.

MATLAB originals
----------------
  sigproc/freqwin.m
"""
from __future__ import annotations

import math

import numpy as np

from ..core._core import setnorm

# ---------------------------------------------------------------------------
# freqwin – frequency-domain window (Gauss / Roex / Butterworth / Gammatone)
# ---------------------------------------------------------------------------

def freqwin(name, L: int, bw: float, *,
            fs: float = 2.0,
            order: int | None = None,
            norm: str = "null") -> np.ndarray:
    """Frequency-response window.

    Returns a length-*L* window representing the frequency response of
    a band-pass filter with -6 dB bandwidth *bw*.

    Parameters
    ----------
    name : str
        ``'gauss'``, ``'butterworth'``, ``'roex'``, or ``'gammatone'``.

        ``'roex'`` — rounded exponential window with transfer function
        ``H(f) = (1 + i f/d)^{-n}`` with peak modulation to centre the
        magnitude envelope.  The MATLAB LTFAT calls this ``'gammatone'``
        inside ``freqwin``, but it is **not** a true gammatone — it is a
        roex(p) auditory filter shape with a symmetric magnitude.

        ``'gammatone'`` — true (analytic) gammatone frequency response:
        ``H(f) = 1 / (1 + i f/b)^n``, the Fourier transform of the
        causal gammatone impulse response
        ``g(t) = t^{n-1} exp(-2π b t) cos(2π fc t)``, centred at DC.
        This produces an **asymmetric** magnitude response (steeper on
        the low-frequency side) matching the auditory nerve tuning
        curve shape.
    L : int
        Number of frequency bins.
    bw : float
        Bandwidth (in Hz if *fs* given, else normalised to Nyquist = 1).
    fs : float
        Sampling rate (default 2 → normalised frequencies).
    order : int or None
        Filter order for ``'butterworth'``, ``'roex'``, and
        ``'gammatone'`` (default 4 for all three).
    norm : str
        Normalisation (default ``'null'``).

    Returns
    -------
    H : (L,) complex or float ndarray
    """
    name = name.lower().strip()

    step = fs / L
    bwrelheight = 10.0 ** (-3.0 / 10.0)  # -6 dB ≈ half-height

    # DFT frequency indices
    k = np.arange(L, dtype=float)
    k[k >= L // 2 + 1] -= L  # centred: 0..L/2, -(L/2-1)..-1
    H = k.copy()

    if name == "gauss":
        H = np.exp(4.0 * H ** 2 * math.log(bwrelheight) / (bw / step) ** 2)  # type: ignore[assignment]

    elif name == "butterworth":
        n = order if order is not None else 4
        # `bw` means the width at `bwrelheight` for every other window in this
        # function.  The textbook Butterworth form 1/sqrt(1 + (f/fc)^2n) puts
        # its half-power point at fc, i.e. |H| = 1/sqrt(2) there, not
        # `bwrelheight` — so until v0.1.1 a butterworth window came out 14 %
        # wider than requested while gauss/roex/gammatone were accurate to
        # 0.4 %.  Rescale the cutoff so the response really does reach
        # `bwrelheight` at bw/2.
        cutoff = (bw / step / 2.0) / (bwrelheight ** (-2.0) - 1.0) ** (1.0 / (2 * n))
        H = 1.0 / np.sqrt(1.0 + (H / cutoff) ** (2 * n))  # type: ignore[assignment]

    elif name == "roex":
        n = order if order is not None else 4
        if n <= 1:
            raise ValueError("Roex order must be > 1")

        def _roex_inverse(yn):
            return math.sqrt(yn ** (-2.0 / n) - 1.0)

        dilation = bw / 2.0 / _roex_inverse(bwrelheight) / step
        peakpos = (n - 1) / (2.0 * np.pi * dilation)
        peakmod = np.exp(2.0j * np.pi * H * peakpos)
        H = (1.0 + 1.0j * H / dilation) ** (-n) * peakmod

    elif name == "gammatone":
        # True analytic gammatone: H(f) = 1 / (1 + i f/b)^n
        #
        # The magnitude is |H(f)| = (1 + (f/b)^2)^{-n/2}, which is
        # asymmetric after shifting to a non-zero centre frequency
        # (steeper skirt on the low-frequency side).
        #
        # Bandwidth parameter b is derived from the -6 dB half-width:
        #   |H(bw/2)| = bwrelheight
        #   (1 + (bw/2/b)^2)^{-n/2} = bwrelheight
        #   b = (bw/2) / sqrt(bwrelheight^{-2/n} - 1)
        n = order if order is not None else 4
        if n < 1:
            raise ValueError("Gammatone order must be >= 1")

        half_bw_bins = (bw / 2.0) / step
        denom = math.sqrt(bwrelheight ** (-2.0 / n) - 1.0)
        b = half_bw_bins / denom

        H = (1.0 + 1.0j * H / b) ** (-n)  # type: ignore[assignment]

    else:
        raise ValueError(f"Unknown freqwin type: {name!r}")

    if norm.lower() not in ("null", "none", ""):
        H, _ = setnorm(H, norm)

    return H


# ---------------------------------------------------------------------------
# magresp – magnitude response (returns dB array; no plotting)
# ---------------------------------------------------------------------------
