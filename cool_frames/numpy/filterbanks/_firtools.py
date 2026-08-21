"""FIR filter manipulation and ramp utilities.

This module contains functions for working with FIR filters and creating
ramp signals (fade-in/fade-out envelopes). Includes FIR coefficient
manipulation, Kaiser-Bessel windows, and periodic group delay estimation.

MATLAB originals
----------------
  sigproc/transferfunction.m, pgrpdelay.m
  sigproc/fir2long.m, long2fir.m
  sigproc/rampup.m, rampdown.m, rampsignal.m
"""
from __future__ import annotations

import math

import numpy as np

from ..filters._filters import filter_freqresp
from ..filters._firwin import firwin as _firwin
from ..filters._gabfilters import _fir2long as _fir2long_internal


# ---------------------------------------------------------------------------
# transferfunction – frequency response of a single filter
# ---------------------------------------------------------------------------

def transferfunction(g: dict, L: int) -> np.ndarray:
    """Compute the transfer function (frequency response) of a filter.

    Parameters
    ----------
    g : dict
        Filter dict as returned by ``blfilter``, ``firfilter``, etc.
    L : int
        DFT length (number of frequency bins).

    Returns
    -------
    H : (L,) complex ndarray
        Transfer function sampled at the *L* DFT frequencies.
    """
    H, _ = filter_freqresp(g, L)
    return H


# ---------------------------------------------------------------------------
# pgrpdelay – periodic group delay
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# fir2long – zero-pad FIR to LONG
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# long2fir – cut LONG window to FIR
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# rampup / rampdown – rising / falling ramp
# ---------------------------------------------------------------------------





# ---------------------------------------------------------------------------
# rampsignal – apply fade-in / fade-out
# ---------------------------------------------------------------------------

def rampsignal(f, L=None, wintype: str = "hann",
               *, dim: int | None = None) -> np.ndarray:
    """Apply fade-in and fade-out ramps to a signal.

    Parameters
    ----------
    f : array_like
        Input signal.
    L : int, (2,) array_like, or None
        Ramp length.  Scalar → same length for rise and fall.
        Length-2 → ``[L_rise, L_fall]``.  ``None`` → half the signal.
    wintype : str
        Window type (default ``'hann'``).
    dim : int or None
        Axis along which to apply.  ``None`` → first non-singleton.

    Returns
    -------
    outsig : ndarray
    """
    f = np.asarray(f, dtype=float)
    if dim is None:
        dim = next((i for i in range(f.ndim) if f.shape[i] > 1), 0)

    Lsig = f.shape[dim]

    if L is None:
        L1 = L2 = Lsig // 2
    else:
        L = np.atleast_1d(np.asarray(L, dtype=int))
        if L.size == 1:
            L1 = L2 = int(L[0])
        else:
            L1, L2 = int(L[0]), int(L[1])

    if Lsig < L1 + L2:
        raise ValueError(
            f"Signal length {Lsig} < ramp lengths {L1}+{L2}")

    def _ramp(n: int, *, rising: bool) -> np.ndarray:
        # Half-window ramp: rising -> left half of a length-2n window (0->1);
        # falling -> right half (1->0).
        win_c = np.fft.fftshift(_firwin(wintype, 2 * n, norm="inf"))
        return win_c[:n] if rising else win_c[n:]

    r1 = _ramp(L1, rising=True)
    r2 = _ramp(L2, rising=False)
    envelope = np.concatenate([r1, np.ones(Lsig - L1 - L2), r2])

    # Broadcast along the correct dimension
    shape = [1] * f.ndim
    shape[dim] = Lsig
    envelope = envelope.reshape(shape)

    return f * envelope  # type: ignore[no-any-return]


def resize_fir(g, L: int) -> np.ndarray:
    """Resize a FIR window to length *L* (DFT ordering). Fuses fir2long + long2fir.

    ``L > len(g)``: zero-pad the middle (extend; was ``fir2long``).
    ``L < len(g)``: discard the zero-padded middle (cut; was ``long2fir``).
    ``L == len(g)``: return a copy. The direction is inferred from *L*, so the
    operation is its own inverse:
    ``resize_fir(resize_fir(g, Lbig), len(g)) == g``.

    Parameters
    ----------
    g : array_like, shape (Lfir,)
        Window in DFT ordering (DC at index 0).
    L : int
        Target length.

    Returns
    -------
    gout : (L,) ndarray
    """
    g = np.asarray(g)
    Lg = len(g)
    if L == Lg:
        return g.copy()
    if L > Lg:
        return _fir2long_internal(g, L)
    out = np.zeros(L, dtype=g.dtype)
    n1 = int(math.ceil(L / 2))
    out[:n1] = g[:n1]
    n2 = L - n1
    if n2 > 0:
        out[n1:] = g[Lg - n2:]
    return out


