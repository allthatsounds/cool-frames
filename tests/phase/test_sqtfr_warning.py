"""The magnitude path says so when it falls back to zero phase gradients.

``sqtfr`` is documented as required on the magnitude path, and omitting it is
a documented fallback rather than an error: the gradients become zero and PGHI
degenerates to zero-phase reconstruction. What the documentation cannot do is
reach a caller who simply forgot the argument, and the cost is not subtle --
on a 24 kHz Gabor bank it is -0.7 dB of spectral convergence against -26.3 dB
with the ratios supplied, which reads as a broken codec rather than a missing
keyword.

So the fallback warns once. Asking for zero gradients on purpose, by passing
``tgrad`` and ``fgrad``, stays silent, and so does the signal path.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from cool_frames.numpy.filters import gabfilters
from cool_frames.numpy.filters._gabfilters import _fir2long, _gabwin
from cool_frames.numpy.gabor import dgtreal
from cool_frames.numpy.phase import filterbankconstphase

FS, W, A, M = 24000, 480, 60, 960
L = 960 * 20
NEEDLE = "no sqtfr on the magnitude path"


def _bank_magnitudes():
    win = _fir2long(_gabwin("hann", W, norm="energy"), M)
    rng = np.random.default_rng(0)
    t = np.arange(L) / FS
    x = sum(np.sin(2 * np.pi * k * 120 * t) / k for k in range(1, 20))
    x = x + 0.02 * rng.standard_normal(L)
    x /= np.abs(x).max()
    edge = np.ones(M // 2 + 1)
    edge[0] = edge[-1] = 1.0 / np.sqrt(2.0)
    mag = np.abs(np.asarray(dgtreal(x, win, A, M))) * edge[:, None]
    return x, win, [m for m in mag.astype(float)]


def _args():
    return np.full(M // 2 + 1, A), np.arange(M // 2 + 1) * FS / M


def test_missing_sqtfr_warns_on_the_magnitude_path():
    _, _, mags = _bank_magnitudes()
    hops, fc = _args()
    with pytest.warns(UserWarning, match=NEEDLE):
        filterbankconstphase(mags, hops, fc, fs=FS)


def test_supplying_sqtfr_is_silent():
    _, _, mags = _bank_magnitudes()
    hops, fc = _args()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        filterbankconstphase(mags, hops, fc, fs=FS,
                             sqtfr=np.sqrt(np.full(M // 2 + 1, 2.0 / M)))


def test_asking_for_zero_gradients_on_purpose_is_silent():
    """The documented way to want this: pass the gradients yourself."""
    _, _, mags = _bank_magnitudes()
    hops, fc = _args()
    zeros = [np.zeros_like(m) for m in mags]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        filterbankconstphase(mags, hops, fc, fs=FS, tgrad=zeros, fgrad=zeros)


def test_the_signal_path_does_not_warn():
    """It builds its own gradients from the filters, so sqtfr is irrelevant."""
    x, win, _ = _bank_magnitudes()
    g, a, *_ = gabfilters(FS, L, window=win, M=M, a=A, real=True)
    caught = []
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        filterbankconstphase(x, g, a, L)
        caught = [w for w in rec if NEEDLE in str(w.message)]
    assert not caught, [str(w.message) for w in caught]
