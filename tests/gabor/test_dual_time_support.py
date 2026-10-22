"""Duals whose window support is constrained in time.

Latency is the point. Synthesis reaches ``back`` samples before a
coefficient's centre, so the back extent of the dual is what a decoder pays:
on the 24 kHz bank below, the canonical dual puts total latency at 32.5 ms and
a dual held to 120 samples back puts it at 27.5 ms, either side of a 30 ms
deadline.
"""

import numpy as np
import pytest

from cool_frames.numpy.filterbanks import filterbank, ifilterbank, filterbankdual
from cool_frames.numpy.filters import audfilters
from cool_frames.numpy.filters._gabfilters import _fir2long, _gabwin, gabfilters

FS, W, A, M, L = 24000, 480, 60, 960, 24000


def _bank():
    """A 20 ms Hann window on a 2.5 ms hop, zero-padded to a 960-point FFT."""
    win = _fir2long(_gabwin("hann", W, norm="energy"), M)
    g, a, fc, _, _ = gabfilters(FS, L, window=win, M=M, a=A, real=True)
    return g, a


def _prototype(gd):
    """The synthesis window and its signed lags, from the stored response."""
    wv = np.real(np.fft.ifft(np.asarray(gd[0]["H"]).ravel()))
    n = len(wv)
    nz = np.nonzero(np.abs(wv) > 1e-12 * np.abs(wv).max())[0]
    lag = np.where(nz > n // 2, nz - n, nz)
    return wv, int(lag.min()), int(lag.max())


def _round_trip_db(gd, a, g, seed=0):
    x = np.random.default_rng(seed).standard_normal(L)
    y = np.real(ifilterbank(filterbank(x, g, a, L), gd, a, L))
    return 20 * np.log10(np.linalg.norm(x - y) / np.linalg.norm(x))


def test_constrained_dual_still_reconstructs():
    g, a = _bank()
    gd = filterbankdual(g, a, L, time_support=(120, 239))
    assert _round_trip_db(gd, a, g) < -200.0


def test_support_is_actually_narrower():
    g, a = _bank()
    wide = _prototype(filterbankdual(g, a, L, time_support=(None, None)))
    tight = _prototype(filterbankdual(g, a, L, time_support=(120, 239)))
    assert (tight[2] - tight[1]) < (wide[2] - wide[1])


def test_cost_of_the_constraint_matches_the_closed_form():
    """The minimum-norm dual on a support has a closed form whenever the
    window is painless in time, and the implementation is that closed form.
    What this pins is the price of the constraint: the noise gain predicted
    for narrowing the support, and its absolute value on this bank."""
    g, a = _bank()

    def closed_form(back=None, fwd=None):
        gl = _fir2long(_gabwin("hann", W, norm="energy"), L)
        lag = np.arange(L)
        lag = np.where(lag > L // 2, lag - L, lag)
        nz = np.abs(gl) > 0
        lo = lag[nz].min() if back is None else -int(back)
        hi = lag[nz].max() if fwd is None else int(fwd)
        S = (lag >= lo) & (lag <= hi) & nz
        per = np.where(S, gl**2, 0.0).reshape(L // A, A).sum(axis=0)
        return np.where(S, gl / (M * per[np.arange(L) % A]), 0.0)

    predicted = 10 * np.log10(
        np.sum(closed_form(120, 239) ** 2) / np.sum(closed_form() ** 2)
    )
    wide = _prototype(filterbankdual(g, a, L, time_support=(None, None)))[0]
    tight = _prototype(filterbankdual(g, a, L, time_support=(120, 239)))[0]
    measured = 10 * np.log10(np.sum(tight**2) / np.sum(wide**2))
    assert predicted == pytest.approx(measured, abs=1e-6)
    assert measured == pytest.approx(0.167, abs=0.01)

    narrow = _prototype(filterbankdual(g, a, L, time_support=(60, 239)))[0]
    assert 10 * np.log10(np.sum(narrow**2) / np.sum(wide**2)) == pytest.approx(
        0.984, abs=0.01
    )


def test_exact_outside_the_frequency_painless_limit():
    """Painless *in time* is what the closed form needs. These lengths put
    N = L/a at 400 and 800 against M = 960, so the filters' frequency support
    exceeds the filterbank painless limit and gabfilters warns the canonical
    dual is approximate; the time-support dual is exact there regardless."""
    for Lx in (24000, 48000):
        win = _fir2long(_gabwin("hann", W, norm="energy"), M)
        g, a, _, _, _ = gabfilters(FS, Lx, window=win, M=M, a=A, real=True)
        assert Lx // A < M
        x = np.random.default_rng(0).standard_normal(Lx)
        for ts in ((None, None), (120, 239), (60, 239)):
            gd = filterbankdual(g, a, Lx, time_support=ts)
            y = np.real(ifilterbank(filterbank(x, g, a, Lx), gd, a, Lx))
            y = np.asarray(y).ravel()[:Lx]
            err = 20 * np.log10(np.linalg.norm(x - y) / np.linalg.norm(x))
            assert err < -250.0, (Lx, ts, err)


def test_impossible_support_is_refused():
    g, a = _bank()
    with pytest.raises(ValueError, match="residue class"):
        filterbankdual(g, a, L, time_support=(5, 5))


def test_non_gabor_bank_is_refused():
    g, a, _, La, _ = audfilters(16000, 16000)
    with pytest.raises(ValueError, match="Gabor bank"):
        filterbankdual(g, a, La, time_support=(100, 100))


def test_default_path_is_unchanged():
    g, a = _bank()
    assert _round_trip_db(filterbankdual(g, a, L), a, g) < -200.0


def test_torch_backend_agrees():
    torch = pytest.importorskip("torch")
    from cool_frames.torch.filterbanks import (
        filterbank as t_filterbank,
        ifilterbank as t_ifilterbank,
        filterbankdual as t_filterbankdual,
    )
    from cool_frames.torch.filters import gabfilters as t_gabfilters

    win = _fir2long(_gabwin("hann", W, norm="energy"), M)
    g, a, _, _, _ = t_gabfilters(FS, L, window=win, M=M, a=A, real=True)
    x = torch.randn(L, dtype=torch.float64)
    c = t_filterbank(x, g, a)
    gd = t_filterbankdual(g, a, L, time_support=(120, 239))
    y = torch.as_tensor(t_ifilterbank(c, gd, a))
    y = y.real if torch.is_complex(y) else y
    n = min(len(x), len(y))
    err = 20 * torch.log10((x[:n] - y[:n]).norm() / x[:n].norm()).item()
    assert err < -200.0
