"""
Regression tests for the window defects found on 2026-09-26.

- ``firwin`` / ``firwin_eval`` promise whole-point even windows peaking at
  index 0, but 'gauss', 'truncgauss', 'butterworth' and 'roex' peaked at the
  middle sample (``firwin``) or were one-sided (``firwin_eval``);
- 'truncgauss' was not LTFAT's (exp(4 log(p/100) x^2)), so ``pghi_findgamma``
  of it was 2.2 times the constant tabulated for that name;
- 'gammatone' was a causal envelope with a spurious unit first sample, so its
  bandwidth factor was 1e-4 and ``audfilters(window='gammatone')`` built
  filters 3750 times too wide; 'roex' was a Gaussian.  Neither is a window
  shape: both now raise and name ``freqwin`` / ``gammatonefir``;
- the ``audfilters`` docstring called its filters gammatone filters; they are
  frequency-domain windows (Hann by default) of 1 ERB equivalent bandwidth.
"""

from __future__ import annotations

import pytest

import numpy as np
from cool_frames.numpy.filters import audfilters, cqtfilters
from cool_frames.numpy.filters._filters import comp_transferfunction
from cool_frames.numpy.filters._firwin import firwin, firwin_eval, window_winbw
from cool_frames.numpy.filters._freqwin import freqwin
from cool_frames.numpy.filters.lowlevel import blfilter, freqfilter
from cool_frames.numpy.phase._findgamma import pghi_findgamma

WINDOWS = ["hann", "sine", "hamming", "blackman", "blackman2", "rect", "tria", "sqrttria",
           "itersine", "nuttall", "nuttall01", "nuttall11", "nuttall20", "gauss",
           "truncgauss", "truncgauss20", "butterworth"]


@pytest.mark.parametrize("name", WINDOWS)
@pytest.mark.parametrize("M", [63, 64, 256])
def test_peak_at_index_zero_and_whole_point_even(name, M):
    g = firwin(name, M)
    assert np.argmax(g) == 0 and g[0] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(g[1:], g[1:][::-1], atol=1e-12)


@pytest.mark.parametrize("name", WINDOWS)
def test_firwin_eval_matches_firwin(name):
    M = 64
    np.testing.assert_allclose(firwin_eval(name, np.arange(M) / M), firwin(name, M), atol=1e-12)


def test_kaiser_peak_at_index_zero():
    g = firwin("kaiser", 64, beta=5.0)
    assert np.argmax(g) == 0


@pytest.mark.parametrize("name,height", [("truncgauss", 0.01), ("truncgauss20", 0.20)])
def test_truncgauss_is_ltfats(name, height):
    M = 128
    n = np.arange(M)
    x = np.where(n < M / 2, n / M, n / M - 1)
    np.testing.assert_allclose(firwin(name, M), np.exp(4 * np.log(height) * x ** 2), atol=1e-12)
    assert firwin(name, M)[M // 2] == pytest.approx(height, rel=1e-12)


def test_truncgauss_findgamma_agrees_with_its_table_entry():
    _, cg_numeric = pghi_findgamma(firwin("truncgauss", 1024))
    _, cg_table = pghi_findgamma("truncgauss", 1024)
    assert cg_numeric == pytest.approx(cg_table, rel=5e-3)


@pytest.mark.parametrize("name", ["gammatone", "roex", "Gammatone"])
def test_frequency_responses_are_not_windows(name):
    with pytest.raises(ValueError, match="freqwin"):
        firwin(name, 64)
    with pytest.raises(ValueError, match="freqwin"):
        firwin_eval(name, np.arange(64) / 64)


def test_audfilters_refuses_a_gammatone_window():
    with pytest.raises(ValueError, match="not a window"):
        audfilters(16000, 16000, window="gammatone")


def test_audfilters_filters_have_one_erb_equivalent_bandwidth():
    _, _, fc, _, info = audfilters(16000, 16000)
    fc = np.asarray(fc, float)[1:-1]
    erb = 24.7 + fc / 9.265
    np.testing.assert_allclose(np.ravel(info["fsupp"])[1:-1] * window_winbw("hann"), erb, rtol=1e-3)


# ---------------------------------------------------------------------------
# where the window goes: blfilter, audfilters, cqtfilters, freqfilter
# ---------------------------------------------------------------------------

PLACED = ["hann", "blackman", "nuttall", "hamming", "gauss", "truncgauss", "tria", "itersine"]


def _centred(H, centre_bin):
    """|H| peaks at centre_bin and is symmetric about it."""
    H = np.abs(H)
    k = int(np.argmax(H))
    assert k == centre_bin
    w = np.nonzero(H > 1e-12 * H.max())[0]
    half = min(k - w.min(), w.max() - k)
    np.testing.assert_allclose(H[k - half:k], H[k + 1:k + half + 1][::-1], rtol=1e-9, atol=1e-12 * H.max())


@pytest.mark.parametrize("name", PLACED)
@pytest.mark.parametrize("fsupp", [0.1, 0.13])
def test_blfilter_centres_every_window_on_fc(name, fsupp):
    L, fc = 512, 0.3
    _centred(comp_transferfunction(blfilter(name, fsupp, fc), L), int(np.floor(L / 2 * fc + 0.5)))


@pytest.mark.parametrize("name", ["hann", "blackman", "nuttall", "gauss"])
def test_audfilters_and_cqtfilters_centre_every_window(name):
    for g, _, fc, L, _ in (audfilters(16000, 16000, window=name),
                           cqtfilters(16000, 16000, bins=12, window=name)):
        for m in (5, len(g) // 2, len(g) - 5):
            H = np.abs(comp_transferfunction(g[m], L))
            assert abs(int(np.argmax(H)) - fc[m] / 16000 * L) <= 0.5 + 1e-9


def test_blfilter_pedantic_moves_the_window_by_the_sub_bin_offset():
    L, fc = 512, 0.3013            # L/2 fc = 77.13: 0.13 bins above bin 77
    H = np.abs(comp_transferfunction(blfilter("gauss", 0.1, fc, pedantic=True), L))
    k = np.arange(L)
    centroid = np.sum(k * H ** 2) / np.sum(H ** 2)
    assert centroid == pytest.approx(L / 2 * fc, abs=0.02)
    H0 = np.abs(comp_transferfunction(blfilter("gauss", 0.1, fc), L))
    assert np.sum(k * H0 ** 2) / np.sum(H0 ** 2) == pytest.approx(77.0, abs=0.02)


@pytest.mark.parametrize("name", ["gauss", "butterworth", "roex", "gammatone"])
def test_freqfilter_is_ltfats_construction(name):
    # LTFAT: H = fftshift(freqwin(name, Lw, bw, fs/L*Lw)), Lw = round(4 bw L / fs),
    # foff = round(L/2 fc) - floor(Lw/2)
    L, bw, fc = 1024, 0.05, 0.3
    lw = int(np.floor(4 * bw * L / 2 + 0.5))
    ref = np.zeros(L, complex)
    w = np.fft.fftshift(freqwin(name, lw, bw, fs=2.0 / L * lw))
    ref[(np.floor(L / 2 * fc + 0.5).astype(int) - lw // 2 + np.arange(lw)) % L] = w / np.abs(w).max()
    np.testing.assert_allclose(comp_transferfunction(freqfilter(name, bw, fc, "peak"), L), ref,
                               atol=1e-12)
    H = comp_transferfunction(freqfilter(name, bw, fc), L)          # 'energy'
    assert np.sum(np.abs(H) ** 2) / L == pytest.approx(1.0, rel=1e-12)


def test_freqfilter_gauss_has_its_bandwidth_at_minus_6_db():
    L, bw, fc = 4096, 0.05, 0.3
    H = np.abs(comp_transferfunction(freqfilter("gauss", bw, fc, "peak"), L))
    above = np.nonzero(H >= 0.5)[0]
    assert (above.max() - above.min() + 1) == pytest.approx(bw * L / 2, abs=2)


def test_freqwin_orders_even_lengths_as_ltfat():
    # LTFAT orders the bins 0..ceil(L/2)-1, -floor(L/2)..-1: index L/2 of an
    # even L is -L/2, where the asymmetric shapes differ from +L/2
    L, bw, n = 64, 0.3, 4
    step = 2.0 / L
    yn = 10.0 ** (-3.0 / 10.0)
    dil = bw / 2.0 / np.sqrt(yn ** (-2.0 / n) - 1.0) / step
    peakpos = (n - 1) / (2.0 * np.pi * dil)
    k = np.r_[0:L // 2, -L // 2:0].astype(float)
    ref = (1.0 + 1j * k / dil) ** (-n) * np.exp(2j * np.pi * k * peakpos)
    np.testing.assert_allclose(freqwin("roex", L, bw), ref, rtol=1e-12)
    # and a shift s evaluates the shape at k - s
    np.testing.assert_allclose(
        freqwin("roex", L, bw, shift=0.25),
        (1.0 + 1j * (k - 0.25) / dil) ** (-n) * np.exp(2j * np.pi * (k - 0.25) * peakpos), rtol=1e-12)
