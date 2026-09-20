"""
test_benchmark_findings.py
==========================
Regression tests for the defects the comparative benchmark (W52, 2026-09-19)
found in cool-frames.

The benchmark ran cool-frames against eleven other time-frequency libraries
and graded every result against an independent reference.  Five of the
defects it exposed were silently wrong numbers or silently lost exactness, and
one was a 300x slowdown on a path that should have been a shortcut:

1. ``filterbankreassign`` moved every coefficient to about **twice** its
   instantaneous frequency: ``filterbankphasegrad`` returns the *absolute*
   instantaneous frequency, and the kernel added the channel's centre on top.
   A 440 Hz tone ended up in the 926 Hz channel.
2. ``filterbanksynchrosqueeze`` zeroed the instantaneous frequency instead of
   the group delay, so it moved coefficients in *time* only -- the opposite of
   synchrosqueezing.
3. Channel centres for reassignment were an arithmetic mean over [0, 2), which
   puts the DC complement (bins near 0 *and* near L) at Nyquist; a filter cell
   passed in place of ``fc`` fell back to M evenly spaced frequencies.
4. ``cqtfilters(sampling='fractional')`` built a Nyquist complement one bin
   wider than its decimated length (999 non-zero bins on N = 998): not
   painless, reconstruction 1.4e-4 instead of 1e-16.  The painless check let
   it through because it allowed one bin of slack on the *stored* length.
5. ``audfilters`` sized the hops of its DC and Nyquist complements from the
   bandwidth of the whole bank rather than of the complements themselves,
   oversampling them up to 24x (2592 coefficients for 109 non-zero bins).
6. ``gabframebounds(g, a, M, L)`` sent every zero-padded window down the
   factorised path -- 152 ms at L = 2**16 for a Hann window whose frame
   operator is diagonal (0.5 ms).  Its numbers were right; only the time was
   wrong.
7. ``reassigned_spectrogram`` reported the instantaneous frequency off by a
   factor of pi and as an absolute frequency under a key that promises the
   deviation from the channel centre (the torch version also swapped the
   instantaneous frequency and the group delay).
8. ``waveletfilters`` capped every lowpass hop by ``L / aprecise`` -- a hop
   taken for a width -- and gave the Nyquist complement the smallest wavelet
   hop: the default bank sampled a 707-bin DC channel at every sample and a
   3-bin Nyquist channel at every other one (16 % of all coefficients), and
   ``lowpass='repeat'`` oversampled its lowpass channels 47x.

Every defect has tests that fail on the commit before its fix (1f581bc;
1d0773b for item 8); the others guard what a fix must not change -- exact
reconstruction, the frame operator, agreement with a dense oracle.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np
from cool_frames.numpy.filterbanks import filterbank, filterbankdual, ifilterbank
from cool_frames.numpy.filterbanks._utils import normalise_a, prepare_filters
from cool_frames.numpy.filters import audfilters, cqtfilters
from cool_frames.numpy.filters._painless import nonzero_support
from cool_frames.numpy.phase import (
    filterbankphasegrad,
    filterbankreassign,
    filterbanksynchrosqueeze,
)

FS = 16000
LS = 8000


def _tone(f0, fs=FS, n=LS):
    return np.cos(2 * np.pi * f0 * np.arange(n) / fs)


def _channel_energy(sr):
    return np.array([float(np.sum(np.asarray(s))) for s in sr])


def _supports_and_lengths(g, a, L):
    an = normalise_a(a, len(g))
    gp, _, _, _ = prepare_filters(g, an, L)
    N = L * an[:, 1] / an[:, 0]
    s = np.array([nonzero_support(gm["H"]) for gm in gp])
    return s, N


# ---------------------------------------------------------------------------
# 1, 3. Reassignment lands in the tone's own channel
# ---------------------------------------------------------------------------


class TestReassignment:
    @pytest.fixture(scope="class")
    def bank(self):
        g, a, fc, L, _ = audfilters(FS, LS)
        return g, a, np.asarray(fc, dtype=float), L

    @pytest.mark.parametrize("f0", [440.0, 2500.0, 6000.0])
    def test_tone_lands_in_nearest_channel(self, bank, f0):
        """Was: 440 Hz -> 926 Hz channel, 2500 -> 5007, 6000 -> Nyquist."""
        g, a, fc, L = bank
        E = _channel_energy(filterbankreassign(_tone(f0), g, a, L))
        m = int(np.argmax(E))
        assert m == int(np.argmin(np.abs(fc - f0))), (
            f"{f0} Hz tone reassigned to channel {m} ({fc[m]:.0f} Hz)"
        )
        assert E[m] / E.sum() > 0.95

    def test_dc_complement_keeps_low_frequencies(self, bank):
        """The DC channel's centre is 0, not Nyquist (arithmetic mean over [0, 2))."""
        g, a, _fc, L = bank
        E = _channel_energy(filterbankreassign(_tone(8.0), g, a, L))
        assert int(np.argmax(E)) == 0
        assert E[0] / E.sum() > 0.9

    def test_filter_cell_gives_the_signal_path_result(self, bank):
        """Pre-computed gradients with ``fc=g`` must use the filters' centres."""
        g, a, _fc, L = bank
        x = np.random.default_rng(3).standard_normal(LS)
        sr_sig = filterbankreassign(x, g, a, L)
        tg, fg, s, _ = filterbankphasegrad(x, g, a, L)
        sr_pc, _repos, _Lc = filterbankreassign(s, tg, fg, a, g)
        for m, (u, v) in enumerate(zip(sr_sig, sr_pc)):
            np.testing.assert_allclose(
                v, u, rtol=1e-12, atol=1e-12 * float(np.max(u) + 1), err_msg=f"channel {m}"
            )

    def test_inconsistent_lengths_are_refused(self, bank):
        g, a, _fc, L = bank
        x = np.random.default_rng(4).standard_normal(LS)
        tg, fg, s, _ = filterbankphasegrad(x, g, a, L)
        s_bad = [np.asarray(sm)[:-1] if m == 3 else sm for m, sm in enumerate(s)]
        with pytest.raises(ValueError, match="transform length"):
            filterbankreassign(s_bad, tg, fg, a, g)


# ---------------------------------------------------------------------------
# 2. Synchrosqueezing moves coefficients in frequency only
# ---------------------------------------------------------------------------


class TestSynchrosqueeze:
    @pytest.fixture(scope="class")
    def uniform(self):
        g, _a, fc, _L, _ = audfilters(8000, 1024)
        a1 = np.ones(len(g), dtype=int)
        x = np.random.default_rng(1).standard_normal(1024)
        c = filterbank(x, g, a1)
        tg, fg, _s, _ = filterbankphasegrad(x, g, a1)
        return g, a1, np.asarray(fc, float) / 8000 * 2, c, tg, fg

    def test_time_marginal_is_preserved(self, uniform):
        """With every hop 1, frequency-only moves keep each instant's energy.

        Was: relative error 0.39 (the coefficients moved in time instead).
        """
        _g, a1, fc_n, c, tg, fg = uniform
        cr, _, _ = filterbanksynchrosqueeze(c, tg, fg, a1, fc_n)
        before = np.sum([np.abs(np.asarray(cm)) ** 2 for cm in c], axis=0)
        after = np.sum([np.asarray(cm) for cm in cr], axis=0)
        assert np.linalg.norm(after - before) / np.linalg.norm(before) < 1e-12

    def test_frequency_marginal_moves(self, uniform):
        """...and it does move energy between channels (was: not at all)."""
        _g, a1, fc_n, c, tg, fg = uniform
        cr, _, _ = filterbanksynchrosqueeze(c, tg, fg, a1, fc_n)
        before = np.array([np.sum(np.abs(np.asarray(cm)) ** 2) for cm in c])
        after = np.array([np.sum(np.asarray(cm)) for cm in cr])
        assert np.linalg.norm(after - before) / np.linalg.norm(before) > 1e-3

    def test_group_delay_is_ignored(self, uniform):
        _g, a1, fc_n, c, tg, fg = uniform
        cr1, _, _ = filterbanksynchrosqueeze(c, tg, fg, a1, fc_n)
        fg_noise = [np.asarray(f) + 50.0 for f in fg]
        cr2, _, _ = filterbanksynchrosqueeze(c, tg, fg_noise, a1, fc_n)
        for u, v in zip(cr1, cr2):
            np.testing.assert_array_equal(np.asarray(u), np.asarray(v))


# ---------------------------------------------------------------------------
# 4. Fractional constant-Q is painless and exact; the check has no slack
# ---------------------------------------------------------------------------


class TestFractionalPainless:
    @pytest.mark.parametrize(
        "fs, Ls, bins",
        [
            (22050, 65536, 24),  # the benchmark's setting: rt 1.4e-4, no warning
            (16000, 16000, 12),  # rt 1.6e-5
            (44100, 32768, 48),
        ],
    )
    def test_cqt_fractional_exact(self, fs, Ls, bins):
        g, a, _fc, L, _ = cqtfilters(
            fs, Ls, fmin=32.70319566257483, bins=bins, sampling="fractional"
        )
        s, N = _supports_and_lengths(g, a, L)
        assert not np.any(s > N + 1e-9), f"channels wider than L/a: {np.flatnonzero(s > N)}"
        x = np.random.default_rng(0).standard_normal(Ls)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*painless.*")
            gd = filterbankdual(g, a, L)
        y = np.real(ifilterbank(filterbank(x, g, a, L=L), gd, a, Ls, real=True))[:Ls]
        assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-12

    def test_one_bin_too_wide_warns(self):
        """A support of N + 1 non-zero bins is not painless and must say so.

        Was: silent, because the check compared the *stored* length with N + 1.
        """
        g, a, _fc, L, _ = cqtfilters(
            22050, 65536, fmin=32.70319566257483, bins=24, sampling="fractional"
        )
        s, _N = _supports_and_lengths(g, a, L)
        # The Nyquist complement stores exactly its non-zero support (999
        # bins), so the old test -- stored length > N + 1 -- passed it on
        # N = 998.  Inner channels carry zero end bins and warned either way.
        m = len(g) - 1
        a_bad = np.array(a, copy=True)
        a_bad[m, 1] = s[m] - 1
        with pytest.warns(UserWarning, match="painless"):
            filterbankdual(g, a_bad, L)

    @pytest.mark.parametrize("designer", ["aud", "cqt"])
    def test_default_banks_do_not_warn(self, designer):
        """Stored L/a + 1 bins whose end bins are exact zeros are painless."""
        if designer == "aud":
            g, a, _fc, L, _ = audfilters(22050, 16384)
        else:
            g, a, _fc, L, _ = cqtfilters(22050, 16384, fmin=32.70319566257483, bins=24)
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*painless.*")
            filterbankdual(g, a, L)


# ---------------------------------------------------------------------------
# 5. audfilters' edge channels are sized by their own support
# ---------------------------------------------------------------------------


class TestAudfiltersEdges:
    @pytest.mark.parametrize("fs", [16000, 22050, 44100])
    @pytest.mark.parametrize("Ls", [4096, 16000])
    @pytest.mark.parametrize("scale", ["erb", "mel"])
    def test_fractional_edges_are_tight(self, fs, Ls, scale):
        """N of the DC and Nyquist channels is at most 3 bins over their support.

        Was: 5-24x over (e.g. 2592 coefficients for 109 non-zero bins).
        """
        g, a, _fc, L, _ = audfilters(fs, Ls, scale=scale, sampling="fractional")
        s, N = _supports_and_lengths(g, a, L)
        assert not np.any(s > N + 1e-9)
        for m in (0, len(g) - 1):
            assert N[m] - s[m] <= 3, f"channel {m}: N = {N[m]:.0f} for {s[m]} bins"

    @pytest.mark.parametrize("sampling", ["regsampling", "fractional"])
    @pytest.mark.parametrize("scale", ["erb", "mel"])
    def test_round_trip_exact(self, sampling, scale):
        fs, Ls = 16000, 16000
        g, a, _fc, L, _ = audfilters(fs, Ls, scale=scale, sampling=sampling)
        s, N = _supports_and_lengths(g, a, L)
        assert not np.any(s > N + 1e-9)
        x = np.random.default_rng(2).standard_normal(Ls)
        gd = filterbankdual(g, a, L)
        y = np.real(ifilterbank(filterbank(x, g, a, L=L), gd, a, Ls, real=True))[:Ls]
        assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-12


# ---------------------------------------------------------------------------
# 6. gabframebounds: the padded painless window takes the diagonal
# ---------------------------------------------------------------------------


def _dense_gabor_bounds(g, a, M, L):
    from cool_frames.gabor import dgt

    A = np.zeros((M * (L // a), L), dtype=complex)
    for k in range(L):
        e = np.zeros(L)
        e[k] = 1.0
        A[:, k] = np.asarray(dgt(e, g, a, M, L)).ravel()
    ev = np.linalg.eigvalsh(A.conj().T @ A)
    return float(ev.min()), float(ev.max())


class TestGaborFrameBounds:
    @pytest.mark.parametrize(
        "case",
        ["hann64_M64", "hann64_M32", "hamming63_M64", "split_middle"],
    )
    def test_matches_dense_oracle(self, case):
        from cool_frames.filters import firwin
        from cool_frames.gabor import gabframebounds

        a, L = 16, 256
        if case == "hann64_M64":
            g, M = firwin("hann", 64), 64
        elif case == "hann64_M32":
            g, M = firwin("hann", 64), 32
        elif case == "hamming63_M64":
            g, M = firwin("hamming", 63), 64
        else:
            # Even window with a non-zero middle sample: middlepad splits it,
            # the support becomes M + 1, and the operator is not diagonal.
            g, M = np.random.default_rng(0).random(64) + 0.1, 64
        A, B = gabframebounds(g, a, M, L)
        A0, B0 = _dense_gabor_bounds(g, a, M, L)
        assert abs(A - A0) <= 1e-12 * B0 and abs(B - B0) <= 1e-12 * B0

    def test_padded_painless_window_uses_the_diagonal(self, monkeypatch):
        """Was: 16,384 1x1 eigenproblems (152 ms) for a diagonal operator."""
        from cool_frames.filters import firwin
        from cool_frames.gabor import gabframebounds
        from cool_frames.numpy.gabor import _factorised

        calls = []
        real = _factorised._gabframediag

        def spy(*args, **kwargs):
            calls.append(args[1:3])
            return real(*args, **kwargs)

        monkeypatch.setattr(_factorised, "_gabframediag", spy)
        g = firwin("hann", 1024)
        A, B = gabframebounds(g, 256, 1024, 2**14)
        assert calls, "painless padded window went down the factorised path"
        np.testing.assert_allclose([A, B], [1536.0, 1536.0], rtol=1e-12)

    def test_split_middle_stays_general(self, monkeypatch):
        from cool_frames.gabor import gabframebounds
        from cool_frames.numpy.gabor import _factorised

        calls = []
        monkeypatch.setattr(
            _factorised, "_gabframediag", lambda *a, **k: calls.append(1) or np.ones(1)
        )
        g = np.random.default_rng(0).random(64) + 0.1
        gabframebounds(g, 16, 64, 256)
        assert not calls


# ---------------------------------------------------------------------------
# 7. reassigned_spectrogram reports the deviation from the centre, in Hz
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("f0", [1000.0, 3000.0])
def test_reassigned_spectrogram_deviation(f0):
    """Was: +258 Hz for a 1000 Hz tone in a 1058 Hz channel (should be -58)."""
    from cool_frames.numpy.diagnostics import reassigned_spectrogram

    spec = reassigned_spectrogram(_tone(f0), FS)
    fc = np.asarray(spec["fc"], float)
    m = int(np.argmin(np.abs(fc - f0)))
    dev = float(np.asarray(spec["instfreq_deviation"])[m])
    assert abs(dev - (f0 - fc[m])) < 0.02 * abs(f0 - fc[m]) + 1.0


# ---------------------------------------------------------------------------
# 8. waveletfilters sizes its lowpass and complement hops by their own width
# ---------------------------------------------------------------------------


class TestWaveletEdges:
    @pytest.mark.parametrize("sampling", ["regsampling", "fractional"])
    @pytest.mark.parametrize("fs, Ls", [(8000, 4096), (22050, 16000)])
    def test_complements_are_not_oversampled(self, fs, Ls, sampling):
        """Was: DC channel N = L (hop 1), Nyquist N = L/2 for a 3-bin support."""
        from cool_frames.numpy.filters import waveletfilters

        g, a, _fc, L, info = waveletfilters(fs, Ls, sampling=sampling)
        s, N = _supports_and_lengths(g, a, L)
        assert info["painless"] and not np.any(s > N + 1e-9)
        for m in (0, len(g) - 1):
            if sampling == "fractional":
                assert N[m] == s[m], f"channel {m}: N = {N[m]:.0f} for {s[m]} bins"
            else:
                # an integer hop must divide L, so N can exceed the support
                assert N[m] <= 1.5 * s[m] + 4, f"channel {m}: N = {N[m]:.0f} for {s[m]} bins"

    def test_repeat_lowpass_is_not_oversampled(self):
        """Was: 16 lowpass channels at N = 3456 for 74 non-zero bins each."""
        from cool_frames.numpy.filters import waveletfilters

        g, a, _fc, L, info = waveletfilters(16000, 16000, lowpass="repeat")
        s, N = _supports_and_lengths(g, a, L)
        lp = int(info["startindex"])
        assert lp > 1 and not np.any(s > N + 1e-9)
        assert np.all(N[:lp] <= 2 * s[:lp]), list(zip(N[:lp], s[:lp]))

    @pytest.mark.parametrize("lowpass", ["single", "repeat"])
    def test_hops_leave_the_frame_operator_alone(self, lowpass):
        """The response is scaled with the hop, so the frame operator does not
        depend on the hops; ``uniform``, whose hops are not fitted, is the
        reference.  (``regsampling`` is left out: its L differs.)"""
        from cool_frames.numpy.filterbanks import filterbankbounds
        from cool_frames.numpy.filters import waveletfilters

        bounds = {}
        for sampling in ("uniform", "fractional", "fractionaluniform"):
            g, a, _fc, L, _ = waveletfilters(8000, 4096, sampling=sampling, lowpass=lowpass)
            assert L == 4096
            bounds[sampling] = filterbankbounds(g, a, L)
        for sampling in ("fractional", "fractionaluniform"):
            np.testing.assert_allclose(bounds[sampling], bounds["uniform"], rtol=1e-9)

    @pytest.mark.parametrize("sampling", ["regsampling", "fractional", "fractionaluniform"])
    def test_round_trip_exact(self, sampling):
        from cool_frames.numpy.filters import waveletfilters

        g, a, _fc, L, _ = waveletfilters(8000, 4096, sampling=sampling)
        x = np.random.default_rng(5).standard_normal(4096)
        y = np.real(
            ifilterbank(filterbank(x, g, a, L=L), filterbankdual(g, a, L), a, 4096, real=True)
        )
        assert np.linalg.norm(x - y[:4096]) / np.linalg.norm(x) < 1e-12

    def test_two_sided_bank_is_painless_without_warning(self):
        """Its wavelets store tails of ~1e-11 past N, which alias onto nothing.

        Was: ``filterbankdual`` warned that 75 of 155 channels exceed the
        painless limit (on 1f581bc and 1d0773b alike), for a bank that
        reconstructs to 5e-16; counting non-zero bins instead would have
        cost 29 % more coefficients to silence it.
        """
        from cool_frames.numpy.filters import waveletfilters

        g, a, _fc, L, info = waveletfilters(8000, 4096, freqrange="complex")
        assert info["painless"]
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*painless.*")
            gd = filterbankdual(g, a, L, real=False)
        rng = np.random.default_rng(6)
        x = rng.standard_normal(4096) + 1j * rng.standard_normal(4096)
        y = ifilterbank(filterbank(x, g, a, L=L), gd, a, 4096, real=False)[:4096]
        assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-12


def test_aliasing_counts_what_aliases():
    """Painless is about pairs of bins N apart, not about the non-zero count."""
    from cool_frames.numpy.filters._painless import ALIAS_TOL, aliasing, painless_length

    h = np.hanning(9)[1:-1]  # 7 live bins
    assert aliasing(h, 7) == 0.0 and aliasing(h, 6) > 1e-3
    # Two tails of 1e-9 on each side: 11 non-zero bins.  At N = 9 every
    # aliased pair is tail against tail (1e-18): painless.  At N = 8 a tail
    # meets a live bin (2e-10): not.
    t = [1e-9, 1e-9]
    tails = np.concatenate([t, h, t])
    assert aliasing(tails, 9) <= ALIAS_TOL < aliasing(tails, 8)
    assert painless_length(tails) == 9 and painless_length(h) == 7


# ---------------------------------------------------------------------------
# 9. The reassignment kernel, vectorised, is the loop kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bank", ["erb", "cqt24", "gabor", "wavelet"])
def test_vectorised_reassignment_is_the_loop(bank, monkeypatch):
    """Same channels, same time indices, same sums, same ``repos`` order."""
    from cool_frames.numpy.filters import gabfilters, waveletfilters
    from cool_frames.numpy.phase import _reassign as R

    fs = 16000
    make = {
        "erb": lambda: audfilters(fs, 4096),
        "cqt24": lambda: cqtfilters(fs, 4096, fmin=50, bins=24),
        "gabor": lambda: gabfilters(fs, 4096, M=256, a=64),
        "wavelet": lambda: waveletfilters(fs, 4096),
    }[bank]
    g, a, fc, L, _ = make()
    rng = np.random.default_rng(7)
    x = np.sin(2 * np.pi * 440 * np.arange(L) / fs) + 0.3 * rng.standard_normal(L)
    tg, fg, s, _ = filterbankphasegrad(x, g, a, L)
    tg = [t + 0.2 * rng.standard_normal(t.shape) for t in tg]  # exercise every branch
    tg[2] = tg[2].copy()
    tg[2][::5] = np.nan
    fg = [f.copy() for f in fg]
    fg[3][::4] = np.nan
    cf = np.asarray(fc, float) / fs * 2
    fast = R.comp_filterbankreassign(s, tg, fg, a, cf, return_repos=True)
    monkeypatch.setattr(R, "_FORCE_LOOP", True)
    loop = R.comp_filterbankreassign(s, tg, fg, a, cf, return_repos=True)
    for u, v in zip(fast[0], loop[0]):
        np.testing.assert_array_equal(u, v)
    np.testing.assert_array_equal(fast[1], loop[1])


def test_unordered_centres_fall_back_to_the_loop():
    from cool_frames.numpy.phase import _reassign as R

    Lc = [4, 4, 4]
    assert (
        R._reassign_targets(
            [np.zeros(4)] * 3, [np.zeros(4)] * 3, np.ones(3), np.array([0.2, 0.1, 0.5]), Lc
        )
        is None
    )
    s = [np.ones(4)] * 3
    sr, _ = R.comp_filterbankreassign(
        s,
        [np.full(4, 0.1)] * 3,
        [np.zeros(4)] * 3,
        np.ones(3, dtype=int),
        np.array([0.2, 0.1, 0.5]),
    )
    assert np.isclose(sum(float(np.sum(v)) for v in sr), 12.0)


# ---------------------------------------------------------------------------
# 10. Complement responses are cached, and the cache follows the inner bank
# ---------------------------------------------------------------------------


def test_complement_cache_follows_the_inner_filters():
    """Designers rescale inner channels and fit hops after building the
    complements; a cached complement must see that, or the frame response
    has a hole."""
    from cool_frames.numpy.filters._edge_filters import build_complement_lowpass

    g, a, fc, L, _ = audfilters(16000, 4096)
    inner = [dict(gm) for gm in g[1:-1]]
    a_inner = np.asarray(a)[1:-1].copy()

    def build():
        return build_complement_lowpass(
            inner,
            a_inner,
            float(fc[1]),
            16000,
            scal=1.0,
            fsupp_lp=2 * float(fc[1]),
            taper_ratio=0.5,
        )

    lp = build()
    h0 = lp["H"](L)
    h0[:] = 0  # callers get a copy, not the cache
    np.testing.assert_array_equal(lp["H"](L), build()["H"](L))
    inner[0]["H"] = (lambda fn: lambda Lq: np.asarray(fn(Lq)) * 0.5)(inner[0]["H"])
    np.testing.assert_array_equal(lp["H"](L), build()["H"](L))  # new inner filter
    a_inner[0] = a_inner[0] * 2
    np.testing.assert_array_equal(lp["H"](L), build()["H"](L))  # new hop, in place


# ---------------------------------------------------------------------------
# B14-B15: uniform banks that are not painless get their exact canonical
# frames, computed once
# ---------------------------------------------------------------------------


def _dense_analysis(g, a, L):
    """The analysis operator as a matrix (rows: coefficients)."""
    c = filterbank(np.eye(L), g, a, L)
    return np.concatenate([np.asarray(cm).reshape(-1, L) for cm in c], axis=0)


def _nonpainless_gabor_bank(real=True):
    """A uniform ``gabfilters`` bank that is not painless (support 64 bins
    against L/a = 32); single-sided for real signals, or two-sided."""
    from cool_frames.numpy.filters import gabfilters

    g, a, _fc, L, _ = gabfilters(8000, 512, M=64, a=16, real=real)
    return g, int(np.asarray(a).ravel()[0]), L


def _random_coefficients(g, a, L, seed=5):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(L // a) + 1j * rng.standard_normal(L // a) for _ in g]


@pytest.mark.parametrize("real", [False, True])
def test_uniform_dual_is_the_pseudo_inverse(real):
    """Synthesis with the dual of a uniform non-painless bank is the
    pseudo-inverse of the analysis -- for *any* coefficients, not only
    consistent ones, which is what makes it the canonical dual.  For real
    signals (a single-sided bank, ``real=True``) the analysis is the
    real-linear map ``x -> [Re A x; Im A x]``; otherwise a two-sided bank."""
    g, a, L = _nonpainless_gabor_bank(real=real)
    A = _dense_analysis(g, a, L)
    c = _random_coefficients(g, a, L)
    cv = np.concatenate(c)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = ifilterbank(c, filterbankdual(g, a, L, real=real), a, L, real=real)
    if real:
        want = np.linalg.pinv(np.vstack([A.real, A.imag])) @ np.concatenate([cv.real, cv.imag])
    else:
        want = np.linalg.pinv(A) @ cv
    assert np.linalg.norm(f - want) / np.linalg.norm(want) < 1e-10


def test_a_single_sided_bank_with_real_false_gets_a_projection():
    """Not a frame of C^L (it sees no negative frequencies, apart from tails
    of 1e-17 of the peak): the uniform branch raised "not a frame".  It now
    returns the pseudo-inverse on the part of the space the bank covers
    (eigenvalues below 1e-10 of the frame operator's scale count as zero), so
    synthesis followed by analysis is an orthogonal projection: idempotent
    and never longer than its input."""
    g, a, L = _nonpainless_gabor_bank(real=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gd = filterbankdual(g, a, L, real=False)

        def P(c):
            return filterbank(ifilterbank(c, gd, a, L, real=False), g, a, L)

        c0 = _random_coefficients(g, a, L)
        c1 = P(c0)
        c2 = P(c1)
    v0, v1, v2 = (np.concatenate(c) for c in (c0, c1, c2))
    assert np.linalg.norm(v2 - v1) / np.linalg.norm(v1) < 1e-6
    assert np.linalg.norm(v1) <= np.linalg.norm(v0) * (1 + 1e-9)
    assert np.linalg.norm(v1) > 0.25 * np.linalg.norm(v0)  # and does not vanish


@pytest.mark.parametrize("real", [False, True])
def test_uniform_tight_frame_and_bounds_are_exact(real):
    """The tight frame is ``A S^{-1/2}`` with ``S`` the frame operator (for
    real signals, of the real-linear map, and scaled by ``1/sqrt(2)``: the
    real-signal synthesis is ``2 Re(...)``), and the bounds are the extreme
    eigenvalues of ``S``."""
    from cool_frames.numpy.filterbanks import filterbankbounds, filterbanktight

    g, a, L = _nonpainless_gabor_bank(real=real)
    A = _dense_analysis(g, a, L)
    Ar = np.vstack([A.real, A.imag]) if real else A
    lam, V = np.linalg.eigh(Ar.conj().T @ Ar)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gt = filterbanktight(g, a, L, real=real)
        At = _dense_analysis(gt, a, L)
        Atr = np.vstack([At.real, At.imag]) if real else At
        want = Ar @ (V * lam**-0.5) @ V.conj().T
        if real:
            want = want / np.sqrt(2)
        assert np.linalg.norm(Atr - want) / np.linalg.norm(want) < 1e-10
        A_b, B_b = filterbankbounds(g, a, L, real=real)
    scale = 2.0 if real else 1.0
    assert A_b == pytest.approx(scale * lam[0], rel=1e-9)
    assert B_b == pytest.approx(scale * lam[-1], rel=1e-9)


def test_the_dual_is_computed_once_per_content(monkeypatch):
    """``gla`` asks for the same dual on every call; it is cached by the
    content of the evaluated filters and handed out as a copy."""
    import cool_frames.numpy.filterbanks._frame as fr

    g, a, L = _nonpainless_gabor_bank()
    calls = []
    real_uniform = fr._uniform_frame

    def counting(*args, **kwargs):
        calls.append(1)
        return real_uniform(*args, **kwargs)

    monkeypatch.setattr(fr, "_uniform_frame", counting)
    fr._FRAME_CACHE.clear()
    d1 = filterbankdual(g, a, L)
    d2 = filterbankdual(g, a, L)
    assert len(calls) == 1
    for u, v in zip(d1, d2):
        np.testing.assert_array_equal(u["H"], v["H"])
    d1[0]["H"][:] = 0  # the caller's copy, not the cache
    assert np.any(filterbankdual(g, a, L)[0]["H"] != 0)
    assert len(calls) == 1
    g2 = [dict(gm) for gm in g]
    g2[3] = dict(g2[3], H=np.asarray(g2[3]["H"]) * 1.5)  # new content: recomputed
    filterbankdual(g2, a, L)
    assert len(calls) == 2
    filterbankdual(g, a, L, real=False)  # another key
    assert len(calls) == 3


def test_legla_kernel_is_built_once(monkeypatch):
    import cool_frames.numpy.phase._leglakernel as lk
    from cool_frames.numpy.phase import legla

    g, a, _fc, L, _ = audfilters(4000, 512)
    s = [np.abs(cm) for cm in filterbank(np.random.default_rng(0).standard_normal(L), g, a, L)]
    built = []
    real_cls = lk.LeglaKernel

    class Counting(real_cls):
        def __init__(self, *args, **kwargs):
            built.append(1)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(lk, "LeglaKernel", Counting)
    lk._KERNEL_CACHE.clear()
    r1 = legla(s, g, a, L=L, real=True, maxit=3)
    r2 = legla(s, g, a, L=L, real=True, maxit=3)
    assert len(built) == 1
    for u, v in zip(r1[0], r2[0]):
        np.testing.assert_array_equal(u, v)
    legla(s, g, a, L=L, real=True, maxit=3, relthr=1e-2)
    assert len(built) == 2
