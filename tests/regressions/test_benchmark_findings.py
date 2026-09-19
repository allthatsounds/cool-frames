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

Each test below fails on commit 1f581bc and passes after the fix.
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
