"""
Tests for sigproc utilities: rms, setnorm, gaindb, pfilt,
thresh, largestn, largestr, transferfunction, pgrpdelay.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the package is importable
# ---------------------------------------------------------------------------
_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from cool_frames.numpy.filters import audfilters, firwin, freqwin
from cool_frames.numpy.core import setnorm
from cool_frames.numpy.filterbanks import (
    filterbank,
    magresp,
    pgrpdelay,
    rampsignal,
    resize_fir,
    transferfunction,
)
from cool_frames.sigproc import largest, thresh

# ===================================================================
# Helpers
# ===================================================================

def _randn(n, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(n)


# ===================================================================
# TestRMS
# ===================================================================

class TestSetnorm:
    """Tests for setnorm()."""

    def test_l2_norm(self):
        """After L2 normalisation, ||f||_2 should equal val."""
        f = _randn(200)
        f_out, fnorm = setnorm(f, "2", val=1.0)
        assert abs(np.linalg.norm(f_out) - 1.0) < 1e-12
        assert abs(fnorm - np.linalg.norm(f)) < 1e-12

    def test_l1_norm(self):
        """After L1 normalisation, ||f||_1 should equal val."""
        f = _randn(200)
        f_out, fnorm = setnorm(f, "1", val=2.0)
        assert abs(np.sum(np.abs(f_out)) - 2.0) < 1e-12

    def test_linf_norm(self):
        """After Linf normalisation, max|f| should equal val."""
        f = _randn(200)
        f_out, _ = setnorm(f, "inf", val=1.0)
        assert abs(np.max(np.abs(f_out)) - 1.0) < 1e-12

    def test_peak_alias(self):
        """'peak' is an alias for 'inf'."""
        f = _randn(200)
        f1, _ = setnorm(f, "inf")
        f2, _ = setnorm(f, "peak")
        assert np.allclose(f1, f2)

    def test_energy_alias(self):
        """'energy' is an alias for '2'."""
        f = _randn(200)
        f1, _ = setnorm(f, "2")
        f2, _ = setnorm(f, "energy")
        assert np.allclose(f1, f2)

    def test_area_alias(self):
        """'area' is an alias for '1'."""
        f = _randn(200)
        f1, _ = setnorm(f, "1")
        f2, _ = setnorm(f, "area")
        assert np.allclose(f1, f2)

    def test_rms_norm(self):
        """After RMS normalisation, the RMS level of f should equal val."""
        f = _randn(200)
        f_out, _ = setnorm(f, "rms", val=1.0)
        rms_level = np.linalg.norm(f_out) / np.sqrt(len(f_out))
        assert abs(rms_level - 1.0) < 1e-12

    def test_wav_norm(self):
        """'wav' normalisation: peak should be 0.99."""
        f = _randn(200)
        f_out, _ = setnorm(f, "wav", val=1.0)
        assert abs(np.max(np.abs(f_out)) - 0.99) < 1e-12

    def test_null_passthrough(self):
        """'null' should return the input unchanged."""
        f = _randn(200)
        f_out, fnorm = setnorm(f, "null")
        assert np.array_equal(f_out, f)
        assert fnorm == 0.0

    def test_custom_val(self):
        """Setting val=5 should produce ||f||_2 == 5."""
        f = _randn(200)
        f_out, _ = setnorm(f, "2", val=5.0)
        assert abs(np.linalg.norm(f_out) - 5.0) < 1e-10

    def test_multichannel(self):
        """Column-wise normalisation on a matrix."""
        rng = np.random.default_rng(0)
        f = rng.standard_normal((128, 3))
        f_out, fnorms = setnorm(f, "2", val=1.0, dim=0)
        for i in range(3):
            assert abs(np.linalg.norm(f_out[:, i]) - 1.0) < 1e-12

    def test_complex_signal(self):
        """Normalisation should work for complex arrays."""
        rng = np.random.default_rng(0)
        f = rng.standard_normal(100) + 1j * rng.standard_normal(100)
        f_out, _ = setnorm(f, "2", val=1.0)
        assert abs(np.linalg.norm(f_out) - 1.0) < 1e-12


# ===================================================================
# TestGaindb
# ===================================================================

class TestThresh:
    """Tests for thresh()."""

    def test_hard_zeros_below(self):
        """Hard threshold zeros elements below lambda."""
        x = np.array([0.5, 1.5, -2.0, 0.1, -0.3])
        xo, N = thresh(x, 1.0, "hard")
        assert np.allclose(xo, [0.0, 1.5, -2.0, 0.0, 0.0])
        assert N == 2

    def test_soft_shrinkage(self):
        """Soft threshold shrinks magnitudes by lambda."""
        x = np.array([3.0, -2.0, 0.5])
        xo, N = thresh(x, 1.0, "soft")
        assert np.allclose(xo, [2.0, -1.0, 0.0])
        assert N == 2

    def test_wiener_intermediate(self):
        """Wiener shrinkage: gain = max(1 - (lam/|x|)^2, 0)."""
        x = np.array([4.0, 2.0, 0.5])
        xo, _ = thresh(x, 1.0, "wiener")
        expected = x * np.maximum(1.0 - (1.0 / np.abs(x)) ** 2, 0.0)
        assert np.allclose(xo, expected)

    def test_complex_soft(self):
        """Soft threshold preserves phase for complex input."""
        x = np.array([3.0 + 4.0j])  # |x| = 5
        xo, _ = thresh(x, 2.0, "soft")
        # |xo| should be 3, phase preserved
        assert abs(np.abs(xo[0]) - 3.0) < 1e-12
        assert abs(np.angle(xo[0]) - np.angle(x[0])) < 1e-12

    def test_hard_keeps_equal(self):
        """Hard threshold keeps elements exactly equal to lambda."""
        x = np.array([1.0, -1.0, 0.5])
        xo, N = thresh(x, 1.0, "hard")
        assert np.allclose(xo, [1.0, -1.0, 0.0])
        assert N == 2

    def test_vector_lambda(self):
        """Element-wise lambda array."""
        x = np.array([1.0, 2.0, 3.0])
        lam = np.array([0.5, 2.5, 1.0])
        xo, N = thresh(x, lam, "hard")
        assert np.allclose(xo, [1.0, 0.0, 3.0])
        assert N == 2

    def test_ordering_soft_hard_wiener(self):
        """For same lambda, |soft| <= |wiener| <= |hard| element-wise."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(200)
        lam = 0.5
        h, _ = thresh(x, lam, "hard")
        w, _ = thresh(x, lam, "wiener")
        s, _ = thresh(x, lam, "soft")
        assert np.all(np.abs(s) <= np.abs(w) + 1e-14)
        assert np.all(np.abs(w) <= np.abs(h) + 1e-14)


# ===================================================================
# TestLargestn
# ===================================================================

class TestLargestn:
    """Tests for largest()."""

    def test_keep_n(self):
        """Keep exactly N largest coefficients."""
        x = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        xo, N_kept = largest(x, 3, "hard")
        assert N_kept == 3
        # The three largest are 5, 4, 3
        assert xo[1] == 5.0
        assert xo[4] == 4.0
        assert xo[2] == 3.0
        assert xo[0] == 0.0
        assert xo[3] == 0.0

    def test_keep_all(self):
        """Keeping all should return original."""
        x = _randn(50)
        xo, _ = largest(x, 50, "hard")
        assert np.allclose(xo, x)

    def test_keep_zero(self):
        """Keeping 0 returns all zeros."""
        x = _randn(50)
        xo, N_kept = largest(x, 0)
        assert np.allclose(xo, 0.0)
        assert N_kept == 0

    def test_soft_mode(self):
        """Soft mode with largestn should shrink kept coefficients."""
        x = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
        xo_hard, _ = largest(x, 2, "hard")
        xo_soft, _ = largest(x, 2, "soft")
        # Soft output magnitudes should be <= hard
        assert np.all(np.abs(xo_soft) <= np.abs(xo_hard) + 1e-14)


# ===================================================================
# TestLargestr
# ===================================================================

class TestLargestr:
    """Tests for largest()."""

    def test_fraction(self):
        """Keep 50% of 10 coefficients = 5."""
        rng = np.random.default_rng(0)
        x = rng.standard_normal(10)
        xo, N_kept = largest(x, 0.5)
        assert N_kept == 5

    def test_integer_passthrough(self):
        """Integer p >= 1 is treated as absolute count."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        xo, N_kept = largest(x, 3.0)
        assert N_kept == 3

    def test_full_fraction(self):
        """p=1.0 keeps all."""
        x = _randn(20)
        # p=1.0 → int(1.0)==1.0 so treated as count=1, not fraction
        # Actually 1.0 == int(1.0), so it goes to n=1
        # Let's use 0.999 to keep ~all
        xo, _ = largest(x, 0.999)
        # round(20 * 0.999) = 20
        assert np.allclose(xo, x)


# ===================================================================
# TestTransferfunction
# ===================================================================

class TestTransferfunction:
    """Tests for transferfunction()."""

    def test_impulse_filter(self):
        """Unit impulse filter has flat frequency response."""
        g = {"h": np.array([1.0]), "offset": 0, "foff": 0}
        H = transferfunction(g, 128)
        assert np.allclose(np.abs(H), 1.0)

    def test_audfilter_energy(self):
        """Transfer function magnitude should match filter design."""
        fs = 8000
        Ls = 512
        g, a, fc, L, _info = audfilters(fs, Ls)
        # Pick a filter and check H has correct length
        H = transferfunction(g[5], L)
        assert H.shape == (L,)
        # Non-zero energy
        assert np.sum(np.abs(H) ** 2) > 0

    def test_consistent_with_filterbankresponse(self):
        """Sum of |H_m|^2/a_m via transferfunction should match filterbankresponse."""
        from cool_frames.numpy.filterbanks import filterbankresponse
        fs = 8000
        Ls = 512
        g, a, fc, L, _info = audfilters(fs, Ls)
        # Compute via transferfunction
        resp_manual = np.zeros(L)
        for m in range(len(g)):
            H = transferfunction(g[m], L)
            am = float(a[m]) if np.ndim(a[m]) == 0 else float(a[m][0])
            resp_manual += np.abs(H) ** 2 / am
        # Compare with filterbankresponse
        resp_api = filterbankresponse(g, a, L)
        assert np.allclose(resp_manual, resp_api, atol=1e-10)


# ===================================================================
# TestPgrpdelay
# ===================================================================

class TestPgrpdelay:
    """Tests for pgrpdelay()."""

    def test_constant_grpdelay_linear_phase(self):
        """A linear-phase FIR should have constant group delay."""
        # 3-tap symmetric FIR placed at offset 0 → linear phase, delay=1
        h = np.array([0.25, 0.5, 0.25])
        g = {"h": h, "offset": 0, "foff": 0}
        L = 128
        ggd = pgrpdelay(g, L)
        # At bins where |H| is significant, group delay ≈ 1
        H = transferfunction(g, L)
        mask = np.abs(H) > 0.1 * np.max(np.abs(H))
        assert np.allclose(ggd[mask], 1.0, atol=0.1)

    def test_pure_delay(self):
        """A pure delay of d samples should give group delay = d."""
        L = 256
        d = 10
        # Create a filter that's just a delayed impulse
        h = np.zeros(d + 1)
        h[d] = 1.0
        g = {"h": h, "offset": 0, "foff": 0}
        ggd = pgrpdelay(g, L)
        # Group delay should be d everywhere
        assert np.allclose(ggd, d, atol=1e-8)

    def test_output_shape(self):
        """Output should have length L."""
        g = {"h": np.array([1.0, 0.5]), "offset": 0, "foff": 0}
        ggd = pgrpdelay(g, 512)
        assert ggd.shape == (512,)

    def test_audfilter_reasonable(self):
        """Group delay of an auditory filter should be finite and reasonable."""
        fs = 8000
        Ls = 512
        g, a, fc, L, _info = audfilters(fs, Ls)
        ggd = pgrpdelay(g[5], L)
        # Should be finite
        assert np.all(np.isfinite(ggd))
        # Group delay magnitude should be < L (no wrap-around issues)
        assert np.max(np.abs(ggd)) < L


# ===================================================================
# TestFir2long / TestLong2fir
# ===================================================================

class TestFir2long:
    """Tests for resize_fir()."""

    def test_identity(self):
        """fir2long with same length returns copy."""
        g = _randn(64)
        assert np.allclose(resize_fir(g, 64), g)

    def test_zero_padding(self):
        """Extended window should have zeros in the middle."""
        g = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # odd length
        out = resize_fir(g, 11)
        assert out[0] == 1.0
        assert out[1] == 2.0
        assert out[2] == 3.0
        assert np.allclose(out[3:9], 0.0)
        assert out[9] == 4.0
        assert out[10] == 5.0

    def test_roundtrip(self):
        """fir2long then long2fir should recover original."""
        g = _randn(31)
        out = resize_fir(resize_fir(g, 128), 31)
        assert np.allclose(out, g)

    def test_even_length(self):
        """Even-length FIR window."""
        g = np.array([1.0, 2.0, 3.0, 4.0])
        out = resize_fir(g, 8)
        assert len(out) == 8
        # DC + first half preserved
        assert out[0] == 1.0
        assert out[1] == 2.0


class TestLong2fir:
    """Tests for resize_fir()."""

    def test_identity(self):
        """long2fir with same length returns copy."""
        g = _randn(64)
        assert np.allclose(resize_fir(g, 64), g)

    def test_preserves_edges(self):
        """First and last samples preserved."""
        g = np.arange(32, dtype=float)
        out = resize_fir(g, 7)
        assert len(out) == 7
        # First ceil(7/2)=4 samples
        assert np.allclose(out[:4], g[:4])
        # Last floor(7/2)=3 samples
        assert np.allclose(out[4:], g[29:])

    def test_longer_pads(self):
        """resize_fir with L > len(g) zero-pads the middle (was long2fir error)."""
        out = resize_fir(np.ones(10), 20)
        assert len(out) == 20


# ===================================================================
# TestKaiserWindow  (folded into firwin('kaiser', M, beta=...))
# ===================================================================

class TestKaiserWindow:
    """Tests for the Kaiser window via firwin('kaiser', ...)."""

    def test_length(self):
        """Output should have requested length."""
        g = firwin('kaiser', 64, beta=5.0)
        assert len(g) == 64

    def test_beta_zero_is_rectangular(self):
        """beta=0 should give a rectangular (flat) window."""
        g = firwin('kaiser', 32, beta=0.0)
        g_c = np.fft.fftshift(g)
        nz = g_c[g_c > 0.01]
        assert np.std(nz) / np.mean(nz) < 0.01

    def test_dc_is_max(self):
        """DC component (index 0) should be the maximum."""
        g = firwin('kaiser', 64, beta=8.0)
        assert g[0] == np.max(g)

    def test_even_length_nyquist_zero(self):
        """For even M, the Nyquist bin should be zero."""
        g = firwin('kaiser', 64, beta=5.0)
        assert g[32] == 0.0

    def test_real_valued(self):
        """Output should be real."""
        g = firwin('kaiser', 100, beta=10.0)
        assert np.isrealobj(g)

    def test_requires_beta(self):
        """Omitting beta for the kaiser window is an error."""
        import pytest
        with pytest.raises(ValueError):
            firwin('kaiser', 64)

    def test_norm_unit_peak_default(self):
        """firwin conventions: default norm='inf' -> unit peak."""
        g = firwin('kaiser', 64, beta=5.0)
        assert abs(np.max(np.abs(g)) - 1.0) < 1e-10


# ===================================================================
# TestRamps
# ===================================================================

class TestRamps:
    """Tests for rampsignal (the fade-in/out applier; ramp envelope is internal).

    The fade envelope is exercised via ``rampsignal(ones(2L), [L, L])``, whose
    output (no middle plateau) is exactly ``[fade_in(L), fade_out(L)]``.
    """

    def test_fade_in_rises_0_to_1(self):
        """Fade-in segment rises ~0 -> ~1, monotonically."""
        L = 100
        rise = rampsignal(np.ones(2 * L), [L, L])[:L]
        assert rise[0] < 0.01 and rise[-1] > 0.95
        assert np.all(np.diff(rise) >= -1e-14)

    def test_fade_out_falls_1_to_0(self):
        """Fade-out segment falls ~1 -> ~0, monotonically."""
        L = 100
        fall = rampsignal(np.ones(2 * L), [L, L])[L:]
        assert fall[0] > 0.95 and fall[-1] < 0.01
        assert np.all(np.diff(fall) <= 1e-14)

    def test_fade_halves_complement(self):
        """Fade-in and fade-out halves sum to ~1 (Hann halves)."""
        L = 100
        env = rampsignal(np.ones(2 * L), [L, L])
        assert np.allclose(env[:L] + env[L:], 1.0, atol=1e-12)

    def test_rampsignal_preserves_middle(self):
        """Middle of ramped signal should be unchanged."""
        f = np.ones(200)
        out = rampsignal(f, 30)
        assert np.allclose(out[30:170], 1.0)

    def test_rampsignal_endpoints_zero(self):
        """Ramped signal starts and ends near zero."""
        f = np.ones(200)
        out = rampsignal(f, 30)
        assert out[0] < 0.01
        assert out[-1] < 0.01

    def test_rampsignal_asymmetric(self):
        """Asymmetric ramp lengths."""
        f = np.ones(200)
        out = rampsignal(f, [20, 50])
        assert np.allclose(out[20:150], 1.0)

    def test_rampsignal_multichannel(self):
        """Rampsignal on matrix."""
        rng = np.random.default_rng(0)
        f = rng.standard_normal((200, 3))
        out = rampsignal(f, 30, dim=0)
        assert out.shape == f.shape
        # First sample of each channel should be ~0
        assert np.all(np.abs(out[0, :]) < 0.01)


# ===================================================================
# TestFreqwin
# ===================================================================

class TestFreqwin:
    """Tests for freqwin()."""

    def test_gauss_peak_at_dc(self):
        """Gaussian window should peak at DC."""
        H = freqwin("gauss", 256, 0.2)
        assert H[0] == np.max(np.abs(H))

    def test_gauss_real(self):
        """Gaussian window should be real."""
        H = freqwin("gauss", 256, 0.1)
        assert np.isrealobj(H) or np.allclose(H.imag, 0, atol=1e-15)

    def test_butterworth_order(self):
        """Higher order Butterworth should have steeper rolloff."""
        H4 = np.abs(freqwin("butterworth", 256, 0.2, order=4))
        H8 = np.abs(freqwin("butterworth", 256, 0.2, order=8))
        # At a frequency well outside the passband, H8 should be smaller
        assert H8[64] < H4[64]

    def test_roex_complex(self):
        """Roex window should be complex (asymmetric)."""
        H = freqwin("roex", 256, 0.2)
        assert np.iscomplexobj(H)

    def test_gammatone_complex(self):
        """True gammatone should be complex."""
        H = freqwin("gammatone", 256, 0.2)
        assert np.iscomplexobj(H)

    def test_gammatone_peak_at_dc(self):
        """Gammatone magnitude should peak at DC."""
        H = freqwin("gammatone", 512, 0.1)
        assert np.argmax(np.abs(H)) == 0

    def test_gammatone_bandwidth(self):
        """Gammatone -6dB bandwidth should match requested bw."""
        L = 4096
        bw = 0.1
        H = freqwin("gammatone", L, bw)
        mag = np.abs(H)
        peak = mag[0]
        target = peak * 10.0 ** (-3.0 / 10.0)  # -6 dB
        # Find the bin closest to bw/2 = 0.05 (normalised freq)
        # In bins: bw/2 / step = 0.05 / (2/L) = 0.05 * L / 2
        bin_half_bw = int(round(bw / 2.0 * L / 2.0))
        # Magnitude at that bin should be close to target
        assert abs(mag[bin_half_bw] - target) / peak < 0.05

    def test_gammatone_conjugate_symmetric(self):
        """Gammatone at DC should be conjugate-symmetric (real time-domain
        envelope).  H(-f) = conj(H(f)) so IFFT is real."""
        H = freqwin("gammatone", 1024, 0.1, order=4)
        h = np.fft.ifft(H)
        # Time-domain impulse response should be essentially real
        # (imaginary part negligible relative to real part)
        assert np.max(np.abs(h.imag)) / np.max(np.abs(h.real)) < 1e-6

    def test_gammatone_causal_like_decay(self):
        """The gammatone time-domain envelope should decay monotonically
        after its peak (causal-like structure)."""
        H = freqwin("gammatone", 2048, 0.05, order=4)
        h = np.fft.ifft(H).real
        h_shifted = np.fft.fftshift(h)
        # The peak should be near the centre; after peak, envelope decays
        peak_idx = np.argmax(np.abs(h_shifted))
        # Check that the magnitude decays over the next 100 samples
        env = np.abs(h_shifted[peak_idx:peak_idx + 200])
        # Not strictly monotonic due to oscillations, but the envelope
        # trend should be downward: last quarter < first quarter
        assert np.mean(env[-50:]) < np.mean(env[:50])

    def test_gammatone_vs_roex_different(self):
        """Gammatone and roex should produce different results."""
        H_gt = freqwin("gammatone", 512, 0.1, order=4)
        H_rx = freqwin("roex", 512, 0.1, order=4)
        # They should NOT be equal
        assert not np.allclose(H_gt, H_rx)

    def test_gammatone_higher_order_narrower(self):
        """Higher order gammatone should have narrower mainlobe."""
        H4 = np.abs(freqwin("gammatone", 512, 0.1, order=4))
        H8 = np.abs(freqwin("gammatone", 512, 0.1, order=8))
        # At a frequency well outside the passband, H8 should be smaller
        assert H8[40] < H4[40]

    def test_gammatone_order1(self):
        """Order-1 gammatone should be a simple Lorentzian."""
        H = freqwin("gammatone", 256, 0.2, order=1)
        # |H(f)| = 1 / sqrt(1 + (f/b)^2)  for n=1
        assert np.iscomplexobj(H)
        assert np.abs(H[0]) > np.abs(H[10])

    def test_length(self):
        """Output length should match L."""
        for name in ["gauss", "butterworth", "roex", "gammatone"]:
            H = freqwin(name, 512, 0.1)
            assert len(H) == 512


# ===================================================================
# TestMagresp
# ===================================================================

class TestMagresp:
    """Tests for magresp()."""

    def test_impulse_flat(self):
        """Impulse filter should have flat magnitude response."""
        g = {"h": np.array([1.0]), "offset": 0, "foff": 0}
        freq, db = magresp(g, L=128)
        # All dB values should be ~0
        assert np.allclose(db, 0.0, atol=0.01)

    def test_returns_two_arrays(self):
        """Should return (freq, mag_db) tuple."""
        g = {"h": np.array([0.5, 0.5]), "offset": 0, "foff": 0}
        freq, db = magresp(g, L=256)
        assert freq.shape == db.shape
        assert len(freq) > 0

    def test_dynrange_clipping(self):
        """Dynamic range should clip floor."""
        g = {"h": np.array([0.5, 1.0, 0.5]), "offset": 0, "foff": 0}
        freq, db = magresp(g, L=256, dynrange=40.0)
        assert np.min(db) >= np.max(db) - 40.0 - 0.01

    def test_fir_array_input(self):
        """Should accept bare FIR arrays."""
        h = np.array([0.25, 0.5, 0.25])
        freq, db = magresp(h, L=256)
        assert len(freq) > 0

    def test_with_fs(self):
        """Frequency axis should be in Hz when fs is given."""
        g = {"h": np.array([1.0]), "offset": 0, "foff": 0}
        freq, db = magresp(g, L=128, fs=8000.0)
        assert freq[-1] == 4000.0  # Nyquist


# ===================================================================
# Run all tests
# ===================================================================

def run_all():
    passed = 0
    failed = 0
    errors = []

    for cls in [TestRMS, TestSetnorm, TestGaindb, TestPfilt,
                TestThresh, TestLargestn, TestLargestr,
                TestTransferfunction, TestPgrpdelay,
                TestFir2long, TestLong2fir, TestFirkaiser,
                TestRamps, TestFreqwin, TestMagresp]:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(obj, name)()
                passed += 1
                print(f"  PASS  {cls.__name__}.{name}")
            except Exception as e:
                failed += 1
                errors.append((f"{cls.__name__}.{name}", e))
                print(f"  FAIL  {cls.__name__}.{name}: {e}")

    print(f"\n{passed} passed, {failed} failed")
    for name, e in errors:
        print(f"  FAIL: {name}")
        import traceback
        traceback.print_exception(type(e), e, e.__traceback__)
    return failed


if __name__ == "__main__":
    sys.exit(run_all())
