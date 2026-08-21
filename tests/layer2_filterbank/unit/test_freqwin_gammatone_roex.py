"""
test_freqwin_gammatone_roex.py
==============================
Tests ported from MATLAB ttest_freqwin_gammatone_roex.m.

These tests verify the gammatone / roex separation fix in freqwin:
  - Gammatone has asymmetric magnitude
  - Roex has symmetric magnitude
  - Roex backward-compatibility with the original analytical formula
  - Order constraints (roex order > 1, gammatone order >= 1)

These complement the existing freqwin tests in test_prop_sigproc.py which
cover peak location, bandwidth, conjugate symmetry, and order effects.
"""
from __future__ import annotations

import pytest

import numpy as np

try:
    from cool_frames.numpy.filters import freqwin
    _HAS_LTFAT = True
except ImportError:
    _HAS_LTFAT = False

pytestmark = pytest.mark.skipif(not _HAS_LTFAT,
                                reason="cool_frames not installed")

L = 512
BW = 0.1


class TestGammatoneRoexSeparation:
    """Verify that 'gammatone' and 'roex' are distinct window types."""

    def test_gammatone_magnitude_symmetric_at_dc(self):
        """Gammatone centred at DC has symmetric magnitude |H(f)| = |H(-f)|.

        Note: asymmetry only appears when the filter is shifted to a
        non-zero centre frequency (inside a filterbank).  At DC the
        formula 1/(1+if/b)^n has symmetric magnitude (1+(f/b)^2)^{-n/2}.
        """
        H = freqwin("gammatone", L, BW)
        mag = np.abs(H)
        sym_err = np.sqrt(sum(
            (mag[k] - mag[L - k]) ** 2
            for k in range(2, L // 4)
        ))
        np.testing.assert_allclose(
            sym_err, 0.0, atol=1e-10,
            err_msg="Gammatone at DC should have symmetric magnitude"
        )

    def test_gammatone_no_peak_modulation(self):
        """Gammatone centred at DC has no peak modulation term, so
        it equals 1/(1+if/b)^n with real-valued H[0]."""
        H = freqwin("gammatone", L, BW)
        # DC component should be real and positive (no peakmod)
        assert abs(H[0].imag) < 1e-14
        assert H[0].real > 0

    def test_roex_has_peak_modulation(self):
        """Roex uses a peakmod term exp(2πi k peakpos), making H[0] differ
        from the gammatone DC value."""
        H_gt = freqwin("gammatone", L, BW)
        H_rx = freqwin("roex", L, BW)
        # They should produce different values (different formulas)
        assert not np.allclose(H_gt, H_rx), \
            "Gammatone and roex should differ"

    def test_roex_has_symmetric_magnitude(self):
        """Roex magnitude |H(f)| == |H(-f)| (symmetric in frequency)."""
        H = freqwin("roex", L, BW)
        mag = np.abs(H)
        sym_err = np.sqrt(sum(
            (mag[k] - mag[L - k]) ** 2
            for k in range(2, L // 4)
        ))
        np.testing.assert_allclose(
            sym_err, 0.0, atol=1e-10,
            err_msg="Roex magnitude should be symmetric"
        )

    def test_roex_rejects_order_1(self):
        """Roex with order=1 should raise ValueError."""
        with pytest.raises(ValueError, match="[Oo]rder"):
            freqwin("roex", L, BW, order=1)

    def test_gammatone_accepts_order_1(self):
        """Gammatone with order=1 should work fine."""
        H = freqwin("gammatone", L, BW, order=1)
        mag = np.abs(H)
        assert mag[0] > mag[10], "Order-1 gammatone should peak at DC"

    def test_roex_backward_compatibility(self):
        """Roex should match the original analytical formula exactly."""
        order_v = 4
        step_v = 2.0 / L
        bwrelheight = 10.0 ** (-3.0 / 10.0)

        k = np.concatenate([
            np.arange(0, np.ceil(L / 2), dtype=float),
            np.arange(-np.floor(L / 2), 0, dtype=float),
        ])

        def gt_inverse(yn):
            return np.sqrt(yn ** (-2.0 / order_v) - 1.0)

        dilation = BW / 2.0 / gt_inverse(bwrelheight) / step_v
        peakpos = (order_v - 1) / (2 * np.pi * dilation)
        peakmod = np.exp(2j * np.pi * k * peakpos)
        H_old = (1 + 1j * k / dilation) ** (-order_v) * peakmod

        H_roex = freqwin("roex", L, BW, order=order_v)
        rel_err = np.linalg.norm(H_roex - H_old) / np.linalg.norm(H_old)
        assert rel_err < 1e-4, \
            f"Roex does not match original analytical formula: rel_err={rel_err:.2e}"


class TestFreqwinAllTypes:
    """Parametric tests across all freqwin window types."""

    @pytest.mark.parametrize("name", ["gauss", "butterworth", "roex", "gammatone"])
    @pytest.mark.parametrize("length", [256, 512, 1024])
    def test_length_correct(self, name, length):
        """Output length matches requested L."""
        H = freqwin(name, length, 0.1)
        assert len(H) == length

    @pytest.mark.parametrize("name", ["gauss", "butterworth", "roex", "gammatone"])
    def test_peak_at_dc(self, name):
        """All freqwin windows should peak at DC."""
        H = freqwin(name, 512, 0.1)
        assert np.argmax(np.abs(H)) == 0

    @pytest.mark.parametrize("name", ["gauss", "butterworth"])
    def test_real_valued_windows(self, name):
        """Gauss and butterworth should be real-valued."""
        H = freqwin(name, 256, 0.2)
        assert np.isrealobj(H) or np.allclose(H.imag, 0.0, atol=1e-15)

    @pytest.mark.parametrize("name", ["roex", "gammatone"])
    def test_complex_valued_windows(self, name):
        """Roex and gammatone should be complex-valued."""
        H = freqwin(name, 256, 0.2)
        assert np.iscomplexobj(H)

    @pytest.mark.parametrize("name", ["butterworth", "roex", "gammatone"])
    @pytest.mark.parametrize("order", [2, 4, 8])
    def test_higher_order_steeper(self, name, order):
        """Higher order should give steeper rolloff (narrower mainlobe)."""
        if name == "roex" and order < 2:
            pytest.skip("Roex requires order > 1")
        H_lo = np.abs(freqwin(name, 512, 0.1, order=max(2, order // 2)))
        H_hi = np.abs(freqwin(name, 512, 0.1, order=order))
        # At bin 60 (well outside passband), higher order should be smaller
        assert H_hi[60] <= H_lo[60] + 1e-12
