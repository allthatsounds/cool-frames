"""
tests/layer1_filters/unit/test_waveletfilters.py
================================================
Python unit tests for waveletfilters and freqwavelet.

Mirrors TestWaveletFilters.m, covering:
  1. Return-value structure
  2. Wavelet types (cauchy, morse, morlet, fbsp, analyticsp, cplxsp)
  3. Sampling modes (regsampling, uniform, fractional, fractionaluniform)
  4. Lowpass modes (single, repeat, none)
  5. Frequency ranges (real, complex)
  6. Reconstruction test
  7. freqwavelet output formats
  8. Wavelet generator functions
  9. Lambert W function
 10. firwin_eval
"""
from __future__ import annotations

import pytest

import numpy as np

try:
    from cool_frames.numpy.filterbanks import filterbank, filterbankdual, ifilterbank
    from cool_frames.numpy.filters import waveletfilters
    from cool_frames.numpy.filters._firwin import firwin_eval
    from cool_frames.numpy.filters._freqwavelet import freqwavelet
    from cool_frames.numpy.filters._wavelet import (
        WAVELET_TYPES,
        lambertw,
        wavelet_generator_func,
    )
    _HAS_LTFAT = True
except ImportError:
    _HAS_LTFAT = False

pytestmark = pytest.mark.skipif(not _HAS_LTFAT,
                                reason="cool_frames not installed")

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

LS = 4096
SCALES = np.linspace(10, 0.1, 50)


# ---------------------------------------------------------------------------
# 1. Return-value structure
# ---------------------------------------------------------------------------

class TestReturnStructure:

    def test_returns_five_outputs(self):
        gout, a, fc, L, info = waveletfilters(2.0, LS, scales=SCALES)
        assert len(gout) > 0
        assert a is not None
        assert len(fc) > 0
        assert L >= LS
        assert isinstance(info, dict)

    def test_fc_length_matches_filters(self):
        gout, _, fc, _, _ = waveletfilters(2.0, LS, scales=SCALES)
        assert len(fc) == len(gout)

    def test_filter_descriptor_has_H_and_foff(self):
        gout, _, _, _, _ = waveletfilters(2.0, LS, scales=SCALES)
        for g in gout:
            assert "H" in g or "h" in g
            assert "foff" in g

    def test_info_keys(self):
        _, _, _, _, info = waveletfilters(2.0, LS, scales=SCALES)
        for key in ["fc", "foff", "fsupp", "scale", "aprecise"]:
            assert key in info


# ---------------------------------------------------------------------------
# 2. Wavelet types
# ---------------------------------------------------------------------------

class TestWaveletTypes:

    @pytest.mark.parametrize("name_args", [
        ["cauchy", 300],
        ["morse", 100, 0, 3],
        ["morlet", 6],
        ["fbsp", 4, 3],
        ["analyticsp", 3, 2],
        ["cplxsp", 3, 2],
    ])
    def test_wavelet_type(self, name_args):
        gout, a, fc, L, info = waveletfilters(2.0, LS, scales=SCALES, wavelet=name_args)
        assert len(gout) > 0


# ---------------------------------------------------------------------------
# 3. Sampling modes
# ---------------------------------------------------------------------------

class TestSamplingModes:

    def test_regsampling(self):
        gout, a, fc, L, _ = waveletfilters(2.0, LS, scales=SCALES, sampling="regsampling")
        assert L >= LS

    def test_uniform(self):
        gout, a, fc, L, _ = waveletfilters(2.0, LS, scales=SCALES, sampling="uniform")
        a_flat = np.asarray(a).ravel()
        # All subsampling rates should be the same
        # (for uniform, first column should be equal)

    def test_fractional(self):
        gout, a, fc, L, _ = waveletfilters(2.0, LS, scales=SCALES, sampling="fractional")
        assert L == LS
        assert np.asarray(a).shape[1] == 2


# ---------------------------------------------------------------------------
# 4. Lowpass modes
# ---------------------------------------------------------------------------

class TestLowpassModes:

    def test_single_lowpass_has_dc(self):
        gout, _, fc, _, _ = waveletfilters(2.0, LS, scales=SCALES, lowpass="single")
        assert fc[0] == pytest.approx(0.0)

    def test_repeat_adds_filters(self):
        gout, _, _, _, _ = waveletfilters(2.0, LS, scales=SCALES, lowpass="repeat")
        assert len(gout) > len(SCALES)

    def test_none_keeps_scale_count(self):
        gout, _, _, _, _ = waveletfilters(2.0, LS, scales=SCALES, lowpass="none")
        assert len(gout) == len(SCALES)


# ---------------------------------------------------------------------------
# 5. Frequency ranges
# ---------------------------------------------------------------------------

class TestFrequencyRanges:

    def test_complex_has_negative_fc(self):
        gout, _, fc, _, _ = waveletfilters(
            2.0, LS, scales=SCALES, lowpass="repeat", freqrange="complex"
        )
        assert np.any(fc < 0)


# ---------------------------------------------------------------------------
# 6. Reconstruction test
# ---------------------------------------------------------------------------

def _roundtrip(g, a, L, Ls, seed=42):
    """Analyse and resynthesise white noise through the painless dual."""
    gd = filterbankdual(g, a, L)
    f = np.random.RandomState(seed).randn(L)
    r = ifilterbank(filterbank(f, g, a), gd, a, real=True)
    return float(np.linalg.norm(np.real(r[:Ls]) - f[:Ls]) / np.linalg.norm(f[:Ls]))


class TestReconstruction:
    """Painless-frame reconstruction via filterbankdual + ifilterbank(real=True).

    History
    -------
    This class used to skip ``waveletfilters`` entirely, on the stated ground
    that "waveletfilters with linspace scales and uniform sampling does NOT
    form a painless frame".  That was true of the two configurations named,
    and false as a reason to leave the designer untested: it conflates two
    independent failures (a ``linspace`` scale set does not *cover* the
    spectrum, which is an admissibility problem; a uniform hop violates the
    *painless* condition, which is a hop problem) and neither is a property of
    ``waveletfilters`` as such.  Meanwhile the designer's own default shipped
    with a 75 % reconstruction error that nothing here would have caught.

    The cqtfilters case that stayed behind asserted ``rel_err < 1.0`` -- a
    100 % relative error passes -- rationalised as "clearly not divergent".
    It measures 4.7e-16.
    """

    def test_reconstruction_waveletfilters(self):
        """The default wavelet bank inverts exactly through its painless dual.

        Regression test.  Until v0.1.1 ``painless`` defaulted to ``False``,
        the bank was 22x over its painless limit, and this round trip lost
        75 % of the signal while ``filterbankbounds`` reported a healthy
        ``kappa = 2.28``.
        """
        fs, Ls = 8000, 4096
        scales = 4 * 2.0 ** (-np.arange(64) / 12)
        g, a, _fc, L, info = waveletfilters(fs, Ls, scales=scales)
        assert info["painless"] is True
        assert info["painless_ratio"] <= 1.0
        rel_err = _roundtrip(g, a, L, Ls)
        assert rel_err < 1e-10, f"waveletfilters reconstruction error: {rel_err}"

    def test_reconstruction_waveletfilters_fmin_fmax(self):
        """Same, through the ``fmin``/``fmax``/``bins`` entry point rather
        than an explicit scale vector."""
        fs, Ls = 8000, 4096
        g, a, _fc, L, info = waveletfilters(fs, Ls, fmin=50.0, fmax=3900.0,
                                            bins=12)
        assert info["painless"] is True
        rel_err = _roundtrip(g, a, L, Ls)
        assert rel_err < 1e-10, f"waveletfilters reconstruction error: {rel_err}"

    @pytest.mark.parametrize("sampling", ["regsampling", "uniform",
                                          "fractional", "fractionaluniform"])
    def test_reconstruction_waveletfilters_all_sampling_modes(self, sampling):
        """``painless=True`` used to be honoured by ``regsampling`` only and
        silently ignored by the other three, each of which then failed to
        reconstruct (0.09 to 0.82 relative error)."""
        fs, Ls = 8000, 4096
        scales = 4 * 2.0 ** (-np.arange(64) / 12)
        g, a, _fc, L, info = waveletfilters(fs, Ls, scales=scales,
                                            sampling=sampling)
        assert info["painless"] is True, sampling
        rel_err = _roundtrip(g, a, L, Ls)
        assert rel_err < 1e-10, f"{sampling}: {rel_err}"

    def test_reconstruction_waveletfilters_sparse_scales(self):
        """A sparse scale set leaves a wide gap up to Nyquist, so the
        complement filter appended to fill it is wide too.

        It is built after the hops are chosen and inherits one, which at
        ``24`` scales at ``6``/octave put a 1113-bin filter on a hop of 6
        (``aW/L = 3.87``) -- one channel over the limit, enough to send the
        exact lower frame bound to 0 while the diagonal estimator still
        reported ``kappa = 4.4``.
        """
        fs, Ls = 8000, 1024
        scales = 4 * 2.0 ** (-np.arange(24) / 6)
        g, a, _fc, L, info = waveletfilters(fs, Ls, scales=scales)
        assert info["painless"] is True
        rel_err = _roundtrip(g, a, L, Ls)
        assert rel_err < 1e-10, f"sparse-scale reconstruction error: {rel_err}"

    def test_non_painless_bank_warns(self):
        """``painless=False`` is a legitimate choice for analysis-only work.
        Being quiet about the dual it breaks is not."""
        import warnings as _w

        fs, Ls = 8000, 4096
        scales = 4 * 2.0 ** (-np.arange(64) / 12)
        with _w.catch_warnings(record=True) as rec:
            _w.simplefilter("always")
            g, a, _fc, L, info = waveletfilters(fs, Ls, scales=scales,
                                                painless=False)
        assert info["painless"] is False
        assert info["painless_ratio"] > 1.0
        assert any("painless condition" in str(r.message) for r in rec)
        # ... and the warning is not a false alarm: the dual really does fail.
        assert _roundtrip(g, a, L, Ls) > 0.1

    def test_reconstruction_cqtfilters(self):
        """CQT filterbank reconstruction via the painless dual is exact.

        The assertion here was ``< 1.0`` until v0.1.1, described as "we only
        require the error to be < 1.0 (clearly not divergent)" on the grounds
        that kappa was about 64.  The condition number bounds the *dual's*
        norm, not the round-trip error: a painless bank round-trips to machine
        precision at any kappa.  It measures 4.7e-16.
        """
        from cool_frames.filters import cqtfilters

        fs, Ls = 16000, 512
        g, a, fc, L, _info = cqtfilters(fs, Ls, fmin=50, fmax=7000, bins=12)
        rel_err = _roundtrip(g, a, L, Ls)
        assert rel_err < 1e-10, f"CQT reconstruction error: {rel_err}"

    def test_reconstruction_audfilters(self):
        """Auditory filterbank gives perfect reconstruction via painless dual."""
        from cool_frames.filters import audfilters

        fs, Ls = 16000, 4000
        g, a, fc, L, _info = audfilters(fs, Ls)
        rel_err = _roundtrip(g, a, L, Ls)
        assert rel_err < 1e-10, f"Audfilters reconstruction error: {rel_err}"


# ---------------------------------------------------------------------------
# 7. freqwavelet output formats
# ---------------------------------------------------------------------------

class TestFreqwavelet:

    def test_full_format(self):
        H, info = freqwavelet(["cauchy", 300], 1024, scale=1.0,
                              output_format="full")
        assert H.shape[0] == 1024

    def test_econ_format(self):
        H, info = freqwavelet(["cauchy", 300], 1024,
                              scale=np.array([1.0, 2.0, 4.0]),
                              output_format="econ")
        assert isinstance(H, list)
        assert len(H) == 3

    def test_asfreqfilter_format(self):
        H, info = freqwavelet(["cauchy", 300], 1024,
                              scale=np.array([1.0, 2.0, 4.0]),
                              output_format="asfreqfilter")
        assert isinstance(H, list)
        assert len(H) == 3
        for g in H:
            assert "H" in g
            assert "foff" in g
            assert callable(g["H"])
            assert callable(g["foff"])

    def test_peak_at_expected_frequency(self):
        L = 4096
        H, info = freqwavelet(["cauchy", 300], L, scale=1.0,
                               output_format="full")
        peak_idx = np.argmax(np.abs(H.ravel()))
        expected_bin = round(float(np.asarray(info["fc"]).ravel()[0]) * L / 2)
        assert abs(peak_idx - expected_bin) < 3


# ---------------------------------------------------------------------------
# 8. Wavelet generator functions
# ---------------------------------------------------------------------------

class TestWaveletGenerator:

    def test_cauchy_generator(self):
        fun, fsupp, peakpos, ca = wavelet_generator_func(["cauchy", 300])
        assert peakpos > 0
        assert ca == 300
        # Evaluate at peak
        val = fun(np.array([peakpos]))
        assert abs(val) > 0.9  # Should be near 1 at peak

    def test_morlet_generator(self):
        fun, fsupp, peakpos, _ = wavelet_generator_func(["morlet", 6])
        assert peakpos > 0
        val = fun(np.array([peakpos]))
        assert abs(float(np.real(val).item()) - 1.0) < 0.01

    @pytest.mark.parametrize("wtype", WAVELET_TYPES)
    def test_all_types_run(self, wtype):
        if wtype in ("cauchy", "morse"):
            name = [wtype, 100]
        elif wtype == "morlet":
            name = [wtype, 4]
        else:
            name = [wtype, 3, 2]
        fun, fsupp, peakpos, _ = wavelet_generator_func(name)
        assert peakpos > 0
        assert len(fsupp) == 5

    def test_support_ordering(self):
        """fsupp should be monotonically increasing."""
        fun, fsupp, _, _ = wavelet_generator_func(["cauchy", 300])
        assert fsupp[0] <= fsupp[1] <= fsupp[2] <= fsupp[3] <= fsupp[4]


# ---------------------------------------------------------------------------
# 9. Lambert W function
# ---------------------------------------------------------------------------

class TestLambertW:

    def test_principal_branch(self):
        w = lambertw(1.0, b=0)
        # W(1) * exp(W(1)) should equal 1
        assert abs(float(np.real(w).item()) * np.exp(float(np.real(w).item())) - 1.0) < 1e-10

    def test_branch_minus1(self):
        w = lambertw(-0.1, b=-1)
        # W(-0.1) * exp(W(-0.1)) should equal -0.1
        val = float(np.real(w).item()) * np.exp(float(np.real(w).item()))
        assert abs(val - (-0.1)) < 1e-10

    def test_known_value(self):
        # W(0) = 0
        w = lambertw(0.0, b=0)
        assert abs(float(np.real(w).item())) < 1e-10


# ---------------------------------------------------------------------------
# 10. firwin_eval
# ---------------------------------------------------------------------------

class TestFirwinEval:

    def test_hann_at_zero(self):
        val = firwin_eval("hann", np.array([0.0]))
        assert abs(val[0] - 1.0) < 1e-10

    def test_hann_outside_support(self):
        """firwin_eval should return 0 for x outside [0, 1) (WPE convention)."""
        val = firwin_eval("hann", np.array([1.1, -0.1]))
        assert val[0] == 0.0
        assert val[1] == 0.0

    @pytest.mark.parametrize("winname", [
        "hann", "hamming", "blackman", "rect", "tria", "nuttall",
    ])
    def test_window_at_center(self, winname):
        val = firwin_eval(winname, np.array([0.0]))
        assert val[0] > 0

    def test_consistency_with_firwin(self):
        """firwin_eval at WPE positions n/M should match firwin output."""
        from cool_frames.numpy.filters._firwin import firwin
        M = 64
        x = np.arange(M) / M       # WPE positions: [0, 1/M, …, (M-1)/M]
        g_eval = firwin_eval("hann", x)
        g_int = firwin("hann", M, norm="inf")
        np.testing.assert_allclose(g_eval, g_int, atol=1e-10)
