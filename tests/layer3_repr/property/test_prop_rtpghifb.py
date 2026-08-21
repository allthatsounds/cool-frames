"""
test_prop_rtpghifb.py
=====================
Python port of:
    layer3_repr/property/PropRtpghifbConsistency.m

Properties verified
-------------------
Magnitude invariance
    (1)  |c(m, n)| == s(m, n) for 10 independent random signals.
    (2)  Magnitude invariance holds for the 'causal' variant (10 signals).
    (3)  Magnitude invariance holds for a range of standard test signals.

Phase consistency
    (4)  c == s * exp(1j * newphase) exactly (rel_err < 1e-10) for 5 signals.
    (5)  newphase is real for 10 signals.
    (6)  tgrad and fgrad are real for 10 signals.

Output structure
    (7)  All four outputs have shape (M, N) for 5 signals.
    (8)  c, newphase, tgrad, fgrad are finite for 10 signals.

Tolerance sensitivity
    (9)  Magnitude invariance holds across tol ∈ {1e-10, 1e-8, 1e-6, 1e-4, 1e-2}.

Gradient scaling
   (10)  tgrad is not identically zero for a non-trivial signal.
   (11)  fgrad is not identically zero for a non-trivial signal.
   (12)  RMS of tgrad is within 100× of a * π.

Notes
-----
- All tests carry ``@pytest.mark.requires_impl`` and skip until
  ``cool_frames.layer3.rtpghifb`` is available.
- We use 32 wavelet channels at Ls=512 for a fast but representative bank.
- For the uniform-hop assumption the ``a`` from ``waveletfilters`` is a
  constant integer; the magnitude matrix is (M × N) for a common N.
"""

from __future__ import annotations

import pytest

# rtpghifb (real-time PGHI filterbank) was moved out of cool_frames.phase to the
# audioeffects package in the 2026-06 consolidation; these tests belong there.
pytest.skip(
    "rtpghifb moved to audioeffects; relocate this test there",
    allow_module_level=True,
)

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LS     = 512
_SCALES = 2.0 ** np.linspace(5, -2, 32)


def _build_fb_from_signal(f: np.ndarray):
    """Return (s, a_scal, fc_vec, tfr) for signal f."""
    from cool_frames.numpy.filterbanks import filterbank  # type: ignore
    from cool_frames.numpy.filters import waveletfilters  # type: ignore

    g, a_arr, _, _, info = waveletfilters(2.0, _LS, scales=_SCALES)
    M      = len(g)
    a_scal = int(np.asarray(a_arr).ravel()[0])
    fc_vec = np.asarray(info["fc"])
    tfr    = np.asarray(info["tfr"])

    corig  = filterbank(f, g, a_arr)
    N_lens = [len(np.asarray(corig[m])) for m in range(M)]
    N      = N_lens[0]
    s = np.abs(np.vstack([
        np.asarray(corig[m]).ravel()[:N] for m in range(M)
    ]))
    return s, a_scal, fc_vec, tfr


def _random_signal(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(_LS)


def _sine_signal(freq_hz: float, fs: float = 8000.0) -> np.ndarray:
    t = np.arange(_LS) / fs
    return np.sin(2 * np.pi * freq_hz * t)


# ---------------------------------------------------------------------------
# (1–3) Magnitude invariance
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPropRtpghifbMagnitudeInvariantSignals:
    """
    PropRtpghifbConsistency: |c| == s for multiple random signals.
    Properties (1), (2), (3).
    """

    def test_magnitude_invariant_10_random_signals(self, needs_impl):
        """Property (1): |c| == s for 10 independent random signals."""
        from cool_frames.phase import rtpghifb  # type: ignore
        for trial in range(10):
            f = _random_signal(7 + trial)
            s, a, fc, tfr = _build_fb_from_signal(f)
            c, _, _, _ = rtpghifb(s, a, fc, tfr)
            mag_err = np.linalg.norm(np.abs(c.ravel()) - s.ravel()) / \
                      (np.linalg.norm(s.ravel()) + np.finfo(float).eps)
            assert mag_err < 1e-6, \
                f"Trial {trial}: rtpghifb magnitude invariance rel_err={mag_err:.2e}"

    def test_magnitude_invariant_causal_10_signals(self, needs_impl):
        """Property (2): causal variant satisfies |c| == s for 10 signals."""
        from cool_frames.phase import rtpghifb  # type: ignore
        for trial in range(10):
            f = _random_signal(11 + trial)
            s, a, fc, tfr = _build_fb_from_signal(f)
            c, _, _, _ = rtpghifb(s, a, fc, tfr, variant="causal")
            mag_err = np.linalg.norm(np.abs(c.ravel()) - s.ravel()) / \
                      (np.linalg.norm(s.ravel()) + np.finfo(float).eps)
            assert mag_err < 1e-6, \
                f"Trial {trial} (causal): magnitude invariance rel_err={mag_err:.2e}"

    def test_magnitude_invariant_standard_signals(self, needs_impl):
        """Property (3): magnitude invariance for sine and noise signals."""
        from cool_frames.phase import rtpghifb  # type: ignore
        standard_signals = {
            "noise":    _random_signal(42),
            "sine_440": _sine_signal(440.0),
            "sine_1k":  _sine_signal(1000.0),
        }
        for name, f in standard_signals.items():
            s, a, fc, tfr = _build_fb_from_signal(f)
            c, _, _, _ = rtpghifb(s, a, fc, tfr)
            mag_err = np.linalg.norm(np.abs(c.ravel()) - s.ravel()) / \
                      (np.linalg.norm(s.ravel()) + np.finfo(float).eps)
            assert mag_err < 1e-6, \
                f'Signal "{name}": magnitude invariance rel_err={mag_err:.2e}'


# ---------------------------------------------------------------------------
# (4–6) Phase consistency
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPropRtpghifbPhaseConsistency:
    """
    PropRtpghifbConsistency: c == s * exp(1j * newphase), real outputs.
    Properties (4), (5), (6).
    """

    def test_c_equals_magnitude_times_exp_phase_5_signals(self, needs_impl):
        """Property (4): c == s * exp(1j * newphase) for 5 signals (rtol < 1e-10)."""
        from cool_frames.phase import rtpghifb  # type: ignore
        for trial in range(5):
            f = _random_signal(13 + trial)
            s, a, fc, tfr = _build_fb_from_signal(f)
            c, newphase, _, _ = rtpghifb(s, a, fc, tfr)
            c_expected = s * np.exp(1j * newphase)
            rel_err = np.linalg.norm((c - c_expected).ravel()) / \
                      (np.linalg.norm(c_expected.ravel()) + np.finfo(float).eps)
            assert rel_err < 1e-10, \
                f"Trial {trial}: c must equal s*exp(1j*newphase), rel_err={rel_err:.2e}"

    def test_newphase_is_real_10_signals(self, needs_impl):
        """Property (5): newphase is real for 10 different signals."""
        from cool_frames.phase import rtpghifb  # type: ignore
        for trial in range(10):
            f = _random_signal(17 + trial)
            s, a, fc, tfr = _build_fb_from_signal(f)
            _, newphase, _, _ = rtpghifb(s, a, fc, tfr)
            assert np.isrealobj(np.asarray(newphase)), \
                f"Trial {trial}: newphase must be real"

    def test_gradients_real_10_signals(self, needs_impl):
        """Property (6): tgrad and fgrad are real for 10 different signals."""
        from cool_frames.phase import rtpghifb  # type: ignore
        for trial in range(10):
            f = _random_signal(19 + trial)
            s, a, fc, tfr = _build_fb_from_signal(f)
            _, _, tgrad, fgrad = rtpghifb(s, a, fc, tfr)
            assert np.isrealobj(np.asarray(tgrad)), \
                f"Trial {trial}: tgrad must be real"
            assert np.isrealobj(np.asarray(fgrad)), \
                f"Trial {trial}: fgrad must be real"


# ---------------------------------------------------------------------------
# (7–8) Output structure
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPropRtpghifbOutputStructure:
    """
    PropRtpghifbConsistency: correct shape and finiteness across signals.
    Properties (7), (8).
    """

    def test_output_size_consistent_5_signals(self, needs_impl):
        """Property (7): all outputs have shape (M, N) for 5 signals."""
        from cool_frames.phase import rtpghifb  # type: ignore
        for trial in range(5):
            f = _random_signal(23 + trial)
            s, a, fc, tfr = _build_fb_from_signal(f)
            c, newphase, tgrad, fgrad = rtpghifb(s, a, fc, tfr)
            for name, arr in [("c", c), ("newphase", newphase),
                               ("tgrad", tgrad), ("fgrad", fgrad)]:
                assert np.asarray(arr).shape == s.shape, \
                    f"Trial {trial}: {name} shape {np.asarray(arr).shape} != {s.shape}"

    def test_output_finite_10_signals(self, needs_impl):
        """Property (8): c, newphase, tgrad, fgrad are finite for 10 signals."""
        from cool_frames.phase import rtpghifb  # type: ignore
        for trial in range(10):
            f = _random_signal(29 + trial)
            s, a, fc, tfr = _build_fb_from_signal(f)
            c, newphase, tgrad, fgrad = rtpghifb(s, a, fc, tfr)
            for name, arr in [("c", c), ("newphase", newphase),
                               ("tgrad", tgrad), ("fgrad", fgrad)]:
                assert np.all(np.isfinite(arr)), \
                    f"Trial {trial}: {name} must be finite everywhere"


# ---------------------------------------------------------------------------
# (9) Tolerance sensitivity
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPropRtpghifbToleranceSensitivity:
    """
    PropRtpghifbConsistency: magnitude invariance across tol range.
    Property (9).
    """

    def test_magnitude_invariant_across_tolerances(self, needs_impl):
        """Magnitude invariance |c| == s holds for tol ∈ {1e-10, …, 1e-2}."""
        from cool_frames.phase import rtpghifb  # type: ignore
        f = _random_signal(42)
        s, a, fc, tfr = _build_fb_from_signal(f)
        for tol_val in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
            c, _, _, _ = rtpghifb(s, a, fc, tfr, tol=tol_val)
            mag_err = np.linalg.norm(np.abs(c.ravel()) - s.ravel()) / \
                      (np.linalg.norm(s.ravel()) + np.finfo(float).eps)
            assert mag_err < 1e-6, \
                f"tol={tol_val}: magnitude invariance violated (rel_err={mag_err:.2e})"


# ---------------------------------------------------------------------------
# (10–12) Gradient scaling
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPropRtpghifbGradientScaling:
    """
    PropRtpghifbConsistency: gradients are non-zero and correctly scaled.
    Properties (10), (11), (12).
    """

    def test_tgrad_not_identically_zero(self, needs_impl):
        """Property (10): tgrad is non-zero for a non-trivial signal."""
        from cool_frames.phase import rtpghifb  # type: ignore
        f = _sine_signal(440.0)
        s, a, fc, tfr = _build_fb_from_signal(f)
        _, _, tgrad, _ = rtpghifb(s, a, fc, tfr)
        assert np.linalg.norm(tgrad.ravel()) > 0, \
            "rtpghifb: tgrad must not be identically zero for a sine signal"

    def test_fgrad_not_identically_zero(self, needs_impl):
        """Property (11): fgrad is non-zero for a non-trivial signal."""
        from cool_frames.phase import rtpghifb  # type: ignore
        f = _random_signal(42)
        s, a, fc, tfr = _build_fb_from_signal(f)
        _, _, _, fgrad = rtpghifb(s, a, fc, tfr)
        assert np.linalg.norm(fgrad.ravel()) > 0, \
            "rtpghifb: fgrad must not be identically zero for a noise signal"

    def test_tgrad_rms_consistent_with_hop_size(self, needs_impl):
        """Property (12): RMS of tgrad is within 100× of a * π."""
        from cool_frames.phase import rtpghifb  # type: ignore
        f = _random_signal(42)
        s, a, fc, tfr = _build_fb_from_signal(f)
        _, _, tgrad, _ = rtpghifb(s, a, fc, tfr)
        rms_tgrad     = float(np.sqrt(np.mean(tgrad.ravel() ** 2)))
        expected_scale = a * np.pi
        assert rms_tgrad > expected_scale / 100, \
            f"rtpghifb: tgrad RMS {rms_tgrad:.3e} is unexpectedly small " \
            f"(expected ~ {expected_scale:.3e})"
        assert rms_tgrad < expected_scale * 100, \
            f"rtpghifb: tgrad RMS {rms_tgrad:.3e} is unexpectedly large " \
            f"(expected ~ {expected_scale:.3e})"
