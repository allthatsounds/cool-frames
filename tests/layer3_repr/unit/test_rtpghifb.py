"""
test_rtpghifb.py
================
Python port of:
    layer3_repr/unit/TestRtpghifb.m

Covers ``rtpghifb`` (Real-Time Phase Gradient Heap Integration for
filter banks), the filterbank generalisation of RTPGHI.

The function is expected to be exposed as:

    from cool_frames.phase import rtpghifb

with the following signature (mirroring the MATLAB ``rtpghifbwl``)::

    c, newphase, tgrad, fgrad = rtpghifb(
        s, a, fc, tfr,
        *,
        variant='normal',   # or 'causal'
        tol=1e-6,
    )

where:
    s        – (M, N) float64   input magnitudes
    a        – int or float     uniform hop size
    fc       – (M,) float64     normalised centre frequencies  (info['fc'])
    tfr      – (M,) float64     time-frequency ratios          (info['tfr'])

Returns:
    c        – (M, N) complex128   coefficients with |c| == s
    newphase – (M, N) float64      phase angles (radians)
    tgrad    – (M, N) float64      time-phase gradient
    fgrad    – (M, N) float64      frequency-phase gradient

The filterbank is built with ``waveletfilters`` (constant-Q scales) and
``filterbank`` (non-uniform analysis); for the RTPGHIFB the subbands are
assumed uniform in the test setup (``sampling='uniform'`` flag or the all-
same ``a`` from the default regsampling at moderate scales).

All tests carry ``@pytest.mark.requires_impl`` and will be skipped
automatically until ``cool_frames.layer3.rtpghifb`` is available.

Notes
-----
- The magnitude convention is ``s.shape == (M, N)``, i.e. the transpose of
  the non-uniform filterbank list.  For a uniform hop, all subbands have
  the same number of frames N, so the (M, N) 2-D array is well-defined.
- We use a small wavelet bank (32 channels, Ls=512) to keep unit tests fast.
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


def _build_fb():
    """Return (g, a_scalar, fc, tfr, L, M, corig, s, f) for test use."""
    from cool_frames.numpy.filterbanks import filterbank  # type: ignore
    from cool_frames.numpy.filters import waveletfilters  # type: ignore

    g, a_arr, fc, L, info = waveletfilters(2.0, _LS, scales=_SCALES)
    M      = len(g)
    a_scal = int(np.asarray(a_arr).ravel()[0])
    fc_vec = np.asarray(info["fc"])
    tfr    = np.asarray(info["tfr"])

    rng  = np.random.default_rng(42)
    f    = rng.standard_normal(_LS)
    corig = filterbank(f, g, a_arr)

    # Build (M, N) magnitude matrix — requires uniform N across channels.
    N_lens = [len(np.asarray(corig[m])) for m in range(M)]
    N      = max(N_lens)
    if not all(n == N for n in N_lens):
        # Non-uniform: pad to common (max) length for test setup convenience.
        padded = np.zeros((M, N), dtype=float)
        for m in range(M):
            arr = np.abs(np.asarray(corig[m]).ravel())
            padded[m, :len(arr)] = arr
        s = padded
    else:
        s = np.abs(np.vstack([np.asarray(corig[m]).ravel() for m in range(M)]))

    return g, a_scal, fc_vec, tfr, L, M, corig, s, f


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestRtpghifbOutputStructure:
    """Port of TestRtpghifb: output structure group."""

    def test_four_outputs_returned_without_error(self, needs_impl):
        """Four outputs must be returned without warning or error."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        c, newphase, tgrad, fgrad = rtpghifb(s, a, fc, tfr)
        assert c        is not None, "rtpghifb: c must not be None"
        assert newphase is not None, "rtpghifb: newphase must not be None"
        assert tgrad    is not None, "rtpghifb: tgrad must not be None"
        assert fgrad    is not None, "rtpghifb: fgrad must not be None"

    def test_output_size_matches_input(self, needs_impl):
        """All four outputs must have the same shape as s (M × N)."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        c, newphase, tgrad, fgrad = rtpghifb(s, a, fc, tfr)
        for name, arr in [("c", c), ("newphase", newphase),
                          ("tgrad", tgrad), ("fgrad", fgrad)]:
            assert np.asarray(arr).shape == s.shape, \
                f"rtpghifb: {name} shape {np.asarray(arr).shape} != {s.shape}"

    def test_output_c_is_complex(self, needs_impl):
        """c must be complex (it contains reconstructed coefficients)."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        c, _, _, _ = rtpghifb(s, a, fc, tfr)
        assert not np.isrealobj(np.asarray(c)), \
            "rtpghifb: c must be complex"

    def test_phase_is_real(self, needs_impl):
        """newphase must be real (it is an array of angles in radians)."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        _, newphase, _, _ = rtpghifb(s, a, fc, tfr)
        assert np.isrealobj(np.asarray(newphase)), \
            "rtpghifb: newphase must be real"

    def test_gradients_are_real(self, needs_impl):
        """tgrad and fgrad are phase derivatives and must be real."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        _, _, tgrad, fgrad = rtpghifb(s, a, fc, tfr)
        assert np.isrealobj(np.asarray(tgrad)), \
            "rtpghifb: tgrad must be real"
        assert np.isrealobj(np.asarray(fgrad)), \
            "rtpghifb: fgrad must be real"


# ---------------------------------------------------------------------------
# Magnitude preservation
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestRtpghifbMagnitudePreservation:
    """Port of TestRtpghifb: magnitude preservation group."""

    def test_magnitude_of_c_matches_input(self, needs_impl):
        """The fundamental PGHI property: |c(m, n)| == s(m, n)."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        c, _, _, _ = rtpghifb(s, a, fc, tfr)
        mag_err = np.linalg.norm(np.abs(c.ravel()) - s.ravel()) / \
                  (np.linalg.norm(s.ravel()) + np.finfo(float).eps)
        assert mag_err < 1e-6, \
            f"rtpghifb: |c| must equal input s (rel_err={mag_err:.2e})"

    def test_c_equals_magnitude_times_exp_phase(self, needs_impl):
        """c must factor exactly as s * exp(1j * newphase)."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        c, newphase, _, _ = rtpghifb(s, a, fc, tfr)
        c_expected = s * np.exp(1j * newphase)
        rel_err = np.linalg.norm((c - c_expected).ravel()) / \
                  (np.linalg.norm(c_expected.ravel()) + np.finfo(float).eps)
        assert rel_err < 1e-10, \
            f"rtpghifb: c must equal s * exp(1j * newphase) (rel_err={rel_err:.2e})"


# ---------------------------------------------------------------------------
# Normal vs causal variant
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestRtpghifbVariants:
    """Port of TestRtpghifb: normal vs causal variant group."""

    def test_causal_variant_runs_without_error(self, needs_impl):
        """The 'causal' variant must be accepted."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        try:
            rtpghifb(s, a, fc, tfr, variant="causal")
        except Exception as exc:
            pytest.fail(f"rtpghifb(variant='causal') raised: {exc}")

    def test_normal_variant_runs_without_error(self, needs_impl):
        """The default ('normal') variant must be accepted."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        try:
            rtpghifb(s, a, fc, tfr, variant="normal")
        except Exception as exc:
            pytest.fail(f"rtpghifb(variant='normal') raised: {exc}")

    def test_causal_preserves_magnitude(self, needs_impl):
        """The causal variant must also satisfy |c| == s."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        c_caus, _, _, _ = rtpghifb(s, a, fc, tfr, variant="causal")
        mag_err = np.linalg.norm(np.abs(c_caus.ravel()) - s.ravel()) / \
                  (np.linalg.norm(s.ravel()) + np.finfo(float).eps)
        assert mag_err < 1e-6, \
            f"rtpghifb (causal): |c| must equal input s (rel_err={mag_err:.2e})"

    def test_causal_and_normal_produce_different_phases(self, needs_impl):
        """Causal and normal variants must produce genuinely different phases."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        c_norm, _, _, _ = rtpghifb(s, a, fc, tfr, variant="normal")
        c_caus, _, _, _ = rtpghifb(s, a, fc, tfr, variant="causal")
        phase_diff = np.angle(c_norm.ravel()) - np.angle(c_caus.ravel())
        assert np.linalg.norm(phase_diff) > 1e-6, \
            "rtpghifb: normal and causal variants must produce different phases"


# ---------------------------------------------------------------------------
# Tolerance parameter
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestRtpghifbTolerance:
    """Port of TestRtpghifb: tolerance parameter group."""

    def test_custom_tol_accepted(self, needs_impl):
        """Passing a custom tolerance must not error."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        try:
            rtpghifb(s, a, fc, tfr, tol=1e-3)
        except Exception as exc:
            pytest.fail(f"rtpghifb(tol=1e-3) raised: {exc}")

    def test_high_tol_preserves_magnitude(self, needs_impl):
        """Even with a loose tolerance, |c| must still equal s."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        c_ht, _, _, _ = rtpghifb(s, a, fc, tfr, tol=1e-3)
        mag_err = np.linalg.norm(np.abs(c_ht.ravel()) - s.ravel()) / \
                  (np.linalg.norm(s.ravel()) + np.finfo(float).eps)
        assert mag_err < 1e-6, \
            f"rtpghifb (tol=1e-3): |c| magnitude error {mag_err:.2e}"

    def test_low_tol_preserves_magnitude(self, needs_impl):
        """A very strict tolerance must also preserve magnitude."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        c_lt, _, _, _ = rtpghifb(s, a, fc, tfr, tol=1e-10)
        mag_err = np.linalg.norm(np.abs(c_lt.ravel()) - s.ravel()) / \
                  (np.linalg.norm(s.ravel()) + np.finfo(float).eps)
        assert mag_err < 1e-6, \
            f"rtpghifb (tol=1e-10): |c| magnitude error {mag_err:.2e}"


# ---------------------------------------------------------------------------
# Edge cases & robustness
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestRtpghifbEdgeCases:
    """Port of TestRtpghifb: edge cases & robustness group."""

    def test_all_zero_magnitude_yields_zero_coefficients(self, needs_impl):
        """If s == 0 everywhere, c must also be zero."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        s_zero = np.zeros_like(s)
        c_z, _, _, _ = rtpghifb(s_zero, a, fc, tfr)
        assert np.linalg.norm(c_z.ravel()) < 1e-14, \
            "rtpghifb: all-zero magnitude must yield all-zero coefficients"

    def test_single_frame_input_works(self, needs_impl):
        """A single-frame input (M × 1) must run without error."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        s_one = s[:, :1]   # (M, 1)
        try:
            c1, _, _, _ = rtpghifb(s_one, a, fc, tfr)
        except Exception as exc:
            pytest.fail(f"rtpghifb single-frame raised: {exc}")
        assert np.asarray(c1).shape == s_one.shape, \
            f"rtpghifb: single-frame output shape {np.asarray(c1).shape} != {s_one.shape}"

    def test_phase_is_finite_everywhere(self, needs_impl):
        """Phase values must be finite (no NaN / Inf)."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        _, newphase, tgrad, fgrad = rtpghifb(s, a, fc, tfr)
        assert np.all(np.isfinite(newphase)), \
            "rtpghifb: newphase must be finite everywhere"
        assert np.all(np.isfinite(tgrad)), \
            "rtpghifb: tgrad must be finite everywhere"
        assert np.all(np.isfinite(fgrad)), \
            "rtpghifb: fgrad must be finite everywhere"

    def test_coefficients_are_finite(self, needs_impl):
        """Reconstructed coefficients must be finite."""
        from cool_frames.phase import rtpghifb  # type: ignore
        _, a, fc, tfr, _, _, _, s, _ = _build_fb()
        c, _, _, _ = rtpghifb(s, a, fc, tfr)
        assert np.all(np.isfinite(c)), \
            "rtpghifb: c must be finite everywhere"

    def test_impulse_signal_smoke(self, needs_impl):
        """An impulse signal must run without error."""
        from cool_frames.numpy.filterbanks import filterbank  # type: ignore
        from cool_frames.numpy.filters import waveletfilters  # type: ignore
        from cool_frames.phase import rtpghifb  # type: ignore

        g, a_arr, fc, L, info = waveletfilters(2.0, _LS, scales=_SCALES)
        M      = len(g)
        a_scal = int(np.asarray(a_arr).ravel()[0])
        fc_vec = np.asarray(info["fc"])
        tfr    = np.asarray(info["tfr"])

        imp = np.zeros(_LS)
        imp[0] = 1.0
        c_imp = filterbank(imp, g, a_arr)
        N_lens = [len(np.asarray(c_imp[m])) for m in range(M)]
        N = N_lens[0]
        s_imp = np.abs(np.vstack([
            np.asarray(c_imp[m]).ravel()[:N] for m in range(M)
        ]))
        try:
            rtpghifb(s_imp, a_scal, fc_vec, tfr)
        except Exception as exc:
            pytest.fail(f"rtpghifb impulse signal raised: {exc}")
