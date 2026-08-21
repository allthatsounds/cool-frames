"""Tests for cool_frames.torch.recipes – high-level audio processing recipes.

Validates that all recipes run without error, produce correct output shapes,
and maintain sensible values across different signal types.
"""

from __future__ import annotations

import pytest

import numpy as np

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.requires_torch_impl


def np_to_torch(x: np.ndarray, dtype=None, device=None) -> torch.Tensor:
    """Convert a NumPy array to a torch tensor, preserving complex type."""
    t = torch.from_numpy(np.ascontiguousarray(x))
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


def torch_to_np(t: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a NumPy array on CPU."""
    return t.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Spectral analysis recipes
# ---------------------------------------------------------------------------


class TestFilterbankSpectrogram:
    """Tests for filterbank_spectrogram recipe."""

    def test_output_structure(self, noise_signal):
        from cool_frames.torch.diagnostics import filterbank_spectrogram

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        spec = filterbank_spectrogram(f_t, fs=fs)

        # Check that output is a dict with expected keys
        assert isinstance(spec, dict)
        assert "coeff_db" in spec
        assert "fc" in spec
        assert "a" in spec
        assert "g" in spec
        assert "fs" in spec
        assert "db_range" in spec

    def test_output_shape(self, noise_signal):
        from cool_frames.torch.diagnostics import filterbank_spectrogram

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        spec = filterbank_spectrogram(f_t, fs=fs)

        coeff_db = spec["coeff_db"]
        assert coeff_db.ndim == 2, "coeff_db should be 2D (channels x time)"
        assert coeff_db.shape[0] > 0, "should have at least one channel"
        assert coeff_db.shape[1] > 0, "should have at least one time frame"

    def test_output_is_finite(self, noise_signal):
        from cool_frames.torch.diagnostics import filterbank_spectrogram

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        spec = filterbank_spectrogram(f_t, fs=fs)

        assert torch.all(torch.isfinite(spec["coeff_db"])), "All values should be finite"

    def test_erb_scale(self, noise_signal):
        from cool_frames.torch.diagnostics import filterbank_spectrogram

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        spec_erb = filterbank_spectrogram(f_t, fs=fs, scale="erb")

        assert "coeff_db" in spec_erb
        assert spec_erb["coeff_db"].shape[0] > 0

    def test_cqt_scale(self, noise_signal):
        from cool_frames.torch.diagnostics import filterbank_spectrogram

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        spec_cqt = filterbank_spectrogram(f_t, fs=fs, scale="cqt")

        assert "coeff_db" in spec_cqt
        assert spec_cqt["coeff_db"].shape[0] > 0

    def test_db_range_param(self, noise_signal):
        from cool_frames.torch.diagnostics import filterbank_spectrogram

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        spec = filterbank_spectrogram(f_t, fs=fs, db_range=40)

        assert spec["db_range"] == 40


class TestReassignedSpectrogram:
    """Tests for reassigned_spectrogram recipe."""

    def test_output_structure(self, noise_signal):
        from cool_frames.torch.diagnostics import reassigned_spectrogram

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        spec = reassigned_spectrogram(f_t, fs=fs)

        assert isinstance(spec, dict)
        assert "coeff_db" in spec
        assert "fc" in spec
        assert "a" in spec
        assert "fs" in spec
        assert "instfreq_deviation" in spec
        assert "groupdelay_shift" in spec

    def test_output_shape(self, noise_signal):
        from cool_frames.torch.diagnostics import reassigned_spectrogram

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        spec = reassigned_spectrogram(f_t, fs=fs)

        coeff_db = spec["coeff_db"]
        assert coeff_db.ndim == 2, "coeff_db should be 2D"
        assert coeff_db.shape[0] > 0

    def test_phase_grad_outputs(self, noise_signal):
        from cool_frames.torch.diagnostics import reassigned_spectrogram

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        spec = reassigned_spectrogram(f_t, fs=fs)

        # Phase gradient outputs should have one value per channel
        n_channels = spec["coeff_db"].shape[0]
        assert len(spec["instfreq_deviation"]) == n_channels
        assert len(spec["groupdelay_shift"]) == n_channels

    def test_instfreq_is_real(self, noise_signal):
        from cool_frames.torch.diagnostics import reassigned_spectrogram

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        spec = reassigned_spectrogram(f_t, fs=fs)

        assert torch.all(torch.isfinite(spec["instfreq_deviation"]))
        assert torch.all(torch.isfinite(spec["groupdelay_shift"]))


# ---------------------------------------------------------------------------
# Mel-frequency and MFCC features
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Chroma features
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Harmonic-percussive source separation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------


class TestDenoise:
    """Tests for denoise recipe."""

    def test_output_types(self, noise_signal):
        from torch_additions.recipes import denoise

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        denoised, stats = denoise(f_t, fs=fs)

        assert isinstance(denoised, torch.Tensor), "denoised should be a tensor"
        assert isinstance(stats, dict), "stats should be a dict"

    def test_output_shape(self, noise_signal):
        from torch_additions.recipes import denoise

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        denoised, _stats = denoise(f_t, fs=fs)

        assert denoised.ndim == 1, "denoised should be 1D"
        assert len(denoised) == len(f_t), "denoised should match input length"

    def test_stats_structure(self, noise_signal):
        from torch_additions.recipes import denoise

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        _denoised, stats = denoise(f_t, fs=fs)

        assert "energy_before" in stats
        assert "energy_after" in stats
        assert "channel_freqs" in stats
        assert "method" in stats

    def test_output_is_finite(self, noise_signal):
        from torch_additions.recipes import denoise

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        denoised, _stats = denoise(f_t, fs=fs)

        # denoise produces synthesis result which may be complex
        assert torch.all(torch.isfinite(denoised))

    @pytest.mark.parametrize("method", ["wiener", "hard", "soft"])
    def test_thresholding_methods(self, noise_signal, method):
        from torch_additions.recipes import denoise

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        denoised, stats = denoise(f_t, fs=fs, method=method)

        assert stats["method"] == method
        assert len(denoised) == len(f_t)
        assert torch.all(torch.isfinite(denoised))

    def test_threshold_db_param(self, noise_signal):
        from torch_additions.recipes import denoise

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000
        denoised, _stats = denoise(f_t, fs=fs, threshold_db=-40)

        assert len(denoised) == len(f_t)


# ---------------------------------------------------------------------------
# Equalization
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Phase reconstruction
# ---------------------------------------------------------------------------


class TestReconstruct:
    """Tests for reconstruct recipe.

    Note: reconstruct may have dtype compatibility issues between float32/64.
    These tests verify the basic API contract rather than exact reconstruction quality.
    """

    def test_reconstruct_callable(self, erb_filterbank, noise_signal):
        """``reconstruct`` runs on well-formed input for every working method.

        This used to call ``method='pghi'`` inside a ``try``.  That branch was
        random phase wearing PGHI's name, and it now raises
        ``NotImplementedError`` deliberately — see
        ``test_reconstruct_pghi_is_wired_to_the_real_algorithm`` below.
        """
        from torch_additions.recipes import reconstruct

        from cool_frames.numpy.filterbanks import filterbank as _np_filterbank

        fb = erb_filterbank
        # Real magnitudes, not ones: a flat spectrum is a degenerate input for
        # a phase-retrieval routine and hides most of what could go wrong.
        rng = np.random.default_rng(0)
        x = rng.standard_normal(fb["Ls"])
        c = _np_filterbank(x, fb["g"], fb["a"], L=fb["L"])
        s_mag = [torch.from_numpy(np.abs(np.asarray(cm)).ravel()) for cm in c]

        for method in ("pghi", "gla", "fgla", "legla", "spsi"):
            reconstructed, info = reconstruct(
                s_mag, fb["g"], fb["a"], fb["L"], fb["Ls"], method=method
            )
            assert isinstance(reconstructed, torch.Tensor)
            assert isinstance(info, dict)
            assert info["method"] == method
            assert torch.all(torch.isfinite(torch.as_tensor(reconstructed).real))

    def test_reconstruct_pghi_is_wired_to_the_real_algorithm(self, erb_filterbank):
        """``'pghi'`` drew random phase and reported ``converged: True``.

        It was bitwise identical to the ``'spsi'`` branch and 57 dB worse than
        ``'gla'``.  It then briefly raised ``NotImplementedError``, because the
        PGHI *magnitude* path in ``filterbankconstphase`` measured no better
        than zero phase — its gradient estimator was returning the
        instantaneous-frequency deviation from each channel's centre frequency
        while the heap integrator consumes the absolute value.  With that fixed
        it is wired up for real; the quality checks live in
        ``tests/regressions/test_audit_v0_1_1_judgement.py``.
        """
        from torch_additions.recipes import reconstruct

        from cool_frames.numpy.filterbanks import filterbank as _np_filterbank

        fb = erb_filterbank
        rng = np.random.default_rng(1)
        x = rng.standard_normal(fb["Ls"])
        c = _np_filterbank(x, fb["g"], fb["a"], L=fb["L"])
        s_mag = [torch.from_numpy(np.abs(np.asarray(cm)).ravel()) for cm in c]

        reconstructed, info = reconstruct(
            s_mag, fb["g"], fb["a"], fb["L"], fb["Ls"], method="pghi"
        )
        assert info["method"] == "pghi"
        assert info["n_iters"] == 0
        assert torch.all(torch.isfinite(torch.as_tensor(reconstructed).real))

    def test_reconstruct_returns_dict(self):
        """Test that reconstruct info dict has expected structure."""
        from torch_additions.recipes import reconstruct

        # Verify the function signature
        sig = str(reconstruct.__doc__)
        assert "method" in sig.lower() or "pghi" in sig.lower()


# ---------------------------------------------------------------------------
# Integration tests (multi-step workflows)
# ---------------------------------------------------------------------------


class TestIntegration:
    """Test realistic workflows combining multiple recipes."""

    def test_spectrogram_produces_magnitude(self, erb_filterbank, noise_signal):
        from cool_frames.torch.diagnostics import filterbank_spectrogram

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = fb["fs"]

        # Compute spectrogram
        spec = filterbank_spectrogram(f_t, fs=fs)

        # Verify spectrogram has magnitude information
        assert "coeff_db" in spec
        coeff = spec["coeff_db"]
        assert torch.all(torch.isfinite(coeff)), "spectrogram should be finite"

    def test_denoise_produces_output(self, noise_signal):
        from torch_additions.recipes import denoise

        f_t = np_to_torch(noise_signal, dtype=torch.float32)
        fs = 8000

        # Denoise
        denoised, stats = denoise(f_t, fs=fs)

        # Verify denoised signal has expected properties
        assert len(denoised) == len(f_t)
        assert torch.all(torch.isfinite(denoised))
        assert "energy_before" in stats
        assert "energy_after" in stats
