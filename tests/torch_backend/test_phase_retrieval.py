"""
Phase 4 – Differentiable phase retrieval tests.

Validates:
  - diff_admm / diff_raar / diff_dm: ADMM-family phase retrieval
  - gla: Griffin-Lim algorithm (standard and fast variants)
  - magnitudeerr / magnitudeerrdb: spectral convergence metrics
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
# helpers
# ---------------------------------------------------------------------------


def _make_magnitude_input(fb, signal_np):
    """Compute magnitude spectrogram using NumPy backend."""
    from cool_frames.numpy.filterbanks import filterbank

    c_np = filterbank(signal_np, fb["g"], fb["a"], L=fb["L"])
    s_np = [np.abs(cm) for cm in c_np]
    N_frames = [cm.shape[0] for cm in c_np]
    return s_np, N_frames, c_np


def _flatten_magnitudes(s_list):
    """Flatten per-channel magnitudes into a single vector."""
    return np.concatenate([s.ravel() for s in s_list])


# ---------------------------------------------------------------------------
# Griffin-Lim algorithm
# ---------------------------------------------------------------------------


class TestGLA:
    """gla: Griffin-Lim phase retrieval."""

    def test_output_structure(self, erb_filterbank, noise_signal):
        from cool_frames.torch.phase import gla

        fb = erb_filterbank
        s_np, _, _ = _make_magnitude_input(fb, noise_signal)
        s_t = [np_to_torch(sm, dtype=torch.float64) for sm in s_np]

        c, f, relres, niter = gla(
            s_t,
            fb["g"],
            fb["a"],
            L=fb["L"],
            Ls=fb["Ls"],
            real=True,
            maxit=5,
        )

        assert isinstance(c, list)
        assert len(c) == fb["M"]
        assert isinstance(f, torch.Tensor)
        assert f.shape[0] == fb["Ls"]
        assert isinstance(relres, (torch.Tensor, np.ndarray))
        assert len(relres) <= 5
        assert isinstance(niter, int)

    def test_residual_decreasing(self, erb_filterbank, noise_signal):
        """Spectral convergence error should not increase."""
        from cool_frames.torch.phase import gla

        fb = erb_filterbank
        s_np, _, _ = _make_magnitude_input(fb, noise_signal)
        s_t = [np_to_torch(sm, dtype=torch.float64) for sm in s_np]

        _, _, relres, _ = gla(
            s_t,
            fb["g"],
            fb["a"],
            L=fb["L"],
            Ls=fb["Ls"],
            real=True,
            maxit=20,
        )

        rr = np.asarray(relres)
        # Allow tiny numerical increase
        assert np.all(np.diff(rr) < 1e-8), f"residual increased: {rr}"

    def test_more_iterations_better(self, erb_filterbank, noise_signal):
        """More iterations should give equal or better convergence."""
        from cool_frames.torch.phase import gla

        fb = erb_filterbank
        s_np, _, _ = _make_magnitude_input(fb, noise_signal)
        s_t = [np_to_torch(sm, dtype=torch.float64) for sm in s_np]

        _, _, rr5, _ = gla(s_t, fb["g"], fb["a"], L=fb["L"], real=True, maxit=5)
        _, _, rr20, _ = gla(s_t, fb["g"], fb["a"], L=fb["L"], real=True, maxit=20)

        assert np.asarray(rr20)[-1] <= np.asarray(rr5)[-1] + 1e-8

    def test_magnitude_preservation(self, erb_filterbank, noise_signal):
        """GLA output magnitudes should be close to input magnitudes."""
        from cool_frames.torch.phase import gla, magnitudeerr

        fb = erb_filterbank
        s_np, _, _ = _make_magnitude_input(fb, noise_signal)
        s_t = [np_to_torch(sm, dtype=torch.float64) for sm in s_np]

        c, _, _, _ = gla(s_t, fb["g"], fb["a"], L=fb["L"], real=True, maxit=50)

        err = magnitudeerr(s_t, c)
        assert err < 0.5, f"magnitude error {err:.4f} too large after 50 iterations"

    def test_agreement_with_numpy(self, erb_filterbank, noise_signal):
        """Torch GLA produces similar results to NumPy GLA."""
        from cool_frames.numpy.phase import gla as np_gla
        from cool_frames.torch.phase import gla as torch_gla

        fb = erb_filterbank
        s_np, _, _ = _make_magnitude_input(fb, noise_signal)
        s_t = [np_to_torch(sm, dtype=torch.float64) for sm in s_np]

        _, _, rr_np, _ = np_gla(
            s_np, fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"], real=True, maxit=10
        )
        _, _, rr_t, _ = torch_gla(
            s_t, fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"], real=True, maxit=10
        )

        # Final residuals should be comparable (not identical due to
        # floating-point differences in FFT implementations)
        np.testing.assert_allclose(
            np.asarray(rr_t)[-1],
            np.asarray(rr_np)[-1],
            rtol=0.1,  # within 10%
            err_msg="GLA residual diverged from NumPy",
        )

    @pytest.mark.parametrize("method", ["gla", "fgla"])
    def test_methods(self, erb_filterbank, noise_signal, method):
        """Both standard and fast GLA methods work."""
        from cool_frames.torch.phase import gla

        fb = erb_filterbank
        s_np, _, _ = _make_magnitude_input(fb, noise_signal)
        s_t = [np_to_torch(sm, dtype=torch.float64) for sm in s_np]

        c, f, _relres, _niter = gla(
            s_t,
            fb["g"],
            fb["a"],
            L=fb["L"],
            Ls=fb["Ls"],
            real=True,
            maxit=5,
            method=method,
        )
        assert len(c) == fb["M"]
        assert torch.all(torch.isfinite(f))


# ---------------------------------------------------------------------------
# ADMM-family phase retrieval
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="diff_admm moved to audioeffects in the 2026-06 consolidation")
class TestDiffADMM:
    """diff_admm: differentiable ADMM phase retrieval."""

    def test_output_structure(self, erb_filterbank, noise_signal):
        from cool_frames.torch.phase import diff_admm

        fb = erb_filterbank
        s_np, N_frames, _ = _make_magnitude_input(fb, noise_signal)
        mag_flat = np_to_torch(
            _flatten_magnitudes(s_np),
            dtype=torch.float64,
        )

        phase, mag_out = diff_admm(
            mag_flat,
            fb["g"],
            fb["a"],
            N_frames,
            L=fb["L"],
            real=True,
            maxit=5,
        )

        assert phase.shape == mag_flat.shape
        assert mag_out.shape == mag_flat.shape
        assert torch.all(torch.isfinite(phase))
        assert torch.all(torch.isfinite(mag_out))

    def test_agreement_with_numpy(self, erb_filterbank, noise_signal):
        from cool_frames.numpy.phase import diff_admm as np_fn
        from cool_frames.torch.phase import diff_admm as torch_fn

        fb = erb_filterbank
        s_np, N_frames, _ = _make_magnitude_input(fb, noise_signal)
        mag_flat_np = _flatten_magnitudes(s_np)

        _phase_np, mag_np = np_fn(
            mag_flat_np,
            fb["g"],
            fb["a"],
            N_frames,
            L=fb["L"],
            real=True,
            maxit=5,
        )

        mag_flat_t = np_to_torch(mag_flat_np, dtype=torch.float64)
        _phase_t, mag_t = torch_fn(
            mag_flat_t,
            fb["g"],
            fb["a"],
            N_frames,
            L=fb["L"],
            real=True,
            maxit=5,
        )

        np.testing.assert_allclose(
            torch_to_np(mag_t),
            mag_np,
            rtol=1e-6,
            atol=1e-8,
            err_msg="ADMM magnitude output mismatch",
        )


@pytest.mark.skip(reason="diff_raar moved to audioeffects in the 2026-06 consolidation")
class TestDiffRAAR:
    """diff_raar: RAAR variant."""

    def test_output_structure(self, erb_filterbank, noise_signal):
        from cool_frames.torch.phase import diff_raar

        fb = erb_filterbank
        s_np, N_frames, _ = _make_magnitude_input(fb, noise_signal)
        mag_flat = np_to_torch(
            _flatten_magnitudes(s_np),
            dtype=torch.float64,
        )

        phase, _mag_out = diff_raar(
            mag_flat,
            fb["g"],
            fb["a"],
            N_frames,
            L=fb["L"],
            real=True,
            maxit=5,
        )

        assert phase.shape == mag_flat.shape
        assert torch.all(torch.isfinite(phase))


@pytest.mark.skip(reason="diff_dm moved to audioeffects in the 2026-06 consolidation")
class TestDiffDM:
    """diff_dm: Difference Map variant."""

    def test_output_structure(self, erb_filterbank, noise_signal):
        from cool_frames.torch.phase import diff_dm

        fb = erb_filterbank
        s_np, N_frames, _ = _make_magnitude_input(fb, noise_signal)
        mag_flat = np_to_torch(
            _flatten_magnitudes(s_np),
            dtype=torch.float64,
        )

        phase, _mag_out = diff_dm(
            mag_flat,
            fb["g"],
            fb["a"],
            N_frames,
            L=fb["L"],
            real=True,
            maxit=5,
        )

        assert phase.shape == mag_flat.shape
        assert torch.all(torch.isfinite(phase))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMagnitudeErr:
    """magnitudeerr / magnitudeerrdb: spectral convergence metrics."""

    def test_zero_error_for_identical(self):
        from cool_frames.torch.phase import magnitudeerr

        a = [torch.randn(10, dtype=torch.float64) for _ in range(3)]
        err = magnitudeerr(a, a)
        assert err < 1e-14

    def test_positive_for_different(self):
        from cool_frames.torch.phase import magnitudeerr

        a = [torch.randn(10, dtype=torch.float64) for _ in range(3)]
        b = [torch.randn(10, dtype=torch.float64) for _ in range(3)]
        err = magnitudeerr(a, b)
        assert err > 0

    def test_db_relation(self):
        from cool_frames.torch.phase import magnitudeerr, magnitudeerrdb

        a = [torch.randn(10, dtype=torch.float64) for _ in range(3)]
        b = [torch.randn(10, dtype=torch.float64) for _ in range(3)]
        err_lin = magnitudeerr(a, b)
        err_db = magnitudeerrdb(a, b)

        expected_db = 20 * np.log10(err_lin)
        assert abs(err_db - expected_db) < 1e-10

    def test_agreement_with_numpy(self, erb_filterbank, noise_signal):
        from cool_frames.numpy.filterbanks import filterbank
        from cool_frames.numpy.phase import magnitudeerr as np_fn
        from cool_frames.torch.phase import magnitudeerr as torch_fn

        fb = erb_filterbank
        c1_np = filterbank(noise_signal, fb["g"], fb["a"], L=fb["L"])

        rng = np.random.default_rng(500)
        noise = rng.standard_normal(fb["Ls"]) * 0.01
        c2_np = filterbank(noise_signal + noise, fb["g"], fb["a"], L=fb["L"])

        err_np = np_fn(c1_np, c2_np)

        c1_t = [np_to_torch(cm, dtype=torch.complex128) for cm in c1_np]
        c2_t = [np_to_torch(cm, dtype=torch.complex128) for cm in c2_np]
        err_t = torch_fn(c1_t, c2_t)

        assert abs(err_t - err_np) < 1e-10
