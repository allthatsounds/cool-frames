"""
Phase 1 – High-level filterbank analysis/synthesis tests.

Validates that ``cool_frames.torch.filterbanks.{filterbank, ifilterbank}`` agree
with the NumPy backend and satisfy mathematical invariants (perfect
reconstruction, linearity, energy bounds).
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
# Structural tests
# ---------------------------------------------------------------------------


class TestFilterbankStructure:
    """Basic shape and type checks for filterbank analysis."""

    def test_output_is_list_of_tensors(self, erb_filterbank, noise_signal):
        from cool_frames.torch.filterbanks import filterbank

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])

        assert isinstance(c, list)
        assert len(c) == fb["M"]
        for cm in c:
            assert isinstance(cm, torch.Tensor)

    def test_mono_output_shapes(self, erb_filterbank, noise_signal):
        from cool_frames.torch.filterbanks import filterbank

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])

        for m, cm in enumerate(c):
            assert cm.ndim == 1, f"channel {m}: expected 1-D, got {cm.ndim}-D"
            assert cm.shape[0] > 0

    def test_stereo_output_shapes(self, erb_filterbank, stereo_signal):
        from cool_frames.torch.filterbanks import filterbank

        fb = erb_filterbank
        f_t = np_to_torch(stereo_signal, dtype=torch.float64)
        c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])

        for m, cm in enumerate(c):
            assert cm.ndim == 2, f"channel {m}: expected 2-D, got {cm.ndim}-D"
            assert cm.shape[1] == 2, "expected W=2 for stereo"


# ---------------------------------------------------------------------------
# Agreement with NumPy backend
# ---------------------------------------------------------------------------


class TestFilterbankAgreement:
    """Torch filterbank matches NumPy output coefficient-by-coefficient."""

    def test_analysis_agreement(self, erb_filterbank, noise_signal, erb_analysis_ref):
        from cool_frames.torch.filterbanks import filterbank

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c_t = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])

        for m in range(fb["M"]):
            np.testing.assert_allclose(
                torch_to_np(c_t[m]),
                erb_analysis_ref[m],
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"analysis channel {m} mismatch",
            )

    def test_synthesis_agreement(self, erb_filterbank, erb_analysis_ref, erb_synthesis_ref):
        from cool_frames.torch.filterbanks import ifilterbank

        fb = erb_filterbank
        c_t = [np_to_torch(cm, dtype=torch.complex128) for cm in erb_analysis_ref]
        f_rec_t = ifilterbank(c_t, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)

        np.testing.assert_allclose(
            torch_to_np(f_rec_t),
            erb_synthesis_ref,
            rtol=1e-10,
            atol=1e-12,
            err_msg="synthesis mismatch with NumPy",
        )


# ---------------------------------------------------------------------------
# Perfect reconstruction
# ---------------------------------------------------------------------------


class TestPerfectReconstruction:
    """Analysis → dual synthesis → time domain ≈ original."""

    @pytest.mark.parametrize("signal_name", ["noise", "tone", "impulse"])
    def test_erb_perfect_reconstruction(
        self, erb_filterbank, noise_signal, tone_signal, impulse_signal, signal_name
    ):
        from cool_frames.torch.filterbanks import filterbank, ifilterbank

        signals = {"noise": noise_signal, "tone": tone_signal, "impulse": impulse_signal}
        x_np = signals[signal_name]
        fb = erb_filterbank

        f_t = np_to_torch(x_np, dtype=torch.float64)
        c_t = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])
        f_rec_t = ifilterbank(c_t, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)
        f_rec = torch_to_np(f_rec_t)

        rel_err = np.linalg.norm(f_rec[: len(x_np)] - x_np) / (np.linalg.norm(x_np) + 1e-30)
        assert rel_err < 1e-8, f"reconstruction error {rel_err:.2e} for {signal_name}"

    def test_100_random_trials(self, erb_filterbank):
        """Monte Carlo: perfect reconstruction for 100 random signals."""
        from cool_frames.torch.filterbanks import filterbank, ifilterbank

        fb = erb_filterbank
        rng = np.random.default_rng(123)

        for trial in range(100):
            x_np = rng.standard_normal(fb["Ls"])
            f_t = np_to_torch(x_np, dtype=torch.float64)
            c_t = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])
            f_rec_t = ifilterbank(c_t, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)
            f_rec = torch_to_np(f_rec_t)

            rel_err = np.linalg.norm(f_rec[: len(x_np)] - x_np) / np.linalg.norm(x_np)
            assert rel_err < 1e-8, f"trial {trial}: rel_err={rel_err:.2e}"


# ---------------------------------------------------------------------------
# Linearity
# ---------------------------------------------------------------------------


class TestLinearity:
    """Filterbank analysis is a linear operator."""

    def test_superposition(self, erb_filterbank):
        from cool_frames.torch.filterbanks import filterbank

        fb = erb_filterbank
        rng = np.random.default_rng(50)
        f1 = rng.standard_normal(fb["Ls"])
        f2 = rng.standard_normal(fb["Ls"])
        alpha, beta = 2.7, -1.3

        c1 = filterbank(np_to_torch(f1, dtype=torch.float64), fb["g"], fb["a"], L=fb["L"])
        c2 = filterbank(np_to_torch(f2, dtype=torch.float64), fb["g"], fb["a"], L=fb["L"])
        c_combo = filterbank(
            np_to_torch(alpha * f1 + beta * f2, dtype=torch.float64),
            fb["g"],
            fb["a"],
            L=fb["L"],
        )

        for m in range(fb["M"]):
            expected = alpha * torch_to_np(c1[m]) + beta * torch_to_np(c2[m])
            np.testing.assert_allclose(
                torch_to_np(c_combo[m]),
                expected,
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"linearity violated at channel {m}",
            )


# ---------------------------------------------------------------------------
# Frame energy bounds
# ---------------------------------------------------------------------------


class TestFrameEnergyBounds:
    """Verify A·||x||² ≤ Σ||c_m||² ≤ B·||x||²."""

    def test_energy_bounds(self, erb_filterbank):
        from cool_frames.torch.filterbanks import filterbank

        fb = erb_filterbank
        # frame_bounds are the folded real-frame bounds; the raw coefficient
        # energy sum(|c|^2) is governed by the operator eigenvalues, which for
        # these single-sided real-audio banks are exactly half the folded ones.
        A, B = fb["frame_bounds"]
        A, B = A / 2.0, B / 2.0
        rng = np.random.default_rng(60)

        for _ in range(20):
            x_np = rng.standard_normal(fb["Ls"])
            f_t = np_to_torch(x_np, dtype=torch.float64)
            c_t = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])

            energy_x = np.sum(x_np**2)
            energy_c = sum(torch.sum(torch.abs(cm) ** 2).item() for cm in c_t)

            # Allow small numerical tolerance
            assert energy_c >= A * energy_x * (1 - 1e-8), (
                f"lower bound violated: {energy_c:.6f} < {A * energy_x:.6f}"
            )
            assert energy_c <= B * energy_x * (1 + 1e-8), (
                f"upper bound violated: {energy_c:.6f} > {B * energy_x:.6f}"
            )


# ---------------------------------------------------------------------------
# CUDA device placement
# ---------------------------------------------------------------------------


class TestFilterbankCuda:
    """Analysis/synthesis on CUDA device."""

    @pytest.mark.requires_cuda
    def test_analysis_cuda(self, erb_filterbank, noise_signal):
        from cool_frames.torch.filterbanks import filterbank

        fb = erb_filterbank
        device = torch.device("cuda")
        f_t = np_to_torch(noise_signal, dtype=torch.float64).to(device)

        c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])

        for cm in c:
            assert cm.device.type == "cuda"
            assert torch.all(torch.isfinite(cm))

    @pytest.mark.requires_cuda
    def test_perfect_reconstruction_cuda(self, erb_filterbank, noise_signal):
        from cool_frames.torch.filterbanks import filterbank, ifilterbank

        fb = erb_filterbank
        device = torch.device("cuda")
        x_np = noise_signal
        f_t = np_to_torch(x_np, dtype=torch.float64).to(device)

        c_t = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])
        f_rec_t = ifilterbank(c_t, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)

        f_rec = f_rec_t.cpu().numpy()
        rel_err = np.linalg.norm(f_rec[: len(x_np)] - x_np) / np.linalg.norm(x_np)
        assert rel_err < 1e-7
