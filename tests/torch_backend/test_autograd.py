"""
Autograd and differentiability integration tests.

These tests verify that gradients flow through the torch backend's
differentiable operations.  This is the core value proposition of the
PyTorch backend: enabling end-to-end training through filterbank
analysis, phase retrieval, and synthesis.

Test categories:
  1. Forward-pass gradient flow through analysis/synthesis
  2. Straight-through estimator for argsort in phase integration
  3. Unrolled ADMM / GLA backward pass
  4. End-to-end: signal → analysis → magnitude → phase retrieval → synthesis → loss
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


# ---------------------------------------------------------------------------
# Analysis / synthesis gradient flow
# ---------------------------------------------------------------------------


class TestAnalysisSynthesisGrad:
    """Gradients flow through filterbank analysis and synthesis."""

    def test_analysis_backward(self, erb_filterbank):
        """Loss on analysis coefficients → gradient w.r.t. input signal."""
        from cool_frames.torch.filterbanks import filterbank

        fb = erb_filterbank
        rng = np.random.default_rng(1000)
        x_np = rng.standard_normal(fb["Ls"])
        x_t = np_to_torch(x_np, dtype=torch.float64).requires_grad_(True)

        c = filterbank(x_t, fb["g"], fb["a"], L=fb["L"])

        # Scalar loss: sum of squared magnitudes
        loss = sum(torch.sum(torch.abs(cm) ** 2) for cm in c)
        loss.backward()

        assert x_t.grad is not None, "no gradient on input signal"
        assert x_t.grad.shape == x_t.shape
        assert torch.all(torch.isfinite(x_t.grad)), "gradient has non-finite values"

    def test_synthesis_backward(self, erb_filterbank):
        """Loss on synthesised signal → gradient w.r.t. coefficients."""
        from cool_frames.torch.filterbanks import ifilterbank

        fb = erb_filterbank
        np.random.default_rng(1001)

        # Create coefficient tensors with gradients
        c_list = []
        for m in range(fb["M"]):
            a_frac = fb["a"][m, 0] / fb["a"][m, 1] if fb["a"].ndim == 2 else fb["a"][m]
            N_m = int(round(fb["L"] / a_frac))
            cm = torch.randn(N_m, dtype=torch.complex128, requires_grad=True)
            c_list.append(cm)

        f_rec = ifilterbank(c_list, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)
        loss = torch.sum(f_rec**2)
        loss.backward()

        for m, cm in enumerate(c_list):
            assert cm.grad is not None, f"no gradient on coefficients[{m}]"
            assert torch.all(torch.isfinite(cm.grad)), (
                f"channel {m} gradient has non-finite values"
            )

    def test_round_trip_backward(self, erb_filterbank):
        """Loss on reconstruction error → gradient through full pipeline."""
        from cool_frames.torch.filterbanks import filterbank, ifilterbank

        fb = erb_filterbank
        rng = np.random.default_rng(1002)
        x_np = rng.standard_normal(fb["Ls"])
        x_t = np_to_torch(x_np, dtype=torch.float64).requires_grad_(True)

        c = filterbank(x_t, fb["g"], fb["a"], L=fb["L"])
        x_rec = ifilterbank(c, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)

        loss = torch.nn.functional.mse_loss(x_rec, x_t.detach())
        loss.backward()

        assert x_t.grad is not None
        assert torch.all(torch.isfinite(x_t.grad))


# ---------------------------------------------------------------------------
# Magnitude manipulation + synthesis (typical training scenario)
# ---------------------------------------------------------------------------


class TestMagnitudeManipulationGrad:
    """Gradients flow through magnitude modification in coefficient domain."""

    def test_learnable_gain(self, erb_filterbank):
        """Learnable per-channel gain: gradient w.r.t. gain parameters."""
        from cool_frames.torch.filterbanks import filterbank, ifilterbank

        fb = erb_filterbank
        rng = np.random.default_rng(1010)
        x_np = rng.standard_normal(fb["Ls"])
        x_t = np_to_torch(x_np, dtype=torch.float64)
        target = np_to_torch(x_np * 0.5, dtype=torch.float64)

        # Learnable gain: one per channel
        gain = torch.ones(fb["M"], dtype=torch.float64, requires_grad=True)

        c = filterbank(x_t, fb["g"], fb["a"], L=fb["L"])
        # Apply gain
        c_scaled = [cm * gain[m] for m, cm in enumerate(c)]
        x_rec = ifilterbank(c_scaled, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)

        loss = torch.nn.functional.mse_loss(x_rec, target)
        loss.backward()

        assert gain.grad is not None, "no gradient on gain parameter"
        assert gain.grad.shape == (fb["M"],)
        assert torch.all(torch.isfinite(gain.grad))


# ---------------------------------------------------------------------------
# Phase integration gradcheck
# ---------------------------------------------------------------------------


class TestPhaseIntegrationGrad:
    """Gradient checks for differentiable phase reconstruction."""

    @pytest.mark.xfail(
        reason="constphase uses .item()/float() for sorting and control flow, breaking autograd graph"
    )
    def test_constphase_nonuniform_gradcheck(self, erb_filterbank, noise_signal):
        """torch.autograd.gradcheck on constphase_nonuniform."""
        from cool_frames.torch.filterbanks import filterbank
        from cool_frames.torch.phase import constphase_nonuniform

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])

        # Use only first few channels for speed
        n_test = min(3, fb["M"])
        s = [torch.abs(c[m]).requires_grad_(True) for m in range(n_test)]

        a_int = fb["a"][:n_test, 0] if fb["a"].ndim == 2 else fb["a"][:n_test]
        fc_n = fb["fc"][:n_test] / fb["fs"] * 2
        tfr = np.ones(n_test)

        def fn(*magnitudes):
            _c_list, phase_list, _, _ = constphase_nonuniform(
                list(magnitudes),
                torch.from_numpy(np.asarray(a_int)),
                torch.from_numpy(fc_n),
                torch.from_numpy(tfr),
            )
            # Return sum of phases as scalar
            return sum(torch.sum(p) for p in phase_list)

        # gradcheck with relaxed tolerance (phase integration is
        # only piecewise smooth due to sorting)
        try:
            torch.autograd.gradcheck(
                fn,
                tuple(s),
                eps=1e-4,
                atol=1e-3,
                rtol=1e-2,
                nondet_tol=1e-2,
            )
        except RuntimeError as e:
            if "not implemented" in str(e).lower():
                pytest.skip("gradcheck not yet supported")
            raise


# ---------------------------------------------------------------------------
# ADMM backward pass
# ---------------------------------------------------------------------------


class TestADMMGrad:
    """Gradient flow through unrolled ADMM phase retrieval."""

    @pytest.mark.skip(reason="diff_admm moved to audioeffects in the 2026-06 consolidation")
    def test_admm_backward(self, erb_filterbank, noise_signal):
        """Loss on ADMM output → gradient w.r.t. input magnitudes."""
        from cool_frames.torch.phase import diff_admm

        fb = erb_filterbank
        from cool_frames.numpy.filterbanks import filterbank as np_fb

        c_np = np_fb(noise_signal, fb["g"], fb["a"], L=fb["L"])
        s_np = [np.abs(cm) for cm in c_np]
        N_frames = [cm.shape[0] for cm in c_np]
        mag_flat_np = np.concatenate([s.ravel() for s in s_np])

        mag_flat = np_to_torch(mag_flat_np, dtype=torch.float64).requires_grad_(True)

        phase, _mag_out = diff_admm(
            mag_flat,
            fb["g"],
            fb["a"],
            N_frames,
            L=fb["L"],
            real=True,
            maxit=3,
        )

        loss = torch.sum(phase**2)
        loss.backward()

        assert mag_flat.grad is not None, "no gradient through ADMM"
        assert torch.all(torch.isfinite(mag_flat.grad)), "ADMM gradient has non-finite values"


# ---------------------------------------------------------------------------
# GLA backward pass
# ---------------------------------------------------------------------------


class TestGLAGrad:
    """Gradient flow through unrolled Griffin-Lim."""

    def test_gla_backward(self, erb_filterbank, noise_signal):
        """Loss on GLA output → gradient w.r.t. input magnitudes."""
        from cool_frames.torch.phase import gla

        fb = erb_filterbank
        from cool_frames.numpy.filterbanks import filterbank as np_fb

        c_np = np_fb(noise_signal, fb["g"], fb["a"], L=fb["L"])
        s_np = [np.abs(cm) for cm in c_np]

        s_t = [np_to_torch(sm, dtype=torch.float64).requires_grad_(True) for sm in s_np]

        _c, f, _relres, _niter = gla(
            s_t,
            fb["g"],
            fb["a"],
            L=fb["L"],
            Ls=fb["Ls"],
            real=True,
            maxit=3,
        )

        loss = torch.sum(f**2)
        loss.backward()

        for m, sm in enumerate(s_t):
            assert sm.grad is not None, f"no gradient on magnitude[{m}]"
            assert torch.all(torch.isfinite(sm.grad)), (
                f"GLA gradient channel {m} has non-finite values"
            )


# ---------------------------------------------------------------------------
# End-to-end training test
# ---------------------------------------------------------------------------


class TestEndToEndTraining:
    """Full pipeline: signal → analysis → |·| → phase retrieval → synthesis → loss."""

    @pytest.mark.skip(
        reason="constphase_nonuniform moved to audioeffects in the 2026-06 consolidation"
    )
    def test_denoising_gradient(self, erb_filterbank):
        """Simulate one step of a denoising training loop."""
        from cool_frames.torch.filterbanks import filterbank, ifilterbank
        from cool_frames.torch.phase import constphase_nonuniform

        fb = erb_filterbank
        rng = np.random.default_rng(2000)

        # Clean signal
        clean = rng.standard_normal(fb["Ls"])
        # Noisy signal
        noisy = clean + 0.1 * rng.standard_normal(fb["Ls"])

        clean_t = np_to_torch(clean, dtype=torch.float64)
        noisy_t = np_to_torch(noisy, dtype=torch.float64)

        # Learnable magnitude mask
        mask = torch.ones(fb["M"], dtype=torch.float64, requires_grad=True)

        # Forward pass
        c_noisy = filterbank(noisy_t, fb["g"], fb["a"], L=fb["L"])
        s = [torch.abs(cm) * mask[m] for m, cm in enumerate(c_noisy)]

        a_int = fb["a"][:, 0] if fb["a"].ndim == 2 else fb["a"]
        fc_n = fb["fc"] / fb["fs"] * 2
        tfr = np.ones(fb["M"])

        c_masked, _phase, _, _ = constphase_nonuniform(
            s,
            torch.from_numpy(np.asarray(a_int)),
            torch.from_numpy(fc_n),
            torch.from_numpy(tfr),
        )

        x_rec = ifilterbank(c_masked, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)

        # Time-domain loss
        loss = torch.nn.functional.mse_loss(x_rec, clean_t)
        loss.backward()

        assert mask.grad is not None, "no gradient on mask parameter"
        assert torch.all(torch.isfinite(mask.grad)), "end-to-end gradient has non-finite values"
        # Gradient should be non-zero (mask affects output)
        assert torch.any(mask.grad != 0), "gradient is all zeros"

    def test_multi_step_optimisation(self, erb_filterbank):
        """Multiple optimisation steps converge (loss decreases)."""
        from cool_frames.torch.filterbanks import filterbank, ifilterbank

        fb = erb_filterbank
        rng = np.random.default_rng(2001)
        x = rng.standard_normal(fb["Ls"])
        target = x * 0.7  # simple gain target

        x_t = np_to_torch(x, dtype=torch.float64)
        target_t = np_to_torch(target, dtype=torch.float64)

        gain = torch.ones(fb["M"], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([gain], lr=0.01)

        losses = []
        for _step in range(5):
            optimizer.zero_grad()
            c = filterbank(x_t, fb["g"], fb["a"], L=fb["L"])
            c_scaled = [cm * gain[m] for m, cm in enumerate(c)]
            x_rec = ifilterbank(c_scaled, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)
            loss = torch.nn.functional.mse_loss(x_rec, target_t)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease (at least from step 0 to step 4)
        assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}"


# ---------------------------------------------------------------------------
# Precision tests
# ---------------------------------------------------------------------------


class TestPrecision:
    """Verify both float32 and float64 work through the gradient pipeline."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_analysis_precision(self, erb_filterbank, dtype):
        from cool_frames.torch.filterbanks import filterbank

        fb = erb_filterbank
        rng = np.random.default_rng(3000)
        x_np = rng.standard_normal(fb["Ls"]).astype(
            np.float32 if dtype == torch.float32 else np.float64
        )
        x_t = np_to_torch(x_np).requires_grad_(True)

        c = filterbank(x_t, fb["g"], fb["a"], L=fb["L"])
        loss = sum(torch.sum(torch.abs(cm) ** 2) for cm in c)
        loss.backward()

        assert x_t.grad is not None
        assert x_t.grad.dtype == dtype
