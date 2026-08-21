"""Tests for the differentiable torch biquad (``cool_frames.torch.filters._biquad``).

Coverage:
  1. Numerical agreement with the NumPy reference ``comp_biquad`` across pole
     locations, lengths, and normalisations (float64).
  2. ``biquad_response`` stability + equivalence to ``comp_biquad`` evaluated
     at ``r = sigmoid(rho)``, ``theta = pi sigmoid(phi)``.
  3. ``torch.autograd.gradcheck`` on the pole parameters ``(r, theta)`` and on
     the unconstrained ML parameters ``(rho, phi)``.
  4. Device compatibility: CUDA result matches CPU (skipped without a GPU).
  5. ``biquadfilter`` descriptor shape and end-to-end gradient flow to
     ``rho``/``phi`` through the ``H(L)`` callable.
"""

from __future__ import annotations

import math

import pytest

import numpy as np

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.requires_torch_impl

CASES = [
    (0.90, 0.50, 64, "energy"),
    (0.50, 1.20, 128, "peak"),
    (0.99, 0.05, 256, "1"),
    (0.30, 2.90, 32, "none"),
    (0.80, math.pi / 2, 100, "energy"),
]


# ---------------------------------------------------------------------------
# 1. numerical agreement with numpy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r,theta,L,norm", CASES)
def test_matches_numpy(r, theta, L, norm):
    from cool_frames.numpy.filters._filters import comp_biquad as np_comp_biquad
    from cool_frames.torch.filters import comp_biquad

    H_np = np_comp_biquad(r, theta, L, norm)
    H_t = comp_biquad(r, theta, L, norm, dtype=torch.float64)
    assert H_t.shape == (L,)
    assert torch.is_complex(H_t)
    err = np.max(np.abs(H_t.detach().cpu().numpy() - H_np))
    assert err < 1e-11, f"max|Δ|={err:.2e} for r={r} theta={theta} norm={norm}"


def test_float32_close():
    from cool_frames.numpy.filters._filters import comp_biquad as np_comp_biquad
    from cool_frames.torch.filters import comp_biquad

    H32 = comp_biquad(0.9, 0.5, 128, "energy", dtype=torch.float32)
    assert H32.dtype == torch.complex64
    H_np = np_comp_biquad(0.9, 0.5, 128, "energy")
    assert np.max(np.abs(H32.detach().cpu().numpy() - H_np)) < 1e-4


# ---------------------------------------------------------------------------
# 2. ML-parameter entry point
# ---------------------------------------------------------------------------
def test_biquad_response_equiv():
    from cool_frames.torch.filters import biquad_response, comp_biquad

    rho, phi, L = 1.3, -0.7, 96
    r = 1.0 / (1.0 + math.exp(-rho))
    theta = math.pi / (1.0 + math.exp(-phi))
    a = biquad_response(rho, phi, L, "energy", dtype=torch.float64)
    b = comp_biquad(r, theta, L, "energy", dtype=torch.float64)
    assert torch.allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("rho", [-5.0, 0.0, 5.0])
@pytest.mark.parametrize("phi", [-5.0, 0.0, 5.0])
def test_biquad_response_stable(rho, phi):
    """Any real (rho, phi) yields a finite, bounded response (stable pole)."""
    from cool_frames.torch.filters import biquad_response

    H = biquad_response(rho, phi, 64, "none", dtype=torch.float64)
    assert torch.isfinite(H).all()


# ---------------------------------------------------------------------------
# 3. gradcheck
# ---------------------------------------------------------------------------
def test_gradcheck_r_theta():
    from cool_frames.torch.filters import comp_biquad

    L = 48
    r = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    theta = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)

    def f(r_, th_):
        return torch.view_as_real(comp_biquad(r_, th_, L, "none", dtype=torch.float64))

    assert torch.autograd.gradcheck(f, (r, theta), atol=1e-6)


def test_gradcheck_rho_phi():
    from cool_frames.torch.filters import biquad_response

    L = 48
    rho = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(-0.3, dtype=torch.float64, requires_grad=True)

    def f(rho_, phi_):
        return torch.view_as_real(biquad_response(rho_, phi_, L, "energy", dtype=torch.float64))

    assert torch.autograd.gradcheck(f, (rho, phi), atol=1e-6)


def test_gradients_nonzero():
    from cool_frames.torch.filters import biquad_response

    rho = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    H = biquad_response(rho, phi, 64, "energy", dtype=torch.float64)
    (H.abs().sum()).backward()
    assert rho.grad is not None and torch.abs(rho.grad) > 0
    assert phi.grad is not None and torch.abs(phi.grad) > 0


# ---------------------------------------------------------------------------
# 4. device compatibility
# ---------------------------------------------------------------------------
@pytest.mark.requires_cuda
def test_cuda_matches_cpu():
    from cool_frames.torch.filters import comp_biquad

    cpu = comp_biquad(0.85, 0.9, 128, "energy", device="cpu", dtype=torch.float64)
    gpu = comp_biquad(0.85, 0.9, 128, "energy", device="cuda", dtype=torch.float64)
    assert gpu.device.type == "cuda"
    assert torch.allclose(cpu, gpu.cpu(), atol=1e-12)


# ---------------------------------------------------------------------------
# 5. biquadfilter descriptor + gradient flow through H(L)
# ---------------------------------------------------------------------------
def test_biquadfilter_descriptor_keys():
    from cool_frames.torch.filters import biquadfilter

    g = biquadfilter(0.3, 0.05, dtype=torch.float64)
    assert isinstance(g, dict)
    for key in ("H", "foff", "realonly", "delay", "fc", "bw", "r", "theta", "rho", "phi"):
        assert key in g
    H = g["H"](128)
    assert H.shape == (128,) and torch.is_complex(H)
    assert g["foff"](128) == 0


def test_biquadfilter_vector_returns_list():
    from cool_frames.torch.filters import biquadfilter

    gs = biquadfilter([0.2, 0.4, 0.6], 0.05, dtype=torch.float64)
    assert isinstance(gs, list) and len(gs) == 3


def test_biquadfilter_trainable():
    """rho/phi tensors with grad propagate through the H(L) callable."""
    from cool_frames.torch.filters import biquadfilter

    rho = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    phi = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    g = biquadfilter(0.0, 0.05, rho=rho, phi=phi, dtype=torch.float64)
    H = g["H"](64)
    (H.abs().sum()).backward()
    assert rho.grad is not None and torch.abs(rho.grad) > 0
    assert phi.grad is not None and torch.abs(phi.grad) > 0
