"""
Phase 6 – Signal processing utility tests.

Validates:
  - rms: root-mean-square with AC coupling
  - gaindb: decibel gain scaling
  - thresh: hard/soft/wiener thresholding
  - compand / expand: dynamic range compression round-trip
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
# RMS
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# gaindb
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# thresh
# ---------------------------------------------------------------------------


class TestThresh:
    """thresh: coefficient thresholding."""

    def test_hard_zeros_small(self):
        from cool_frames.torch.sigproc import thresh

        x = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
        xo, N = thresh(x, 0.5, mode="hard")
        assert N == 3  # 0.5, 1.0, 2.0 survive
        assert xo[0].item() == 0.0

    def test_soft_shrinkage(self):
        from cool_frames.torch.sigproc import thresh

        x = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
        xo, _N = thresh(x, 0.5, mode="soft")
        # 0.1 → 0, 0.5 → 0, 1.0 → 0.5, 2.0 → 1.5
        np.testing.assert_allclose(torch_to_np(xo), [0, 0, 0.5, 1.5], atol=1e-12)

    def test_wiener(self):
        from cool_frames.torch.sigproc import thresh

        x = torch.tensor([0.1, 0.5, 1.0, 3.0], dtype=torch.float64)
        xo, _N = thresh(x, 0.5, mode="wiener")
        # gain = max(1 - (lam/|x|)^2, 0) → for x=3: 1-(0.5/3)^2 ≈ 0.972
        assert torch.all(torch.isfinite(xo))
        assert xo[3].item() > 0

    def test_complex_soft(self):
        from cool_frames.torch.sigproc import thresh

        x = torch.tensor([1 + 1j, 0.1 + 0j, 2 - 1j], dtype=torch.complex128)
        xo, _N = thresh(x, 0.5, mode="soft")
        # Phase should be preserved for surviving coefficients
        for i in range(len(x)):
            if torch.abs(xo[i]).item() > 0:
                orig_angle = torch.angle(x[i]).item()
                new_angle = torch.angle(xo[i]).item()
                assert abs(orig_angle - new_angle) < 1e-10

    def test_agreement_with_numpy(self):
        from cool_frames.numpy.sigproc import thresh as np_thresh
        from cool_frames.torch.sigproc import thresh as torch_thresh

        rng = np.random.default_rng(300)
        x_np = rng.standard_normal(50)

        for mode in ["hard", "soft", "wiener"]:
            xo_np, N_np = np_thresh(x_np, 0.3, mode=mode)
            xo_t, N_t = torch_thresh(np_to_torch(x_np, dtype=torch.float64), 0.3, mode=mode)

            np.testing.assert_allclose(
                torch_to_np(xo_t), xo_np, atol=1e-12, err_msg=f"thresh mode={mode} mismatch"
            )
            assert N_t == N_np, f"thresh mode={mode}: count mismatch"


# ---------------------------------------------------------------------------
# compand / expand
# ---------------------------------------------------------------------------
