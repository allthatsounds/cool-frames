"""
PyTorch backend benchmarks.

Run with:
    pytest benchmarks/bench_torch.py --benchmark-only
"""

import pytest

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")

if HAS_TORCH:
    import cool_frames.torch as lfb_t
    from cool_frames.numpy.filterbanks import filterbankdual
    from cool_frames.numpy.filters import audfilters
    from cool_frames.torch.filters import biquad_response

# Devices to benchmark over: always CPU, plus CUDA when a GPU is present.
DEVICES = (
    (["cpu"] + (["cuda"] if (HAS_TORCH and torch.cuda.is_available()) else []))
    if HAS_TORCH
    else []
)


def _sync(device):
    """Make CUDA timings honest by waiting for the queue to drain."""
    if HAS_TORCH and str(device).startswith("cuda"):
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch_erb_setup():
    fs, Ls = 16000, 16000
    g, a, _fc, L, _info = audfilters(fs, Ls)
    gd = filterbankdual(g, a, L, real=True)
    rng = np.random.default_rng(0)
    f_np = rng.standard_normal(Ls)
    f = torch.tensor(f_np, dtype=torch.float64)
    return dict(g=g, a=a, gd=gd, L=L, Ls=Ls, f=f)


@pytest.fixture(scope="module")
def torch_erb_setup_long():
    fs, Ls = 16000, 16000 * 10
    g, a, _fc, L, _info = audfilters(fs, Ls)
    gd = filterbankdual(g, a, L, real=True)
    rng = np.random.default_rng(1)
    f_np = rng.standard_normal(Ls)
    f = torch.tensor(f_np, dtype=torch.float64)
    return dict(g=g, a=a, gd=gd, L=L, Ls=Ls, f=f)


# ---------------------------------------------------------------------------
# Torch filterbank benchmarks
# ---------------------------------------------------------------------------


class BenchTorchAnalysis:
    """filterbank() — analysis (torch, CPU)."""

    def test_erb_1s(self, benchmark, torch_erb_setup):
        s = torch_erb_setup
        benchmark(lfb_t.filterbanks.filterbank, s["f"], s["g"], s["a"], L=s["L"])

    def test_erb_10s(self, benchmark, torch_erb_setup_long):
        s = torch_erb_setup_long
        benchmark(lfb_t.filterbanks.filterbank, s["f"], s["g"], s["a"], L=s["L"])


class BenchTorchSynthesis:
    """ifilterbank() — synthesis (torch, CPU)."""

    def test_erb_1s(self, benchmark, torch_erb_setup):
        s = torch_erb_setup
        c = lfb_t.filterbanks.filterbank(s["f"], s["g"], s["a"], L=s["L"])
        benchmark(lfb_t.filterbanks.ifilterbank, c, s["gd"], s["a"], Ls=s["Ls"], real=True)


class BenchTorchRoundTrip:
    """Full analysis + synthesis round-trip (torch, CPU)."""

    def test_erb_1s(self, benchmark, torch_erb_setup):
        s = torch_erb_setup

        def roundtrip():
            c = lfb_t.filterbanks.filterbank(s["f"], s["g"], s["a"], L=s["L"])
            return lfb_t.filterbanks.ifilterbank(c, s["gd"], s["a"], Ls=s["Ls"], real=True)

        benchmark(roundtrip)


class BenchTorchBackward:
    """Backward pass through the filterbank (autograd)."""

    def test_backward_erb_1s(self, benchmark, torch_erb_setup):
        s = torch_erb_setup
        f = s["f"].clone().requires_grad_(True)

        def forward_backward():
            c = lfb_t.filterbanks.filterbank(f, s["g"], s["a"], L=s["L"])
            rec = lfb_t.filterbanks.ifilterbank(c, s["gd"], s["a"], Ls=s["Ls"], real=True)
            loss = rec.abs().sum()
            loss.backward()
            f.grad = None

        benchmark(forward_backward)


class BenchTorchGLA:
    """Griffin-Lim phase retrieval (torch)."""

    def test_gla_10iter(self, benchmark, torch_erb_setup):
        s = torch_erb_setup
        c_ref = lfb_t.filterbanks.filterbank(s["f"], s["g"], s["a"], L=s["L"])
        s_abs = [cm.abs() for cm in c_ref]

        def run_gla():
            return lfb_t.phase.gla(
                s_abs, s["g"], s["a"], L=s["L"], Ls=s["Ls"], real=True, maxit=10
            )

        benchmark(run_gla)


# ---------------------------------------------------------------------------
# Device-parametrised benchmarks (CPU + CUDA when available)
# ---------------------------------------------------------------------------


@pytest.fixture(params=DEVICES)
def bench_device(request):
    """Each device that should be benchmarked (CPU always; CUDA if present)."""
    return request.param


class BenchTorchBiquad:
    """Differentiable IIR biquad — builds a bank of resonators on `device`.

    Self-contained and device-correct by construction, so it gives a clean
    CPU-vs-GPU throughput comparison for the newly ported biquad.
    """

    def test_biquad_bank_64(self, benchmark, bench_device):
        L = 4096
        rho = torch.linspace(-2, 2, 64, dtype=torch.float64, device=bench_device)
        phi = torch.linspace(-2, 2, 64, dtype=torch.float64, device=bench_device)

        def build_bank():
            H = torch.stack(
                [
                    biquad_response(
                        rho[i], phi[i], L, "energy", device=bench_device, dtype=torch.float64
                    )
                    for i in range(rho.numel())
                ]
            )
            _sync(bench_device)
            return H

        benchmark(build_bank)

    def test_biquad_backward(self, benchmark, bench_device):
        """Forward + backward through a trainable resonator on `device`."""
        L = 4096
        rho = torch.tensor(0.5, dtype=torch.float64, device=bench_device, requires_grad=True)
        phi = torch.tensor(0.1, dtype=torch.float64, device=bench_device, requires_grad=True)

        def fwd_bwd():
            H = biquad_response(rho, phi, L, "energy", device=bench_device, dtype=torch.float64)
            H.abs().sum().backward()
            rho.grad = None
            phi.grad = None
            _sync(bench_device)

        benchmark(fwd_bwd)


class BenchTorchFFT:
    """On-device rFFT/irFFT primitive — representative GPU throughput."""

    def test_rfft_roundtrip(self, benchmark, bench_device):
        x = torch.randn(1 << 18, dtype=torch.float64, device=bench_device)

        def roundtrip():
            y = torch.fft.irfft(torch.fft.rfft(x), n=x.numel())
            _sync(bench_device)
            return y

        benchmark(roundtrip)
