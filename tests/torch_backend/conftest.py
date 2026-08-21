"""
Shared fixtures for the PyTorch backend test suite.

Every test in this directory requires both ``torch`` and the NumPy backend of
``cool_frames`` to be importable.  The main conftest markers are:

* ``requires_torch`` — skip if ``torch`` is not installed.
* ``requires_torch_impl`` — skip if ``cool_frames.torch`` has no real
  implementation yet (i.e. ``__all__`` is still empty).
* ``requires_cuda`` — skip if no CUDA device is available.

Fixtures generate test filterbanks and signals using the **NumPy** backend as
the ground-truth reference.  The torch tests then compare against these.
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# availability flags
# ---------------------------------------------------------------------------

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cool_frames.numpy as lfb_np  # noqa: F401

    HAS_NUMPY_IMPL = True
except ImportError:
    HAS_NUMPY_IMPL = False

try:
    import cool_frames.torch as lfb_torch

    # The backend is "real" once it exports at least one symbol.
    HAS_TORCH_IMPL = bool(getattr(lfb_torch, "__all__", None))
except ImportError:
    HAS_TORCH_IMPL = False


def _has_cuda() -> bool:
    return HAS_TORCH and torch.cuda.is_available()


# ---------------------------------------------------------------------------
# pytest markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_torch: skip when torch is not installed")
    config.addinivalue_line(
        "markers", "requires_torch_impl: skip when cool_frames.torch is not implemented"
    )
    config.addinivalue_line("markers", "requires_cuda: skip when CUDA is not available")


def pytest_collection_modifyitems(items):
    skip_torch = pytest.mark.skip(reason="torch not installed")
    skip_impl = pytest.mark.skip(reason="cool_frames.torch not yet implemented")
    skip_cuda = pytest.mark.skip(reason="CUDA not available")

    for item in items:
        if "requires_torch" in item.keywords and not HAS_TORCH:
            item.add_marker(skip_torch)
        if "requires_torch_impl" in item.keywords and not HAS_TORCH_IMPL:
            item.add_marker(skip_impl)
        if "requires_cuda" in item.keywords and not _has_cuda():
            item.add_marker(skip_cuda)


# ---------------------------------------------------------------------------
# convenience: auto-apply requires_torch to the whole directory
# ---------------------------------------------------------------------------


def pytest_itemcollected(item):
    """Every test in tests/torch_backend/ implicitly requires torch."""
    if not HAS_TORCH:
        item.add_marker(pytest.mark.skip(reason="torch not installed"))
    if not HAS_NUMPY_IMPL:
        item.add_marker(pytest.mark.skip(reason="cool_frames.numpy not installed"))


# ---------------------------------------------------------------------------
# type helpers
# ---------------------------------------------------------------------------


@pytest.fixture(params=[torch.float32, torch.float64] if HAS_TORCH else [])
def float_dtype(request):
    """Parametrise over single / double precision."""
    return request.param


@pytest.fixture(params=[torch.complex64, torch.complex128] if HAS_TORCH else [])
def complex_dtype(request):
    """Parametrise over single / double complex precision."""
    return request.param


# ---------------------------------------------------------------------------
# numpy ↔ torch conversion helpers
# ---------------------------------------------------------------------------


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


def coeff_list_np_to_torch(c_np: list[np.ndarray], **kw) -> list:
    """Convert a list of NumPy coefficient arrays to torch tensors."""
    return [np_to_torch(cm, **kw) for cm in c_np]


def coeff_list_torch_to_np(c_t: list) -> list[np.ndarray]:
    """Convert a list of torch coefficient tensors to NumPy arrays."""
    return [torch_to_np(cm) for cm in c_t]


# ---------------------------------------------------------------------------
# filterbank geometry fixtures (computed once via NumPy)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def fb_params() -> dict:
    """Basic filterbank parameters shared across tests."""
    return dict(fs=8000, Ls=1024)


@pytest.fixture(scope="session")
def erb_filterbank(fb_params):
    """ERB filterbank designed with the NumPy backend.

    Returns a dict with keys: g, a, L, M, fc, fs, Ls, gd
    """
    if not HAS_NUMPY_IMPL:
        pytest.skip("cool_frames.numpy not available")

    from cool_frames.numpy.filterbanks import (
        filterbankbounds,
        filterbankdual,
    )
    from cool_frames.numpy.filters import audfilters

    fs, Ls = fb_params["fs"], fb_params["Ls"]
    g, a, fc, L, _info = audfilters(fs, Ls)
    gd = filterbankdual(g, a, L)
    A, B = filterbankbounds(g, a, L)
    M = len(g)

    return dict(
        g=g,
        a=a,
        fc=fc,
        L=L,
        M=M,
        fs=fs,
        Ls=Ls,
        gd=gd,
        frame_bounds=(A, B),
    )


@pytest.fixture(scope="session")
def cqt_filterbank(fb_params):
    """Bark-scale auditory filterbank (second filterbank type for PR tests).

    Uses audfilters with Bark scale — produces a proper frame with
    positive lower bound, unlike CQT with default parameters.
    """
    if not HAS_NUMPY_IMPL:
        pytest.skip("cool_frames.numpy not available")

    from cool_frames.numpy.filterbanks import (
        filterbankbounds,
        filterbankdual,
    )
    from cool_frames.numpy.filters import audfilters

    fs, Ls = fb_params["fs"], fb_params["Ls"]
    g, a, fc, L, _info = audfilters(fs, Ls, scale="bark")
    gd = filterbankdual(g, a, L)
    A, B = filterbankbounds(g, a, L)
    M = len(g)

    return dict(
        g=g,
        a=a,
        fc=fc,
        L=L,
        M=M,
        fs=fs,
        Ls=Ls,
        gd=gd,
        frame_bounds=(A, B),
    )


@pytest.fixture(scope="session")
def uniform_gabor_fb():
    """Small uniform Gabor-like filterbank for fast tests.

    M=8 modulated Hann windows, hop=16, L=128.
    """
    if not HAS_NUMPY_IMPL:
        pytest.skip("cool_frames.numpy not available")

    from cool_frames.numpy.filterbanks import (
        filterbankbounds,
        filterbankdual,
    )
    from cool_frames.numpy.filters.lowlevel import blfilter

    M, a_hop, L = 8, 16, 128
    N = L // a_hop  # frames per channel
    fsupp = L // M
    g = []
    fc_list = []
    for m in range(M):
        fc_m = m / M
        fc_list.append(fc_m)
        g.append(blfilter("hann", fsupp, fc_m))

    a = np.full((M, 2), [[a_hop, 1]], dtype=int)
    gd = filterbankdual(g, a, L)
    A, B = filterbankbounds(g, a, L)

    return dict(
        g=g,
        a=a,
        L=L,
        M=M,
        N=N,
        a_hop=a_hop,
        gd=gd,
        frame_bounds=(A, B),
        fc=np.array(fc_list),
    )


# ---------------------------------------------------------------------------
# test signals (NumPy reference)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def noise_signal(fb_params):
    """Deterministic noise signal, shape (Ls,)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(fb_params["Ls"])


@pytest.fixture(scope="session")
def tone_signal(fb_params):
    """Sine-tone test signal: 440 Hz + 0.5 × 1000 Hz."""
    fs, Ls = fb_params["fs"], fb_params["Ls"]
    t = np.arange(Ls) / fs
    return np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1000 * t)


@pytest.fixture(scope="session")
def stereo_signal(fb_params):
    """Two-channel test signal, shape (Ls, 2)."""
    rng = np.random.default_rng(99)
    Ls = fb_params["Ls"]
    return rng.standard_normal((Ls, 2))


@pytest.fixture(scope="session")
def impulse_signal(fb_params):
    """Unit impulse at sample 0."""
    x = np.zeros(fb_params["Ls"])
    x[0] = 1.0
    return x


# ---------------------------------------------------------------------------
# NumPy reference computations (cached per session)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def erb_analysis_ref(erb_filterbank, noise_signal):
    """NumPy filterbank analysis of the noise signal with ERB bank."""
    from cool_frames.numpy.filterbanks import filterbank

    fb = erb_filterbank
    c = filterbank(noise_signal, fb["g"], fb["a"], L=fb["L"])
    return c


@pytest.fixture(scope="session")
def erb_synthesis_ref(erb_filterbank, erb_analysis_ref):
    """NumPy filterbank synthesis (perfect reconstruction reference)."""
    from cool_frames.numpy.filterbanks import ifilterbank

    fb = erb_filterbank
    f_rec = ifilterbank(erb_analysis_ref, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)
    return f_rec
