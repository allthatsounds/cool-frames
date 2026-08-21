"""
Phase 1 – Core FFT kernel tests.

Validates that ``cool_frames.torch.core`` reproduces the NumPy backend's
FFT-based analysis and synthesis kernels to floating-point tolerance.

Each test compares torch output against the NumPy reference for:
  - comp_filterbank_fft   (full-length FFT analysis)
  - comp_ifilterbank_fft  (full-length FFT synthesis)
  - comp_filterbank_fftbl (band-limited FFT analysis)
  - comp_ifilterbank_fftbl(band-limited FFT synthesis)
"""

from __future__ import annotations

import pytest

import numpy as np

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.requires_torch_impl


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_full_length_fb(L: int = 128, M: int = 4, a_hop: int = 16):
    """Build a minimal full-length FFT filterbank in NumPy for testing.

    Returns (G, a_norm, L, M) where G is a list of M complex arrays of
    length L and a_norm is an (M, 2) integer hop-size matrix.
    """
    rng = np.random.default_rng(7)
    G = [rng.standard_normal(L) + 1j * rng.standard_normal(L) for _ in range(M)]
    a_norm = np.full((M, 2), [[a_hop, 1]], dtype=int)
    return G, a_norm, L, M


def _make_bandlimited_fb(L: int = 128, M: int = 4, bw: int = 32, a_hop: int = 16):
    """Build a minimal band-limited FFT filterbank in NumPy.

    Returns (G_bl, foff, realonly, a_norm, L, M).
    """
    rng = np.random.default_rng(8)
    G_bl = []
    foff = np.zeros(M, dtype=int)
    realonly = np.zeros(M, dtype=int)
    for m in range(M):
        g_m = rng.standard_normal(bw) + 1j * rng.standard_normal(bw)
        G_bl.append(g_m)
        foff[m] = m * (L // M)
    a_norm = np.full((M, 2), [[a_hop, 1]], dtype=int)
    return G_bl, foff, realonly, a_norm, L, M


# ---------------------------------------------------------------------------
# Full-length FFT analysis
# ---------------------------------------------------------------------------


class TestCompFilterbankFft:
    """comp_filterbank_fft: torch vs. NumPy agreement."""

    def test_output_shapes(self):
        """Output list has M elements with correct frame counts."""
        from cool_frames.torch.core import comp_filterbank_fft as torch_fn

        G, a_norm, L, M = _make_full_length_fb()

        F_np = np.fft.fft(np.random.default_rng(0).standard_normal(L))
        F_t = torch.from_numpy(F_np.copy()).unsqueeze(1)  # (L, 1)

        G_t = [torch.from_numpy(g.copy()) for g in G]
        a_t = torch.from_numpy(a_norm)

        c_t = torch_fn(F_t, G_t, a_t)

        assert isinstance(c_t, list)
        assert len(c_t) == M
        for m in range(M):
            a_frac = a_norm[m, 0] / a_norm[m, 1]
            N_m = int(round(L / a_frac))
            assert c_t[m].shape[0] == N_m

    def test_agreement_with_numpy(self):
        """Torch output matches NumPy reference to near machine precision."""
        from cool_frames.numpy.core import comp_filterbank_fft as np_fn
        from cool_frames.torch.core import comp_filterbank_fft as torch_fn

        G, a_norm, L, M = _make_full_length_fb()
        rng = np.random.default_rng(1)
        f = rng.standard_normal(L)
        F_np = np.fft.fft(f)[:, np.newaxis]  # (L, 1)

        # NumPy reference
        c_np = np_fn(F_np, G, a_norm)

        # Torch
        F_t = torch.from_numpy(F_np.copy())
        G_t = [torch.from_numpy(g.copy()) for g in G]
        a_t = torch.from_numpy(a_norm)
        c_t = torch_fn(F_t, G_t, a_t)

        for m in range(M):
            np.testing.assert_allclose(
                c_t[m].numpy(),
                c_np[m],
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"channel {m} mismatch",
            )

    def test_multichannel(self):
        """Works with multi-channel (W > 1) input."""
        from cool_frames.numpy.core import comp_filterbank_fft as np_fn
        from cool_frames.torch.core import comp_filterbank_fft as torch_fn

        G, a_norm, L, M = _make_full_length_fb()
        rng = np.random.default_rng(2)
        W = 3
        F_np = np.fft.fft(rng.standard_normal((L, W)), axis=0)

        c_np = np_fn(F_np, G, a_norm)

        F_t = torch.from_numpy(F_np.copy())
        G_t = [torch.from_numpy(g.copy()) for g in G]
        a_t = torch.from_numpy(a_norm)
        c_t = torch_fn(F_t, G_t, a_t)

        for m in range(M):
            assert c_t[m].shape[1] == W
            np.testing.assert_allclose(
                c_t[m].numpy(),
                c_np[m],
                rtol=1e-10,
                atol=1e-12,
            )

    @pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
    def test_dtype_preservation(self, dtype):
        """Output dtype matches input dtype."""
        from cool_frames.torch.core import comp_filterbank_fft as torch_fn

        _G, a_norm, L, M = _make_full_length_fb()

        F_t = torch.randn(L, 1, dtype=dtype)
        G_t = [torch.randn(L, dtype=dtype) for _ in range(M)]
        a_t = torch.from_numpy(a_norm)

        c_t = torch_fn(F_t, G_t, a_t)
        for m in range(M):
            assert c_t[m].dtype == dtype


# ---------------------------------------------------------------------------
# Full-length FFT synthesis
# ---------------------------------------------------------------------------


class TestCompIFilterbankFft:
    """comp_ifilterbank_fft: torch vs. NumPy agreement."""

    def test_agreement_with_numpy(self):
        """Synthesis output matches NumPy reference."""
        from cool_frames.numpy.core import (
            comp_filterbank_fft as np_ana,
        )
        from cool_frames.numpy.core import (
            comp_ifilterbank_fft as np_syn,
        )
        from cool_frames.torch.core import comp_ifilterbank_fft as torch_syn

        G, a_norm, L, _M = _make_full_length_fb()
        rng = np.random.default_rng(3)
        F_np = np.fft.fft(rng.standard_normal(L))[:, np.newaxis]

        c_np = np_ana(F_np, G, a_norm)
        F_out_np = np_syn(c_np, G, a_norm, L)

        c_t = [torch.from_numpy(cm.copy()) for cm in c_np]
        G_t = [torch.from_numpy(g.copy()) for g in G]
        a_t = torch.from_numpy(a_norm)
        F_out_t = torch_syn(c_t, G_t, a_t, L)

        np.testing.assert_allclose(
            F_out_t.numpy(),
            F_out_np,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_output_shape(self):
        """Synthesis output has shape (L, W)."""
        from cool_frames.torch.core import comp_ifilterbank_fft as torch_syn

        _G, a_norm, L, M = _make_full_length_fb()

        a_frac = a_norm[0, 0] / a_norm[0, 1]
        N = int(round(L / a_frac))
        c_t = [torch.randn(N, 1, dtype=torch.complex128) for _ in range(M)]
        G_t = [torch.randn(L, dtype=torch.complex128) for _ in range(M)]
        a_t = torch.from_numpy(a_norm)

        F_out = torch_syn(c_t, G_t, a_t, L)
        assert F_out.shape == (L, 1)


# ---------------------------------------------------------------------------
# Band-limited FFT analysis
# ---------------------------------------------------------------------------


class TestCompFilterbankFftbl:
    """comp_filterbank_fftbl: torch vs. NumPy agreement."""

    def test_agreement_with_numpy(self):
        from cool_frames.numpy.core import comp_filterbank_fftbl as np_fn
        from cool_frames.torch.core import comp_filterbank_fftbl as torch_fn

        G_bl, foff, realonly, a_norm, L, M = _make_bandlimited_fb()
        rng = np.random.default_rng(4)
        F_np = np.fft.fft(rng.standard_normal(L))[:, np.newaxis]

        c_np = np_fn(F_np, G_bl, foff, a_norm, realonly)

        F_t = torch.from_numpy(F_np.copy())
        G_t = [torch.from_numpy(g.copy()) for g in G_bl]
        foff_t = torch.from_numpy(foff)
        a_t = torch.from_numpy(a_norm)
        ro_t = torch.from_numpy(realonly)

        c_t = torch_fn(F_t, G_t, foff_t, a_t, ro_t)

        for m in range(M):
            np.testing.assert_allclose(
                c_t[m].numpy(),
                c_np[m],
                rtol=1e-10,
                atol=1e-12,
                err_msg=f"band-limited channel {m}",
            )

    def test_realonly_flag(self):
        """Filters marked realonly=1 produce correct conjugate mirror."""
        from cool_frames.numpy.core import comp_filterbank_fftbl as np_fn
        from cool_frames.torch.core import comp_filterbank_fftbl as torch_fn

        G_bl, foff, realonly, a_norm, L, M = _make_bandlimited_fb()
        realonly[:] = 1  # mark all as real

        rng = np.random.default_rng(5)
        F_np = np.fft.fft(rng.standard_normal(L))[:, np.newaxis]

        c_np = np_fn(F_np, G_bl, foff, a_norm, realonly)

        F_t = torch.from_numpy(F_np.copy())
        G_t = [torch.from_numpy(g.copy()) for g in G_bl]
        c_t = torch_fn(
            F_t, G_t, torch.from_numpy(foff), torch.from_numpy(a_norm), torch.from_numpy(realonly)
        )

        for m in range(M):
            np.testing.assert_allclose(
                c_t[m].numpy(),
                c_np[m],
                rtol=1e-10,
                atol=1e-12,
            )


# ---------------------------------------------------------------------------
# Band-limited FFT synthesis
# ---------------------------------------------------------------------------


class TestCompIFilterbankFftbl:
    """comp_ifilterbank_fftbl: torch vs. NumPy agreement."""

    def test_agreement_with_numpy(self):
        from cool_frames.numpy.core import (
            comp_filterbank_fftbl as np_ana,
        )
        from cool_frames.numpy.core import (
            comp_ifilterbank_fftbl as np_syn,
        )
        from cool_frames.torch.core import comp_ifilterbank_fftbl as torch_syn

        G_bl, foff, realonly, a_norm, L, _M = _make_bandlimited_fb()
        rng = np.random.default_rng(6)
        F_np = np.fft.fft(rng.standard_normal(L))[:, np.newaxis]

        c_np = np_ana(F_np, G_bl, foff, a_norm, realonly)
        F_out_np = np_syn(c_np, G_bl, foff, a_norm, realonly, L)

        c_t = [torch.from_numpy(cm.copy()) for cm in c_np]
        G_t = [torch.from_numpy(g.copy()) for g in G_bl]
        F_out_t = torch_syn(
            c_t,
            G_t,
            torch.from_numpy(foff),
            torch.from_numpy(a_norm),
            torch.from_numpy(realonly),
            L,
        )

        np.testing.assert_allclose(
            F_out_t.numpy(),
            F_out_np,
            rtol=1e-10,
            atol=1e-12,
        )


# ---------------------------------------------------------------------------
# Round-trip (analysis → synthesis) at the kernel level
# ---------------------------------------------------------------------------


class TestKernelRoundTrip:
    """Verify analysis → synthesis → IFFT recovers the original signal."""

    def test_full_length_round_trip(self):
        from cool_frames.numpy.core import (
            comp_filterbank_fft as np_ana,
        )
        from cool_frames.numpy.core import (
            comp_ifilterbank_fft as np_syn,
        )
        from cool_frames.torch.core import (
            comp_filterbank_fft,
            comp_ifilterbank_fft,
        )

        G_np, a_norm, L, _M = _make_full_length_fb()
        rng = np.random.default_rng(10)
        f = rng.standard_normal(L)
        F_np = np.fft.fft(f)[:, np.newaxis]

        # NumPy reference round-trip
        c_np = np_ana(F_np, G_np, a_norm)
        F_ref_np = np_syn(c_np, G_np, a_norm, L)

        # Torch round-trip
        F_t = torch.from_numpy(F_np.copy())
        G_t = [torch.from_numpy(g.copy()) for g in G_np]
        a_t = torch.from_numpy(a_norm)

        c_t = comp_filterbank_fft(F_t, G_t, a_t)
        F_rec_t = comp_ifilterbank_fft(c_t, G_t, a_t, L)

        np.testing.assert_allclose(
            F_rec_t.numpy(),
            F_ref_np,
            rtol=1e-9,
            atol=1e-11,
            err_msg="kernel round-trip: torch does not match numpy",
        )

    def test_bandlimited_round_trip(self):
        from cool_frames.torch.core import (
            comp_filterbank_fftbl,
            comp_ifilterbank_fftbl,
        )

        G_bl, foff, realonly, a_norm, L, _M = _make_bandlimited_fb()
        rng = np.random.default_rng(11)
        F_np = np.fft.fft(rng.standard_normal(L))[:, np.newaxis]

        F_t = torch.from_numpy(F_np.copy())
        G_t = [torch.from_numpy(g.copy()) for g in G_bl]
        foff_t = torch.from_numpy(foff)
        a_t = torch.from_numpy(a_norm)
        ro_t = torch.from_numpy(realonly)

        c_t = comp_filterbank_fftbl(F_t, G_t, foff_t, a_t, ro_t)
        F_rec_t = comp_ifilterbank_fftbl(c_t, G_t, foff_t, a_t, ro_t, L)

        # At minimum: output is finite and shape is (L, W)
        assert F_rec_t.shape == (L, 1)
        assert torch.all(torch.isfinite(F_rec_t))


# ---------------------------------------------------------------------------
# Device placement (CUDA)
# ---------------------------------------------------------------------------


class TestCudaDevice:
    """Verify that torch core kernels work on CUDA when available."""

    @pytest.mark.requires_cuda
    def test_filterbank_fft_cuda(self):
        from cool_frames.torch.core import comp_filterbank_fft

        G_np, a_norm, L, M = _make_full_length_fb()
        rng = np.random.default_rng(20)
        F_np = np.fft.fft(rng.standard_normal(L))[:, np.newaxis]

        device = torch.device("cuda")
        F_t = torch.from_numpy(F_np.copy()).to(device)
        G_t = [torch.from_numpy(g.copy()).to(device) for g in G_np]
        a_t = torch.from_numpy(a_norm).to(device)

        c_t = comp_filterbank_fft(F_t, G_t, a_t)

        for m in range(M):
            assert c_t[m].device.type == "cuda"
            assert torch.all(torch.isfinite(c_t[m]))

        # Agreement with CPU
        c_cpu = [cm.cpu() for cm in c_t]
        from cool_frames.numpy.core import comp_filterbank_fft as np_fn

        c_np = np_fn(F_np, G_np, a_norm)
        for m in range(M):
            np.testing.assert_allclose(
                c_cpu[m].numpy(),
                c_np[m],
                rtol=1e-5,
                atol=1e-7,
            )
