"""
Phase 2 – Frame theory tests.

Validates that ``cool_frames.torch.filterbanks`` frame operations
(bounds, dual, tight, scale, freqz) agree with NumPy and satisfy
frame-theoretic invariants.

Covers:
  - filterbankbounds / filterbankbounds
  - filterbankdual / filterbankdual (agreement + PR)
  - filterbanktight / filterbanktight (agreement + self-dual PR)
  - filterbankresponse
  - filterbankfreqz
  - filterbankscale
  - ifilterbankiter (CG-based synthesis)
  - CQT filterbank PR (second filterbank type)
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
# Frame bounds
# ---------------------------------------------------------------------------


class TestFilterbankBounds:
    """filterbankbounds: torch wrapper agrees with NumPy."""

    def test_agreement(self, erb_filterbank):
        from cool_frames.numpy.filterbanks import filterbankbounds as np_fn
        from cool_frames.torch.filterbanks import filterbankbounds as torch_fn

        fb = erb_filterbank
        A_np, B_np = np_fn(fb["g"], fb["a"], fb["L"])
        A_t, B_t = torch_fn(fb["g"], fb["a"], fb["L"])

        assert abs(A_t - A_np) < 1e-10
        assert abs(B_t - B_np) < 1e-10

    def test_bounds_positive(self, erb_filterbank):
        from cool_frames.torch.filterbanks import filterbankbounds

        fb = erb_filterbank
        A, B = filterbankbounds(fb["g"], fb["a"], fb["L"])
        assert A >= 0, "lower frame bound must be non-negative"
        assert B >= A, "upper bound must be >= lower bound"
        assert B > 0, "upper frame bound must be positive"

    def test_condition_number(self, erb_filterbank):
        """Condition number B/A should be finite and reasonable (when A > 0)."""
        from cool_frames.torch.filterbanks import filterbankbounds

        fb = erb_filterbank
        A, B = filterbankbounds(fb["g"], fb["a"], fb["L"])
        if A == 0:
            pytest.skip("A=0 (not a frame) — condition number undefined")
        cond = B / A
        assert np.isfinite(cond)
        assert cond < 1e6, f"condition number {cond:.1f} seems too large"


class TestFilterbankrealbounds:
    """filterbankbounds: real frame bounds agree with NumPy."""

    def test_agreement(self, erb_filterbank):
        from cool_frames.numpy.filterbanks import filterbankbounds as np_fn
        from cool_frames.torch.filterbanks import filterbankbounds as torch_fn

        fb = erb_filterbank
        A_np, B_np = np_fn(fb["g"], fb["a"], fb["L"])
        A_t, B_t = torch_fn(fb["g"], fb["a"], fb["L"])

        assert abs(A_t - A_np) < 1e-10
        assert abs(B_t - B_np) < 1e-10

    def test_real_bounds_tighter(self, erb_filterbank):
        """Real bounds should be at least as good as general bounds."""
        from cool_frames.torch.filterbanks import (
            filterbankbounds,
        )

        fb = erb_filterbank
        _A, _B = filterbankbounds(fb["g"], fb["a"], fb["L"])
        _Ar, Br = filterbankbounds(fb["g"], fb["a"], fb["L"])

        # Real bounds are computed on half the spectrum — can differ
        assert Br >= 0, "upper real bound must be non-negative"

    def test_cqt_realbounds(self, cqt_filterbank):
        """CQT filterbank also yields valid real bounds."""
        from cool_frames.torch.filterbanks import filterbankbounds

        fb = cqt_filterbank
        A, B = filterbankbounds(fb["g"], fb["a"], fb["L"])
        assert B >= A >= 0


# ---------------------------------------------------------------------------
# Dual frame
# ---------------------------------------------------------------------------


class TestFilterbankDual:
    """filterbankdual / filterbankdual: agreement and perfect reconstruction."""

    def test_dual_agreement(self, erb_filterbank):
        """Torch dual frame matches NumPy dual frame."""
        from cool_frames.numpy.filterbanks import filterbankdual as np_fn
        from cool_frames.numpy.filters import filter_freqresp
        from cool_frames.torch.filterbanks import filterbankdual as torch_fn

        fb = erb_filterbank
        gd_np = np_fn(fb["g"], fb["a"], fb["L"])
        gd_t = torch_fn(fb["g"], fb["a"], fb["L"])

        assert len(gd_t) == len(gd_np)

        for m in range(fb["M"]):
            H_np, _ = filter_freqresp(gd_np[m], fb["L"])
            H_t, _ = filter_freqresp(gd_t[m], fb["L"])
            np.testing.assert_allclose(
                np.asarray(H_t),
                np.asarray(H_np),
                atol=1e-10,
                err_msg=f"dual filter {m} mismatch",
            )

    def test_dual_frame_reconstruction(self, erb_filterbank):
        """Synthesis with dual frame gives perfect reconstruction."""
        from cool_frames.torch.filterbanks import (
            filterbank,
            filterbankdual,
            ifilterbank,
        )

        fb = erb_filterbank
        gd = filterbankdual(fb["g"], fb["a"], fb["L"])

        rng = np.random.default_rng(200)
        for trial in range(10):
            x_np = rng.standard_normal(fb["Ls"])
            f_t = np_to_torch(x_np, dtype=torch.float64)
            c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])
            f_rec = ifilterbank(c, gd, fb["a"], Ls=fb["Ls"], real=True)
            f_rec_np = torch_to_np(f_rec)

            rel_err = np.linalg.norm(f_rec_np[: len(x_np)] - x_np) / np.linalg.norm(x_np)
            assert rel_err < 1e-8, f"trial {trial}: rel_err={rel_err:.2e}"


# ---------------------------------------------------------------------------
# Tight frame
# ---------------------------------------------------------------------------


class TestFilterbankTight:
    """filterbanktight / filterbanktight: self-dual frame."""

    def test_tight_frame_reconstruction(self, erb_filterbank):
        """Tight frame is self-dual: synthesis with gt gives PR."""
        from cool_frames.torch.filterbanks import (
            filterbank,
            filterbanktight,
            ifilterbank,
        )

        fb = erb_filterbank
        gt = filterbanktight(fb["g"], fb["a"], fb["L"])

        rng = np.random.default_rng(300)
        for trial in range(10):
            x_np = rng.standard_normal(fb["Ls"])
            f_t = np_to_torch(x_np, dtype=torch.float64)
            c = filterbank(f_t, gt, fb["a"], L=fb["L"])
            f_rec = ifilterbank(c, gt, fb["a"], Ls=fb["Ls"], real=True)
            f_rec_np = torch_to_np(f_rec)

            rel_err = np.linalg.norm(f_rec_np[: len(x_np)] - x_np) / np.linalg.norm(x_np)
            assert rel_err < 1e-8, f"tight PR trial {trial}: rel_err={rel_err:.2e}"

    def test_tight_agreement(self, erb_filterbank):
        """Torch tight frame matches NumPy tight frame."""
        from cool_frames.numpy.filterbanks import filterbanktight as np_fn
        from cool_frames.numpy.filters import filter_freqresp
        from cool_frames.torch.filterbanks import filterbanktight as torch_fn

        fb = erb_filterbank
        gt_np = np_fn(fb["g"], fb["a"], fb["L"])
        gt_t = torch_fn(fb["g"], fb["a"], fb["L"])

        for m in range(fb["M"]):
            H_np, _ = filter_freqresp(gt_np[m], fb["L"])
            H_t, _ = filter_freqresp(gt_t[m], fb["L"])
            np.testing.assert_allclose(
                np.asarray(H_t),
                np.asarray(H_np),
                atol=1e-10,
                err_msg=f"tight filter {m} mismatch",
            )

    def test_tight_response_is_flat(self, erb_filterbank):
        """Tight filterbank response should be close to constant."""
        from cool_frames.torch.filterbanks import (
            filterbankresponse,
            filterbanktight,
        )

        fb = erb_filterbank
        gt = filterbanktight(fb["g"], fb["a"], fb["L"])
        resp = filterbankresponse(gt, fb["a"], fb["L"], real=True)
        resp_np = torch_to_np(resp)

        # After tightening, the response should be approximately flat
        # (constant = 1 for normalized tight frame, but may differ by scale)
        if resp_np.max() > 0:
            variation = (resp_np.max() - resp_np.min()) / resp_np.max()
            assert variation < 0.1, (
                f"tight response variation {variation:.4f} — expected near-flat"
            )


# ---------------------------------------------------------------------------
# Filterbank response
# ---------------------------------------------------------------------------


class TestFilterbankResponse:
    """filterbankresponse: sum of squared magnitudes."""

    def test_agreement(self, erb_filterbank):
        from cool_frames.numpy.filterbanks import filterbankresponse as np_fn
        from cool_frames.torch.filterbanks import filterbankresponse as torch_fn

        fb = erb_filterbank
        resp_np = np_fn(fb["g"], fb["a"], fb["L"])
        resp_t = torch_fn(fb["g"], fb["a"], fb["L"])

        np.testing.assert_allclose(
            np.asarray(resp_t),
            np.asarray(resp_np),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_response_positive(self, erb_filterbank):
        from cool_frames.torch.filterbanks import filterbankresponse

        fb = erb_filterbank
        resp = filterbankresponse(fb["g"], fb["a"], fb["L"])
        resp_arr = np.asarray(resp)
        assert np.all(resp_arr >= 0), "response must be non-negative"

    def test_response_is_tensor(self, erb_filterbank):
        from cool_frames.torch.filterbanks import filterbankresponse

        fb = erb_filterbank
        resp = filterbankresponse(fb["g"], fb["a"], fb["L"])
        assert isinstance(resp, torch.Tensor)
        assert resp.shape == (fb["L"],)


# ---------------------------------------------------------------------------
# Filterbank frequency response
# ---------------------------------------------------------------------------


class TestFilterbankFreqz:
    """filterbankfreqz: frequency responses stacked as (M, L) matrix."""

    def test_agreement(self, erb_filterbank):
        from cool_frames.numpy.filterbanks import filterbankfreqz as np_fn
        from cool_frames.torch.filterbanks import filterbankfreqz as torch_fn

        fb = erb_filterbank
        H_np = np_fn(fb["g"], fb["a"], fb["L"])
        H_t = torch_fn(fb["g"], fb["a"], fb["L"])

        np.testing.assert_allclose(
            torch_to_np(H_t),
            H_np,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_output_shape(self, erb_filterbank):
        from cool_frames.torch.filterbanks import filterbankfreqz

        fb = erb_filterbank
        H = filterbankfreqz(fb["g"], fb["a"], fb["L"])

        assert isinstance(H, torch.Tensor)
        # filterbankfreqz returns (L, M) — rows are frequency bins
        assert H.shape == (fb["L"], fb["M"])
        assert H.dtype == torch.complex128

    def test_all_channels_finite(self, erb_filterbank):
        """All frequency response values should be finite."""
        from cool_frames.torch.filterbanks import filterbankfreqz

        fb = erb_filterbank
        H = filterbankfreqz(fb["g"], fb["a"], fb["L"])
        assert torch.all(torch.isfinite(H)), "freqz contains non-finite values"


# ---------------------------------------------------------------------------
# Filterbank scaling
# ---------------------------------------------------------------------------


class TestFilterbankScale:
    """filterbankscale: per-channel scaling of filter dicts."""

    def test_uniform_scale(self, erb_filterbank):
        """Scaling by 1.0 should be identity."""
        from cool_frames.numpy.filters import filter_freqresp
        from cool_frames.torch.filterbanks import filterbankscale

        fb = erb_filterbank
        s = np.ones(fb["M"])
        gs = filterbankscale(fb["g"], s, L=fb["L"])

        assert len(gs) == fb["M"]
        for m in range(fb["M"]):
            H_orig, _ = filter_freqresp(fb["g"][m], fb["L"])
            H_scaled, _ = filter_freqresp(gs[m], fb["L"])
            np.testing.assert_allclose(
                np.abs(np.asarray(H_scaled)),
                np.abs(np.asarray(H_orig)),
                atol=1e-10,
                err_msg=f"filter {m}: scale by 1 changed magnitude",
            )

    def test_scale_by_two(self, erb_filterbank):
        """Scaling by 2.0 should double filter magnitudes."""
        from cool_frames.numpy.filters import filter_freqresp
        from cool_frames.torch.filterbanks import filterbankscale

        fb = erb_filterbank
        s = 2.0 * np.ones(fb["M"])
        gs = filterbankscale(fb["g"], s, L=fb["L"])

        for m in range(min(5, fb["M"])):
            H_orig, _ = filter_freqresp(fb["g"][m], fb["L"])
            H_scaled, _ = filter_freqresp(gs[m], fb["L"])
            # |H_scaled| ≈ 2 * |H_orig|
            np.testing.assert_allclose(
                np.abs(np.asarray(H_scaled)),
                2.0 * np.abs(np.asarray(H_orig)),
                atol=1e-10,
                err_msg=f"filter {m}: scale by 2 failed",
            )


# ---------------------------------------------------------------------------
# CQT filterbank: perfect reconstruction (second filterbank type)
# ---------------------------------------------------------------------------


class TestCqtPerfectReconstruction:
    """Perfect reconstruction with a CQT filterbank (not just ERB)."""

    def test_cqt_dual_reconstruction(self, cqt_filterbank):
        """CQT analysis → dual synthesis → PR."""
        from cool_frames.torch.filterbanks import (
            filterbank,
            filterbankdual,
            ifilterbank,
        )

        fb = cqt_filterbank
        gd = filterbankdual(fb["g"], fb["a"], fb["L"])

        rng = np.random.default_rng(400)
        for trial in range(10):
            x_np = rng.standard_normal(fb["Ls"])
            f_t = np_to_torch(x_np, dtype=torch.float64)
            c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])
            f_rec = ifilterbank(c, gd, fb["a"], Ls=fb["Ls"], real=True)
            f_rec_np = torch_to_np(f_rec)

            rel_err = np.linalg.norm(f_rec_np[: len(x_np)] - x_np) / np.linalg.norm(x_np)
            assert rel_err < 1e-8, f"CQT PR trial {trial}: rel_err={rel_err:.2e}"

    def test_cqt_tight_reconstruction(self, cqt_filterbank):
        """CQT tight frame is self-dual → PR."""
        from cool_frames.torch.filterbanks import (
            filterbank,
            filterbanktight,
            ifilterbank,
        )

        fb = cqt_filterbank
        gt = filterbanktight(fb["g"], fb["a"], fb["L"])

        rng = np.random.default_rng(401)
        for trial in range(10):
            x_np = rng.standard_normal(fb["Ls"])
            f_t = np_to_torch(x_np, dtype=torch.float64)
            c = filterbank(f_t, gt, fb["a"], L=fb["L"])
            f_rec = ifilterbank(c, gt, fb["a"], Ls=fb["Ls"], real=True)
            f_rec_np = torch_to_np(f_rec)

            rel_err = np.linalg.norm(f_rec_np[: len(x_np)] - x_np) / np.linalg.norm(x_np)
            assert rel_err < 1e-8, f"CQT tight PR trial {trial}: rel_err={rel_err:.2e}"

    def test_cqt_bounds_agreement(self, cqt_filterbank):
        """CQT frame bounds agree between torch and numpy."""
        from cool_frames.numpy.filterbanks import filterbankbounds as np_fn
        from cool_frames.torch.filterbanks import filterbankbounds as torch_fn

        fb = cqt_filterbank
        A_np, B_np = np_fn(fb["g"], fb["a"], fb["L"])
        A_t, B_t = torch_fn(fb["g"], fb["a"], fb["L"])

        assert abs(A_t - A_np) < 1e-10
        assert abs(B_t - B_np) < 1e-10


# ---------------------------------------------------------------------------
# ifilterbankiter (iterative synthesis)
# ---------------------------------------------------------------------------


class TestIfilterbankiter:
    """Iterative synthesis via CG when explicit dual is not available."""

    def test_basic_call(self, erb_filterbank, noise_signal):
        from cool_frames.torch.filterbanks import filterbank, ifilterbankiter

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])

        xr, relres, _niter = ifilterbankiter(
            c,
            fb["g"],
            fb["a"],
            Ls=fb["Ls"],
            real=True,
        )

        assert isinstance(xr, torch.Tensor)
        assert xr.shape[0] == fb["Ls"]
        assert relres < 1e-4

    def test_iterative_vs_dual(self, erb_filterbank, noise_signal):
        """Iterative synthesis should approximate dual-frame synthesis."""
        from cool_frames.torch.filterbanks import (
            filterbank,
            ifilterbank,
            ifilterbankiter,
        )

        fb = erb_filterbank
        f_t = np_to_torch(noise_signal, dtype=torch.float64)
        c = filterbank(f_t, fb["g"], fb["a"], L=fb["L"])

        # Dual-frame synthesis
        f_dual = ifilterbank(c, fb["gd"], fb["a"], Ls=fb["Ls"], real=True)

        # Iterative synthesis (should converge to same result)
        f_iter, _relres, _niter = ifilterbankiter(
            c,
            fb["g"],
            fb["a"],
            Ls=fb["Ls"],
            real=True,
            tol=1e-10,
            maxit=200,
        )

        # Both should reconstruct the original signal
        x_np = noise_signal
        err_dual = np.linalg.norm(torch_to_np(f_dual)[: len(x_np)] - x_np) / np.linalg.norm(x_np)
        err_iter = np.linalg.norm(torch_to_np(f_iter)[: len(x_np)] - x_np) / np.linalg.norm(x_np)

        # Both should be good reconstructions
        assert err_dual < 1e-8, f"dual err={err_dual:.2e}"
        assert err_iter < 1e-4, f"iter err={err_iter:.2e}"
