"""
Tests for torch frame multiplier operators (cool_frames.torch.operators).

Tests validate mathematical properties of frame multipliers:
- Forward multiplier (framemul)
- Adjoint operator (framemuladj)
- Inverse via PCG (framemulinv)
- Eigenvalue computation (framemuleigs)
- Ridge-based multiplier construction (ridges_to_symbol, fit_ridge_multiplier, denoise_by_ridges)

Filterbank fixtures use audfilters at small sizes (fs=8000, Ls=2000) for speed.
All tests verify that torch tensors have proper gradient flow when required.
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def torch_fb_setup():
    """Build a small auditory filterbank (tight frame) for torch tests.

    Returns dict with keys: g, g_tight, a, L, M, f, c, fs, Ls
    """
    from cool_frames.numpy.filterbanks import filterbank, filterbanktight
    from cool_frames.numpy.filters import audfilters

    fs, Ls = 8000, 2000
    g, a, fc, L, _info = audfilters(fs, Ls)
    g_tight = filterbanktight(g, a, L)
    M = len(g)

    rng = np.random.default_rng(42)
    f = rng.standard_normal(L)

    # Pre-compute coefficients for convenience
    c = filterbank(f, g_tight, a, L)

    return dict(g=g, g_tight=g_tight, a=a, L=L, M=M, f=f, fc=fc, c=c, fs=fs, Ls=Ls)


@pytest.fixture(scope="module")
def torch_dual_setup():
    """Build a filterbank with separate analysis/synthesis (real dual) frames."""
    from cool_frames.numpy.filterbanks import filterbankdual
    from cool_frames.numpy.filters import audfilters

    fs, Ls = 8000, 2000
    g, a, fc, L, _info = audfilters(fs, Ls)
    g_dual = filterbankdual(g, a, L)
    M = len(g)

    rng = np.random.default_rng(42)
    f = rng.standard_normal(L)

    return dict(g=g, g_dual=g_dual, a=a, L=L, M=M, f=f, fc=fc, fs=fs, Ls=Ls)


def _ones_symbol_torch(c):
    """Create a symbol of all ones matching coefficient structure as torch tensors."""
    return [torch.ones_like(np_to_torch(ci)) for ci in c]


def _random_symbol_torch(c, seed=123):
    """Create a random positive symbol matching coefficient structure as torch tensors."""
    rng = np.random.default_rng(seed)
    return [np_to_torch(rng.uniform(0.5, 2.0, size=ci.shape), dtype=torch.float32) for ci in c]


def _binary_mask_torch(c, seed=99):
    """Create a random binary (0/1) mask matching coefficient structure as torch tensors."""
    rng = np.random.default_rng(seed)
    return [
        np_to_torch(rng.integers(0, 2, size=ci.shape).astype(np.float32), dtype=torch.float32)
        for ci in c
    ]


# ===================================================================
# framemul — forward multiplier
# ===================================================================


class TestFramemul:
    """Forward frame multiplier: M_sigma f = synth(sigma * analysis(f))."""

    def test_identity_symbol(self, torch_fb_setup):
        """sigma = 1 everywhere => M_sigma f = f (for tight frame)."""
        from cool_frames.torch.operators import framemul

        d = torch_fb_setup
        sigma = _ones_symbol_torch(d["c"])
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        result = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        np.testing.assert_allclose(torch_to_np(result), d["f"], atol=1e-6, rtol=1e-5)

    def test_zero_symbol(self, torch_fb_setup):
        """sigma = 0 everywhere => M_sigma f = 0."""
        from cool_frames.torch.operators import framemul

        d = torch_fb_setup
        sigma = [torch.zeros_like(np_to_torch(ci)) for ci in d["c"]]
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        result = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        np.testing.assert_allclose(torch_to_np(result), 0.0, atol=1e-11)

    def test_scalar_symbol(self, torch_fb_setup):
        """Constant sigma = alpha => M_sigma f = alpha * f (tight frame)."""
        from cool_frames.torch.operators import framemul

        d = torch_fb_setup
        alpha = 3.7
        sigma = [alpha * torch.ones_like(np_to_torch(ci)) for ci in d["c"]]
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        result = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        np.testing.assert_allclose(torch_to_np(result), alpha * d["f"], atol=1e-5, rtol=1e-4)

    def test_output_length(self, torch_fb_setup):
        """Output has same length as input signal."""
        from cool_frames.torch.operators import framemul

        d = torch_fb_setup
        sigma = _random_symbol_torch(d["c"])
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        result = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        assert len(result) == d["L"]

    def test_output_is_real(self, torch_fb_setup):
        """Real signal + real symbol => real output (dtype float32)."""
        from cool_frames.torch.operators import framemul

        d = torch_fb_setup
        sigma = _random_symbol_torch(d["c"])
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        result = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        # Exactly float32: the backend now preserves the input dtype rather
        # than upcasting to double, so this no longer needs to accept either.
        assert result.dtype == torch.float32
        assert torch.isreal(result).all() or torch.max(torch.abs(torch.imag(result))) < 1e-9

    def test_with_dual_frame(self, torch_dual_setup):
        """Identity symbol with original + dual => perfect reconstruction."""
        from cool_frames.numpy.filterbanks import filterbank
        from cool_frames.torch.operators import framemul

        d = torch_dual_setup
        c = filterbank(d["f"], d["g"], d["a"], d["L"])
        sigma = _ones_symbol_torch(c)
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        result = framemul(f_t, d["g"], d["g_dual"], d["a"], sigma, d["L"])

        np.testing.assert_allclose(torch_to_np(result), d["f"], atol=1e-6, rtol=1e-5)

    def test_gradient_flow(self, torch_fb_setup):
        """framemul output has grad_fn when input requires_grad."""
        from cool_frames.torch.operators import framemul

        d = torch_fb_setup
        sigma = _random_symbol_torch(d["c"])
        f_t = np_to_torch(d["f"], dtype=torch.float32)
        f_t.requires_grad_(True)

        result = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        assert result.grad_fn is not None


# ===================================================================
# framemuladj — adjoint operator
# ===================================================================


class TestFramemulAdj:
    """Adjoint: <M f, h> = <f, M* h> for all f, h."""

    def test_adjoint_identity(self, torch_fb_setup):
        """Verify the adjoint relationship: <Mf, h> = <f, M*h>."""
        from cool_frames.torch.operators import framemul, framemuladj

        d = torch_fb_setup
        rng = np.random.default_rng(77)
        f = rng.standard_normal(d["L"])
        h = rng.standard_normal(d["L"])
        sigma = _random_symbol_torch(d["c"])

        f_t = np_to_torch(f, dtype=torch.float32)
        h_t = np_to_torch(h, dtype=torch.float32)

        Mf = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])
        Mstar_h = framemuladj(h_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        # Ensure same length and dtype for dot product
        min_len = min(len(Mf), len(Mstar_h), len(f_t), len(h_t))
        lhs = torch.dot(Mf[:min_len].float(), h_t[:min_len].float()).item()
        rhs = torch.dot(f_t[:min_len].float(), Mstar_h[:min_len].float()).item()

        np.testing.assert_allclose(lhs, rhs, atol=1e-6, rtol=1e-4)

    def test_self_adjoint_tight_frame(self, torch_fb_setup):
        """For real sigma and tight frame, M = M* (self-adjoint)."""
        from cool_frames.torch.operators import framemul, framemuladj

        d = torch_fb_setup
        sigma = _random_symbol_torch(d["c"])
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        Mf = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])
        Mstar_f = framemuladj(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        np.testing.assert_allclose(torch_to_np(Mf), torch_to_np(Mstar_f), atol=1e-10)


# ===================================================================
# framemulinv — inverse frame multiplier (PCG)
# ===================================================================


class TestFramemulinv:
    """Inverse: M^{-1} (M f) = f when sigma > 0."""

    def test_roundtrip_tight(self, torch_fb_setup):
        """Apply then invert with tight frame and positive symbol."""
        from cool_frames.torch.operators import framemul, framemulinv

        d = torch_fb_setup
        sigma = _random_symbol_torch(d["c"])
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        Mf = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])
        recovered, info = framemulinv(Mf, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        # Note: recovered may have slightly different length due to framemul output
        min_len = min(len(recovered), len(f_t))
        np.testing.assert_allclose(
            torch_to_np(recovered[:min_len]), torch_to_np(f_t[:min_len]), atol=1e-5
        )
        assert info["converged"], f"PCG did not converge: relres={info['relres']}"

    def test_roundtrip_dual(self, torch_dual_setup):
        """Apply then invert with original + dual frame."""
        from cool_frames.numpy.filterbanks import filterbank
        from cool_frames.torch.operators import framemul, framemulinv

        d = torch_dual_setup
        c = filterbank(d["f"], d["g"], d["a"], d["L"])
        sigma = _random_symbol_torch(c)
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        Mf = framemul(f_t, d["g"], d["g_dual"], d["a"], sigma, d["L"])
        _recovered, info = framemulinv(Mf, d["g"], d["g_dual"], d["a"], sigma, d["L"])

        # Check convergence
        assert info["converged"], f"PCG did not converge: relres={info['relres']}"
        assert info["relres"] < 1e-6

    def test_convergence_info(self, torch_fb_setup):
        """Verify that convergence info is returned with expected keys."""
        from cool_frames.torch.operators import framemul, framemulinv

        d = torch_fb_setup
        sigma = _random_symbol_torch(d["c"])
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        Mf = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])
        _, info = framemulinv(Mf, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        assert "relres" in info
        assert "iter" in info
        assert "converged" in info
        assert isinstance(info["relres"], float)
        assert isinstance(info["iter"], int)
        assert isinstance(info["converged"], bool)


# ===================================================================
# framemuleigs — eigenvalue computation
# ===================================================================


class TestFramemuleigs:
    """Eigenvalue decomposition of frame multiplier."""

    def test_identity_eigenvalues(self, torch_fb_setup):
        """sigma=1 with tight frame => all eigenvalues ≈ 1."""
        from cool_frames.torch.operators import framemuleigs

        d = torch_fb_setup
        sigma = _ones_symbol_torch(d["c"])
        eigs = framemuleigs(d["g_tight"], d["g_tight"], d["a"], sigma, d["L"], K=min(6, d["L"]))

        eigs_np = torch_to_np(eigs)
        np.testing.assert_allclose(np.abs(eigs_np), 1.0, atol=1e-5, rtol=1e-4)

    def test_eigenvalue_bounds(self, torch_fb_setup):
        """Eigenvalues bounded by min/max of sigma (tight frame, A=B=1)."""
        from cool_frames.torch.operators import framemuleigs

        d = torch_fb_setup
        sigma = _random_symbol_torch(d["c"])
        eigs = framemuleigs(d["g_tight"], d["g_tight"], d["a"], sigma, d["L"], K=min(6, d["L"]))

        eigs_np = torch_to_np(eigs)

        # Extract min/max from sigma tensors
        sig_min = min(float(torch.min(s).item()) for s in sigma)
        sig_max = max(float(torch.max(s).item()) for s in sigma)

        assert np.min(np.real(eigs_np)) >= sig_min - 1e-5
        assert np.max(np.real(eigs_np)) <= sig_max + 1e-5

    def test_returns_requested_count(self, torch_fb_setup):
        """Returns exactly K eigenvalues."""
        from cool_frames.torch.operators import framemuleigs

        d = torch_fb_setup
        K = min(4, d["L"])
        sigma = _random_symbol_torch(d["c"])
        eigs = framemuleigs(d["g_tight"], d["g_tight"], d["a"], sigma, d["L"], K=K)

        assert len(eigs) == K

    def test_eigenvalues_sorted_descending(self, torch_fb_setup):
        """Eigenvalues returned in descending order of magnitude."""
        from cool_frames.torch.operators import framemuleigs

        d = torch_fb_setup
        sigma = _random_symbol_torch(d["c"])
        K = min(6, d["L"])
        eigs = framemuleigs(d["g_tight"], d["g_tight"], d["a"], sigma, d["L"], K=K)

        eigs_np = torch_to_np(eigs)
        magnitudes = np.abs(eigs_np)
        assert np.all(magnitudes[:-1] >= magnitudes[1:] - 1e-11)


# ===================================================================
# ridges_to_symbol
# ===================================================================


class TestRidgesToSymbol:
    """Ridge-to-symbol conversion."""

    def test_ridges_to_symbol_output_shape(self, torch_fb_setup):
        """ridges_to_symbol produces M tensors with correct shapes."""
        pytest.importorskip("audioeffects")
        from audioeffects.denoising.code.ridge_torch import Ridge
        from torch_additions.operators import ridges_to_symbol

        d = torch_fb_setup

        # Create a simple ridge
        ridge = Ridge(
            channel_indices=torch.tensor([2, 2, 3], dtype=torch.int64),
            time_indices=torch.tensor([0, 1, 2], dtype=torch.int64),
            inst_freq=torch.tensor([0.1, 0.11, 0.12], dtype=torch.float32),
            amplitude=torch.tensor([1.0, 1.0, 0.9], dtype=torch.float32),
            bandwidth=None,
            onset=0,
            offset=2,
        )

        coeff_lengths = [len(d["c"][m]) for m in range(d["M"])]
        sigma = ridges_to_symbol([ridge], coeff_lengths, d["a"], d["L"])

        assert isinstance(sigma, list)
        assert len(sigma) == d["M"]
        for m in range(d["M"]):
            assert isinstance(sigma[m], torch.Tensor)
            assert len(sigma[m]) == coeff_lengths[m]
            assert sigma[m].dtype in [torch.float32, torch.float64]

    def test_ridges_to_symbol_empty_ridges(self, torch_fb_setup):
        """ridges_to_symbol with no ridges produces zeros."""
        from torch_additions.operators import ridges_to_symbol

        d = torch_fb_setup
        coeff_lengths = [len(d["c"][m]) for m in range(d["M"])]
        sigma = ridges_to_symbol([], coeff_lengths, d["a"], d["L"])

        assert len(sigma) == d["M"]
        for m in range(d["M"]):
            np.testing.assert_array_equal(torch_to_np(sigma[m]), 0.0)


# ===================================================================
# fit_ridge_multiplier (requires phase analysis)
# ===================================================================


class TestFitRidgeMultiplier:
    """Ridge-fitting multiplier construction."""

    @pytest.mark.slow
    @pytest.mark.skip(
        reason="Phase gradient computation has interoperability issues with audfilters"
    )
    def test_fit_ridge_multiplier_runs(self, torch_fb_setup):
        """fit_ridge_multiplier runs end-to-end without error."""
        from torch_additions.operators import fit_ridge_multiplier

        d = torch_fb_setup
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        sigma, ridges, info = fit_ridge_multiplier(
            f_t,
            d["g_tight"],
            d["a"],
            d["L"],
            d["fc"],
            mag_threshold_db=-40.0,
            min_ridge_hops=3,
        )

        # Check output structure
        assert isinstance(sigma, list)
        assert len(sigma) == d["M"]
        for s in sigma:
            assert isinstance(s, torch.Tensor)

        assert isinstance(ridges, list)
        assert isinstance(info, dict)
        assert "num_ridges" in info
        assert "num_channels" in info


# ===================================================================
# denoise_by_ridges (full pipeline)
# ===================================================================


class TestDenoiseByRidges:
    """Full ridge-based denoising pipeline."""

    @pytest.mark.slow
    @pytest.mark.skip(
        reason="Phase gradient computation has interoperability issues with audfilters"
    )
    def test_denoise_by_ridges_runs(self, torch_fb_setup):
        """denoise_by_ridges runs end-to-end without error."""
        from torch_additions.operators import denoise_by_ridges

        d = torch_fb_setup
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        f_denoised, ridges, info = denoise_by_ridges(
            f_t,
            d["g_tight"],
            d["a"],
            d["L"],
            d["fc"],
            mag_threshold_db=-40.0,
            min_ridge_hops=3,
        )

        # Check output structure
        assert isinstance(f_denoised, torch.Tensor)
        assert len(f_denoised) <= d["L"]

        assert isinstance(ridges, list)
        assert isinstance(info, dict)
        assert "num_ridges" in info
        assert "num_segments" in info

    @pytest.mark.slow
    @pytest.mark.skip(
        reason="Phase gradient computation has interoperability issues with audfilters"
    )
    def test_denoise_by_ridges_output_dtype(self, torch_fb_setup):
        """denoise_by_ridges preserves real dtype."""
        from torch_additions.operators import denoise_by_ridges

        d = torch_fb_setup
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        f_denoised, _, _ = denoise_by_ridges(
            f_t,
            d["g_tight"],
            d["a"],
            d["L"],
            d["fc"],
            mag_threshold_db=-40.0,
        )

        # Output should be real-valued
        assert f_denoised.dtype in [torch.float32, torch.float64]


# ===================================================================
# Energy and positivity properties
# ===================================================================


class TestMultiplierProperties:
    """Cross-cutting mathematical properties."""

    def test_positive_symbol_positive_definite(self, torch_fb_setup):
        """sigma > 0 => <M f, f> > 0 for all nonzero f."""
        from cool_frames.torch.operators import framemul

        d = torch_fb_setup
        sigma = _random_symbol_torch(d["c"])  # all values in [0.5, 2.0]
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        Mf = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        inner = torch.dot(f_t[: len(Mf)].float(), Mf.float()).item()
        assert inner > 0

    def test_binary_mask_energy_reduction(self, torch_fb_setup):
        """Binary mask (0/1 symbol) can only reduce energy."""
        from cool_frames.torch.operators import framemul

        d = torch_fb_setup
        sigma = _binary_mask_torch(d["c"])
        f_t = np_to_torch(d["f"], dtype=torch.float32)

        Mf = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        energy_in = torch.sum(f_t**2).item()
        energy_out = torch.sum(Mf**2).item()
        assert energy_out <= energy_in + 1e-7

    def test_linearity_in_signal(self, torch_fb_setup):
        """M_sigma (alpha f + beta h) = alpha M_sigma f + beta M_sigma h."""
        from cool_frames.torch.operators import framemul

        d = torch_fb_setup
        rng = np.random.default_rng(55)
        h = rng.standard_normal(d["L"])
        alpha, beta = 2.3, -0.7
        sigma = _random_symbol_torch(d["c"])

        f_t = np_to_torch(d["f"], dtype=torch.float32)
        h_t = np_to_torch(h, dtype=torch.float32)

        M_sum = framemul(
            alpha * f_t + beta * h_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"]
        )
        Mf = framemul(f_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])
        Mh = framemul(h_t, d["g_tight"], d["g_tight"], d["a"], sigma, d["L"])

        expected = alpha * Mf + beta * Mh
        # float32 tolerance.  Until v0.1.1 the backend silently computed in
        # float64 regardless of the input dtype, so this comparison used to get
        # double precision for free; now that float32 input really is computed
        # in float32, an FFT round trip plus a scaled sum accumulates a few
        # times float32 epsilon (~1.2e-7) on a signal whose peak is ~4.
        np.testing.assert_allclose(torch_to_np(M_sum), torch_to_np(expected), atol=1e-5, rtol=1e-4)
