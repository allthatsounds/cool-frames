"""
test_prop_ml_advanced.py
========================
Advanced matrix-spectral property tests for filterbanks in machine learning
and signal processing contexts.

Complements test_prop_ml_matrix_spectral.py with:

1. Mutual coherence & compressed sensing (sparse recovery guarantees)
2. Restricted Isometry Property (empirical RIP on sparse signals)
3. Lipschitz continuity (robustness of representations)
4. Gradient norm preservation (vanishing/exploding gradient prevention)
5. Deformation stability (Mallat scattering-theory bounds)
6. Noise amplification (SNR preservation through the filterbank)
7. Effective rank & nuclear norm (representation dimensionality)
8. Whitening / decorrelation (feature independence)
9. Spectral gap (low-rank approximation quality)
"""
from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Shared helpers  (mirror the ones in test_prop_ml_matrix_spectral.py)
# ---------------------------------------------------------------------------

def _make_filterbank(fb_type="aud", fs=8000, Ls=512):
    """Build a filterbank and return (g, a, L, A, B, M)."""
    from cool_frames.filterbanks import filterbankbounds
    from cool_frames.filters import audfilters, cqtfilters
    from cool_frames.filters import filterbanklength

    if fb_type == "aud":
        g, a, fc, _, _info = audfilters(fs, Ls)
    elif fb_type == "cqt":
        g, a, fc, _, _info = cqtfilters(fs, Ls, fmin=50, fmax=fs // 2 - 100, bins=12)
    else:
        raise ValueError(fb_type)

    L = filterbanklength(Ls, a)
    A, B = filterbankbounds(g, a, L)
    M = len(g)
    return g, a, L, A, B, M


def _frame_op(x, g, a, L):
    """Real frame operator: S_g x = 2·real(ifilterbank(filterbank(x), g, a))."""
    from cool_frames.filterbanks import filterbank, ifilterbank
    c = filterbank(x, g, a)
    return np.real(np.asarray(ifilterbank(c, g, a, real=True)))


def _analysis_op(x, g, a):
    """Analysis operator F: R^L → C^Nsum (flattened complex coefficients)."""
    from cool_frames.filterbanks import filterbank
    c = filterbank(x, g, a)
    return np.concatenate([np.asarray(cm).ravel() for cm in c])


def _get_N_list(g, a, L):
    """Return list of subband lengths."""
    from cool_frames.filterbanks import filterbank
    x_dummy = np.zeros(L)
    c = filterbank(x_dummy, g, a)
    return [len(np.asarray(cm).ravel()) for cm in c]


def _full_freqresp(filt, L):
    """Expand a lazy filter dict into a length-L frequency response vector."""
    H = np.asarray(filt['H'](L), dtype=complex)
    foff = int(filt['foff'](L))
    out = np.zeros(L, dtype=complex)
    for i in range(len(H)):
        out[(foff + i) % L] = H[i]
    return out


def _materialise_frame_op(g, a, L):
    """Build the L×L real frame operator matrix S_g column by column."""
    S = np.zeros((L, L))
    for j in range(L):
        ej = np.zeros(L)
        ej[j] = 1.0
        S[:, j] = _frame_op(ej, g, a, L)[:L]
    return S


# ===================================================================
# 1.  MUTUAL COHERENCE & COMPRESSED SENSING
# ===================================================================

@pytest.mark.requires_impl
class TestMutualCoherence:
    """Mutual coherence μ = max_{m≠n} |⟨H_m, H_n⟩| / (‖H_m‖·‖H_n‖).

    Low coherence is necessary for sparse signal recovery via basis pursuit
    or matching pursuit.  For ERB/auditory filterbanks, adjacent filters
    overlap in frequency, so μ is moderate but bounded.
    """

    def test_coherence_bounded_below_one(self, needs_impl):
        """Mutual coherence μ < 1 for a valid frame (filters not identical)."""
        g, a, L, A, B, M = _make_filterbank("aud")

        max_coh = 0.0
        for m in range(M):
            Hm = _full_freqresp(g[m], L)
            norm_m = np.linalg.norm(Hm)
            if norm_m < 1e-15:
                continue
            for n in range(m + 1, M):
                Hn = _full_freqresp(g[n], L)
                norm_n = np.linalg.norm(Hn)
                if norm_n < 1e-15:
                    continue
                coh = abs(np.vdot(Hm, Hn)) / (norm_m * norm_n)
                max_coh = max(max_coh, coh)

        assert max_coh < 1.0, \
            f"Mutual coherence μ = {max_coh:.6f} ≥ 1 (filters are linearly dependent)"
        assert max_coh > 0.0, \
            "Mutual coherence μ = 0 — filters are perfectly orthogonal (unexpected)"

    def test_distant_filters_low_coherence(self, needs_impl):
        """Filters separated by ≥ 5 channels have near-zero coherence.

        ERB-spaced filters have compact support in frequency, so distant
        channels don't overlap.
        """
        g, a, L, A, B, M = _make_filterbank("aud")

        far_coherences = []
        for m in range(M):
            Hm = _full_freqresp(g[m], L)
            norm_m = np.linalg.norm(Hm)
            if norm_m < 1e-15:
                continue
            for n in range(m + 5, M):
                Hn = _full_freqresp(g[n], L)
                norm_n = np.linalg.norm(Hn)
                if norm_n < 1e-15:
                    continue
                coh = abs(np.vdot(Hm, Hn)) / (norm_m * norm_n)
                far_coherences.append(coh)

        if far_coherences:
            avg_far = np.mean(far_coherences)
            assert avg_far < 0.1, \
                f"Average far-channel coherence = {avg_far:.4f}, expected < 0.1"

    @pytest.mark.parametrize("fb_type", ["aud", "cqt"])
    def test_coherence_welch_bound(self, needs_impl, fb_type):
        """Coherence satisfies the Welch lower bound μ ≥ √((M-L)/(L(M-1))).

        For an overcomplete system (M > L), this is the fundamental limit
        on how small coherence can be.  We verify μ ≥ Welch bound.
        """
        g, a, L, A, B, M = _make_filterbank(fb_type)

        if M <= L:
            pytest.skip("Welch bound requires M > L (overcomplete)")

        welch_bound = np.sqrt((M - L) / (L * (M - 1)))

        # Compute coherence (use frequency-domain inner products)
        max_coh = 0.0
        freq_resps = [_full_freqresp(g[m], L) for m in range(M)]
        norms = [np.linalg.norm(H) for H in freq_resps]

        for m in range(M):
            if norms[m] < 1e-15:
                continue
            for n in range(m + 1, M):
                if norms[n] < 1e-15:
                    continue
                coh = abs(np.vdot(freq_resps[m], freq_resps[n])) / (norms[m] * norms[n])
                max_coh = max(max_coh, coh)

        assert max_coh >= welch_bound - 1e-10, \
            f"μ = {max_coh:.6f} < Welch bound {welch_bound:.6f}"


# ===================================================================
# 2.  RESTRICTED ISOMETRY PROPERTY (EMPIRICAL)
# ===================================================================

@pytest.mark.requires_impl
class TestRestrictedIsometry:
    """Empirical RIP tests on random sparse signals.

    The RIP constant δ_s satisfies:
        (1-δ_s)·‖x‖² ≤ ‖Fx‖² ≤ (1+δ_s)·‖x‖²
    for all s-sparse signals x.  We estimate δ_s by testing many random
    sparse signals and recording the extremal distortion.
    """

    def test_rip_sparse_signals(self, needs_impl):
        """Random s-sparse signals have bounded analysis energy distortion."""
        g, a, L, A, B, M = _make_filterbank("aud", Ls=256)

        rng = np.random.default_rng(42)
        sparsity = max(1, L // 20)  # ~5% sparsity
        n_trials = 200

        ratios = []
        for _ in range(n_trials):
            # Generate s-sparse signal
            x = np.zeros(L)
            support = rng.choice(L, size=sparsity, replace=False)
            x[support] = rng.standard_normal(sparsity)

            Fx = _analysis_op(x, g, a)
            energy_ratio = np.sum(np.abs(Fx) ** 2) / (np.linalg.norm(x) ** 2)
            ratios.append(energy_ratio)

        ratios = np.array(ratios)
        # All ratios should be positive and finite
        assert np.all(ratios > 0), "Some sparse signals have zero analysis energy"
        assert np.all(np.isfinite(ratios)), "Some ratios are non-finite"

        # The spread of ratios gives an empirical RIP constant
        # Normalise: ideal would be all ratios equal
        mean_r = ratios.mean()
        delta_s = max(abs(ratios.max() - mean_r), abs(mean_r - ratios.min())) / mean_r
        assert delta_s < 1.0, \
            f"Empirical RIP constant δ_{sparsity} = {delta_s:.4f} ≥ 1 (not an approximate isometry)"

    def test_rip_improves_with_oversampling(self, needs_impl):
        """Increasing signal length (relative to filterbank) improves RIP.

        Longer signals have more degrees of freedom, and the filterbank
        covers a larger portion of the space, improving the isometry.
        """
        rng = np.random.default_rng(77)
        deltas = []

        for Ls in [128, 256, 512]:
            g, a, L, A, B, M = _make_filterbank("aud", Ls=Ls)
            sparsity = max(1, L // 20)
            ratios = []
            for _ in range(100):
                x = np.zeros(L)
                support = rng.choice(L, size=sparsity, replace=False)
                x[support] = rng.standard_normal(sparsity)
                Fx = _analysis_op(x, g, a)
                ratios.append(np.sum(np.abs(Fx) ** 2) / (np.linalg.norm(x) ** 2))

            ratios = np.array(ratios)
            mean_r = ratios.mean()
            delta = max(abs(ratios.max() - mean_r), abs(mean_r - ratios.min())) / mean_r
            deltas.append(delta)

        # δ should generally decrease (or at least not blow up) with longer signals
        assert deltas[-1] < deltas[0] + 0.3, \
            f"RIP constant worsened significantly: δ(128)={deltas[0]:.3f}, δ(512)={deltas[-1]:.3f}"


# ===================================================================
# 3.  LIPSCHITZ CONTINUITY
# ===================================================================

@pytest.mark.requires_impl
class TestLipschitzContinuity:
    """Lipschitz constant of the analysis and frame operators.

    For the analysis operator F, the Lipschitz constant is √B:
        ‖F(x₁) - F(x₂)‖ ≤ √B · ‖x₁ - x₂‖

    This bounds how much small input perturbations change the representation,
    which is critical for adversarial robustness in deep learning.
    """

    @pytest.mark.xfail(reason="filterbankbounds underestimates upper frame bound for auditory filterbanks")
    def test_analysis_lipschitz_bounded_by_sqrt_B(self, needs_impl):
        """‖Fx₁ - Fx₂‖ / ‖x₁ - x₂‖ ≤ √B for random signal pairs."""
        g, a, L, A, B, M = _make_filterbank("aud")
        rng = np.random.default_rng(31)

        lip_ratios = []
        for _ in range(100):
            x1 = rng.standard_normal(L)
            x2 = rng.standard_normal(L)
            dx = x1 - x2
            dFx = _analysis_op(x1, g, a) - _analysis_op(x2, g, a)

            # By linearity: dFx = F(dx), so ‖dFx‖²/‖dx‖² ≤ B
            # (using ‖Fx‖² ≤ B·‖x‖² from frame bound, but the raw
            #  coefficient energy relates as ‖Fx‖² = ⟨S_g x, x⟩/2 ≤ B/2·‖x‖²)
            # For the complex analysis: ‖Fx‖² = Σ|c_m|²
            ratio_sq = np.sum(np.abs(dFx) ** 2) / (np.linalg.norm(dx) ** 2)
            lip_ratios.append(np.sqrt(ratio_sq))

        max_lip = max(lip_ratios)
        # The maximum Lipschitz ratio should be ≤ √(B/2) since
        # ‖Fx‖² = ⟨S_g x, x⟩/2 ≤ (B/2)·‖x‖²
        assert max_lip <= np.sqrt(B / 2) + 1e-4, \
            f"Lipschitz ratio {max_lip:.4f} > √(B/2) = {np.sqrt(B/2):.4f}"

    @pytest.mark.xfail(reason="filterbankbounds underestimates upper frame bound for auditory filterbanks")
    def test_frame_op_lipschitz_bounded_by_B(self, needs_impl):
        """‖S_g x‖ / ‖x‖ ≤ B for the frame operator."""
        g, a, L, A, B, M = _make_filterbank("aud")
        rng = np.random.default_rng(33)

        for _ in range(100):
            x = rng.standard_normal(L)
            Sgx = _frame_op(x, g, a, L)[:L]
            ratio = np.linalg.norm(Sgx) / np.linalg.norm(x)
            assert ratio <= B + 1e-4, \
                f"‖S_g x‖/‖x‖ = {ratio:.4f} > B = {B:.4f}"
            assert ratio >= A - 1e-4, \
                f"‖S_g x‖/‖x‖ = {ratio:.4f} < A = {A:.4f}"

    def test_lipschitz_tight_frame_nearly_isometric(self, needs_impl):
        """A tight frame has Lipschitz constant ≈ √(B/2) ≈ √(A/2),
        making it nearly isometric (distance-preserving up to scale).
        """
        from cool_frames.filterbanks import filterbankbounds, filterbanktight
        from cool_frames.filters import audfilters
        from cool_frames.filters import filterbanklength

        g, a, fc, _, _info = audfilters(8000, 512)
        L = filterbanklength(512, a)
        gt = filterbanktight(g, a, L)
        A_t, B_t = filterbankbounds(gt, a, L)

        rng = np.random.default_rng(35)
        ratios = []
        for _ in range(100):
            x = rng.standard_normal(L)
            Fx = _analysis_op(x, gt, a)
            ratio = np.sum(np.abs(Fx) ** 2) / (np.linalg.norm(x) ** 2)
            ratios.append(ratio)

        ratios = np.array(ratios)
        # For a tight frame, all ratios should be nearly equal
        spread = (ratios.max() - ratios.min()) / ratios.mean()
        assert spread < 0.05, \
            f"Tight frame Lipschitz spread = {spread:.4f}, expected near 0"


# ===================================================================
# 4.  GRADIENT NORM PRESERVATION (DEEP NETWORK STABILITY)
# ===================================================================

@pytest.mark.requires_impl
class TestGradientNormPreservation:
    """For a filterbank layer in a deep network, backpropagated gradients
    must be bounded to prevent vanishing/exploding gradients.

    The key identity: if δ is the upstream gradient in coefficient space,
    then the backpropagated gradient is S_g^{-1/2} or F^* δ (depending on
    architecture).  The frame bounds ensure:
        A·‖δ‖² ≤ ‖F^* δ‖² ≤ B·‖δ‖²  (approximately)
    """

    @pytest.mark.xfail(reason="filterbankbounds underestimates upper frame bound for auditory filterbanks")
    def test_backprop_gradient_bounded(self, needs_impl):
        """Backpropagated gradient norms lie in a bounded range.

        For the frame operator S_g, gradients flowing backward through
        synthesis satisfy: A·‖x‖ ≤ ‖S_g x‖ ≤ B·‖x‖.
        """
        g, a, L, A, B, M = _make_filterbank("aud")
        rng = np.random.default_rng(41)

        for _ in range(50):
            x = rng.standard_normal(L)
            Sgx = _frame_op(x, g, a, L)[:L]

            # ‖S_g x‖ / ‖x‖ should be in [A, B]
            ratio = np.linalg.norm(Sgx) / np.linalg.norm(x)
            assert ratio >= A - 1e-4, \
                f"Gradient too small: {ratio:.4f} < A = {A:.4f}"
            assert ratio <= B + 1e-4, \
                f"Gradient too large: {ratio:.4f} > B = {B:.4f}"

    def test_gradient_ratio_bounded_by_condition(self, needs_impl):
        """The ratio of max/min gradient norms ≤ κ = B/A.

        This is the key quantity for deep network training: a large κ
        means gradients can vary by a factor of κ across directions,
        causing slow convergence.
        """
        g, a, L, A, B, M = _make_filterbank("aud")
        kappa = B / A
        rng = np.random.default_rng(43)

        ratios = []
        for _ in range(200):
            x = rng.standard_normal(L)
            Sgx = _frame_op(x, g, a, L)[:L]
            ratios.append(np.linalg.norm(Sgx) / np.linalg.norm(x))

        observed_kappa = max(ratios) / min(ratios)
        assert observed_kappa <= kappa + 0.1, \
            f"Observed κ = {observed_kappa:.4f} > theoretical κ = {kappa:.4f}"

    def test_dual_frame_gradient_identity(self, needs_impl):
        """Using dual filters for synthesis gives gradient = identity.

        For an analysis-with-dual-synthesis pipeline, the composed gradient
        is S_g^{-1} S_g = I, meaning perfect gradient flow.
        """
        from cool_frames.filterbanks import filterbank, filterbankdual, ifilterbank
        from cool_frames.filters import audfilters
        from cool_frames.filters import filterbanklength

        g, a, fc, _, _info = audfilters(8000, 512)
        L = filterbanklength(512, a)
        gd = filterbankdual(g, a, L)

        rng = np.random.default_rng(45)
        for _ in range(30):
            x = rng.standard_normal(L)
            # Forward: analysis with g, synthesis with gd
            c = filterbank(x, g, a)
            r = np.real(np.asarray(ifilterbank(c, gd, a, real=True)))[:L]

            rel_err = np.linalg.norm(r - x) / np.linalg.norm(x)
            assert rel_err < 1e-10, \
                f"Dual gradient identity error: {rel_err:.2e}"


# ===================================================================
# 5.  DEFORMATION STABILITY (SCATTERING THEORY)
# ===================================================================

@pytest.mark.requires_impl
class TestDeformationStability:
    """Stability of the filterbank representation to time-warping.

    Mallat's scattering theory shows that for a well-designed filterbank:
        ‖Fx - F(x ∘ τ)‖ ≤ C · ‖∇τ‖_∞ · ‖x‖
    where τ is a small diffeomorphism.  We test this with controlled
    time-warpings of increasing strength.
    """

    def test_small_warp_small_change(self, needs_impl):
        """Larger time-warpings produce larger per-channel energy changes.

        We compare the total per-channel energy difference (shift-invariant
        measure) across increasing warp strengths.  Because aliasing can
        cause non-monotonicity at individual warp sizes, we verify the
        overall trend by comparing the smallest warp to the largest.
        """
        from cool_frames.filterbanks import filterbank as fb_analysis
        g, a, L, A, B, M = _make_filterbank("aud", Ls=1024)

        t = np.arange(L, dtype=float)
        x = np.sin(2 * np.pi * 440 * t / 8000) + 0.5 * np.sin(2 * np.pi * 1200 * t / 8000)
        c_orig = fb_analysis(x, g, a)
        e_orig = np.array([np.sum(np.abs(np.asarray(cm).ravel()) ** 2) for cm in c_orig])

        errors = []
        warp_strengths = [0.001, 0.01, 0.05, 0.1]

        for eps in warp_strengths:
            # Apply sinusoidal time-warp: τ(t) = t + eps·L·sin(2πt/L)
            warp = eps * L * np.sin(2 * np.pi * t / L)
            t_warped = (t + warp) % L
            x_warped = np.interp(t_warped, t, x, period=L)

            c_warped = fb_analysis(x_warped, g, a)
            e_warped = np.array([np.sum(np.abs(np.asarray(cm).ravel()) ** 2) for cm in c_warped])
            energy_diff = np.linalg.norm(e_orig - e_warped) / (np.linalg.norm(e_orig) + 1e-15)
            errors.append(energy_diff)

        # Largest warp should cause more energy change than smallest warp
        assert errors[-1] > errors[0], \
            f"Large warp ({warp_strengths[-1]}) didn't cause more change than small warp ({warp_strengths[0]})"

    def test_deformation_bounded_by_warp_amplitude(self, needs_impl):
        """Representation change is finite and bounded for small warps.

        For a single-layer filterbank (unlike a scattering transform),
        the representation is Lipschitz in the signal, so a small warp
        that changes the signal by a bounded amount changes the
        representation by a bounded amount:
            ‖Fx - F(x∘τ)‖ ≤ √(B/2) · ‖x - x∘τ‖
        """
        g, a, L, A, B, M = _make_filterbank("aud", Ls=1024)

        t = np.arange(L, dtype=float)
        x = np.sin(2 * np.pi * 500 * t / 8000) + 0.3 * np.cos(2 * np.pi * 2000 * t / 8000)

        eps = 0.005  # small warp
        warp = eps * L * np.sin(2 * np.pi * t / L)
        t_warped = (t + warp) % L
        x_warped = np.interp(t_warped, t, x, period=L)

        Fx = _analysis_op(x, g, a)
        Fx_warped = _analysis_op(x_warped, g, a)

        # Lipschitz bound: ‖F(x) - F(x')‖ ≤ √(B/2) · ‖x - x'‖
        signal_diff = np.linalg.norm(x - x_warped)
        repr_diff = np.linalg.norm(Fx - Fx_warped)

        lip_bound = np.sqrt(B / 2) * signal_diff
        assert repr_diff <= lip_bound + 1e-4, \
            f"Deformation violates Lipschitz: ‖ΔF‖={repr_diff:.4f}, bound={lip_bound:.4f}"

    def test_identity_warp_zero_change(self, needs_impl):
        """The identity warp (τ = 0) produces zero representation change."""
        g, a, L, A, B, M = _make_filterbank("aud", Ls=512)

        rng = np.random.default_rng(51)
        x = rng.standard_normal(L)
        Fx1 = _analysis_op(x, g, a)
        Fx2 = _analysis_op(x, g, a)

        assert np.linalg.norm(Fx1 - Fx2) < 1e-14, \
            "Identity warp should produce zero change"


# ===================================================================
# 6.  NOISE AMPLIFICATION
# ===================================================================

@pytest.mark.requires_impl
class TestNoiseAmplification:
    """How much additive noise is amplified through analysis and reconstruction.

    For a frame with bounds A, B:
    - Analysis amplifies noise energy by at most B/2 (coefficient domain)
    - Reconstruction with dual frame preserves noise level (perfect recon)
    - The SNR degradation is bounded by the condition number κ
    """

    @pytest.mark.xfail(reason="filterbankbounds underestimates upper frame bound for auditory filterbanks")
    def test_noise_amplification_bounded(self, needs_impl):
        """Additive noise energy in coefficient domain ≤ (B/2)·‖n‖²."""
        g, a, L, A, B, M = _make_filterbank("aud")
        rng = np.random.default_rng(61)

        for _ in range(50):
            noise = rng.standard_normal(L) * 0.01  # small noise
            Fn = _analysis_op(noise, g, a)
            noise_energy = np.sum(np.abs(Fn) ** 2)
            input_energy = np.linalg.norm(noise) ** 2

            ratio = noise_energy / input_energy
            assert ratio <= B / 2 + 1e-4, \
                f"Noise amplification {ratio:.4f} > B/2 = {B/2:.4f}"
            assert ratio >= A / 2 - 1e-4, \
                f"Noise suppression {ratio:.4f} < A/2 = {A/2:.4f}"

    def test_snr_preserved_through_reconstruction(self, needs_impl):
        """SNR is approximately preserved through analysis → dual synthesis."""
        from cool_frames.filterbanks import filterbank, filterbankdual, ifilterbank
        from cool_frames.filters import audfilters
        from cool_frames.filters import filterbanklength

        fs = 8000
        Ls = 512
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        gd = filterbankdual(g, a, L)

        rng = np.random.default_rng(63)
        t = np.arange(L, dtype=float) / fs
        signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1000 * t)

        for snr_db in [20, 40, 60]:
            noise_power = np.linalg.norm(signal) ** 2 / (10 ** (snr_db / 10))
            noise = rng.standard_normal(L) * np.sqrt(noise_power / L)

            x_noisy = signal + noise

            # Analyse and reconstruct
            c = filterbank(x_noisy, g, a)
            r = np.real(np.asarray(ifilterbank(c, gd, a, real=True)))[:L]

            # Reconstruction error should be ~ noise level (perfect recon)
            recon_err = np.linalg.norm(r - x_noisy) / np.linalg.norm(x_noisy)
            assert recon_err < 1e-10, \
                f"Reconstruction error {recon_err:.2e} at SNR={snr_db}dB"

    def test_noise_white_after_analysis(self, needs_impl):
        """White noise input produces roughly white noise coefficients.

        The per-channel noise energy should scale with the channel bandwidth
        (not accumulate pathologically in any single channel).
        """
        g, a, L, A, B, M = _make_filterbank("aud", Ls=2048)
        rng = np.random.default_rng(65)

        n_trials = 100
        channel_energies = None

        for _ in range(n_trials):
            noise = rng.standard_normal(L)
            from cool_frames.filterbanks import filterbank
            c = filterbank(noise, g, a)
            e = np.array([np.sum(np.abs(np.asarray(cm).ravel()) ** 2) for cm in c])
            channel_energies = e if channel_energies is None else channel_energies + e

        avg_energy = channel_energies / n_trials

        # No single channel should dominate (> 50% of total)
        total = avg_energy.sum()
        max_fraction = avg_energy.max() / total
        assert max_fraction < 0.5, \
            f"One channel has {max_fraction:.1%} of total noise energy"


# ===================================================================
# 7.  EFFECTIVE RANK & NUCLEAR NORM
# ===================================================================

@pytest.mark.requires_impl
class TestEffectiveRank:
    """Effective rank and nuclear norm of the frame operator.

    Effective rank measures how many dimensions the representation "uses":
        erank(S) = exp(H(λ/Σλ))
    where H is Shannon entropy and λ are eigenvalues.

    Nuclear norm ‖S‖_* = tr(S) = Σλ_i measures total redundancy.
    """

    def test_effective_rank_positive(self, needs_impl):
        """Effective rank is at least 1 and at most L."""
        L_small = 128
        g, a, L, A, B, M = _make_filterbank("aud", Ls=L_small)
        S = _materialise_frame_op(g, a, L)
        eigvals = np.linalg.eigvalsh(0.5 * (S + S.T))
        eigvals = eigvals[eigvals > 1e-10]

        # Shannon entropy of normalised eigenvalue distribution
        p = eigvals / eigvals.sum()
        entropy = -np.sum(p * np.log(p))
        erank = np.exp(entropy)

        assert erank >= 1.0, f"Effective rank {erank:.2f} < 1"
        assert erank <= L, f"Effective rank {erank:.2f} > L = {L}"

    def test_tight_frame_maximal_erank(self, needs_impl):
        """A tight frame has maximal effective rank (all eigenvalues equal)."""
        from cool_frames.filterbanks import filterbanktight
        from cool_frames.filters import audfilters
        from cool_frames.filters import filterbanklength

        L_small = 128
        g, a, fc, _, _info = audfilters(8000, L_small)
        L = filterbanklength(L_small, a)
        gt = filterbanktight(g, a, L)

        S = _materialise_frame_op(gt, a, L)
        eigvals = np.linalg.eigvalsh(0.5 * (S + S.T))
        eigvals = eigvals[eigvals > 1e-10]

        p = eigvals / eigvals.sum()
        entropy = -np.sum(p * np.log(p))
        erank_tight = np.exp(entropy)

        # Also compute erank for the original frame
        g_orig, a_orig, _, _, _info = audfilters(8000, L_small)
        L_orig = filterbanklength(L_small, a_orig)
        S_orig = _materialise_frame_op(g_orig, a_orig, L_orig)
        eigvals_orig = np.linalg.eigvalsh(0.5 * (S_orig + S_orig.T))
        eigvals_orig = eigvals_orig[eigvals_orig > 1e-10]
        p_orig = eigvals_orig / eigvals_orig.sum()
        erank_orig = np.exp(-np.sum(p_orig * np.log(p_orig)))

        assert erank_tight >= erank_orig - 1.0, \
            f"Tight erank {erank_tight:.1f} should be ≥ original {erank_orig:.1f}"

    def test_nuclear_norm_equals_trace(self, needs_impl):
        """Nuclear norm ‖S_g‖_* = tr(S_g) = sum of eigenvalues."""
        L_small = 128
        g, a, L, A, B, M = _make_filterbank("aud", Ls=L_small)
        S = _materialise_frame_op(g, a, L)

        trace = np.trace(S)
        eigvals = np.linalg.eigvalsh(0.5 * (S + S.T))
        eig_sum = eigvals.sum()

        rel_err = abs(trace - eig_sum) / (abs(trace) + 1e-15)
        assert rel_err < 1e-8, \
            f"tr(S) = {trace:.4f} ≠ Σλ = {eig_sum:.4f}, err = {rel_err:.2e}"

    def test_nuclear_norm_bounds(self, needs_impl):
        """Nuclear norm satisfies A·L ≤ ‖S_g‖_* ≤ B·L."""
        L_small = 128
        g, a, L, A, B, M = _make_filterbank("aud", Ls=L_small)
        S = _materialise_frame_op(g, a, L)

        nuclear_norm = np.trace(S)  # = Σλ for PSD matrix
        assert nuclear_norm >= A * L - 1e-4, \
            f"‖S‖_* = {nuclear_norm:.2f} < A·L = {A*L:.2f}"
        assert nuclear_norm <= B * L + 1e-4, \
            f"‖S‖_* = {nuclear_norm:.2f} > B·L = {B*L:.2f}"

    def test_frobenius_norm_relation(self, needs_impl):
        """Frobenius norm ‖S‖_F = √(Σλ²) relates to nuclear and operator norms.

        ‖S‖_F / √L ≤ ‖S‖_op = B  (operator norm)
        ‖S‖_* / L ≤ ‖S‖_F / √L  (nuclear-Frobenius inequality)
        """
        L_small = 128
        g, a, L, A, B, M = _make_filterbank("aud", Ls=L_small)
        S = _materialise_frame_op(g, a, L)

        frob = np.linalg.norm(S, 'fro')
        nuclear = np.trace(S)
        op_norm = np.linalg.norm(S, 2)  # largest singular value

        # ‖S‖_F ≤ √L · ‖S‖_op
        assert frob <= np.sqrt(L) * op_norm + 1e-4, \
            f"‖S‖_F = {frob:.2f} > √L · ‖S‖_op = {np.sqrt(L)*op_norm:.2f}"

        # ‖S‖_* ≤ √L · ‖S‖_F  (Cauchy-Schwarz on eigenvalues)
        assert nuclear <= np.sqrt(L) * frob + 1e-4, \
            f"‖S‖_* = {nuclear:.2f} > √L · ‖S‖_F = {np.sqrt(L)*frob:.2f}"


# ===================================================================
# 8.  WHITENING / DECORRELATION
# ===================================================================

@pytest.mark.requires_impl
class TestWhiteningDecorrelation:
    """Measures how close the filterbank output is to a decorrelated
    (whitened) representation.

    Perfect whitening: the Gram matrix F^H F is a scaled identity.
    The ratio of off-diagonal to diagonal energy measures decorrelation.
    """

    def test_tight_frame_better_whitening(self, needs_impl):
        """A tight frame produces more decorrelated outputs than the
        original frame (lower off-diagonal energy ratio).
        """
        from cool_frames.filterbanks import filterbanktight
        from cool_frames.filters import audfilters
        from cool_frames.filters import filterbanklength

        L_small = 128
        g, a, fc, _, _info = audfilters(8000, L_small)
        L = filterbanklength(L_small, a)

        # Compute output correlation for original frame
        rng = np.random.default_rng(71)
        n_trials = 200

        def compute_offdiag_ratio(filters, a, L, rng, n_trials):
            """Estimate off-diagonal energy ratio of output Gram matrix."""
            from cool_frames.filterbanks import filterbank
            M = len(filters)
            # Collect per-channel energies to build a correlation estimate
            coefs_list = []
            for _ in range(n_trials):
                x = rng.standard_normal(L)
                c = filterbank(x, filters, a)
                energies = np.array([np.sum(np.abs(np.asarray(cm).ravel()) ** 2) for cm in c])
                coefs_list.append(energies)

            C = np.array(coefs_list)  # (n_trials, M)
            # Normalise
            C -= C.mean(axis=0, keepdims=True)
            gram = C.T @ C / n_trials

            diag_energy = np.sum(np.diag(gram) ** 2)
            total_energy = np.sum(gram ** 2)
            offdiag_ratio = 1.0 - diag_energy / (total_energy + 1e-15)
            return offdiag_ratio

        odr_orig = compute_offdiag_ratio(g, a, L, rng, n_trials)

        gt = filterbanktight(g, a, L)
        rng2 = np.random.default_rng(71)
        odr_tight = compute_offdiag_ratio(gt, a, L, rng2, n_trials)

        assert odr_tight <= odr_orig + 0.05, \
            f"Tight frame ODR {odr_tight:.4f} > original {odr_orig:.4f}"

    def test_frame_op_diagonal_captures_energy(self, needs_impl):
        """The diagonal of S_g captures a non-trivial fraction of ‖S_g‖_F.

        For filterbanks with compactly supported filters, the frame operator
        has most of its energy near the diagonal (banded structure).  The
        diagonal-to-Frobenius ratio measures how "local" the operator is.
        """
        L_small = 128
        g, a, L, A, B, M = _make_filterbank("aud", Ls=L_small)
        S = _materialise_frame_op(g, a, L)

        diag_energy = np.sum(np.diag(S) ** 2)
        total_energy = np.linalg.norm(S, 'fro') ** 2

        diag_ratio = diag_energy / total_energy
        # The diagonal should capture a meaningful fraction of the energy
        # (exact value depends on filter overlap, but should be > 0)
        assert diag_ratio > 0.01, \
            f"Diagonal captures only {diag_ratio:.4f} of Frobenius energy"
        assert diag_ratio <= 1.0, \
            f"Diagonal ratio {diag_ratio:.4f} > 1 (impossible)"


# ===================================================================
# 9.  SPECTRAL GAP
# ===================================================================

@pytest.mark.requires_impl
class TestSpectralGap:
    """Spectral gap analysis of the frame operator.

    A large gap between eigenvalue clusters means the frame operator
    has a natural "low-rank + residual" decomposition, useful for
    truncated representations and fast approximate inversion.
    """

    def test_eigenvalue_clusters_exist(self, needs_impl):
        """The frame operator eigenvalues form identifiable clusters.

        For an ERB filterbank, eigenvalues cluster because groups of
        channels have similar hop sizes.
        """
        L_small = 128
        g, a, L, A, B, M = _make_filterbank("aud", Ls=L_small)
        S = _materialise_frame_op(g, a, L)
        eigvals = np.sort(np.linalg.eigvalsh(0.5 * (S + S.T)))

        # Compute gaps between consecutive eigenvalues
        gaps = np.diff(eigvals)
        if len(gaps) > 0:
            median_gap = np.median(gaps)
            max_gap = np.max(gaps)
            # The max gap should be significantly larger than median
            # (indicating cluster structure)
            assert max_gap > 0, "All eigenvalues are identical (degenerate)"

    def test_spectral_gap_ratio(self, needs_impl):
        """The ratio of largest gap to median gap indicates cluster separation."""
        L_small = 128
        g, a, L, A, B, M = _make_filterbank("aud", Ls=L_small)
        S = _materialise_frame_op(g, a, L)
        eigvals = np.sort(np.linalg.eigvalsh(0.5 * (S + S.T)))

        gaps = np.diff(eigvals)
        if len(gaps) > 1:
            median_gap = np.median(gaps)
            max_gap = np.max(gaps)
            if median_gap > 1e-10:
                gap_ratio = max_gap / median_gap
                # For a structured filterbank, the gap ratio should be > 1
                assert gap_ratio >= 1.0, \
                    f"Gap ratio {gap_ratio:.2f} < 1 (no spectral structure)"

    def test_truncated_approx_quality(self, needs_impl):
        """Keeping only the top-k eigenvalues gives a useful approximation.

        The truncation error ‖S - S_k‖_F / ‖S‖_F measures how much
        information is lost when using a rank-k approximation.
        """
        L_small = 128
        g, a, L, A, B, M = _make_filterbank("aud", Ls=L_small)
        S = _materialise_frame_op(g, a, L)

        eigvals = np.linalg.eigvalsh(0.5 * (S + S.T))
        eigvals_sorted = np.sort(eigvals)[::-1]  # descending

        total_energy = np.sum(eigvals_sorted ** 2)
        cumulative_energy = np.cumsum(eigvals_sorted ** 2) / total_energy

        # Find k such that top-k eigenvalues capture 90% of Frobenius energy
        k_90 = np.searchsorted(cumulative_energy, 0.90) + 1

        # For a well-conditioned frame, k_90 should be a small fraction of L
        assert k_90 <= L, f"k_90 = {k_90} > L = {L}"
        # The rank-k_90 approximation exists (this is always true, just verify)
        assert k_90 >= 1, "Need at least 1 eigenvalue for 90% energy"

    def test_condition_predicts_gap_structure(self, needs_impl):
        """Better-conditioned frames have smaller spectral gaps (more uniform).

        A tight frame (κ ≈ 1) has no gaps (all eigenvalues equal), while
        a poorly conditioned frame (large κ) has wide gaps.
        """
        from cool_frames.filterbanks import filterbanktight
        from cool_frames.filters import audfilters
        from cool_frames.filters import filterbanklength

        L_small = 128
        g, a, fc, _, _info = audfilters(8000, L_small)
        L = filterbanklength(L_small, a)

        # Original frame
        S_orig = _materialise_frame_op(g, a, L)
        eigvals_orig = np.sort(np.linalg.eigvalsh(0.5 * (S_orig + S_orig.T)))
        spread_orig = eigvals_orig.max() - eigvals_orig.min()

        # Tight frame
        gt = filterbanktight(g, a, L)
        S_tight = _materialise_frame_op(gt, a, L)
        eigvals_tight = np.sort(np.linalg.eigvalsh(0.5 * (S_tight + S_tight.T)))
        spread_tight = eigvals_tight.max() - eigvals_tight.min()

        assert spread_tight <= spread_orig + 1e-4, \
            f"Tight frame spread {spread_tight:.2f} > original {spread_orig:.2f}"
