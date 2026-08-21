"""
Tests for frame multiplier operators (cool_frames.numpy.operators).

Tests are organised by mathematical property so that each test validates
a theorem from frame multiplier theory (see MATH_REFERENCE.md §15a).

Filterbank fixtures use audfilters at moderate sizes to keep tests fast
while exercising non-uniform hop sizes.
"""
from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=[
    (16000, 4096),
    (22050, 8000),
])
def fb_setup(request):
    """Build an auditory filterbank and tight frame for testing.

    Returns dict with keys: g, g_tight, a, L, M, f, fc
    """
    from cool_frames.numpy.filters import audfilters
    from cool_frames.numpy.filterbanks import filterbanktight, filterbank

    fs, Ls = request.param
    g, a, fc, L, _info = audfilters(fs, Ls)
    g_tight = filterbanktight(g, a, L)
    M = len(g)

    rng = np.random.default_rng(42)
    f = rng.standard_normal(L)

    # Pre-compute coefficients for convenience
    c = filterbank(f, g_tight, a, L)

    return dict(g=g, g_tight=g_tight, a=a, L=L, M=M, f=f, fc=fc, c=c, fs=fs)


@pytest.fixture()
def fb_small():
    """A small tight frame where the FULL normal equations are tractable.

    framemulappr's exact solve is O(Lambda^2) memory and O(Lambda^3) time
    in the total coefficient count. The fb_setup frames give Lambda ~ 10^4,
    i.e. a ~1 GB Gram -- fine for the cheap multiplier tests, hopeless for
    the approximation ones. These use a deliberately small frame instead.
    """
    from cool_frames.numpy.filterbanks import filterbank, filterbanktight
    from cool_frames.numpy.filters import audfilters

    g, a, fc, L, _info = audfilters(8000, 96)
    g_tight = filterbanktight(g, a, L)
    c = filterbank(np.zeros(L), g_tight, a, L)
    return dict(g_tight=g_tight, a=a, L=L, M=len(g_tight),
                Nm=[len(np.asarray(ci).ravel()) for ci in c], c=c)


def _operator_matrix(sigma, d):
    """Materialise M_sigma as an (L, L) matrix, column by column."""
    from cool_frames.numpy.operators import framemul
    L = d['L']
    Q = np.zeros((L, L))
    for k in range(L):
        e_k = np.zeros(L)
        e_k[k] = 1.0
        Q[:, k] = np.real(framemul(e_k, d['g_tight'], d['g_tight'],
                                   d['a'], sigma, L))
    return Q


@pytest.fixture()
def dual_setup():
    """Build a filterbank with separate analysis/synthesis (real dual) frames."""
    from cool_frames.numpy.filters import audfilters
    from cool_frames.numpy.filterbanks import filterbankdual

    fs, Ls = 16000, 4096
    g, a, fc, L, _info = audfilters(fs, Ls)
    g_dual = filterbankdual(g, a, L)
    M = len(g)

    rng = np.random.default_rng(42)
    f = rng.standard_normal(L)

    return dict(g=g, g_dual=g_dual, a=a, L=L, M=M, f=f, fc=fc, fs=fs)


def _ones_symbol(c):
    """Create a symbol of all ones matching coefficient structure."""
    return [np.ones_like(ci) for ci in c]


def _random_symbol(c, seed=123):
    """Create a random positive symbol matching coefficient structure."""
    rng = np.random.default_rng(seed)
    return [rng.uniform(0.5, 2.0, size=ci.shape) for ci in c]


def _binary_mask(c, seed=99):
    """Create a random binary (0/1) mask matching coefficient structure."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 2, size=ci.shape).astype(float) for ci in c]


# ===================================================================
# framemul — forward multiplier
# ===================================================================

class TestFramemul:
    """Forward frame multiplier: M_sigma f = synth(sigma * analysis(f))."""

    def test_identity_symbol(self, fb_setup):
        """sigma = 1 everywhere => M_sigma f = f (for tight frame)."""
        from cool_frames.numpy.operators import framemul

        d = fb_setup
        sigma = _ones_symbol(d['c'])
        result = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                          sigma, d['L'])
        np.testing.assert_allclose(result, d['f'], atol=1e-10)

    def test_zero_symbol(self, fb_setup):
        """sigma = 0 everywhere => M_sigma f = 0."""
        from cool_frames.numpy.operators import framemul

        d = fb_setup
        sigma = [np.zeros_like(ci) for ci in d['c']]
        result = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                          sigma, d['L'])
        np.testing.assert_allclose(result, 0.0, atol=1e-12)

    def test_scalar_symbol(self, fb_setup):
        """Constant sigma = alpha => M_sigma f = alpha * f (tight frame)."""
        from cool_frames.numpy.operators import framemul

        d = fb_setup
        alpha = 3.7
        sigma = [alpha * np.ones_like(ci) for ci in d['c']]
        result = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                          sigma, d['L'])
        np.testing.assert_allclose(result, alpha * d['f'], atol=1e-9)

    def test_output_is_real_for_real_input(self, fb_setup):
        """Real signal + real symbol => real output."""
        from cool_frames.numpy.operators import framemul

        d = fb_setup
        sigma = _random_symbol(d['c'])
        result = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                          sigma, d['L'])
        assert np.isrealobj(result) or np.max(np.abs(np.imag(result))) < 1e-10

    def test_output_length(self, fb_setup):
        """Output has same length as input signal."""
        from cool_frames.numpy.operators import framemul

        d = fb_setup
        sigma = _random_symbol(d['c'])
        result = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                          sigma, d['L'])
        assert len(result) == d['L']

    def test_with_dual_frame(self, dual_setup):
        """Identity symbol with original + dual => perfect reconstruction."""
        from cool_frames.numpy.operators import framemul
        from cool_frames.numpy.filterbanks import filterbank

        d = dual_setup
        c = filterbank(d['f'], d['g'], d['a'], d['L'])
        sigma = _ones_symbol(c)
        result = framemul(d['f'], d['g'], d['g_dual'], d['a'],
                          sigma, d['L'])
        np.testing.assert_allclose(result, d['f'], atol=1e-10)


# ===================================================================
# framemul adjoint
# ===================================================================

class TestFramemulAdj:
    """Adjoint: <M f, h> = <f, M* h> for all f, h."""

    def test_adjoint_identity(self, fb_setup):
        """Verify the adjoint relationship: <Mf, h> = <f, M*h>."""
        from cool_frames.numpy.operators import framemul, framemuladj

        d = fb_setup
        rng = np.random.default_rng(77)
        f = rng.standard_normal(d['L'])
        h = rng.standard_normal(d['L'])
        sigma = _random_symbol(d['c'])

        Mf = framemul(f, d['g_tight'], d['g_tight'], d['a'],
                       sigma, d['L'])
        Mstar_h = framemuladj(h, d['g_tight'], d['g_tight'], d['a'],
                               sigma, d['L'])

        lhs = np.dot(Mf, h)
        rhs = np.dot(f, Mstar_h)
        np.testing.assert_allclose(lhs, rhs, atol=1e-9)

    def test_self_adjoint_for_real_symbol_tight_frame(self, fb_setup):
        """For real sigma and tight frame (Fa=Fs), M = M* (self-adjoint)."""
        from cool_frames.numpy.operators import framemul, framemuladj

        d = fb_setup
        sigma = _random_symbol(d['c'])
        f = d['f']

        Mf = framemul(f, d['g_tight'], d['g_tight'], d['a'],
                       sigma, d['L'])
        Mstar_f = framemuladj(f, d['g_tight'], d['g_tight'], d['a'],
                               sigma, d['L'])
        np.testing.assert_allclose(Mf, Mstar_f, atol=1e-10)


# ===================================================================
# framemulinv — inverse frame multiplier (PCG)
# ===================================================================

class TestFramemulinv:
    """Inverse: M^{-1} (M f) = f when sigma > 0."""

    def test_roundtrip_tight(self, fb_setup):
        """Apply then invert with tight frame and positive symbol."""
        from cool_frames.numpy.operators import framemul, framemulinv

        d = fb_setup
        sigma = _random_symbol(d['c'])

        Mf = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                       sigma, d['L'])
        recovered, info = framemulinv(Mf, d['g_tight'], d['g_tight'],
                                       d['a'], sigma, d['L'])

        np.testing.assert_allclose(recovered, d['f'], atol=1e-6)
        assert info['converged']

    def test_roundtrip_dual(self, dual_setup):
        """Apply then invert with original + dual frame."""
        from cool_frames.numpy.operators import framemul, framemulinv
        from cool_frames.numpy.filterbanks import filterbank

        d = dual_setup
        c = filterbank(d['f'], d['g'], d['a'], d['L'])
        sigma = _random_symbol(c)

        Mf = framemul(d['f'], d['g'], d['g_dual'], d['a'],
                       sigma, d['L'])
        recovered, info = framemulinv(Mf, d['g'], d['g_dual'],
                                       d['a'], sigma, d['L'])

        np.testing.assert_allclose(recovered, d['f'], atol=1e-5)
        assert info['converged']

    def test_convergence_info(self, fb_setup):
        """Verify that convergence info is returned."""
        from cool_frames.numpy.operators import framemul, framemulinv

        d = fb_setup
        sigma = _random_symbol(d['c'])
        Mf = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                       sigma, d['L'])

        _, info = framemulinv(Mf, d['g_tight'], d['g_tight'],
                               d['a'], sigma, d['L'])

        assert 'relres' in info
        assert 'iter' in info
        assert 'converged' in info
        assert info['relres'] < 1e-6


# ===================================================================
# framemulappr — best approximation of operator by multiplier
# ===================================================================

class TestFramemulappr:
    """Best Hilbert-Schmidt approximation of a linear operator."""

    def test_identity_operator(self, fb_small):
        """Approximating the identity should give sigma ≈ 1 (tight frame).

        Uses the small frame: framemulappr's solve is O(Lambda^3), so this is
        not reachable at fb_setup sizes (see TestFramemulapprExact).
        """
        from cool_frames.numpy.operators import framemulappr

        d = fb_small
        T = np.eye(d['L'])
        sigma = framemulappr(T, d['g_tight'], d['g_tight'], d['a'], d['L'])

        # For a perfectly tight frame the best multiplier approximation of I
        # is sigma=1. In practice the diagonal Gram approximation introduces
        # a constant scaling factor, so we check that sigma is approximately
        # constant across time (shape consistency) rather than exactly 1.
        #
        # The non-degeneracy check below is not cosmetic. Comparing each
        # channel against its OWN mean is satisfied trivially by a channel
        # that is uniformly zero, so without it this test cannot distinguish
        # "constant" from "empty" -- which is exactly how a conjugation bug
        # in the lower symbol went unnoticed: it returned ~1e-16 for every
        # channel whose atoms carry phase, leaving only DC and Nyquist.
        for m, s_m in enumerate(sigma):
            assert np.abs(np.mean(s_m)) > 1e-8, (
                f"channel {m} symbol is numerically zero "
                f"(mean {np.mean(s_m):.3e}); the multiplier approximation "
                f"of the identity cannot be empty"
            )
            # All entries should be approximately the same value
            np.testing.assert_allclose(s_m, np.mean(s_m), rtol=0.1)

    def test_zero_operator(self, fb_small):
        """Approximating the zero operator gives sigma = 0."""
        from cool_frames.numpy.operators import framemulappr

        d = fb_small
        T = np.zeros((d['L'], d['L']))
        sigma = framemulappr(T, d['g_tight'], d['g_tight'], d['a'], d['L'])

        for s_m in sigma:
            np.testing.assert_allclose(s_m, 0.0, atol=1e-10)

    def test_approximation_reduces_error(self, fb_small):
        """The optimal symbol should give lower error than a random one.

        Uses the small frame: this is only true of the EXACT solve, and the
        exact solve is O(Lambda^3), so it is not reachable at fb_setup
        sizes. See TestFramemulapprExact for the size discussion.
        """
        from cool_frames.numpy.operators import framemul, framemulappr

        d = fb_small
        # A simple operator: time shift by 1
        T = np.roll(np.eye(d['L']), 1, axis=1)

        sigma_opt = framemulappr(T, d['g_tight'], d['g_tight'],
                                  d['a'], d['L'], method='full')
        sigma_rand = _random_symbol(d['c'], seed=999)

        # Compute Frobenius error for each
        def _frob_err(sigma):
            err_sq = 0.0
            for k in range(d['L']):
                e_k = np.zeros(d['L'])
                e_k[k] = 1.0
                Mek = framemul(e_k, d['g_tight'], d['g_tight'],
                               d['a'], sigma, d['L'])
                Tek = T[:, k]
                err_sq += np.sum((Mek - Tek) ** 2)
            return np.sqrt(err_sq)

        err_opt = _frob_err(sigma_opt)
        err_rand = _frob_err(sigma_rand)
        assert err_opt <= err_rand


# ===================================================================
# framemuleigs — eigenvalue computation
# ===================================================================

class TestFramemuleigs:
    """Eigenvalue decomposition of frame multiplier."""

    def test_identity_eigenvalues(self, fb_setup):
        """sigma=1 with tight frame => all eigenvalues = 1."""
        from cool_frames.numpy.operators import framemuleigs

        d = fb_setup
        sigma = _ones_symbol(d['c'])
        eigs = framemuleigs(d['g_tight'], d['g_tight'], d['a'],
                            sigma, d['L'], K=6)

        np.testing.assert_allclose(np.abs(eigs), 1.0, atol=1e-8)

    def test_eigenvalue_bounds(self, fb_setup):
        """Eigenvalues bounded by A*min(sigma) and B*max(sigma)."""
        from cool_frames.numpy.operators import framemuleigs
        from cool_frames.numpy.filterbanks import filterbankresponse

        d = fb_setup
        sigma = _random_symbol(d['c'])

        eigs = framemuleigs(d['g_tight'], d['g_tight'], d['a'],
                            sigma, d['L'], K=6)

        # For tight frame, A = B = 1, so eigenvalues bounded by
        # min(sigma) and max(sigma)
        sig_min = min(float(np.min(s)) for s in sigma)
        sig_max = max(float(np.max(s)) for s in sigma)

        assert np.min(np.real(eigs)) >= sig_min - 1e-6
        assert np.max(np.real(eigs)) <= sig_max + 1e-6

    def test_returns_requested_count(self, fb_setup):
        """Returns exactly K eigenvalues."""
        from cool_frames.numpy.operators import framemuleigs

        d = fb_setup
        K = 4
        sigma = _random_symbol(d['c'])
        eigs = framemuleigs(d['g_tight'], d['g_tight'], d['a'],
                            sigma, d['L'], K=K)
        assert len(eigs) == K

    def test_eigenvalues_sorted_descending(self, fb_setup):
        """Eigenvalues returned in descending order of magnitude."""
        from cool_frames.numpy.operators import framemuleigs

        d = fb_setup
        sigma = _random_symbol(d['c'])
        eigs = framemuleigs(d['g_tight'], d['g_tight'], d['a'],
                            sigma, d['L'], K=6)

        magnitudes = np.abs(eigs)
        assert np.all(magnitudes[:-1] >= magnitudes[1:] - 1e-12)


# ===================================================================
# Energy and positivity properties
# ===================================================================

class TestMultiplierProperties:
    """Cross-cutting mathematical properties."""

    def test_positive_symbol_positive_definite(self, fb_setup):
        """sigma > 0 => <M f, f> > 0 for all nonzero f."""
        from cool_frames.numpy.operators import framemul

        d = fb_setup
        sigma = _random_symbol(d['c'])  # all values in [0.5, 2.0]

        Mf = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                       sigma, d['L'])
        inner = np.dot(d['f'], np.real(Mf))
        assert inner > 0

    def test_binary_mask_energy_reduction(self, fb_setup):
        """Binary mask (0/1 symbol) can only reduce energy."""
        from cool_frames.numpy.operators import framemul

        d = fb_setup
        sigma = _binary_mask(d['c'])

        Mf = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                       sigma, d['L'])

        energy_in = np.sum(d['f'] ** 2)
        energy_out = np.sum(np.real(Mf) ** 2)
        assert energy_out <= energy_in + 1e-8

    def test_composition(self, fb_setup):
        """M_{sigma1} M_{sigma2} ≈ M_{sigma1 * sigma2} for tight frames."""
        from cool_frames.numpy.operators import framemul

        d = fb_setup
        sigma1 = _random_symbol(d['c'], seed=10)
        sigma2 = _random_symbol(d['c'], seed=20)
        sigma_prod = [s1 * s2 for s1, s2 in zip(sigma1, sigma2)]

        # Apply sigma2 then sigma1
        Mf2 = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                        sigma2, d['L'])
        M1_M2_f = framemul(Mf2, d['g_tight'], d['g_tight'], d['a'],
                            sigma1, d['L'])

        # Apply product symbol
        M_prod_f = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                             sigma_prod, d['L'])

        # For tight frames this is approximate (exact only for DGT with
        # rectangular lattice), but should be close
        np.testing.assert_allclose(M1_M2_f, M_prod_f, atol=0.5)

    def test_linearity_in_signal(self, fb_setup):
        """M_sigma (alpha f + beta h) = alpha M_sigma f + beta M_sigma h."""
        from cool_frames.numpy.operators import framemul

        d = fb_setup
        rng = np.random.default_rng(55)
        h = rng.standard_normal(d['L'])
        alpha, beta = 2.3, -0.7
        sigma = _random_symbol(d['c'])

        M_sum = framemul(alpha * d['f'] + beta * h,
                          d['g_tight'], d['g_tight'], d['a'],
                          sigma, d['L'])
        Mf = framemul(d['f'], d['g_tight'], d['g_tight'], d['a'],
                       sigma, d['L'])
        Mh = framemul(h, d['g_tight'], d['g_tight'], d['a'],
                       sigma, d['L'])

        np.testing.assert_allclose(M_sum, alpha * Mf + beta * Mh, atol=1e-10)


# ===================================================================
# framemulappr — LTFAT acceptance criteria
# ===================================================================

class TestFramemulapprExact:
    """The criteria LTFAT's own test_gabmulappr.m applies.

    LTFAT does not check that the symbol merely has a sensible shape: it
    builds an operator that IS a multiplier with a known symbol and
    requires the routine to reproduce it. These transplant that standard,
    with one necessary change described in test_round_trip_operator.
    """

    def test_identity_on_parseval_frame_gives_one(self, fb_small):
        """For a Parseval frame, I is exactly M_1, so sigma must be 1."""
        from cool_frames.numpy.operators import framemulappr

        d = fb_small
        sigma = framemulappr(np.eye(d['L']), d['g_tight'], d['g_tight'],
                             d['a'], d['L'], method='full')
        for m, s_m in enumerate(sigma):
            np.testing.assert_allclose(
                s_m, 1.0, rtol=1e-4, atol=1e-6,
                err_msg=f"channel {m}: A = B = 1 implies sigma = 1 exactly",
            )

    def test_round_trip_operator(self, fb_small):
        """Build an operator from a known symbol; recover the operator.

        LTFAT compares the recovered SYMBOL against the original. That is
        the right test for a Gabor system, where sigma -> M_sigma is
        injective. For a redundant filterbank it is not: the Gram is
        singular (rank 273 of 343 for this frame), several symbols give
        the same operator, and lstsq returns the minimum-norm one. So the
        operator is what must round-trip, and it does -- to ~1e-12 --
        while the symbol legitimately differs.
        """
        from cool_frames.numpy.operators import framemulappr

        d = fb_small
        rng = np.random.default_rng(0)
        sigma = [rng.uniform(0.5, 2.0, size=(n,)) for n in d['Nm']]

        T = _operator_matrix(sigma, d)
        sigma_rec = framemulappr(T, d['g_tight'], d['g_tight'],
                                 d['a'], d['L'], method='full')
        T_rec = _operator_matrix(sigma_rec, d)

        rel = np.linalg.norm(T - T_rec) / np.linalg.norm(T)
        assert rel < 1e-8, f"operator not reproduced: relative error {rel:.3e}"

    def test_zero_operator_exact(self, fb_small):
        from cool_frames.numpy.operators import framemulappr

        d = fb_small
        sigma = framemulappr(np.zeros((d['L'], d['L'])), d['g_tight'],
                             d['g_tight'], d['a'], d['L'], method='full')
        for s_m in sigma:
            np.testing.assert_allclose(s_m, 0.0, atol=1e-12)

    def test_full_beats_diagonal(self, fb_small):
        """The fallback is genuinely worse, which is why it is a fallback."""
        from cool_frames.numpy.operators import framemulappr

        d = fb_small
        T = np.roll(np.eye(d['L']), 1, axis=1)
        err = {}
        for method in ('full', 'diagonal'):
            sigma = framemulappr(T, d['g_tight'], d['g_tight'], d['a'],
                                 d['L'], method=method)
            err[method] = np.linalg.norm(T - _operator_matrix(sigma, d))
        assert err['full'] < err['diagonal'], (
            f"full={err['full']:.4f} diagonal={err['diagonal']:.4f}"
        )

    def test_auto_falls_back_and_warns(self, fb_small):
        """auto must warn, not silently degrade, when it drops to diagonal."""
        from cool_frames.numpy.operators import framemulappr

        d = fb_small
        with pytest.warns(RuntimeWarning, match="diagonal approximation"):
            framemulappr(np.eye(d['L']), d['g_tight'], d['g_tight'],
                         d['a'], d['L'], method='auto', max_gram=1)
