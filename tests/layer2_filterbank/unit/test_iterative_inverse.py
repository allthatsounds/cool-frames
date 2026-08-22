"""
test_iterative_inverse.py
=========================

Regression tests for ``ifilterbankiter`` (CG/PCG frame inversion).

Guards the 2026-06-13 fix: the fast path used to return the painless *diagonal*
dual unconditionally with a hardcoded ``relres=0.0``. For a NON-painless frame
(e.g. gabfilters whose filters are wider than L/a) that dual is wrong, so the
function reported false convergence and returned an incorrect signal. The fast
path now validates the candidate against the true analysis residual and falls
through to a warm-started CG when it is not exact.
"""
from __future__ import annotations

import pytest

import numpy as np


def _rel(xr, x):
    xr = np.real(np.asarray(xr)).ravel()[: len(x)]
    return np.linalg.norm(xr - x) / np.linalg.norm(x)


@pytest.mark.requires_impl
class TestIterativeInverseImpl:

    def test_painless_fast_path_is_exact(self, needs_impl):
        """A painless auditory bank reconstructs in one step (fast path)."""
        from cool_frames.filterbanks import filterbank, ifilterbankiter
        from cool_frames.filters import audfilters
        fs, Ls = 16_000, 2048
        g, a, fc, L, _info = audfilters(fs, Ls)
        x = np.random.default_rng(0).standard_normal(Ls)
        xr, relres, niter = ifilterbankiter(filterbank(x, g, a, L), g, a,
                                            Ls=Ls, real=True, tol=1e-10)
        assert _rel(xr, x) < 1e-9
        assert relres < 1e-9       # honest residual, not a hardcoded 0
        assert niter == 1          # one-step fast path

    def test_nonpainless_gabor_reconstructs_via_cg(self, needs_impl):
        """A non-painless Gabor frame must reconstruct via CG, not silently fail."""
        from cool_frames.filterbanks import filterbank, ifilterbankiter
        from cool_frames.filters import gabfilters
        Ls = 2048
        # a=64 with M=128 -> filter freq support (128) > L/a (32): non-painless.
        g, a, fc, L, _ = gabfilters(16_000, Ls, window="hann", a=64, M=128)
        x = np.random.default_rng(1).standard_normal(Ls)
        xr, relres, niter = ifilterbankiter(filterbank(x, g, a, L), g, a,
                                            Ls=Ls, real=True, maxit=300, tol=1e-9)
        assert _rel(xr, x) < 1e-6, "non-painless Gabor frame failed to reconstruct"
        assert relres < 1e-6
        assert niter > 1, "should have fallen through to CG, not the fast path"

    def test_reported_relres_matches_reality(self, needs_impl):
        """relres must reflect the actual analysis residual (no false convergence)."""
        from cool_frames.filterbanks import filterbank, ifilterbankiter
        from cool_frames.filters import gabfilters
        Ls = 2048
        g, a, fc, L, _ = gabfilters(16_000, Ls, window="hann", a=64, M=128)
        x = np.random.default_rng(2).standard_normal(Ls)
        # Too few iterations: relres should be reported LARGE, not 0.0.
        xr, relres, niter = ifilterbankiter(filterbank(x, g, a, L), g, a,
                                            Ls=Ls, real=True, maxit=2, tol=1e-12)
        assert relres > 1e-9, "early-stopped CG must report a non-trivial residual"
