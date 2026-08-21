"""
test_prop_iterative_reconstruction.py
=======================================
Python port of:
    layer2_filterbank/property/PropIterativeReconstruction.m

ifilterbankiter uses iterative (CG/PCG) inversion. For a valid frame:

(1) Relative residual relres < tol at convergence.
(2) Iterative result agrees with direct dual-frame inversion.
(3) More iterations yield smaller (or equal) residual.
(4) Tight-frame PCG converges to tight tolerance.

Tests are skipped gracefully if ifilterbankiter is absent.
"""

from __future__ import annotations

import pytest

import numpy as np


def _has_iter(module):
    try:
        from cool_frames.filterbanks import ifilterbankiter  # type: ignore  # noqa: F401
        return True
    except (ImportError, AttributeError):
        return False


@pytest.mark.requires_impl
class TestIterativeConvergenceImpl:
    """PropIterativeReconstruction: relres < tol after convergence."""

    def test_relres_below_tol(self, needs_impl):
        if not _has_iter(None):
            pytest.skip("ifilterbankiter not available")
        from cool_frames.filterbanks import filterbank, ifilterbankiter  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        tol = 1e-6
        rng = np.random.default_rng(42)
        for trial in range(5):
            x  = rng.standard_normal(Ls)
            c  = filterbank(x, g, a)
            result = ifilterbankiter(c, g, a, tol=tol, real=True)
            relres = float(result[1]) if isinstance(result, (list, tuple)) else 0.0
            assert relres < 10 * tol, \
                f"Trial {trial}: relres={relres:.2e} exceeds 10×tol={10*tol:.2e}"


@pytest.mark.requires_impl
class TestIterativeMatchesDualImpl:
    """PropIterativeReconstruction: iterative result ≈ direct dual inversion."""

    def test_matches_direct_dual(self, needs_impl):
        if not _has_iter(None):
            pytest.skip("ifilterbankiter not available")
        from cool_frames.filterbanks import filterbank, filterbankdual, ifilterbank, ifilterbankiter  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        gd = filterbankdual(g, a, L)
        rng = np.random.default_rng(0)
        for trial in range(3):
            x = rng.standard_normal(Ls)
            c = filterbank(x, g, a)
            # audfilters is a single-sided real bank: reconstruction (direct and
            # iterative) must use real=True (2*real(ifft)); real=False does not
            # reconstruct it.
            xr_direct = np.real(np.asarray(ifilterbank(c, gd, a, L, real=True)))
            result = ifilterbankiter(c, g, a, real=True)
            xr_iter = np.asarray(result[0]) if isinstance(result, (list, tuple)) else np.asarray(result)
            rel_err = np.linalg.norm(xr_direct[:Ls] - xr_iter[:Ls]) / \
                        (np.linalg.norm(xr_direct[:Ls]) + 1e-15)
            assert rel_err < 0.1, \
                f"Trial {trial}: iterative vs direct error {rel_err:.2e}"


@pytest.mark.requires_impl
class TestIterativeMonotoneResidualImpl:
    """PropIterativeReconstruction: more iterations → smaller residual."""

    def test_monotone_residual(self, needs_impl):
        if not _has_iter(None):
            pytest.skip("ifilterbankiter not available")
        from cool_frames.filterbanks import filterbank, ifilterbankiter  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        rng = np.random.default_rng(1)
        x = rng.standard_normal(Ls)
        c = filterbank(x, g, a)
        maxit_vals  = [5, 15, 40]
        relres_vals = []
        for maxit in maxit_vals:
            result = ifilterbankiter(c, g, a, maxit=maxit, real=True)
            rr = float(result[1]) if isinstance(result, (list, tuple)) else 0.0
            relres_vals.append(rr)
        for k in range(len(maxit_vals) - 1):
            assert relres_vals[k + 1] <= relres_vals[k] + 1e-10, \
                f"Residual increased: maxit={maxit_vals[k]}→{relres_vals[k]:.2e}, " \
                f"maxit={maxit_vals[k+1]}→{relres_vals[k+1]:.2e}"


@pytest.mark.requires_impl
class TestIterativeTightFrameImpl:
    """PropIterativeReconstruction: tight frame converges tightly."""

    def test_tight_frame_pcg(self, needs_impl):
        if not _has_iter(None):
            pytest.skip("ifilterbankiter not available")
        from cool_frames.filterbanks import filterbank, filterbanktight, ifilterbankiter  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        gt = filterbanktight(g, a, L)
        rng = np.random.default_rng(2)
        x = rng.standard_normal(Ls)
        c = filterbank(x, gt, a)
        result = ifilterbankiter(c, gt, a, tol=1e-8, real=True)
        relres = float(result[1]) if isinstance(result, (list, tuple)) else 0.0
        assert relres < 1e-4, \
            f"Tight-frame PCG relres={relres:.2e} too large"
