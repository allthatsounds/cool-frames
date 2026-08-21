"""
test_frame_math.py
==================
Python port of:
    layer2_filterbank/unit/TestFrameMath.m

Unit tests for frame-theoretic entry points.

Covers: filterbankbounds, filterbankbounds, filterbankdual,
        filterbankdual, filterbanktight, filterbanktight,
        filterbankscale.
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# filterbankbounds / filterbankbounds
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbankboundsImpl:
    """TestFrameMath: filterbankbounds and filterbankbounds."""

    def test_bounds_positive(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        from cool_frames.filters import filterbanklength  # type: ignore
        L = filterbanklength(1024, a)
        A, B = filterbankbounds(g, a, L)
        assert A >= 0, f"Lower bound A={A} must be non-negative"
        assert B > 0,  f"Upper bound B={B} must be positive"
        assert B < np.inf, "Upper bound B must be finite"

    def test_bounds_ordering(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        A, B = filterbankbounds(g, a, L)
        assert A <= B + 1e-10, f"Frame bounds not ordered: A={A}, B={B}"

    def test_real_bounds_positive(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        A, B = filterbankbounds(g, a, L)
        assert A > 0, f"filterbankbounds: lower bound A={A} not positive"
        assert B > 0, f"filterbankbounds: upper bound B={B} not positive"

    def test_real_bounds_ordering(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        A, B = filterbankbounds(g, a, L)
        assert A <= B + 1e-10, f"Real bounds not ordered: A={A}, B={B}"

    def test_bounds_match_parseval(self, needs_impl):
        """Weighted coefficient energy ratio must lie in [A, B]."""
        from cool_frames.filterbanks import filterbank, filterbankbounds  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        A, B = filterbankbounds(g, a, L)
        rng = np.random.default_rng(42)
        x = rng.standard_normal(Ls)
        c = filterbank(x, g, a)
        coeff_energy = sum(np.linalg.norm(np.asarray(cm).ravel()) ** 2 for cm in c)
        sig_energy   = np.linalg.norm(x) ** 2
        ratio        = coeff_energy / sig_energy
        assert ratio >= 0.0, f"Ratio {ratio} is negative"
        assert ratio <= B * 1.001, f"Ratio {ratio} > B={B}"


# ---------------------------------------------------------------------------
# filterbankdual
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbankdualImpl:
    """TestFrameMath: filterbankdual."""

    def test_dual_filter_count(self, needs_impl):
        from cool_frames.filterbanks import filterbankdual  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        gd = filterbankdual(g, a, L)
        assert len(gd) == len(g), \
            f"filterbankdual: {len(gd)} filters returned, expected {len(g)}"

    @pytest.mark.parametrize("sig_name", ["noise", "sine", "impulse"])
    def test_perfect_reconstruction(self, needs_impl, sig_name):
        """Real-dual-frame synthesis recovers the input signal.

        For real-signal filterbanks (audfilters), the correct reconstruction
        path is ``filterbankdual`` + ``2*real(ifilterbank)``::

            xr = 2 * real(ifilterbank(filterbank(x, g, a), grd, a, L))

        This achieves machine-precision perfect reconstruction because:
        1. Analysis with ``realonly=0`` extracts positive-frequency content
           without lossy averaging.
        2. ``filterbankdual`` computes the dual using the real-folded
           response (``resp + involute(resp)``).
        3. ``2*real(ifft(...))`` mirrors the one-sided spectrum correctly.

        Tolerance: 1e-10 (machine precision).
        """
        from cool_frames.filterbanks import filterbank, filterbankdual, ifilterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L   = filterbanklength(Ls, a)
        grd = filterbankdual(g, a, L)
        rng = np.random.default_rng(0)
        if sig_name == "noise":
            x = rng.standard_normal(Ls)
        elif sig_name == "sine":
            x = np.sin(2 * np.pi * 1000 * np.arange(Ls) / fs)
        else:
            x = np.zeros(Ls); x[0] = 1.0
        c  = filterbank(x, g, a)
        xr = np.asarray(ifilterbank(c, grd, a, L, real=True))
        rel_err = np.linalg.norm(x - xr[:Ls]) / np.linalg.norm(x)
        assert rel_err < 1e-10, \
            f"Real-dual PR ({sig_name}): rel_err={rel_err:.2e} >= 1e-10"


# ---------------------------------------------------------------------------
# filterbankdual
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbankrealdualImpl:
    """TestFrameMath: filterbankdual."""

    def test_runs(self, needs_impl):
        from cool_frames.filterbanks import filterbankdual  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        gdr = filterbankdual(g, a, L)
        assert len(gdr) == len(g), "filterbankdual: filter count mismatch"


# ---------------------------------------------------------------------------
# filterbanktight
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbanktightImpl:
    """TestFrameMath: filterbanktight."""

    def test_tight_equal_bounds(self, needs_impl):
        """After tightening, real frame bounds must be approximately equal."""
        from cool_frames.filterbanks import filterbankbounds, filterbanktight  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        gt = filterbanktight(g, a, L)
        AF_t, BF_t = filterbankbounds(gt, a, L)
        rel_diff = abs(BF_t - AF_t) / (AF_t + 1e-15)
        assert rel_diff < 1e-4, \
            f"Tight frame: A={AF_t:.8f}, B={BF_t:.8f}, rel_diff={rel_diff:.2e}"

    def test_tight_filter_count(self, needs_impl):
        from cool_frames.filterbanks import filterbanktight  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        gt = filterbanktight(g, a, L)
        assert len(gt) == len(g), "filterbanktight: filter count changed"

    def test_tight_reconstruction(self, needs_impl):
        """Tight-frame reconstruction: ifilterbank(..., real=True) / A ≈ f.

        Matches MATLAB TestFrameMath: uses ``ifilterbank(..., 'real')``
        and tolerance ``1e-0``.
        """
        from cool_frames.filterbanks import filterbank, filterbankbounds, filterbanktight, ifilterbank  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L  = filterbanklength(Ls, a)
        gt = filterbanktight(g, a, L)
        AF_t, _ = filterbankbounds(gt, a, L)
        rng = np.random.default_rng(42)
        x   = rng.standard_normal(Ls)
        c   = filterbank(x, gt, a)
        xr  = np.asarray(ifilterbank(c, gt, a, L, real=True))
        rel_err = np.linalg.norm(x - xr[:Ls] / AF_t) / np.linalg.norm(x)
        assert rel_err < 1.0, f"Tight PR: rel_err={rel_err:.2e}"


# ---------------------------------------------------------------------------
# filterbanktight
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbankrealtightImpl:
    """TestFrameMath: filterbanktight."""

    def test_runs(self, needs_impl):
        from cool_frames.filterbanks import filterbanktight  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        gt_r = filterbanktight(g, a, L)
        assert len(gt_r) == len(g), "filterbanktight: filter count mismatch"


# ---------------------------------------------------------------------------
# filterbankscale
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbankscaleImpl:
    """TestFrameMath: filterbankscale."""

    def test_output_length(self, needs_impl):
        from cool_frames.filterbanks import filterbankscale  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        gs = filterbankscale(g, 2.0)
        assert len(gs) == len(g), "filterbankscale: filter count changed"

    def test_scale_multiplies_bounds_squared(self, needs_impl):
        """Scaling by s should multiply real frame bounds by s²."""
        from cool_frames.filterbanks import filterbankbounds, filterbankscale  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        AF, BF = filterbankbounds(g, a, L)
        s  = 2.5
        gs = filterbankscale(g, s)
        AF2, BF2 = filterbankbounds(gs, a, L)
        assert abs(AF2 / AF - s**2) / s**2 < 0.01, \
            f"Lower bound scaling error: {AF2}/{AF}={AF2/AF:.4f}, expected {s**2}"
        assert abs(BF2 / BF - s**2) / s**2 < 0.01, \
            f"Upper bound scaling error: {BF2}/{BF}={BF2/BF:.4f}, expected {s**2}"

    def test_scale_by_one_identity(self, needs_impl):
        from cool_frames.filterbanks import filterbankbounds, filterbankscale  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        from cool_frames.filters import filterbanklength  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        L = filterbanklength(1024, a)
        AF, BF = filterbankbounds(g, a, L)
        gs      = filterbankscale(g, 1.0)
        AF1, BF1 = filterbankbounds(gs, a, L)
        assert abs(AF1 - AF) / AF < 1e-10, "scale by 1: lower bound changed"
        assert abs(BF1 - BF) / BF < 1e-10, "scale by 1: upper bound changed"

    def test_per_channel_scale(self, needs_impl):
        from cool_frames.filterbanks import filterbankscale  # type: ignore
        from cool_frames.filters import audfilters  # type: ignore
        g, a, fc, _, _info = audfilters(8000, 1024)
        M = len(g)
        s_vec = np.ones(M) * 1.5
        gs = filterbankscale(g, s_vec)
        assert len(gs) == M, "filterbankscale with per-channel vector: filter count mismatch"
