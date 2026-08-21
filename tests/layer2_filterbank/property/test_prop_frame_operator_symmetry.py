"""
test_prop_frame_operator_symmetry.py
=====================================
Python port of:
    layer2_filterbank/property/PropFrameOperatorSymmetry.m

The frame operator S_g x = ifilterbank(filterbank(x, g, a), g, a, L) is:

(1) Self-adjoint:          <S_g x, y> = <x, S_g y>
(2) Positive semi-definite: <S_g x, x> >= 0
(3) S_g(0) = 0
(4) Linear: S_g(α x + β y) = α S_g x + β S_g y
"""

from __future__ import annotations

import pytest

import numpy as np


def _frame_op(x, g, a, L):
    """Compute S_g x = ifilterbank(filterbank(x, g, a), g, a, L)."""
    from cool_frames.filterbanks import filterbank, ifilterbank  # type: ignore
    c   = filterbank(x, g, a)
    # Frame-operator algebra (self-adjoint, complex-linear) is tested on complex
    # inputs, so use real=False: the real=True (2*real(ifft)) synthesis is not
    # complex-linear and would break these algebraic identities.
    Sgx = np.asarray(ifilterbank(c, g, a, L, real=False))
    return Sgx


@pytest.mark.requires_impl
class TestFrameOperatorSelfAdjointImpl:
    """PropFrameOperatorSymmetry: S_g is self-adjoint."""

    def test_self_adjoint(self, needs_impl):
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(42)
        for trial in range(100):
            x   = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            y   = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            Sgx = _frame_op(x, g, a, L)[:Ls]
            Sgy = _frame_op(y, g, a, L)[:Ls]
            lhs = np.dot(Sgx, np.conj(y))
            rhs = np.dot(x,   np.conj(Sgy))
            err = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-15)
            assert err < 1e-8, \
                f"Trial {trial}: self-adjoint error {err:.2e}"


@pytest.mark.requires_impl
class TestFrameOperatorPSDImpl:
    """PropFrameOperatorSymmetry: S_g is PSD."""

    def test_positive_semi_definite(self, needs_impl):
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(43)
        for trial in range(50):
            x          = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            Sgx        = _frame_op(x, g, a, L)[:Ls]
            inner_prod = float(np.real(np.dot(Sgx, np.conj(x))))
            assert inner_prod >= -1e-10, \
                f"Trial {trial}: <S_g x, x>={inner_prod:.4e} < 0 (not PSD)"


@pytest.mark.requires_impl
class TestFrameOperatorZeroImpl:
    """PropFrameOperatorSymmetry: S_g(0) = 0."""

    def test_zero_input(self, needs_impl):
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L   = filterbanklength(Ls, a)
        Sg0 = _frame_op(np.zeros(Ls), g, a, L)
        assert np.linalg.norm(Sg0) < 1e-12, \
            f"S_g(0) norm={np.linalg.norm(Sg0):.2e}, expected 0"


@pytest.mark.requires_impl
class TestFrameOperatorLinearImpl:
    """PropFrameOperatorSymmetry: S_g is linear."""

    def test_linearity(self, needs_impl):
        from cool_frames.filters import (
            audfilters,  # type: ignore
            filterbanklength,  # type: ignore
        )
        Ls, fs = 1024, 8000
        g, a, fc, _, _info = audfilters(fs, Ls)
        L = filterbanklength(Ls, a)
        rng = np.random.default_rng(44)
        for trial in range(30):
            x     = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            y     = rng.standard_normal(Ls) + 1j * rng.standard_normal(Ls)
            alpha = rng.standard_normal() + 1j * rng.standard_normal()
            beta  = rng.standard_normal() + 1j * rng.standard_normal()
            Sgx  = _frame_op(x, g, a, L)[:Ls]
            Sgy  = _frame_op(y, g, a, L)[:Ls]
            Sgz  = _frame_op(alpha * x + beta * y, g, a, L)[:Ls]
            expected = alpha * Sgx + beta * Sgy
            err = np.linalg.norm(Sgz - expected) / (np.linalg.norm(expected) + 1e-15)
            assert err < 1e-10, \
                f"Trial {trial}: frame operator linearity error {err:.2e}"
