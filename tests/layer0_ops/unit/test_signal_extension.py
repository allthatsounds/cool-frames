"""
test_signal_extension.py
========================
Python port of:
    layer0_ops/unit/TestSignalExtension.m

Covers: comp_extBoundary
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import comp_extBoundary_ref

_ALL_MODES = ("per", "sym", "symw", "asym", "asymw", "sp0", "zpd")


# ---------------------------------------------------------------------------
# comp_extBoundary – reference tests
# ---------------------------------------------------------------------------

class TestCompExtBoundaryReference:
    """
    MATLAB counterpart: TestSignalExtension.
    """

    def test_periodic_left_extension(self):
        """'per' left extension == tail of f."""
        rng    = np.random.default_rng(42)
        f      = rng.standard_normal(64)
        extLen = 16
        f_ext  = comp_extBoundary_ref(f, extLen, "per")

        np.testing.assert_allclose(
            f_ext[:extLen], f[len(f) - extLen:],
            atol=1e-14,
            err_msg="Periodic left extension does not match tail of f"
        )

    def test_periodic_right_extension(self):
        """'per' right extension == head of f."""
        rng    = np.random.default_rng(42)
        f      = rng.standard_normal(64)
        extLen = 16
        L      = len(f)
        f_ext  = comp_extBoundary_ref(f, extLen, "per")

        np.testing.assert_allclose(
            f_ext[L + extLen:],
            f[:extLen],
            atol=1e-14,
            err_msg="Periodic right extension does not match head of f"
        )

    def test_zpd_left_zeros(self):
        """'zpd' left extension is zero."""
        rng    = np.random.default_rng(42)
        f      = rng.standard_normal(64)
        extLen = 16
        f_ext  = comp_extBoundary_ref(f, extLen, "zpd")

        np.testing.assert_allclose(f_ext[:extLen], np.zeros(extLen), atol=1e-14)

    def test_zpd_right_zeros(self):
        """'zpd' right extension is zero."""
        rng    = np.random.default_rng(42)
        f      = rng.standard_normal(64)
        extLen = 16
        L      = len(f)
        f_ext  = comp_extBoundary_ref(f, extLen, "zpd")

        np.testing.assert_allclose(f_ext[L + extLen:], np.zeros(extLen), atol=1e-14)

    def test_sym_output_length(self):
        """'sym' mode: total length == L + 2*extLen."""
        rng    = np.random.default_rng(42)
        f      = rng.standard_normal(64)
        extLen = 16
        f_ext  = comp_extBoundary_ref(f, extLen, "sym")

        assert len(f_ext) == len(f) + 2 * extLen

    @pytest.mark.parametrize("mode", _ALL_MODES)
    def test_all_modes_total_length(self, mode):
        """All modes: total length == L + 2*extLen."""
        rng    = np.random.default_rng(42)
        f      = rng.standard_normal(64)
        extLen = 16
        f_ext  = comp_extBoundary_ref(f, extLen, mode)

        assert len(f_ext) == len(f) + 2 * extLen, \
            f"Mode {mode}: expected {len(f) + 2*extLen}, got {len(f_ext)}"

    def test_zero_extlen_identity(self):
        """extLen=0: output identical to input."""
        rng   = np.random.default_rng(42)
        f     = rng.standard_normal(64)
        f_ext = comp_extBoundary_ref(f, 0, "per")

        assert len(f_ext) == len(f)
        np.testing.assert_allclose(f_ext, f, atol=1e-14)

    @pytest.mark.parametrize("mode", ("per", "sym", "zpd"))
    def test_original_signal_preserved(self, mode):
        """Middle segment f_ext[extLen : extLen+L] == f for common modes."""
        rng    = np.random.default_rng(42)
        L      = 128
        f      = rng.standard_normal(L)
        extLen = L // 4
        f_ext  = comp_extBoundary_ref(f, extLen, mode)

        assert len(f_ext) == L + 2 * extLen
        np.testing.assert_allclose(
            f_ext[extLen: extLen + L], f, atol=1e-14,
            err_msg=f"Mode {mode}: original signal not preserved in the middle"
        )

    def test_extlen_equals_L(self):
        """extLen=L: total output length == 3*L."""
        rng    = np.random.default_rng(42)
        L      = 64
        f      = rng.standard_normal(L)
        f_ext  = comp_extBoundary_ref(f, L, "per")

        assert len(f_ext) == 3 * L

    @pytest.mark.parametrize("mode", _ALL_MODES)
    def test_asymmetric_signal_all_modes(self, mode):
        """All modes produce correct length for a generic random signal."""
        rng    = np.random.default_rng(42)
        f      = rng.standard_normal(64)
        extLen = 8
        f_ext  = comp_extBoundary_ref(f, extLen, mode)

        assert len(f_ext) == 64 + 2 * 8, \
            f"Mode {mode}: length mismatch"


# ---------------------------------------------------------------------------
# comp_extBoundary – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCompExtBoundaryImpl:
    """Verify cool_frames.layer0.comp_extBoundary matches reference."""

    @pytest.mark.parametrize("mode", ("per", "zpd", "sym"))
    def test_matches_reference(self, needs_impl, mode):
        from cool_frames.core import comp_extBoundary  # type: ignore

        rng = np.random.default_rng(42)
        for _ in range(5):
            L      = int(rng.integers(32, 128))
            f      = rng.standard_normal(L)
            extLen = int(rng.integers(4, L // 4 + 1))
            np.testing.assert_allclose(
                comp_extBoundary(f, extLen, mode),
                comp_extBoundary_ref(f, extLen, mode),
                atol=1e-14,
                err_msg=f"mode={mode}"
            )
