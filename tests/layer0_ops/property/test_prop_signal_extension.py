"""
test_prop_signal_extension.py
==============================
Python port of:
    layer0_ops/property/PropSignalExtensionModes.m

Property tests for comp_extBoundary across signal extension modes.
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import comp_extBoundary_ref

# ---------------------------------------------------------------------------
# Reference tests (unconditional)
# ---------------------------------------------------------------------------

class TestSignalExtensionModesReference:
    """
    MATLAB counterpart: PropSignalExtensionModes.
    """

    def test_periodic_mode(self):
        """'per': left extension = tail of f, right extension = head of f."""
        rng = np.random.default_rng(42)
        for trial in range(50):
            L      = int(rng.integers(32, 257))
            f      = rng.standard_normal(L)
            extLen = int(rng.integers(4, max(5, L // 4 + 1)))

            f_ext = comp_extBoundary_ref(f, extLen, "per")

            assert len(f_ext) == L + 2 * extLen, \
                f"Trial {trial}: wrong total length"
            np.testing.assert_allclose(
                f_ext[:extLen], f[L - extLen:], atol=1e-14,
                err_msg=f"Trial {trial}: left extension wrong"
            )
            np.testing.assert_allclose(
                f_ext[L + extLen:], f[:extLen], atol=1e-14,
                err_msg=f"Trial {trial}: right extension wrong"
            )

    def test_zpd_mode(self):
        """'zpd': extended region is exactly zero."""
        rng = np.random.default_rng(42)
        for trial in range(50):
            L      = int(rng.integers(32, 257))
            f      = rng.standard_normal(L)
            extLen = int(rng.integers(4, max(5, L // 4 + 1)))

            f_ext = comp_extBoundary_ref(f, extLen, "zpd")

            np.testing.assert_allclose(
                f_ext[:extLen], np.zeros(extLen), atol=1e-14,
                err_msg=f"Trial {trial}: left extension not zero"
            )
            np.testing.assert_allclose(
                f_ext[L + extLen:], np.zeros(extLen), atol=1e-14,
                err_msg=f"Trial {trial}: right extension not zero"
            )

    def test_sym_mode_length(self):
        """'sym': output has correct length L + 2*extLen."""
        rng = np.random.default_rng(42)
        for trial in range(50):
            L      = int(rng.integers(32, 257))
            f      = rng.standard_normal(L)
            extLen = min(int(rng.integers(4, max(5, L // 4 + 1))), L - 1)

            f_ext = comp_extBoundary_ref(f, extLen, "sym")

            assert len(f_ext) == L + 2 * extLen, \
                f"Trial {trial}: incorrect extended length"

    def test_zero_extlen(self):
        """extLen=0: output identical to input."""
        rng = np.random.default_rng(42)
        for _ in range(30):
            f     = rng.standard_normal(64)
            f_ext = comp_extBoundary_ref(f, 0, "per")
            assert len(f_ext) == len(f)
            np.testing.assert_allclose(f_ext, f, atol=1e-14)

    def test_quarter_length_extlen(self):
        """extLen = L/4: middle segment preserved and total length correct for all modes."""
        rng   = np.random.default_rng(42)
        modes = ("per", "sym", "zpd")
        for _ in range(30):
            L      = 128
            f      = rng.standard_normal(L)
            extLen = L // 4

            for mode in modes:
                f_ext = comp_extBoundary_ref(f, extLen, mode)
                assert len(f_ext) == L + 2 * extLen, \
                    f"mode={mode}: length mismatch"
                np.testing.assert_allclose(
                    f_ext[extLen: extLen + L], f, atol=1e-14,
                    err_msg=f"mode={mode}: original signal not preserved"
                )

    @pytest.mark.parametrize("mode", ("per", "sym", "zpd"))
    def test_asymmetric_signal_all_modes(self, mode):
        """All three core modes produce correct total length for random signals."""
        rng = np.random.default_rng(42)
        for trial in range(50):
            L      = int(rng.integers(32, 257))
            f      = rng.standard_normal(L)
            extLen = int(rng.integers(4, min(33, max(5, L // 4 + 1))))

            f_ext = comp_extBoundary_ref(f, extLen, mode)
            assert len(f_ext) == L + 2 * extLen, \
                f"Trial {trial}, mode={mode}: length mismatch"


# ---------------------------------------------------------------------------
# Implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestSignalExtensionModesImpl:
    """Verify cool_frames.layer0.comp_extBoundary matches reference."""

    @pytest.mark.parametrize("mode", ("per", "zpd", "sym", "symw", "asym", "asymw", "sp0"))
    def test_matches_reference(self, needs_impl, mode):
        from cool_frames.core import comp_extBoundary  # type: ignore

        rng = np.random.default_rng(42)
        for _ in range(10):
            L      = int(rng.integers(32, 128))
            f      = rng.standard_normal(L)
            extLen = int(rng.integers(2, max(3, L // 4 + 1)))

            np.testing.assert_allclose(
                comp_extBoundary(f, extLen, mode),
                comp_extBoundary_ref(f, extLen, mode),
                atol=1e-14,
                err_msg=f"mode={mode}"
            )
