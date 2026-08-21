"""
test_padding.py
===============
Python port of:
    layer0_ops/unit/TestPadding.m

Covers: postpad, middlepad, fir2long, long2fir

Notes
-----
- postpad_ref is implemented in conftest.py and runs unconditionally.
- middlepad, fir2long, long2fir require complex WPE/HPE symmetry-preserving
  logic that is infeasible to replicate as a pure-numpy reference, so those
  tests are @pytest.mark.requires_impl only.
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import postpad_ref

# ---------------------------------------------------------------------------
# postpad – reference tests (no impl required)
# ---------------------------------------------------------------------------

class TestPostpadReference:
    """
    MATLAB counterpart: TestPadding (postpad section).
    """

    def test_target_length(self):
        """postpad(x, 200) has length 200."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            x      = rng.standard_normal(100)
            result = postpad_ref(x, 200)
            assert len(result) == 200

    def test_truncation(self):
        """postpad(x, 100) when len(x)=200 equals x[:100]."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            x      = rng.standard_normal(200)
            result = postpad_ref(x, 100)
            np.testing.assert_array_equal(result, x[:100])

    def test_extension_preserves_input(self):
        """First len(x) elements of postpad(x, 200) equal x."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            x      = rng.standard_normal(100)
            result = postpad_ref(x, 200)
            np.testing.assert_array_equal(result[: len(x)], x)

    def test_extension_pads_zeros(self):
        """Trailing len(x)..200 elements of postpad(x, 200) are zero."""
        rng = np.random.default_rng(42)
        for _ in range(10):
            x      = rng.standard_normal(100)
            result = postpad_ref(x, 200)
            np.testing.assert_array_equal(result[len(x):], np.zeros(100))

    def test_identity_same_length(self):
        """postpad(x, len(x)) == x."""
        rng = np.random.default_rng(99)
        x      = rng.standard_normal(64)
        result = postpad_ref(x, 64)
        np.testing.assert_array_equal(result, x)


# ---------------------------------------------------------------------------
# postpad – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPostpadImpl:
    """Verify cool_frames.layer0.postpad matches reference."""

    def test_matches_reference(self, needs_impl):
        from cool_frames.core import postpad  # type: ignore

        rng = np.random.default_rng(42)
        for L_out in (128, 50, 256):
            x = rng.standard_normal(100)
            np.testing.assert_allclose(postpad(x, L_out), postpad_ref(x, L_out),
                                       atol=1e-14)


# ---------------------------------------------------------------------------
# middlepad – implementation tests only
# (WPE/HPE symmetry-preserving logic is too complex for a numpy reference)
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestMiddlepadImpl:
    """
    MATLAB counterpart: TestPadding (middlepad section).
    Requires cool_frames.layer0.middlepad.
    """

    def test_identity(self, needs_impl):
        """middlepad(x, len(x)) == x."""
        from cool_frames.core import middlepad  # type: ignore

        rng = np.random.default_rng(42)
        for _ in range(10):
            x = rng.standard_normal(128)
            np.testing.assert_allclose(middlepad(x, len(x)), x, atol=1e-14)

    def test_output_length(self, needs_impl):
        """middlepad(x, 200) has length 200."""
        from cool_frames.core import middlepad  # type: ignore

        rng = np.random.default_rng(42)
        for _ in range(10):
            x      = rng.standard_normal(100)
            result = middlepad(x, 200)
            assert len(result) == 200

    def test_wpe_symmetry(self, needs_impl):
        """middlepad preserves whole-point even (WPE) symmetry.

        A WPE signal of even length N satisfies x[k] = x[N-k] for k=1..N-1
        (i.e., result[1:] == result[1:][::-1]).
        Construction: a_half of length 51 -> x = concat(a_half, flip(a_half[1:-1]))
        giving length 100 WPE signal.
        """
        from cool_frames.core import middlepad  # type: ignore

        rng = np.random.default_rng(42)
        for trial in range(10):
            a_half = rng.standard_normal(51)
            x      = np.concatenate([a_half, a_half[1:-1][::-1]])  # length 100, WPE
            result = middlepad(x, 300)
            np.testing.assert_allclose(
                result[1:], result[1:][::-1], atol=1e-12,
                err_msg=f"Trial {trial}: middlepad did not preserve WPE symmetry"
            )


# ---------------------------------------------------------------------------
# fir2long / long2fir – implementation tests only
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFir2LongImpl:
    """
    MATLAB counterpart: TestPadding (fir2long section).
    """

    def test_output_length(self, needs_impl):
        """resize_fir(h, 512) has length 512."""
        from cool_frames.core import resize_fir  # type: ignore

        rng = np.random.default_rng(42)
        for _ in range(10):
            h      = rng.standard_normal(64)
            result = resize_fir(h, 512)
            assert len(result) == 512

    def test_roundtrip(self, needs_impl):
        """resize_fir(resize_fir(h, 512), 64) ≈ h."""
        from cool_frames.core import resize_fir  # type: ignore

        rng = np.random.default_rng(42)
        for trial in range(10):
            h         = rng.standard_normal(64)
            long_h    = resize_fir(h, 512)
            recovered = resize_fir(long_h, len(h))
            np.testing.assert_allclose(
                recovered, h, atol=1e-12,
                err_msg=f"Trial {trial}: fir2long->long2fir roundtrip failed"
            )


@pytest.mark.requires_impl
class TestLong2FirImpl:
    """
    MATLAB counterpart: TestPadding (long2fir section).
    """

    def test_output_length(self, needs_impl):
        """resize_fir(H, 64) has length 64."""
        from cool_frames.core import resize_fir  # type: ignore

        rng = np.random.default_rng(42)
        for _ in range(10):
            H      = rng.standard_normal(512)
            result = resize_fir(H, 64)
            assert len(result) == 64
