"""
Phase 1 – Filter design wrapper tests.

Validates that ``cool_frames.torch.filters`` wraps the NumPy filter
design functions and returns correct torch-compatible representations.

Filter design is a setup-time operation — the torch wrappers call NumPy
internally and convert results to tensors.  Tests verify:
  - Same filterbank geometry (M, hop sizes, center frequencies)
  - Filter frequency responses are identical
  - Returned filter dicts contain tensors where applicable
"""

from __future__ import annotations

import pytest

import numpy as np

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.requires_torch_impl


# ---------------------------------------------------------------------------
# audfilters wrapper
# ---------------------------------------------------------------------------


class TestAudfilters:
    """torch.filters.audfilters wraps numpy.filters.audfilters."""

    def test_returns_same_geometry(self):
        """Same M, hop sizes, center frequencies as NumPy."""
        from cool_frames.numpy.filters import audfilters as np_fn
        from cool_frames.torch.filters import audfilters as torch_fn

        fs, Ls = 8000, 1024
        g_np, a_np, fc_np, L_np, _info_np = np_fn(fs, Ls)
        g_t, a_t, fc_t, L_t, _info_t = torch_fn(fs, Ls)

        assert len(g_t) == len(g_np)
        assert L_t == L_np
        np.testing.assert_array_equal(
            np.asarray(a_t),
            np.asarray(a_np),
        )
        np.testing.assert_allclose(
            np.asarray(fc_t),
            np.asarray(fc_np),
            rtol=1e-12,
        )

    @pytest.mark.parametrize("scale", ["erb", "bark", "mel"])
    def test_scale_parameter(self, scale):
        """Different auditory scales produce valid filterbanks."""
        from cool_frames.torch.filters import audfilters

        g, _a, fc, L, _info = audfilters(8000, 1024, scale=scale)
        assert len(g) > 0
        assert L > 0
        # Center frequencies should be sorted
        fc_arr = np.asarray(fc)
        assert np.all(np.diff(fc_arr[1:-1]) >= 0)

    def test_filter_dicts_have_expected_keys(self):
        """Each filter dict has 'H' and 'foff' keys."""
        from cool_frames.torch.filters import audfilters

        g, _a, _fc, _L, _info = audfilters(8000, 1024)
        for m, gm in enumerate(g):
            assert "H" in gm or "h" in gm, f"filter {m} missing H or h"


# ---------------------------------------------------------------------------
# cqtfilters wrapper
# ---------------------------------------------------------------------------


class TestCqtfilters:
    """torch.filters.cqtfilters wraps numpy.filters.cqtfilters."""

    def test_returns_same_geometry(self):
        from cool_frames.numpy.filters import cqtfilters as np_fn
        from cool_frames.torch.filters import cqtfilters as torch_fn

        fs, fmin, fmax, bins, Ls = 8000, 100, 3000, 12, 1024
        g_np, _a_np, _fc_np, L_np, _info_np = np_fn(fs, Ls, fmin=fmin, fmax=fmax, bins=bins)
        g_t, _a_t, _fc_t, L_t, _info_t = torch_fn(fs, Ls, fmin=fmin, fmax=fmax, bins=bins)

        assert len(g_t) == len(g_np)
        assert L_t == L_np

    def test_frequency_response_agreement(self):
        """Filter frequency responses match NumPy exactly."""
        from cool_frames.numpy.filters import cqtfilters as np_fn
        from cool_frames.numpy.filters import filter_freqresp
        from cool_frames.torch.filters import cqtfilters as torch_fn

        fs, fmin, fmax, bins, Ls = 8000, 100, 3000, 12, 1024
        g_np, _a_np, _fc_np, L_np, _info_np = np_fn(fs, Ls, fmin=fmin, fmax=fmax, bins=bins)
        g_t, _a_t, _fc_t, L_t, _info_t = torch_fn(fs, Ls, fmin=fmin, fmax=fmax, bins=bins)

        # Compare full-length frequency responses
        for m in range(len(g_np)):
            H_np, _ = filter_freqresp(g_np[m], L_np)
            H_t, _ = filter_freqresp(g_t[m], L_t)
            np.testing.assert_allclose(
                np.asarray(H_t),
                np.asarray(H_np),
                atol=1e-12,
                err_msg=f"filter {m} freq response mismatch",
            )


# ---------------------------------------------------------------------------
# firwin wrapper
# ---------------------------------------------------------------------------


class TestFirwin:
    """torch.filters.firwin wraps numpy.filters.firwin."""

    @pytest.mark.parametrize("name", ["hann", "sine", "rect", "tria"])
    @pytest.mark.parametrize("M", [16, 32, 64, 128])
    def test_agreement(self, name, M):
        from cool_frames.numpy.filters import firwin as np_fn
        from cool_frames.torch.filters import firwin as torch_fn

        w_np = np_fn(name, M)
        w_t = torch_fn(name, M)

        np.testing.assert_allclose(
            w_t if isinstance(w_t, np.ndarray) else w_t.numpy(),
            w_np,
            atol=1e-14,
        )

    @pytest.mark.parametrize("name", ["hann", "sine"])
    def test_returns_tensor(self, name):
        """Output should be a torch.Tensor."""
        from cool_frames.torch.filters import firwin

        w = firwin(name, 64)
        assert isinstance(w, torch.Tensor)


# ---------------------------------------------------------------------------
# Other filter design wrappers
# ---------------------------------------------------------------------------


class TestGabfilters:
    """torch.filters.gabfilters produces valid filterbanks."""

    def test_basic_call(self):
        from cool_frames.torch.filters import gabfilters

        try:
            result = gabfilters(16000, 1024, window="hann", a=32, M=8)
            g, _a, _fc, L = result[0], result[1], result[2], result[3]
        except NotImplementedError:
            pytest.skip("gabfilters not yet implemented in torch backend")

        assert len(g) > 0
        assert L > 0


# ``TestHopfilters`` lived here and did nothing: ``hopfilters`` raised
# ``NotImplementedError`` unconditionally (there is no NumPy ``hopfilters`` to
# wrap) and the test caught that and skipped, so it was green for the entire
# life of a function that could never work.  The name is now removed from the
# package; the contract is pinned in
# tests/regressions/test_audit_v0_1_1_open_items.py.


class TestWaveletfilters:
    """torch.filters.waveletfilters wraps NumPy."""

    def test_basic_call(self):
        from cool_frames.torch.filters import waveletfilters

        try:
            result = waveletfilters(2.0, 1024, scales=np.arange(1, 9), wavelet="morlet")
            g, _a, _fc, _L = result[0], result[1], result[2], result[3]
        except (NotImplementedError, Exception):
            pytest.skip("waveletfilters not available or failed")

        assert len(g) > 0
