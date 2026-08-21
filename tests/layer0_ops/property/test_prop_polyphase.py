"""
test_prop_polyphase.py
======================
Python port of:
    layer0_ops/property/PropPolyphaseEquivalence.m

Property tests: FFT path and TD path agree for FIR filters.

For causal FIR filters with periodic boundary extension:
    comp_filterbank_fft(fft(f), G_fft, a)
    should equal
    comp_filterbank_td(f, G_td, a, offset, 'per')
where G_fft[m] = fft(postpad(h_m, L)) and G_td[m] = h_m.

Uses the polyphase_fb fixture (from conftest):
    Ls, M, Lh, a, G_td, G_fft, offset, noise_real
"""

from __future__ import annotations

import pytest

import numpy as np
from conftest import postpad_ref

# ---------------------------------------------------------------------------
# Reference consistency test (unconditional) — zero-input
# ---------------------------------------------------------------------------

class TestPolyphaseReference:
    """
    Structural check: G_fft[m] == fft(postpad(G_td[m], Ls)).
    No impl required.
    """

    def test_fft_filter_consistency(self, polyphase_fb):
        """G_fft[m] == fft(postpad(G_td[m], Ls)) for all bands."""
        Ls    = polyphase_fb["Ls"]
        G_td  = polyphase_fb["G_td"]
        G_fft = polyphase_fb["G_fft"]

        for m, (h, H) in enumerate(zip(G_td, G_fft)):
            H_ref = np.fft.fft(postpad_ref(h, Ls))
            np.testing.assert_allclose(
                H, H_ref, atol=1e-12,
                err_msg=f"Band {m}: G_fft mismatch with fft(postpad(G_td))"
            )

    def test_zero_input_both_paths(self, polyphase_fb):
        """Zero signal -> zero at DFT level for the FFT filter."""
        Ls    = polyphase_fb["Ls"]
        G_fft = polyphase_fb["G_fft"]
        a     = polyphase_fb["a"]
        F_zero = np.zeros(Ls, dtype=complex)

        for m, (H, a_m) in enumerate(zip(G_fft, a)):
            N_m = Ls // a_m
            c_m = np.fft.ifft(F_zero * H)[:N_m]
            np.testing.assert_allclose(
                c_m, np.zeros(N_m, dtype=complex), atol=1e-14,
                err_msg=f"Band {m}: non-zero for zero input (FFT path)"
            )


# ---------------------------------------------------------------------------
# Implementation polyphase equivalence tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestPolyphaseEquivalenceImpl:
    """
    MATLAB counterpart: PropPolyphaseEquivalence.
    """

    @pytest.mark.parametrize("seed", [42, 123, 7, 11, 55])
    def test_fft_vs_td_path(self, needs_impl, seed, polyphase_fb):
        """FFT path and TD path agree to within 1e-8 (relative) for 'per' boundary."""
        from cool_frames.core import comp_filterbank_fft, comp_filterbank_td  # type: ignore

        rng = np.random.default_rng(seed)
        Ls  = polyphase_fb["Ls"]
        x   = rng.standard_normal(Ls)
        F   = np.fft.fft(x)

        c_fft = comp_filterbank_fft(F, polyphase_fb["G_fft"], polyphase_fb["a"])
        c_td  = comp_filterbank_td(x, polyphase_fb["G_td"],
                                   polyphase_fb["a"], polyphase_fb["offset"], "per")

        for m in range(polyphase_fb["M"]):
            c_fft_m = np.asarray(c_fft[m])
            c_td_m  = np.asarray(c_td[m])
            N_min   = min(len(c_fft_m.ravel()), len(c_td_m.ravel()))

            diff  = np.max(np.abs(c_fft_m.ravel()[:N_min] - c_td_m.ravel()[:N_min]))
            scale = np.max(np.abs(c_fft_m.ravel()[:N_min]))
            if scale < 1e-15:
                scale = 1.0

            assert diff / scale < 1e-8, \
                f"seed={seed}, band={m}: FFT vs TD mismatch (rel err={diff/scale:.2e})"

    def test_zero_input_both_paths(self, needs_impl, polyphase_fb):
        """Zero signal -> zero output on both FFT and TD paths."""
        from cool_frames.core import comp_filterbank_fft, comp_filterbank_td  # type: ignore

        Ls     = polyphase_fb["Ls"]
        x_zero = np.zeros(Ls)
        F_zero = np.fft.fft(x_zero)

        c_fft = comp_filterbank_fft(F_zero, polyphase_fb["G_fft"], polyphase_fb["a"])
        c_td  = comp_filterbank_td(x_zero, polyphase_fb["G_td"],
                                   polyphase_fb["a"], polyphase_fb["offset"], "per")

        for m in range(polyphase_fb["M"]):
            np.testing.assert_allclose(
                np.asarray(c_fft[m]), np.zeros_like(np.asarray(c_fft[m])),
                atol=1e-14, err_msg=f"Band {m}: FFT path non-zero for zero input"
            )
            np.testing.assert_allclose(
                np.asarray(c_td[m]), np.zeros_like(np.asarray(c_td[m])),
                atol=1e-14, err_msg=f"Band {m}: TD path non-zero for zero input"
            )
