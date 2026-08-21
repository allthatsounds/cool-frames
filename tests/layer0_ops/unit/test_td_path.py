"""
test_td_path.py
===============
Python port of:
    layer0_ops/unit/TestTDPath.m

Covers: comp_filterbank_td, comp_ifilterbank_td

Calling conventions
-------------------
comp_filterbank_td(f, g, a, offset, ext)
    f      : time-domain signal [L] or [L x W]
    g      : list of M FIR impulse response vectors
    a      : subsampling factors [M]
    offset : filter offset vector [M]; offset=0 => causal (skip=0)
    ext    : boundary extension string ('per', 'zpd', ...)

comp_ifilterbank_td(c, g, a, Ls, offset, ext)
    c      : list of M coefficient arrays
    g      : same list of FIR impulse responses
    a      : upsampling factors [M]
    Ls     : desired output length
    offset : synthesis filter offset vector [M]
    ext    : boundary extension string

The fixture td_fb (from conftest) provides:
    Ls, M, Lh, a, G_td, offset, noise_real, zeros_sig, impulse
"""

from __future__ import annotations

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Structural reference tests (no impl required)
# ---------------------------------------------------------------------------

class TestTDPathReference:
    """
    Structural sanity checks on the synthetic TD fixture.
    MATLAB counterpart: TestTDPath.
    """

    def test_filter_count(self, td_fb):
        """Number of FIR filters equals M."""
        assert len(td_fb["G_td"]) == td_fb["M"]

    def test_filter_lengths(self, td_fb):
        """Each G_td[m] has length Lh."""
        Lh = td_fb["Lh"]
        for m, h in enumerate(td_fb["G_td"]):
            assert len(h) == Lh, f"G_td[{m}] length {len(h)}, expected {Lh}"

    def test_signal_lengths(self, td_fb):
        """Input signals have length Ls."""
        Ls = td_fb["Ls"]
        assert len(td_fb["noise_real"]) == Ls
        assert len(td_fb["zeros_sig"])  == Ls
        assert len(td_fb["impulse"])    == Ls

    def test_impulse_position(self, td_fb):
        """Impulse has non-zero at index 0 and zeros elsewhere."""
        imp = td_fb["impulse"]
        assert imp[0] == 1.0
        assert np.all(imp[1:] == 0.0)

    def test_offset_length(self, td_fb):
        """offset vector has length M."""
        assert len(td_fb["offset"]) == td_fb["M"]


# ---------------------------------------------------------------------------
# comp_filterbank_td – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCompFilterbankTDImpl:
    """
    MATLAB counterpart: TestTDPath (comp_filterbank_td section).
    """

    def test_zero_input(self, needs_impl, td_fb):
        """Zero signal -> all subband outputs zero."""
        from cool_frames.core import comp_filterbank_td  # type: ignore

        c = comp_filterbank_td(td_fb["zeros_sig"], td_fb["G_td"],
                               td_fb["a"], td_fb["offset"], "per")

        for m, c_m in enumerate(c):
            np.testing.assert_allclose(
                c_m, np.zeros_like(c_m), atol=1e-14,
                err_msg=f"Subband {m} not zero for zero input"
            )

    def test_impulse_response(self, needs_impl, td_fb):
        """Impulse input -> at least one subband is non-zero."""
        from cool_frames.core import comp_filterbank_td  # type: ignore

        c = comp_filterbank_td(td_fb["impulse"], td_fb["G_td"],
                               td_fb["a"], td_fb["offset"], "per")

        non_zero = sum(1 for c_m in c if np.any(np.asarray(c_m) != 0))
        assert non_zero > 0, "No subband responded to impulse"

    def test_output_sizes(self, needs_impl, td_fb):
        """Each c[m] has ceil(Ls/a[m]) rows for 'per' extension."""
        from cool_frames.core import comp_filterbank_td  # type: ignore

        Ls = td_fb["Ls"]
        c  = comp_filterbank_td(td_fb["noise_real"], td_fb["G_td"],
                                td_fb["a"], td_fb["offset"], "per")

        for m, (c_m, a_m) in enumerate(zip(c, td_fb["a"])):
            expected = int(np.ceil(Ls / a_m))
            assert np.asarray(c_m).shape[0] == expected, \
                f"Subband {m}: expected {expected} rows, got {np.asarray(c_m).shape[0]}"

    def test_periodic_boundary(self, needs_impl, td_fb):
        """'per' extension runs without error and returns M subbands."""
        from cool_frames.core import comp_filterbank_td  # type: ignore

        c = comp_filterbank_td(td_fb["noise_real"], td_fb["G_td"],
                               td_fb["a"], td_fb["offset"], "per")
        assert len(c) == td_fb["M"]

    def test_zpd_boundary(self, needs_impl, td_fb):
        """'zpd' extension runs without error and returns M subbands."""
        from cool_frames.core import comp_filterbank_td  # type: ignore

        c = comp_filterbank_td(td_fb["noise_real"], td_fb["G_td"],
                               td_fb["a"], td_fb["offset"], "zpd")
        assert len(c) == td_fb["M"]


# ---------------------------------------------------------------------------
# comp_ifilterbank_td – implementation tests
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestCompIFilterbankTDImpl:
    """
    MATLAB counterpart: TestTDPath (comp_ifilterbank_td section).
    """

    def test_forward_inverse_output_length(self, needs_impl, td_fb):
        """Analysis + synthesis output has length Ls."""
        from cool_frames.core import comp_filterbank_td, comp_ifilterbank_td  # type: ignore

        Ls       = td_fb["Ls"]
        Lh       = td_fb["Lh"]
        c        = comp_filterbank_td(td_fb["noise_real"], td_fb["G_td"],
                                      td_fb["a"], td_fb["offset"], "per")
        offset_s = -(Lh - 1) * np.ones(td_fb["M"], dtype=int)
        f_recon  = comp_ifilterbank_td(c, td_fb["G_td"], td_fb["a"],
                                       Ls, offset_s, "per")

        assert np.asarray(f_recon).shape[0] == Ls

    def test_forward_inverse_nonzero(self, needs_impl, td_fb):
        """Synthesis output is non-trivially non-zero for non-zero input."""
        from cool_frames.core import comp_filterbank_td, comp_ifilterbank_td  # type: ignore

        Ls       = td_fb["Ls"]
        Lh       = td_fb["Lh"]
        c        = comp_filterbank_td(td_fb["noise_real"], td_fb["G_td"],
                                      td_fb["a"], td_fb["offset"], "per")
        offset_s = -(Lh - 1) * np.ones(td_fb["M"], dtype=int)
        f_recon  = comp_ifilterbank_td(c, td_fb["G_td"], td_fb["a"],
                                       Ls, offset_s, "per")

        assert np.linalg.norm(np.asarray(f_recon)) > 0, \
            "Synthesis output is zero for non-zero input"

    def test_short_signal_padded(self, needs_impl, td_fb):
        """Short signal zero-padded to Ls: analysis runs without error."""
        from conftest import postpad_ref
        from cool_frames.core import comp_filterbank_td  # type: ignore

        Ls         = td_fb["Ls"]
        short_sig  = np.random.default_rng(7).standard_normal(16)
        padded     = postpad_ref(short_sig, Ls)

        c = comp_filterbank_td(padded, td_fb["G_td"],
                               td_fb["a"], td_fb["offset"], "per")
        assert len(c) == td_fb["M"]
