"""gabfilters kept only M of the L bins of its transformed window (2026-09-28).

The port stored ``gnum[center - M//2 : center + M - M//2]`` on the reasoning
that a window of M samples has its spectrum "within ~M bins".  The support in
bins scales with L/M, not M: the kept band was M**2/L channel spacings wide.
The bank therefore departed from the DGT it documents as L grew past M**2 and
stopped being a frame near L = M**2 -- M = 16, a = 4 at L = 1024, a tight Hann
frame, was reported "not a frame"; at M = 480, a = 120, L = 144000 the
magnitudes were 28 % off ``dgtreal`` and the bounds read (2.53, 3.84) instead
of (4, 4).  LTFAT's gabfilters.m stores all L bins; so does the port now.

Storing a length-L response with a non-zero ``foff`` then exposed a second,
latent defect: ``prepare_filters`` routed any length-L ``H`` to the full-length
kernels, which never read ``foff`` -- so analysis and synthesis applied every
channel at DC while ``filter_freqresp`` (bounds, duals) applied it at its
centre frequency.  No designer produced such a filter before, so the two paths
had never met.  ``prepare_filters`` now rotates it into place.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from cool_frames.numpy.filterbanks import (
    filterbank,
    filterbankbounds,
    filterbankdual,
    ifilterbank,
)
from cool_frames.numpy.filters import gabfilters
from cool_frames.numpy.filters._gabfilters import _gabwin
from cool_frames.numpy.gabor import dgtreal, gabframebounds


def _as_gabfilters(C, a, M):
    """dgtreal's coefficients in gabfilters' convention: time-invariant
    phase, and the 1/sqrt(2) the real-mode bank puts on its edge channels."""
    m = np.arange(C.shape[0])[:, None]
    n = np.arange(C.shape[1])[None, :]
    edge = np.ones(C.shape[0])
    edge[0] = 1 / np.sqrt(2)
    if M % 2 == 0:
        edge[-1] = 1 / np.sqrt(2)
    return C * np.exp(2j * np.pi * m * n * a / M) * edge[:, None]


# (M, a, Ls): from L < M**2, where the old truncation was nearly harmless,
# to L >> M**2, where it was not a frame.
CASES = [(480, 120, 4800), (480, 120, 24000), (256, 64, 16000),
         (64, 16, 8192), (16, 4, 1024), (17, 5, 1000)]


@pytest.mark.parametrize("M,a,Ls", CASES)
def test_gabfilters_is_dgtreal_at_every_length(M, a, Ls):
    """LTFAT's documented identity: ufilterbank(f, gabfilters(...)) equals
    dgtreal(f, g, a, M, 'timeinv')."""
    warnings.simplefilter("ignore")
    g, aout, _fc, L, _info = gabfilters(24000, Ls, M=M, a=a)
    x = np.random.default_rng(M * 1000 + a).standard_normal(L)
    c = np.stack(filterbank(x, g, aout, L))
    C = np.asarray(dgtreal(x, _gabwin("hann", M, norm="energy"), a, M, L))
    if C.shape != c.shape:
        C = C.T
    ref = _as_gabfilters(C, a, M)
    assert np.linalg.norm(c - ref) / np.linalg.norm(ref) < 1e-10


@pytest.mark.parametrize("M,a,Ls", CASES)
def test_gabfilters_bounds_and_round_trip(M, a, Ls):
    warnings.simplefilter("ignore")
    g, aout, _fc, L, info = gabfilters(24000, Ls, M=M, a=a)
    win = _gabwin("hann", M, norm="energy")
    A, B = filterbankbounds(g, aout, L)
    Ad, Bd = gabframebounds(win, a, M, L)
    assert A == pytest.approx(Ad, rel=1e-9) and B == pytest.approx(Bd, rel=1e-9)
    assert info["admissible"]["is_frame"] is True
    x = np.random.default_rng(7).standard_normal(L)
    y = np.real(ifilterbank(filterbank(x, g, aout, L), filterbankdual(g, aout, L), aout, Ls=L))
    assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-12


def test_full_length_response_with_offset_is_placed_where_the_frame_algebra_places_it():
    """A length-L ``H`` with ``foff != 0`` must be analysed at its centre
    frequency, as ``filter_freqresp`` describes it, not at DC."""
    from cool_frames.numpy.filters._filters import filter_freqresp

    L, a = 256, 4
    rng = np.random.default_rng(3)
    H = rng.standard_normal(L) + 1j * rng.standard_normal(L)
    g = [{"H": H, "foff": 37, "realonly": 0, "delay": 0}]
    x = rng.standard_normal(L)
    c = filterbank(x, g, np.array([a]), L)[0]
    Hfull, _ = filter_freqresp(g[0], L)
    expect = np.fft.ifft((np.fft.fft(x) * Hfull).reshape(a, L // a).sum(axis=0)) / a
    np.testing.assert_allclose(c, expect, rtol=1e-12, atol=1e-12)


def test_torch_full_length_response_with_offset_matches_numpy():
    """The torch backend had the same full-length path, with the same blind
    spot for ``foff``: analysis differed from NumPy by O(1) and the round
    trip read +2 dB."""
    torch = pytest.importorskip("torch")
    from cool_frames.numpy.filterbanks import filterbankdual as np_dual
    from cool_frames.torch.filterbanks import filterbank as t_fb
    from cool_frames.torch.filterbanks import ifilterbank as t_ifb
    from cool_frames.torch.filters import gabfilters as t_gab
    from cool_frames.torch.filters._wrappers import numpy_filters_to_torch

    warnings.simplefilter("ignore")
    g, aout, _fc, L, _ = gabfilters(16000, 8192, M=64, a=16)
    gt, *_ = t_gab(16000, 8192, M=64, a=16)
    x = np.random.default_rng(0).standard_normal(L)
    c = filterbank(x, g, aout, L)
    ct = t_fb(torch.tensor(x), gt, aout, L)
    for cm, ctm in zip(c, ct):
        np.testing.assert_allclose(ctm.detach().numpy().ravel(), cm, rtol=0, atol=1e-12)
    y = t_ifb(ct, numpy_filters_to_torch(np_dual(g, aout, L), L), aout, L)
    y = np.real(y.detach().numpy().ravel()[:L])
    assert np.linalg.norm(x - y) / np.linalg.norm(x) < 1e-12
