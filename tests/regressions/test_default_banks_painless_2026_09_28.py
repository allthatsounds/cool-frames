"""Regression: the designers' default banks are painless, and say so.

Found 2026-09-28, once the admissibility verdicts stopped calling banks with
an aliasing channel frames (G6): 1197 of the W03 sweep's default
configurations of ``audfilters``, ``greenwoodfilters`` and ``cqtfilters``
had a channel over its painless limit; the 477 greenwoodfilters and
cqtfilters banks among them reconstructed with the canonical dual to a median
of 3.8e-3, up to 0.18, silently (DEFECT_REGISTER G8); ``audfilters(scale=
'greenwood')`` had defaults for a different unit (G9); an empty wavelet
channel made ``filterbankdual`` raise a misleading FIR error (G10).
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pytest

from cool_frames.numpy.filterbanks import filterbank, filterbankdual, ifilterbank
from cool_frames.numpy.filterbanks._frame import _nonpainless_channels
from cool_frames.numpy.filterbanks._utils import normalise_a, prepare_filters
from cool_frames.numpy.filters import audfilters, cqtfilters, greenwoodfilters, waveletfilters


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def _round_trip(g, a, L):
    x = np.random.default_rng(0).standard_normal(L)
    c = filterbank(x, g, a, L)
    y = np.real(ifilterbank(c, filterbankdual(g, a, L), a, L))[:L]
    return np.linalg.norm(x - y) / np.linalg.norm(x)


def _n_aliasing(g, a, L):
    an = normalise_a(a, len(g))
    return len(_nonpainless_channels(prepare_filters(g, an, L)[0], an, L))


@pytest.mark.parametrize("make,was", [
    (lambda: cqtfilters(8000, 512, fmin=20, bins=1, Qvar=0.5), 1.47e-2),
    (lambda: cqtfilters(8000, 512, fmin=20, bins=12, Qvar=1.5), 3.62e-2),
    (lambda: cqtfilters(8000, 512, fmin=50, bins=24, Qvar=1.0), 6.61e-2),
    (lambda: greenwoodfilters(8000, 512, M=14), 5.32e-3),
    (lambda: greenwoodfilters(8000, 4096, M=5), 2.28e-3),
    (lambda: greenwoodfilters(16000, 512, M=4), 6.50e-2),
    (lambda: greenwoodfilters(16000, 512, M=5), 4.21e-3),
], ids=["cqt-b1", "cqt-b12", "cqt-b24", "gw-M14", "gw-4096-M5", "gw-M4", "gw-16k-M5"])
def test_complement_and_edge_channels_are_repaired(make, was):
    g, a, fc, L, info = _quiet(make)
    assert _n_aliasing(g, a, L) == 0
    assert info["painless"] is True
    assert _round_trip(g, a, L) < 1e-13, f"was {was:.2e}"


def test_repair_keeps_the_frame_response():
    # The hop is lowered and the response rescaled by sqrt(a_new / a_old):
    # each channel's share |H|^2 / a of the frame response is unchanged.
    from cool_frames.numpy.filters._painless import repair_painless_hops
    from cool_frames.numpy.filterbanks import filterbankresponse

    g, a, fc, L, info = _quiet(cqtfilters, 8000, 512, fmin=20, bins=1, Qvar=0.5)
    resp = filterbankresponse(g, a, L, real=True)
    a2 = np.asarray(a).copy()
    a2[0] = 128                       # the hop the designer chose before G8
    g2 = [dict(d) for d in g]
    s = np.sqrt(128 / a[0])
    H0 = g2[0]["H"]
    g2[0]["H"] = (lambda fn: lambda Lq: np.asarray(fn(Lq)) * s)(H0) if callable(H0) else H0 * s
    assert _n_aliasing(g2, a2, L) == 1
    assert repair_painless_hops(g2, a2, L) == 1
    assert np.array_equal(a2, np.asarray(a))
    np.testing.assert_allclose(filterbankresponse(g2, a2, L, real=True), resp, rtol=1e-12)


@pytest.mark.parametrize("designer", ["aud", "greenwood", "cqt"])
def test_sweep_grid_is_painless(designer):
    fs_set = (8000.0, 16000.0, 44100.0)
    ls_set = (512, 2048)
    bad = []
    if designer == "aud":
        for sc, fs, Ls, M in itertools.product(("erb", "mel", "greenwood", "log"),
                                               fs_set, ls_set, (4, 5, 6, 12, 39)):
            g, a, fc, L, info = _quiet(audfilters, fs, Ls, scale=sc, M=M)
            if _n_aliasing(g, a, L):
                bad.append((sc, fs, Ls, M))
    elif designer == "greenwood":
        for fs, Ls, M in itertools.product(fs_set, ls_set, range(4, 40, 3)):
            g, a, fc, L, info = _quiet(greenwoodfilters, fs, Ls, M=M)
            if _n_aliasing(g, a, L):
                bad.append((fs, Ls, M))
    else:
        for fs, Ls, bins, fmin, Q in itertools.product(fs_set, ls_set, (1, 3, 12, 24),
                                                        (20.0, 100.0), (0.5, 1.0, 1.5)):
            try:
                g, a, fc, L, info = _quiet(cqtfilters, fs, Ls, fmin=fmin, bins=bins, Qvar=Q)
            except ValueError:
                continue
            if _n_aliasing(g, a, L):
                bad.append((fs, Ls, bins, fmin, Q))
    assert not bad, bad[:5]


@pytest.mark.parametrize("sampling", ["uniform", "fractional", "fractionaluniform"])
def test_other_samplings_reconstruct(sampling):
    for make in (lambda: audfilters(16000, 2048, M=24, sampling=sampling),
                 lambda: greenwoodfilters(16000, 2048, M=24, sampling=sampling),
                 lambda: cqtfilters(16000, 2048, fmin=100, bins=12, sampling=sampling)):
        g, a, fc, L, info = _quiet(make)
        assert _n_aliasing(g, a, L) == 0
        assert _round_trip(g, a, L) < 1e-13


def test_audfilters_greenwood_scale_matches_greenwoodfilters():
    g, a, fc, L, info = _quiet(audfilters, 16000, 4096, scale="greenwood")
    g2, a2, fc2, L2, info2 = _quiet(greenwoodfilters, 16000, 4096)
    np.testing.assert_allclose(fc, fc2, rtol=1e-12)
    assert info["painless"] is True and info["admissible"]["is_frame"] is True
    assert _round_trip(g, a, L) < 1e-13


def test_audfilters_rejects_fmin_above_fmax():
    with pytest.raises(ValueError, match="not below fmax"):
        audfilters(8000, 512, fmin=5000.0)


def test_empty_wavelet_channel_has_an_empty_dual():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        g, a, fc, L, info = waveletfilters(8000, 512, fmin=20, bins=1)
    assert info["empty_channels"] == [1]
    assert any("analyses nothing" in str(x.message) for x in w)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        err = _round_trip(g, a, L)
    assert err < 1e-12
    assert not any("two-sided" in str(x.message) for x in w)
