"""
test_benchmark_findings_torch.py
================================
The PyTorch half of ``test_benchmark_findings.py``: the same reassignment,
synchrosqueezing and diagnostics defects, which the torch backend shared
(its ``reassigned_spectrogram`` also swapped the instantaneous frequency and
the group delay).  Separate because this module skips wholesale when torch is
absent.  The NumPy backend, fixed and tested on its own, is the reference.
"""

from __future__ import annotations

import pytest

import numpy as np

torch = pytest.importorskip("torch")

FS = 16000
LS = 8000


def _tone(f0, fs=FS, n=LS):
    return np.cos(2 * np.pi * f0 * np.arange(n) / fs)


def _energy(sr):
    return np.array([float(torch.sum(torch.as_tensor(s))) for s in sr])


@pytest.fixture(scope="module")
def banks():
    from cool_frames.numpy.filters import audfilters as np_aud
    from cool_frames.torch.filters import audfilters as t_aud

    g, a, fc, L, _ = np_aud(FS, LS)
    gt, at, _fct, Lt, _ = t_aud(FS, LS)
    return (g, a, np.asarray(fc, float), L), (gt, at, Lt)


@pytest.mark.parametrize("f0", [8.0, 440.0, 2500.0, 6000.0])
def test_tone_reassignment_matches_numpy(banks, f0):
    from cool_frames.numpy.phase import filterbankreassign as np_reassign
    from cool_frames.torch.phase import filterbankreassign as t_reassign

    (g, a, fc, L), (gt, at, Lt) = banks
    x = _tone(f0)
    E = _energy(t_reassign(torch.tensor(x), gt, at, Lt))
    m = int(np.argmax(E))
    assert m == int(np.argmin(np.abs(fc - f0)))
    assert E[m] / E.sum() > 0.9
    E_np = np.array([float(np.sum(s)) for s in np_reassign(x, g, a, L)])
    np.testing.assert_allclose(E, E_np, rtol=1e-9, atol=1e-9 * E.sum())


def test_filter_cell_gives_the_signal_path_result(banks):
    from cool_frames.torch.phase import filterbankphasegrad, filterbankreassign

    _, (gt, at, Lt) = banks
    x = torch.tensor(np.random.default_rng(3).standard_normal(LS))
    sr_sig = filterbankreassign(x, gt, at, Lt)
    tg, fg, s, _ = filterbankphasegrad(x, gt, at, Lt)
    sr_pc, _repos, _Lc = filterbankreassign(s, tg, fg, at, gt)
    for u, v in zip(sr_sig, sr_pc):
        torch.testing.assert_close(torch.as_tensor(v), torch.as_tensor(u))


def test_synchrosqueeze_is_frequency_only():
    from cool_frames.numpy.filters import audfilters as np_aud
    from cool_frames.numpy.phase import filterbanksynchrosqueeze as np_sq
    from cool_frames.torch.filters import audfilters as t_aud
    from cool_frames.torch.phase import filterbankphasegrad
    from cool_frames.torch.phase import filterbanksynchrosqueeze as t_sq

    # Every hop 1, at the length the bank was designed for (torch stores the
    # responses as tensors sampled at that length).
    gt, _a, _fc, L, _ = t_aud(8000, 1024)
    g, _a, _fc, _L, _ = np_aud(8000, 1024)
    a1 = np.ones(len(gt), dtype=int)
    xn = np.random.default_rng(1).standard_normal(L)
    x = torch.tensor(xn)
    _tg, _fg, s, _c = filterbankphasegrad(x, gt, a1, L)
    with pytest.warns(DeprecationWarning):
        sr = t_sq(x, gt, a1, L)
    before = torch.stack([torch.as_tensor(sm) for sm in s]).sum(0)
    after = torch.stack([torch.as_tensor(sm) for sm in sr]).sum(0)
    assert float(torch.linalg.norm(after - before) / torch.linalg.norm(before)) < 1e-12
    with pytest.warns(DeprecationWarning):
        sr_np = np_sq(xn, g, a1, L)
    for u, v in zip(sr, sr_np):
        np.testing.assert_allclose(
            torch.as_tensor(u).numpy(), np.asarray(v), rtol=1e-9, atol=1e-12
        )


@pytest.mark.parametrize("f0", [1000.0, 3000.0])
def test_reassigned_spectrogram_matches_numpy(f0):
    from cool_frames.numpy.diagnostics import reassigned_spectrogram as np_rs
    from cool_frames.torch.diagnostics import reassigned_spectrogram as t_rs

    x = _tone(f0)
    ref = np_rs(x, FS)
    out = t_rs(torch.tensor(x), FS)
    fc = np.asarray(ref["fc"], float)
    m = int(np.argmin(np.abs(fc - f0)))
    dev = np.asarray(out["instfreq_deviation"], float)
    assert abs(dev[m] - (f0 - fc[m])) < 0.02 * abs(f0 - fc[m]) + 1.0
    np.testing.assert_allclose(dev, np.asarray(ref["instfreq_deviation"]), atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(out["groupdelay_shift"], float), np.asarray(ref["groupdelay_shift"]), atol=1e-9
    )
