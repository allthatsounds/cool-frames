"""
test_rtisila_family.py
======================
The RTISI-LA family (``rtisila``, ``gsrtisila``, ``lertisila``) after its
rewrite as PHASERET's algorithm.

What was wrong
--------------
The W52 benchmark recorded ``rtisila`` as not finishing a 3 s excerpt
(22.05 kHz, Gabor bank 1024/256) in 300 s.  That was the benchmark's
timeout helper deadlocking on the returned signal (it took about a minute),
but reading the code showed it was not RTISI-LA:

- every update of every frame re-synthesised and re-analysed the *whole*
  signal, ``maxit`` times per time instant;
- the synthesis included every frame, future ones too, with their initial
  zero phase, so the "real-time" algorithm was not causal;
- frames were grouped by ``a_norm[:, 0]`` -- the numerator of a fractional
  hop -- so a fractional bank was grouped at the wrong times;
- ``lertisila`` claimed Le Roux's truncated kernel and used the same full
  transforms; ``gsrtisila`` claimed Gnann and Spiertz's windows and had none;
- ``relres`` was zero by construction and ``niter`` counted time instants
  (see ``test_phase_retrieval_family.py``); the docstrings cited papers
  that are not the algorithms' sources.

What it is now
--------------
On a Gabor window (``rtisila(s, g, a, M)``), line-by-line ports of
PHASERET's ``rtisila``, ``gsrtisila`` and ``lertisila``, checked here against
PHASERET run in Octave (``tests/phaseret_reference/``; reference in
``tests/reference_data/phaseret_rtisila.mat``).  On a filter bank, the same
schedule over frames of ``frame_hop`` samples, with the partial
reconstruction held as its spectrum and updated exactly.
"""

from __future__ import annotations

import pathlib
import time
import warnings

import pytest

import numpy as np

_REF = pathlib.Path(__file__).resolve().parents[1] / "reference_data" / "phaseret_rtisila.mat"


def _ref():
    sio = pytest.importorskip("scipy.io")
    if not _REF.exists():
        pytest.skip(f"{_REF.name} not generated (tests/phaseret_reference/)")
    return sio.loadmat(str(_REF), squeeze_me=True)


def _case(R, ci):
    p = f"c{ci}_"
    la = R[p + "lookahead"]
    kw = {} if np.size(la) == 0 else {"lookahead": int(la)}
    return (
        p,
        int(R[p + "a"]),
        int(R[p + "M"]),
        np.atleast_1d(R[p + "g"]).astype(float),
        R[p + "s"],
        kw,
    )


def _rel(x, y):
    return float(np.max(np.abs(np.asarray(x) - y)) / np.max(np.abs(y)))


# ---------------------------------------------------------------------------
# Gabor form: PHASERET's algorithms
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ci", [1, 2])
def test_rtisila_is_phaserets(ci):
    """``rtisila(s, g, a, M)`` gives PHASERET's coefficients (both phase
    conventions), including its modified windows for the newest frame and its
    wrap-around at the end; case 2 has ``gl < M`` and a look-ahead of 1."""
    from cool_frames.numpy.phase import rtisila

    R = _ref()
    p, a, M, g, s, kw = _case(R, ci)
    c = rtisila(np.abs(s), g, a, M, phase="timeinv", **kw)[0]
    assert _rel(c, R[p + "rt_c"]) < 1e-9
    c = rtisila(np.abs(s), g, a, M, maxit=3, **kw)[0]  # freqinv, the default
    assert _rel(c, R[p + "rt3fi_c"]) < 1e-9


@pytest.mark.parametrize("ci", [1, 2])
@pytest.mark.parametrize("init", ["zeros", "unwrap", "spsi", "input"])
def test_gsrtisila_is_phaserets(ci, init):
    """``gsrtisila`` with each ported initialisation, as PHASERET's C
    library runs it (the reference patches PHASERET's MATLAB fallback, which
    never reads the initial phase)."""
    from cool_frames.numpy.phase import gsrtisila

    R = _ref()
    p, a, M, g, s, kw = _case(R, ci)
    s_in = s if init == "input" else np.abs(s)
    c = gsrtisila(s_in, g, a, M, startphase=init, phase="timeinv", **kw)[0]
    assert _rel(c, R[p + f"gs_{init}_c"]) < 1e-9


_LE = [
    dict(startphase="zhu"),
    dict(startphase="zero", asymwin=False),
    dict(startphase="zhu", variant="modtrunc"),
    dict(startphase="input", energy_order=True),
    dict(startphase="zhu", onthefly=True),
    dict(startphase="unwrap"),
]


@pytest.mark.parametrize("k", range(1, len(_LE) + 1))
def test_lertisila_is_phaserets(k):
    """``lertisila`` in six configurations.  The truncated projection is not
    a projection, so rounding differences grow along the signal (on longer
    signals, to 1e-2 by the end, while the first frames agree to 1e-15 and
    ``relres`` to 1e-4); on this 32-frame case they stay below 1e-9."""
    from cool_frames.numpy.phase import lertisila

    R = _ref()
    p, a, M, g, s, _kw = _case(R, 1)
    v = _LE[k - 1]
    s_in = s if v["startphase"] == "input" else np.abs(s)
    c, _f, relres, _n = lertisila(s_in, g, a, M, phase="timeinv", **v)
    assert _rel(c, R[p + f"le{k}_c"]) < 1e-8
    assert relres == pytest.approx(float(R[p + f"le{k}_relres"]), rel=1e-8)


def test_gabor_relres_is_the_consistency_and_niter_counts_updates():
    """PHASERET's ``rtisila``/``gsrtisila`` return the norm of the complex
    difference ``dgtreal(f) - s``, a phase measure (1.41 for noise); here it
    is ``|| |dgtreal(f)| - s || / ||s||``.  ``niter`` is the updates a frame
    receives, ``maxit * (lookahead + 1)``, as PHASERET documents (its code
    returns ``maxit * lookahead``)."""
    from cool_frames.numpy.gabor import dgtreal
    from cool_frames.numpy.phase import gsrtisila, lertisila, rtisila

    R = _ref()
    _p, a, M, g, s, _kw = _case(R, 1)
    abss = np.abs(s)
    for fn in (rtisila, gsrtisila, lertisila):
        _c, f, relres, niter = fn(abss, g, a, M, maxit=2)
        want = np.linalg.norm(np.abs(dgtreal(f, g, a, M)) - abss) / np.linalg.norm(abss)
        assert relres == pytest.approx(want, rel=1e-10), fn.__name__
        assert niter == 2 * (M // a), fn.__name__  # default lookahead M/a - 1


def test_gabor_form_is_fast_at_benchmark_scale():
    """W52's T6 case: 65536 samples, Hann 1024, a = 256.  The filter-bank
    code this replaced took about a minute; PHASERET's algorithm on a Gabor
    frame costs a few FFTs of M samples per update."""
    from cool_frames.filters import firwin
    from cool_frames.numpy.gabor import dgtreal
    from cool_frames.numpy.phase import rtisila

    L, a, M = 65536, 256, 1024
    t = np.arange(L) / 22050
    x = np.sin(2 * np.pi * (300 * t + 900 * t**2)) + 0.5 * np.sin(2 * np.pi * 1234.5 * t)
    g = firwin("hann", M)
    s = np.abs(dgtreal(x, g, a, M))
    t0 = time.perf_counter()
    _c, _f, relres, _n = rtisila(s, g, a, M)
    elapsed = time.perf_counter() - t0
    assert elapsed < 20.0, f"rtisila took {elapsed:.1f} s"
    assert relres < 0.1


def test_gabor_form_refuses_a_non_painless_window():
    from cool_frames.filters import firwin
    from cool_frames.numpy.phase import rtisila

    with pytest.raises(ValueError, match="painless"):
        rtisila(np.ones((33, 16)), firwin("hann", 128), 16, 64)
    with pytest.raises(TypeError, match="number of channels M"):
        rtisila(np.ones((33, 16)), firwin("hann", 64), 16)
    with pytest.raises(ValueError, match="filter bank"):
        rtisila(np.ones((33, 16)), firwin("hann", 64), 16, 64, frame_hop=16)


# ---------------------------------------------------------------------------
# Filter-bank form
# ---------------------------------------------------------------------------


def _banks():
    from cool_frames.numpy.filters import audfilters, cqtfilters, gabfilters
    from cool_frames.numpy.filters.lowlevel import firfilter

    fir = [firfilter("hann", 32, fc=f) for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    return {
        "uniform-gabor": lambda: gabfilters(8000, 4096, M=128, a=32)[:4],
        "auditory": lambda: audfilters(8000, 4000)[:4],
        "fractional-cqt": lambda: cqtfilters(8000, 4000, sampling="fractional")[:4],
        "fir": lambda: (fir, 4, None, 256),
    }


@pytest.mark.parametrize("bank", ["uniform-gabor", "auditory", "fractional-cqt", "fir"])
@pytest.mark.parametrize("real", [True, False])
def test_engine_is_filterbank_and_ifilterbank(bank, real):
    """The incremental engine's per-frame analysis and synthesis are the
    bank's own transforms, for band-limited, fractional-hop and FIR banks."""
    from cool_frames.numpy.filterbanks import filterbank, filterbankdual, ifilterbank
    from cool_frames.numpy.filterbanks._utils import normalise_a, prepare_filters
    from cool_frames.numpy.phase._rtisila_fb import FbFrames, default_frame_hop

    g, a, _fc, L = _banks()[bank]()
    an = normalise_a(a, len(g))
    rng = np.random.default_rng(3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gd = filterbankdual(g, a, L, real=real)
        x = (
            rng.standard_normal(L)
            if real
            else rng.standard_normal(L) + 1j * rng.standard_normal(L)
        )
        c = filterbank(x, g, a, L)
        N = [len(cm) for cm in c]
        eng = FbFrames(
            g, gd, an, L, N, real, default_frame_hop(prepare_filters(g, an, L)[0], an, L)
        )
        eng.F = np.fft.fft(x)
        est = [np.zeros((len(cl.chans), cl.N), dtype=complex) for cl in eng.classes]
        for k in range(eng.n_frames):
            for (ci, n0, n1), e in zip(eng.frames[k], eng.analyse(k)):
                est[ci][:, n0:n1] = e
        got = eng.unstack(est)
        scale = max(float(np.max(np.abs(cm))) for cm in c)
        assert max(float(np.max(np.abs(u - v))) for u, v in zip(got, c)) / scale < 1e-13

        y = [rng.standard_normal(n) + 1j * rng.standard_normal(n) for n in N]
        Y = eng.stack(y)
        eng.F[:] = 0
        for k in range(eng.n_frames):
            eng.add(k, [Y[ci][:, n0:n1] for ci, n0, n1 in eng.frames[k]])
        F_ref = np.fft.fft(ifilterbank(y, gd, a, L, real=real))
    assert np.max(np.abs(eng.F - F_ref)) / np.max(np.abs(F_ref)) < 1e-13


def test_filterbank_form_is_phaserets_on_a_gabor_frame():
    """A filter bank that *is* a painless Gabor frame (full-length filters
    ``g(t) exp(2 pi i m t / M)``, DC and Nyquist scaled by ``1/sqrt(2)`` so
    that the real-mode frame operator is the Gabor one) gives PHASERET's
    ``rtisila`` coefficients -- the exact re-analysis, and Zhu's windows
    built from the bank's atoms, reproduce the frame-based algorithm.  The
    first frames agree to rounding; later ones drift (the iteration
    amplifies rounding differences), and the last ``lookahead`` differ by
    design (PHASERET wraps round to the first frames)."""
    from cool_frames.filters import firwin
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.gabor import dgtreal
    from cool_frames.numpy.gabor._dgt import _as_window
    from cool_frames.numpy.phase import rtisila

    L, a, M = 2048, 32, 128
    g = firwin("hann", M)
    gl = _as_window(g, L)
    t = np.arange(L)
    bank = []
    for m in range(M // 2 + 1):
        w = np.sqrt(0.5) if m in (0, M // 2) else 1.0
        atom = gl * np.exp(2j * np.pi * m * t / M)
        bank.append({"H": w * np.conj(np.fft.fft(atom)), "foff": 0, "realonly": 0, "delay": 0})
    x = np.sin(2 * np.pi * (0.01 * t + 2e-6 * t**2)) + 0.5 * np.sin(2 * np.pi * 0.123 * t)
    s_gab = np.abs(dgtreal(x, g, a, M))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s_fb = [np.abs(cm) for cm in filterbank(x, bank, a, L)]
        c_fb = np.array(rtisila(s_fb, bank, a, L=L, real=True, lookahead=3)[0])
    c_fb[[0, M // 2]] *= np.sqrt(2)
    c_gab = rtisila(s_gab, g, a, M, phase="timeinv", lookahead=3)[0]
    d = np.max(np.abs(c_fb - c_gab), axis=0) / np.max(np.abs(c_gab))
    assert np.all(d[:16] < 1e-7), d[:16]


def test_frames_hold_the_coefficients_at_their_true_times():
    """A coefficient ``n`` of a channel with hop ``num/den`` is at time
    ``n num / den`` and sits in frame ``floor(time / frame_hop)``; the old
    grouping used ``num`` as the hop."""
    from cool_frames.numpy.filterbanks import filterbankdual
    from cool_frames.numpy.filterbanks._utils import normalise_a
    from cool_frames.numpy.filters import cqtfilters
    from cool_frames.numpy.phase._rtisila_fb import FbFrames

    g, a, _fc, L, _ = cqtfilters(8000, 4000, sampling="fractional")
    an = normalise_a(a, len(g))
    assert np.any(an[:, 1] != 1), "fixture should have fractional hops"
    N = [int(round(L * an[m, 1] / an[m, 0])) for m in range(len(g))]
    H = 250
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eng = FbFrames(g, filterbankdual(g, a, L), an, L, N, True, H)
    seen = 0
    for k, fl in enumerate(eng.frames):
        for ci, n0, n1 in fl:
            m0 = int(eng.classes[ci].chans[0])
            t = np.arange(n0, n1) * an[m0, 0] / an[m0, 1]
            assert np.all((t >= k * H - 1e-9) & (t < (k + 1) * H - 1e-9))
            seen += (n1 - n0) * len(eng.classes[ci].chans)
    assert seen == sum(N)


@pytest.mark.parametrize("name", ["rtisila", "gsrtisila", "lertisila"])
def test_filterbank_form_is_causal(name):
    """Frames later than the look-ahead never enter the reconstruction:
    changing the magnitudes from frame ``k0`` on leaves every frame
    committed before ``k0 - lookahead`` bit-identical."""
    import cool_frames.numpy.phase as P
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import gabfilters

    g, a, _fc, L, _ = gabfilters(8000, 2048, M=64, a=16)
    x = np.random.default_rng(0).standard_normal(L)
    s = [np.abs(cm) for cm in filterbank(x, g, a, L)]
    la, k0 = 3, 60
    s2 = [sm.copy() for sm in s]
    for sm in s2:
        sm[k0:] *= 1.5
    kw = dict(L=L, real=True, maxit=3, lookahead=la)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c1 = getattr(P, name)(s, g, a, **kw)[0]
        c2 = getattr(P, name)(s2, g, a, **kw)[0]
    for u, v in zip(c1, c2):
        assert np.array_equal(u[: k0 - la], v[: k0 - la])
        assert not np.array_equal(u, v)


def test_filterbank_form_improves_on_a_fractional_bank():
    """The fractional-hop bank the old grouping mis-timed: RTISI-LA's
    consistency beats zero phase by a wide margin."""
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import cqtfilters
    from cool_frames.numpy.phase import rtisila

    g, a, _fc, L, _ = cqtfilters(8000, 4000, sampling="fractional")
    t = np.arange(L) / 8000
    x = np.sin(2 * np.pi * (300 * t + 400 * t**2)) + 0.3 * np.sin(2 * np.pi * 1500 * t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = [np.abs(cm) for cm in filterbank(x, g, a, L)]
        _c, _f, relres0, _ = rtisila(s, g, a, L=L, real=True, maxit=1, lookahead=0)
        _c, _f, relres, _ = rtisila(s, g, a, L=L, real=True, maxit=5)
    assert relres < 0.5 * relres0, (relres, relres0)


def test_default_lookahead_is_phaserets_on_a_gabor_bank():
    """On the benchmark's Gabor bank (Hann 1024, a = 256) the default
    look-ahead is PHASERET's ``ceil(M/a) - 1 = 3``; the auditory bank's DC
    and Nyquist complements, whose atoms span the signal, do not set it."""
    from cool_frames.numpy.filterbanks._utils import normalise_a, prepare_filters
    from cool_frames.numpy.filters import audfilters, gabfilters
    from cool_frames.numpy.phase._rtisila_fb import default_frame_hop, default_lookahead

    for make, want_hop, want_la in (
        (lambda: gabfilters(22050, 65536, M=1024, a=256), 256, 3),
        (lambda: audfilters(22050, 65536), 256, 3),
    ):
        g, a, _fc, L, _ = make()
        an = normalise_a(a, len(g))
        gr = prepare_filters(g, an, L)[0]
        H = default_frame_hop(gr, an, L)
        assert H == want_hop
        assert default_lookahead(gr, an, L, H, 10**6) == want_la


def test_a_single_sided_uniform_bank_has_a_pseudo_inverse_dual():
    """``real=False`` with a single-sided ``gabfilters`` bank (the phase
    functions' default) used to raise "not a frame" in the uniform branch.
    Like LTFAT, which uses ``pinv``, and like the painless formula, it now
    returns the pseudo-inverse dual: re-analysing its synthesis reproduces
    coefficients that are in the range of the analysis."""
    from cool_frames.numpy.filterbanks import filterbank, filterbankdual, ifilterbank
    from cool_frames.numpy.filters import gabfilters

    g, a, _fc, L, _ = gabfilters(8000, 4096, M=128, a=32)
    x = np.random.default_rng(1).standard_normal(L)
    c = filterbank(x, g, a, L)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gd = filterbankdual(g, a, L, real=False)
        c2 = filterbank(ifilterbank(c, gd, a, L, real=False), g, a, L)
    scale = max(float(np.max(np.abs(cm))) for cm in c)
    assert max(float(np.max(np.abs(u - v))) for u, v in zip(c, c2)) / scale < 1e-8
