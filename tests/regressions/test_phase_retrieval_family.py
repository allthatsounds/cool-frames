"""
test_phase_retrieval_family.py
==============================
The iterative phase-retrieval routines, none of which any test executed.

Before this file, ``rtisila``, ``gsrtisila``, ``lertisila``, ``legla``,
``decolbfgs`` and ``spsi`` were covered at 7–11 % in both backends — that is,
the module docstring and the import block ran, and the algorithms did not.  All
twelve implementations (six routines x two backends) shipped in v0.1.0 and
v0.1.1 without a single call from the suite.  ``filterbankconstphase`` was in
exactly that position when the audit found that its gradient estimator had been
returning frequency *deviations* to an integrator that consumes absolute
frequencies, so this is not a hypothetical risk: the untested phase code in
this package has a track record.

What is asserted here
---------------------
**Consistency, not magnitude error.**  ``magnitudeerr`` compares ``|c|`` to the
target magnitudes, and every routine in this family ends by *assigning*
``c = s * exp(i*phase)``.  The magnitudes therefore match by construction, and
``magnitudeerr`` returns ~1e-16 for a result that is audibly wrong.  Zero phase
scores a perfect ``-inf dB`` on it.  The metric that means something is
consistency: resynthesise, re-analyse, and ask whether the magnitudes survived
the round trip.  A coefficient set that is not in the range of the analysis
operator cannot be produced by any signal, and the gap is what you hear.

Measured consistency on the fixture below (redundancy 2.1, a chirp plus a
steady partial), zero phase = +8.4 dB, true phase = -310 dB:

===========  ==========  ==========
routine      1 iter      10 iters
===========  ==========  ==========
gla            +0.6 dB    -12.0 dB
legla          +0.5 dB    -12.0 dB
rtisila       -16.6 dB    -19.2 dB
gsrtisila     -16.6 dB    -19.2 dB
lertisila     -16.6 dB    -19.2 dB
decolbfgs      -5.1 dB    -10.2 dB
===========  ==========  ==========

(The RTISI-LA rows were -7.9 / -13.0 dB, and -8.6 / -12.1 dB for
``lertisila``, before the family was rewritten as PHASERET's algorithm: the
old code re-synthesised every frame, future ones included, on every update.
Here each frame gets ``maxit`` updates in each of the ``lookahead + 1``
steps it spends in the window -- 53 on this bank, whose atoms span most of
the signal -- so "1 iter" is not one pass.)

The thresholds below are set well slack of these numbers.  They are regression
guards, not accuracy claims: the point is that a routine which silently stops
retrieving phase — the ``'pghi'`` branch of the torch recipe drew
``rand_like`` for a release and reported ``converged: True`` — fails here.

This file also uncovered a reporting defect: the RTISI-LA family computed
``relres`` *after* the magnitude projection, so it was ~1e-16 unconditionally.
It now measures consistency; ``test_relres_is_the_consistency`` at the bottom
holds that.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

FS = 4000
LS = 512

# Every routine in the family shares one signature:
#   fn(s_list, g, a, *, L, Ls, real, maxit) -> (c, f, relres, niter)
ITERATIVE = ["gla", "legla", "rtisila", "gsrtisila", "lertisila", "decolbfgs"]


def _fixture():
    """A redundant Gabor bank and the magnitudes of a signal with real phase structure.

    Redundancy matters.  An undercomplete bank makes phase retrieval trivial —
    an earlier draft of this file used ``M=16, a=32`` (redundancy 0.28) and
    every routine converged to 1e-16 on the first iteration, which would have
    made the tests pass against an implementation that did nothing at all.
    """
    from cool_frames.numpy.filterbanks import filterbank, filterbank_is_real
    from cool_frames.numpy.filters import gabfilters

    t = np.arange(LS) / FS
    # A linear chirp (so the phase gradient is non-trivial) plus a steady partial.
    sig = np.sin(2 * np.pi * (300 * t + 40 * t * t * FS / LS)) + 0.4 * np.sin(2 * np.pi * 1100 * t)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, _fc, L, _info = gabfilters(fs=FS, Ls=LS, M=32, a=8, real=True)

    a_np = np.asarray(a)
    real = bool(filterbank_is_real(g, a_np, L))
    c = filterbank(sig, g, a_np, L)
    mag = [np.abs(np.asarray(cm)) for cm in c]
    return g, a_np, L, real, c, mag


def _consistency_db(coefs, g, a_np, L, real, mag):
    """Resynthesise, re-analyse, compare magnitudes.  Lower is better."""
    from cool_frames.numpy.filterbanks import filterbank, filterbankdual, ifilterbank
    from cool_frames.numpy.phase import magnitudeerrdb

    coefs = [np.asarray(x.detach().cpu() if hasattr(x, "detach") else x) for x in coefs]
    gd = filterbankdual(g, a_np, L, real=real)
    f = ifilterbank(coefs, gd, a_np, LS, real=real)
    c2 = filterbank(np.real(np.asarray(f)), g, a_np, L)
    return float(magnitudeerrdb(c2, mag))


def _to_backend(mag, backend):
    if backend == "numpy":
        return mag
    import torch

    return [torch.as_tensor(m) for m in mag]


def _np(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)


# ---------------------------------------------------------------------------
# The routines run at all, and do the thing they are named for
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ITERATIVE)
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_iterative_retrieval_beats_zero_phase(name, backend):
    """Each routine must improve consistency over the zero-phase input it starts from.

    Zero phase measures +8.4 dB on this fixture.  Anything that retrieves phase
    lands far below 0 dB; anything that returns noise, returns its input, or
    reports success without iterating lands at or above it.  The -3 dB
    threshold is ~7 dB of slack against the worst measured routine.
    """
    if backend == "torch":
        pytest.importorskip("torch")
    import importlib

    mod = importlib.import_module(f"cool_frames.{backend}.phase")

    g, a_np, L, real, _c, mag = _fixture()

    zero = [m.astype(complex) for m in mag]
    baseline = _consistency_db(zero, g, a_np, L, real, mag)
    assert baseline > 0.0, f"zero-phase baseline should be poor, got {baseline:.2f} dB"

    fn = getattr(mod, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_out, f_out, _relres, _niter = fn(
            _to_backend(mag, backend), g, a_np, L=L, Ls=LS, real=real, maxit=10
        )

    got = _consistency_db(c_out, g, a_np, L, real, mag)
    assert got < -3.0, (
        f"{backend}.{name} reached only {got:.2f} dB consistency "
        f"(zero phase = {baseline:.2f} dB); it is not retrieving phase"
    )
    assert got < baseline - 8.0, (
        f"{backend}.{name} improved on zero phase by only {baseline - got:.2f} dB"
    )

    # The reconstructed signal must be the length that was asked for, real-valued
    # for a real bank, and finite.
    f_arr = _np(f_out).ravel()
    assert f_arr.size == LS, f"{backend}.{name} returned {f_arr.size} samples, expected {LS}"
    assert np.all(np.isfinite(np.real(f_arr))), f"{backend}.{name} produced non-finite output"


@pytest.mark.parametrize("name", ITERATIVE)
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_more_iterations_do_not_make_it_worse(name, backend):
    """Ten iterations must not be worse than one.

    Monotone improvement is too strong a claim for alternating projections —
    GLA is not guaranteed to decrease the consistency error at every step, and
    L-BFGS line searches can overshoot.  But a routine that gets *worse* with
    more work has its projection or its update sign wrong, and that is worth
    catching.  Half a decibel of tolerance absorbs the non-monotonicity.
    """
    if backend == "torch":
        pytest.importorskip("torch")
    import importlib

    mod = importlib.import_module(f"cool_frames.{backend}.phase")
    fn = getattr(mod, name)
    g, a_np, L, real, _c, mag = _fixture()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c1 = fn(_to_backend(mag, backend), g, a_np, L=L, Ls=LS, real=real, maxit=1)[0]
        c10 = fn(_to_backend(mag, backend), g, a_np, L=L, Ls=LS, real=real, maxit=10)[0]

    e1 = _consistency_db(c1, g, a_np, L, real, mag)
    e10 = _consistency_db(c10, g, a_np, L, real, mag)
    assert e10 <= e1 + 0.5, (
        f"{backend}.{name}: 10 iterations ({e10:.2f} dB) is worse than 1 iteration ({e1:.2f} dB)"
    )


@pytest.mark.parametrize("name", ["gla", "legla", "rtisila", "gsrtisila", "lertisila"])
def test_backends_agree(name):
    """The two backends must produce the same coefficients.

    ``decolbfgs`` is excluded: it runs an L-BFGS line search, and NumPy's and
    torch's differ, so the two backends legitimately land in different local
    minima (-10.25 dB vs -10.16 dB after 10 iterations here).  Its agreement is
    checked by the consistency bound above rather than element-wise.

    The rest are deterministic alternating projections and must agree to
    floating-point accumulation.  This is the assertion that the v0.1.1 audit
    needed and did not have: four separate fixes to the same defect were
    applied one backend at a time, and torch ``ifilterbankiter`` went three
    releases reconstructing the flagship bank with 23 % error because nothing
    compared the two.
    """
    pytest.importorskip("torch")
    import importlib

    np_mod = importlib.import_module("cool_frames.numpy.phase")
    t_mod = importlib.import_module("cool_frames.torch.phase")
    g, a_np, L, real, _c, mag = _fixture()

    kw = dict(L=L, Ls=LS, real=real, maxit=4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_np = getattr(np_mod, name)(mag, g, a_np, **kw)[0]
        c_t = getattr(t_mod, name)(_to_backend(mag, "torch"), g, a_np, **kw)[0]

    assert len(c_np) == len(c_t), f"{name}: channel count differs"
    for m, (x, y) in enumerate(zip(c_np, c_t)):
        x, y = np.asarray(x).ravel(), _np(y).ravel()
        assert x.shape == y.shape, f"{name} channel {m}: shape {x.shape} vs {y.shape}"
        scale = max(float(np.max(np.abs(x))), 1e-30)
        err = float(np.max(np.abs(x - y))) / scale
        assert err < 1e-8, f"{name} channel {m}: backends differ by {err:.3e} (relative)"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_spsi_is_single_pass_and_uses_centre_frequencies(backend):
    """SPSI must produce phase from the peak structure, not from nothing.

    ``spsi`` has its own signature — ``(s_list, a, fc, fs)`` — and returns
    ``(c, phase)`` rather than the four-tuple.  Two things are asserted: the
    returned coefficients carry the target magnitudes, and the phase actually
    depends on the centre frequencies handed in.  The second matters because
    the torch recipe's ``'spsi'`` branch was, until v0.1.1, a block of
    ``rand_like`` bitwise identical to its ``'pghi'`` branch — a routine that
    ignores its ``fc`` argument is indistinguishable from that.
    """
    if backend == "torch":
        pytest.importorskip("torch")
    import importlib

    mod = importlib.import_module(f"cool_frames.{backend}.phase")
    g, a_np, L, _real, _c, mag = _fixture()

    from cool_frames.numpy.phase._centerfreq import filter_center_frequencies

    fc = filter_center_frequencies(g, L)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_out, phase = mod.spsi(_to_backend(mag, backend), a_np, fc, 1.0)
        # Same magnitudes, different centre frequencies.
        _c_alt, phase_alt = mod.spsi(_to_backend(mag, backend), a_np, fc * 0.5, 1.0)

    for m, cm in enumerate(c_out):
        got, want = np.abs(_np(cm).ravel()), mag[m].ravel()
        assert np.allclose(got, want, rtol=1e-9, atol=1e-12), (
            f"{backend}.spsi channel {m}: returned magnitudes are not the target magnitudes"
        )

    p1 = np.concatenate([_np(p).ravel() for p in phase])
    p2 = np.concatenate([_np(p).ravel() for p in phase_alt])
    assert not np.allclose(p1, p2), (
        f"{backend}.spsi produced identical phase for two different centre-frequency "
        "vectors — it is ignoring fc"
    )


# ---------------------------------------------------------------------------
# A reporting defect this file uncovered, fixed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rtisila", "gsrtisila", "lertisila"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_relres_is_the_consistency(name, backend):
    """``relres`` is ``|| |A f| - s || / ||s||``, and ``niter`` counts updates.

    Until this was fixed, ``relres`` was ``|| |c| - s || / ||s||`` computed
    after ``c = s * exp(i*phase)`` -- zero by construction, 1e-16 here while
    the consistency was -13 dB and GLA honestly reported 0.249, so selecting a
    method by ``relres`` read the RTISI-LA family as fifteen orders of
    magnitude better.  (PHASERET's own ``rtisila`` and ``gsrtisila`` get it
    wrong the other way: the norm of the complex difference, a phase
    measure.)  ``niter`` was ``maxit`` times the number of time instants;
    it is now the updates a frame receives, ``maxit * (lookahead + 1)``.
    """
    if backend == "torch":
        pytest.importorskip("torch")
    import importlib

    from cool_frames.numpy.filterbanks import filterbank, filterbankdual, ifilterbank

    mod = importlib.import_module(f"cool_frames.{backend}.phase")
    g, a_np, L, real, _c, mag = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c, _f, relres, niter = getattr(mod, name)(
            _to_backend(mag, backend), g, a_np, L=L, Ls=LS, real=real, maxit=4, lookahead=3
        )
        c = [_np(x) for x in c]
        f = ifilterbank(c, filterbankdual(g, a_np, L, real=real), a_np, L, real=real)
        re = filterbank(np.real(f) if real else f, g, a_np, L)
    num = np.sqrt(sum(np.sum((np.abs(r) - m) ** 2) for r, m in zip(re, mag)))
    den = np.sqrt(sum(np.sum(m**2) for m in mag))
    assert relres == pytest.approx(num / den, rel=1e-9), (
        f"{backend}.{name}: relres {relres} is not the consistency {num / den}"
    )
    assert 1e-3 < relres < 0.5, f"{backend}.{name}: relres {relres} is implausible"
    assert niter == 4 * (3 + 1), f"{backend}.{name}: niter {niter}, expected 16"
