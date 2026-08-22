"""
test_prop_phase_retrieval_family.py
===================================
Invariant properties shared by the iterative phase-retrieval family:
``gla`` / ``fgla``, ``legla`` / ``flegla``, ``lertisila``, ``gsrtisila``,
``decolbfgs`` and the single-pass ``spsi``.

Why *consistency* and not magnitude error
-----------------------------------------
Spectral convergence between the *target* magnitudes and ``|c_out|`` is not a
useful measure of quality here: every method in this family finishes by
projecting onto the magnitude constraint set, so that number is ~1e-16 for all
of them and discriminates nothing.

What discriminates is **consistency** — resynthesise the coefficients, analyse
the result again, and compare *those* magnitudes with the target.  A set of
coefficients that lies in the range of the analysis operator reproduces itself;
one that does not, does not.  On the fixture below:

    oracle (true phase)          5.9e-16
    gla, 20 iterations           0.058
    fgla, 20 iterations          0.042
    zero phase (do nothing)      0.385

Every property that talks about quality is therefore phrased in terms of
consistency, never magnitude error.

Properties verified
-------------------
    (1) Structure: M channels, per-channel lengths equal the analysis lengths.
    (2) Magnitude constraint: ``|c_out| == s`` — for every member, momentum
        variants included.
    (3) Finiteness: no NaN/Inf in coefficients or reconstructed signal.
    (4) Determinism: identical inputs give bit-identical outputs.
    (5) Homogeneity: scaling the target magnitudes by k scales ``c_out`` by k.
    (6) Consistency beats the zero-phase baseline.
    (7) Iterating more does not make consistency worse.
    (8) ``real=True`` yields a real-valued signal of length ``Ls``.

Cost
----
``rtisila``, ``lertisila`` and ``gsrtisila`` are two to three orders of
magnitude slower than the rest of the family (seconds per call, versus tens of
milliseconds), and their cost grows with signal length.  The fixture is
deliberately tiny — ``FS=4000``, ``Ls=512``, 23 channels — and those three
appear only in the cheap structural properties.  Do not enlarge the fixture
without re-timing.
"""

from __future__ import annotations

import pytest

import numpy as np

FS = 4000
LS = 512

# Every member of the family that iterates towards a fixed point.
ITERATIVE_METHODS = ["gla", "fgla", "legla", "flegla", "lertisila", "gsrtisila", "decolbfgs"]

# Every member projects onto the magnitude constraint before returning.  The
# momentum variants gained their final projection in v0.1.1; before that they
# returned the extrapolated point, with |c_out| off by ~16 %.
PROJECTING_METHODS = ["gla", "fgla", "legla", "flegla", "lertisila", "gsrtisila", "decolbfgs"]

# The subset that is cheap enough to call several times per test.
FAST_METHODS = ["gla", "fgla", "legla", "flegla", "decolbfgs"]

# Seconds-per-call methods; kept out of the repeated-call properties, but
# deliberately NOT marked ``slow``: CI runs ``-m "not slow"``, and marking them
# would leave three whole modules at their pre-existing ~8 % coverage.
RTISI_METHODS = ["rtisila", "lertisila", "gsrtisila"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fb():
    """A small ERB filterbank plus the magnitudes of a harmonic test signal."""
    from cool_frames.numpy.filterbanks import filterbank, filterbankdual
    from cool_frames.numpy.filters import audfilters

    t = np.arange(LS) / FS
    x = (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1320 * t)) * np.hanning(LS)

    g, a, fc, L, _info = audfilters(FS, LS)
    gd = filterbankdual(g, a, L)
    c = filterbank(x, g, a, L=L)
    s = [np.abs(ci) for ci in c]
    return dict(
        g=g, gd=gd, a=a, fc=fc, L=L, Ls=LS, x=x, c=c, s=s,
        lengths=[len(ci) for ci in c],
    )


def _run(method, d, maxit=3):
    """Dispatch a family member by name and return its coefficients only."""
    from cool_frames.numpy.phase import (
        decolbfgs,
        gla,
        gsrtisila,
        legla,
        lertisila,
        rtisila,
        spsi,
    )

    common = dict(L=d["L"], Ls=d["Ls"], real=True, maxit=maxit)
    if method == "spsi":
        # spsi takes fc in Hz plus fs, so audfilters' output goes straight in.
        return spsi(d["s"], d["a"], d["fc"], FS)[0]
    if method in ("gla", "fgla"):
        return gla(d["s"], d["g"], d["a"], method=method, **common)[0]
    if method in ("legla", "flegla"):
        return legla(d["s"], d["g"], d["a"], method=method, **common)[0]
    if method == "rtisila":
        return rtisila(d["s"], d["g"], d["a"], **common)[0]
    if method == "lertisila":
        return lertisila(d["s"], d["g"], d["a"], **common)[0]
    if method == "gsrtisila":
        return gsrtisila(d["s"], d["g"], d["a"], **common)[0]
    if method == "decolbfgs":
        return decolbfgs(d["s"], d["g"], d["a"], **common)[0]
    raise ValueError(method)


def _consistency(c_hat, d):
    """Spectral convergence of the *re-analysed* reconstruction.

    See the module docstring for why this, and not ``magnitudeerr(s, |c_hat|)``,
    is the meaningful quality measure for this family.
    """
    from cool_frames.numpy.filterbanks import filterbank, ifilterbank
    from cool_frames.numpy.phase import magnitudeerr

    f = np.real(ifilterbank(c_hat, d["gd"], d["a"], Ls=d["Ls"], real=True))
    c2 = filterbank(f, d["g"], d["a"], L=d["L"])
    return magnitudeerr(d["s"], [np.abs(ci) for ci in c2])


def _max_rel_magnitude_error(c_hat, s):
    return max(
        np.max(np.abs(np.abs(ch) - si)) / max(float(np.max(si)), 1e-30)
        for ch, si in zip(c_hat, s)
    )


# ---------------------------------------------------------------------------
# (1) Structure
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", [*FAST_METHODS, "spsi"])
def test_output_structure_matches_analysis(method, fb):
    """One complex array per channel, each as long as the analysis produced."""
    c_hat = _run(method, fb)
    assert len(c_hat) == len(fb["s"]), f"{method}: wrong channel count"
    assert [len(ci) for ci in c_hat] == fb["lengths"], f"{method}: wrong subband lengths"
    assert all(np.iscomplexobj(ci) for ci in c_hat), f"{method}: coefficients must be complex"


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", RTISI_METHODS)
def test_output_structure_matches_analysis_rtisi_family(method, fb):
    """Same structural contract for the RTISI family (~2 s per call).

    These three get one structural test and one constraint test each rather than
    appearing throughout the file — enough to exercise the module end to end
    without paying for them in every property.
    """
    c_hat = _run(method, fb, maxit=1)
    assert len(c_hat) == len(fb["s"]), f"{method}: wrong channel count"
    assert [len(ci) for ci in c_hat] == fb["lengths"], f"{method}: wrong subband lengths"
    assert all(np.iscomplexobj(ci) for ci in c_hat), f"{method}: coefficients must be complex"
    for m, ci in enumerate(c_hat):
        assert np.all(np.isfinite(ci)), f"{method}: channel {m} contains NaN/Inf"


# ---------------------------------------------------------------------------
# (2) Magnitude constraint
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", ["gla", "fgla", "legla", "flegla", "decolbfgs", "spsi"])
def test_magnitude_constraint_is_satisfied(method, fb):
    """``|c_out|`` reproduces the requested magnitudes to machine precision.

    This is the defining property of the constraint set: these methods only
    ever change phase, never magnitude.  A caller is entitled to rely on it —
    e.g. when applying a gain to ``s`` and expecting it to survive.
    """
    c_hat = _run(method, fb, maxit=5)
    err = _max_rel_magnitude_error(c_hat, fb["s"])
    assert err < 1e-12, f"{method}: |c_out| differs from the target magnitudes by {err:.2e}"


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", RTISI_METHODS)
def test_magnitude_constraint_is_satisfied_rtisi_family(method, fb):
    """``|c_out| == s`` holds for the RTISI family too."""
    c_hat = _run(method, fb, maxit=1)
    err = _max_rel_magnitude_error(c_hat, fb["s"])
    assert err < 1e-12, f"{method}: |c_out| differs from the target magnitudes by {err:.2e}"


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", ["fgla", "flegla"])
def test_momentum_variants_project_before_returning(method, fb):
    """The momentum variants honour the magnitude constraint like everyone else.

    ``fgla``/``flegla`` apply ``c = t_new + alpha*(t_new - t_old)`` after the
    magnitude projection.  Until v0.1.1 they returned that extrapolated point
    directly, so ``|c_out| != s`` — 16 % relative error at 20 iterations, 120 %
    at 2 — while every other member guaranteed equality.  A caller applying the
    family-wide assumption got a silent gain error.

    They now re-project the extrapolate, which also measures at least as well on
    consistency as returning the last projected iterate would.
    """
    for maxit in (2, 20):
        c_hat = _run(method, fb, maxit=maxit)
        err = _max_rel_magnitude_error(c_hat, fb["s"])
        assert err < 1e-12, (
            f"{method} at maxit={maxit}: |c_out| differs from the target "
            f"magnitudes by {err:.2e}"
        )


# ---------------------------------------------------------------------------
# (3) Finiteness
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", [*FAST_METHODS, "spsi"])
def test_output_is_finite(method, fb):
    c_hat = _run(method, fb)
    for m, ci in enumerate(c_hat):
        assert np.all(np.isfinite(ci)), f"{method}: channel {m} contains NaN/Inf"


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", FAST_METHODS)
def test_silent_input_gives_silent_output(method, fb):
    """All-zero magnitudes must not produce NaN through a 0/0 phase division."""
    zeros = dict(fb)
    zeros["s"] = [np.zeros_like(si) for si in fb["s"]]
    c_hat = _run(method, zeros)
    for m, ci in enumerate(c_hat):
        assert np.all(np.isfinite(ci)), f"{method}: channel {m} contains NaN/Inf"
        assert np.allclose(ci, 0.0), f"{method}: zero input gave non-zero channel {m}"


# ---------------------------------------------------------------------------
# (4) Determinism
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", [*FAST_METHODS, "spsi"])
def test_deterministic_for_fixed_start_phase(method, fb):
    """With the default deterministic start phase the algorithms are reproducible.

    Reproducibility is what makes the reference values in the rest of this file
    meaningful, and what lets a caller diff two runs of a pipeline.
    """
    first = _run(method, fb, maxit=4)
    second = _run(method, fb, maxit=4)
    for m, (u, v) in enumerate(zip(first, second)):
        assert np.array_equal(u, v), f"{method}: channel {m} differs between runs"


# ---------------------------------------------------------------------------
# (5) Homogeneity
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", ["gla", "legla", "spsi"])
@pytest.mark.parametrize("k", [3.7, 1e-4])
def test_scaling_the_magnitudes_scales_the_coefficients(method, k, fb):
    """Phase retrieval is scale-equivariant: ``k*s`` in, ``k*c`` out.

    Phase depends only on the *shape* of the spectrogram, so a global gain has
    to pass straight through.  Two very different scales are checked because a
    hidden absolute threshold anywhere in the iteration would break one of them
    while leaving the other intact.
    """
    scaled = dict(fb)
    scaled["s"] = [k * si for si in fb["s"]]

    c_ref = _run(method, fb, maxit=6)
    c_scaled = _run(method, scaled, maxit=6)

    err = max(
        np.max(np.abs(u - k * v)) / max(float(np.max(np.abs(u))), 1e-30)
        for u, v in zip(c_scaled, c_ref)
    )
    assert err < 1e-5, f"{method}: not scale-equivariant at k={k} (rel. error {err:.2e})"


# ---------------------------------------------------------------------------
# (6) Quality: better than doing nothing
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", [*FAST_METHODS, "spsi"])
def test_beats_the_zero_phase_baseline(method, fb):
    """Consistency must improve on simply setting every phase to zero.

    A method that failed this would be actively worse than not running it.  The
    baseline is ~0.385 on this fixture; the family lands between 0.04 and 0.22.
    """
    baseline = _consistency([si.astype(complex) for si in fb["s"]], fb)
    got = _consistency(_run(method, fb, maxit=8), fb)
    assert got < baseline, (
        f"{method}: consistency {got:.4f} is no better than the zero-phase "
        f"baseline {baseline:.4f}"
    )


@pytest.mark.requires_impl
def test_true_phase_is_perfectly_consistent(fb):
    """Sanity check on the metric itself: the oracle must score ~0.

    If this fails, every other consistency assertion in the file is measuring
    the wrong thing.
    """
    assert _consistency(fb["c"], fb) < 1e-10


# ---------------------------------------------------------------------------
# (7) Iterating more does not hurt
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", ["gla", "fgla", "legla", "flegla"])
def test_more_iterations_do_not_worsen_consistency(method, fb):
    """The Griffin-Lim family is non-increasing in the consistency objective.

    A small tolerance is allowed for the momentum variants, which are not
    monotone by construction but must not diverge.
    """
    few = _consistency(_run(method, fb, maxit=2), fb)
    many = _consistency(_run(method, fb, maxit=20), fb)
    assert many <= few + 1e-3, (
        f"{method}: consistency worsened with more iterations "
        f"({few:.4f} at 2 iterations -> {many:.4f} at 20)"
    )


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", ["gla", "legla"])
def test_reported_residual_is_non_increasing(method, fb):
    """The tracked residual sequence must decrease monotonically for plain GLA.

    ``relres`` is what a caller watches to decide when to stop; if it is not
    monotone for the non-accelerated variants, the projection step is wrong.
    """
    from cool_frames.numpy.phase import gla, legla

    fn = gla if method == "gla" else legla
    _c, _f, relres, niter = fn(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                               real=True, maxit=15)
    assert niter == len(relres) > 1, f"{method}: niter {niter} vs {len(relres)} residuals"
    diffs = np.diff(np.asarray(relres))
    assert np.all(diffs <= 1e-12), (
        f"{method}: residual increased at iteration(s) {np.flatnonzero(diffs > 1e-12) + 1}"
    )


# ---------------------------------------------------------------------------
# (8) Reconstructed signal
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", ["gla", "legla", "decolbfgs"])
def test_real_mode_returns_a_real_signal_of_length_ls(method, fb):
    from cool_frames.numpy.phase import decolbfgs, gla, legla

    fn = dict(gla=gla, legla=legla, decolbfgs=decolbfgs)[method]
    _c, f, _relres, _niter = fn(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                                real=True, maxit=3)

    f = np.asarray(f)
    assert f.size == fb["Ls"], f"{method}: expected {fb['Ls']} samples, got {f.size}"
    assert np.all(np.isfinite(f)), f"{method}: reconstruction contains NaN/Inf"
    assert np.max(np.abs(np.imag(f))) < 1e-10, f"{method}: real=True gave a complex signal"


@pytest.mark.requires_impl
@pytest.mark.parametrize("method", ["gla", "legla"])
def test_returned_signal_matches_synthesis_of_returned_coefficients(method, fb):
    """``f`` must be the dual-frame synthesis of ``c`` — not an earlier iterate.

    Callers use one or the other interchangeably; if they disagree, the two
    halves of the return value describe different reconstructions.
    """
    from cool_frames.numpy.filterbanks import ifilterbank
    from cool_frames.numpy.phase import gla, legla

    fn = gla if method == "gla" else legla
    c, f, _relres, _niter = fn(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                               real=True, maxit=5)
    f_expected = np.real(ifilterbank(c, fb["gd"], fb["a"], Ls=fb["Ls"], real=True))
    err = np.max(np.abs(np.asarray(f).ravel() - f_expected.ravel()))
    scale = max(float(np.max(np.abs(f_expected))), 1e-30)
    assert err / scale < 1e-10, f"{method}: returned f is not ifilterbank(c) (rel. {err/scale:.2e})"
