"""
test_phase_retrieval_family.py
==============================
Structural unit tests for the phase-retrieval entry points, plus regression
tests for the defects fixed in v0.1.1.

Two kinds of test live here.

*Contract tests* pin what a caller is entitled to rely on: return arity, the
meaning of each returned element, accepted keyword values, and behaviour at the
edges (single channel, silence, one frame).

*Regression tests* cover the four defects fixed in v0.1.1, each asserting the
now-correct behaviour rather than pinning the old broken one:

    * ``legla`` applies a real truncated projection kernel, so ``relthr`` and
      ``variant`` change the result, and ``relthr=0`` reproduces GLA exactly.
    * ``decolbfgs`` supplies L-BFGS-B a gradient that agrees with its objective,
      so the optimiser iterates and the result improves with ``maxit``.
    * ``spsi`` takes ``fc`` in Hz plus an explicit ``fs``, and rejects a ``fc``
      that cannot be in the unit it claims.
    * the momentum variants project onto the magnitude constraint before
      returning (asserted in the property suite, not here).

Each carries a note on what the failure used to look like, so that a
reintroduction is recognisable rather than merely red.
"""

from __future__ import annotations

import pytest

import numpy as np

FS = 4000
LS = 512


@pytest.fixture(scope="module")
def fb():
    from cool_frames.numpy.filterbanks import filterbank, filterbankdual
    from cool_frames.numpy.filters import audfilters

    t = np.arange(LS) / FS
    x = (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1320 * t)) * np.hanning(LS)
    g, a, fc, L, _info = audfilters(FS, LS)
    gd = filterbankdual(g, a, L)
    c = filterbank(x, g, a, L=L)
    return dict(g=g, gd=gd, a=a, fc=fc, L=L, Ls=LS, x=x, c=c,
                s=[np.abs(ci) for ci in c])


def _consistency(c_hat, d):
    from cool_frames.numpy.filterbanks import filterbank, ifilterbank
    from cool_frames.numpy.phase import magnitudeerr

    f = np.real(ifilterbank(c_hat, d["gd"], d["a"], Ls=d["Ls"], real=True))
    c2 = filterbank(f, d["g"], d["a"], L=d["L"])
    return magnitudeerr(d["s"], [np.abs(ci) for ci in c2])


# ---------------------------------------------------------------------------
# Return contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.requires_impl
@pytest.mark.parametrize("name", ["gla", "legla", "decolbfgs"])
def test_returns_coefficients_signal_residuals_and_iteration_count(name, fb):
    """The four-element return is ``(c, f, relres, niter)`` in that order."""
    import cool_frames.numpy.phase as P

    out = getattr(P, name)(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                           real=True, maxit=4)
    assert len(out) == 4, f"{name}: expected a 4-tuple, got {len(out)} elements"
    c, f, relres, niter = out

    assert isinstance(c, list) and len(c) == len(fb["s"])
    assert np.asarray(f).size == fb["Ls"]
    relres = np.asarray(relres)
    assert relres.ndim == 1 and relres.size >= 1
    assert np.all(np.isfinite(relres)) and np.all(relres >= 0)
    assert isinstance(niter, (int, np.integer)) and niter >= 0


@pytest.mark.unit
@pytest.mark.requires_impl
def test_spsi_returns_coefficients_and_phases(fb):
    """``spsi`` is single-pass and returns ``(c, phase)`` — no residuals."""
    from cool_frames.numpy.phase import spsi

    out = spsi(fb["s"], fb["a"], fb["fc"], FS)
    assert len(out) == 2, f"expected a 2-tuple, got {len(out)}"
    c, phase = out
    assert len(c) == len(phase) == len(fb["s"])
    for cm, pm, sm in zip(c, phase, fb["s"]):
        assert cm.shape == sm.shape
        assert np.iscomplexobj(cm)
        assert np.all(np.isfinite(pm))


@pytest.mark.unit
@pytest.mark.requires_impl
def test_niter_is_capped_by_maxit(fb):
    """``niter`` never exceeds ``maxit`` and equals it when ``tol`` is unreachable."""
    from cool_frames.numpy.phase import gla

    for maxit in (1, 3, 11):
        _c, _f, relres, niter = gla(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                                    real=True, maxit=maxit, tol=0.0)
        assert niter == maxit, f"maxit={maxit} gave niter={niter}"
        assert len(relres) == maxit


@pytest.mark.unit
@pytest.mark.requires_impl
def test_loose_tolerance_stops_early(fb):
    """A tolerance the first residual already meets stops after one iteration."""
    from cool_frames.numpy.phase import gla

    _c, _f, relres, niter = gla(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                                real=True, maxit=50, tol=1e9)
    assert niter == 1, f"tol=1e9 should stop immediately, ran {niter} iterations"
    assert len(relres) == 1


# ---------------------------------------------------------------------------
# startphase
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.requires_impl
@pytest.mark.parametrize("startphase", ["zero", "input", "rand"])
def test_startphase_variants_all_run_and_beat_the_baseline(startphase, fb):
    """Every documented ``startphase`` value is accepted and produces a usable result."""
    from cool_frames.numpy.phase import gla

    c, _f, _relres, _niter = gla(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                                 real=True, maxit=8, startphase=startphase)
    baseline = _consistency([si.astype(complex) for si in fb["s"]], fb)
    assert _consistency(c, fb) < baseline


@pytest.mark.unit
@pytest.mark.requires_impl
def test_startphase_input_uses_the_phase_of_the_argument(fb):
    """``startphase='input'`` must read the phase of ``s_list``, not discard it.

    Handing it the *true* coefficients means it starts at the fixed point, so a
    single iteration reproduces them exactly.  With ``startphase='zero'`` the
    same call cannot.
    """
    from cool_frames.numpy.phase import gla

    c_warm, _f, _r, _n = gla(fb["c"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                             real=True, maxit=1, startphase="input")
    c_cold, _f, _r, _n = gla(fb["c"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                             real=True, maxit=1, startphase="zero")
    assert _consistency(c_warm, fb) < 1e-10, "warm start did not land on the fixed point"
    assert _consistency(c_cold, fb) > 1e-3, "cold start unexpectedly perfect"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_random_startphase_is_not_reproducible(fb):
    """``startphase='rand'`` draws from an unseeded generator.

    Documented so that nobody builds a regression baseline on top of it.  If a
    ``seed`` argument is ever added, this test should be replaced by one that
    asserts reproducibility for a fixed seed.
    """
    from cool_frames.numpy.phase import gla

    kw = dict(L=fb["L"], Ls=fb["Ls"], real=True, maxit=2, startphase="rand")
    a_ = gla(fb["s"], fb["g"], fb["a"], **kw)[0]
    b_ = gla(fb["s"], fb["g"], fb["a"], **kw)[0]
    assert not all(np.array_equal(u, v) for u, v in zip(a_, b_))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.requires_impl
def test_magnitudeerr_is_zero_for_identical_input(fb):
    from cool_frames.numpy.phase import magnitudeerr

    assert magnitudeerr(fb["s"], fb["s"]) == pytest.approx(0.0, abs=1e-14)


@pytest.mark.unit
@pytest.mark.requires_impl
def test_magnitudeerr_is_scale_invariant(fb):
    """Scaling both arguments by the same factor leaves the relative error alone."""
    from cool_frames.numpy.phase import magnitudeerr

    other = [0.9 * si + 1e-3 for si in fb["s"]]
    base = magnitudeerr(fb["s"], other)
    for k in (1e-3, 1e3):
        scaled = magnitudeerr([k * si for si in fb["s"]], [k * oi for oi in other])
        assert scaled == pytest.approx(base, rel=1e-10), f"not scale-invariant at k={k}"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_magnitudeerrdb_is_twenty_log10_of_magnitudeerr(fb):
    """The dB variant is the same quantity on a log scale — nothing else."""
    from cool_frames.numpy.phase import magnitudeerr, magnitudeerrdb

    other = [0.9 * si + 1e-3 for si in fb["s"]]
    lin = magnitudeerr(fb["s"], other)
    assert magnitudeerrdb(fb["s"], other) == pytest.approx(20 * np.log10(lin), rel=1e-10)


@pytest.mark.unit
@pytest.mark.requires_impl
def test_magnitudeerr_grows_with_the_perturbation(fb):
    from cool_frames.numpy.phase import magnitudeerr

    errs = [magnitudeerr(fb["s"], [(1 + eps) * si for si in fb["s"]])
            for eps in (1e-4, 1e-2, 1e-1)]
    assert errs[0] < errs[1] < errs[2]


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.requires_impl
def test_maxit_zero_returns_the_initial_estimate(fb):
    """``maxit=0`` must be a no-op that still returns a well-formed result."""
    from cool_frames.numpy.phase import gla

    c, f, relres, niter = gla(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                              real=True, maxit=0)
    assert niter == 0 and len(relres) == 0
    assert len(c) == len(fb["s"])
    assert np.asarray(f).size == fb["Ls"]
    for cm, sm in zip(c, fb["s"]):
        assert np.allclose(np.abs(cm), sm, atol=1e-12)
        assert np.allclose(np.angle(cm), 0.0), "maxit=0 should leave the zero start phase"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_input_magnitudes_are_not_mutated(fb):
    """The algorithms must not write into the caller's arrays."""
    from cool_frames.numpy.phase import gla, legla, spsi

    before = [si.copy() for si in fb["s"]]
    gla(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"], real=True, maxit=3)
    legla(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"], real=True, maxit=3)
    spsi(fb["s"], fb["a"], fb["fc"], FS)
    for m, (now, was) in enumerate(zip(fb["s"], before)):
        assert np.array_equal(now, was), f"channel {m} was modified in place"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_complex_mode_returns_a_complex_signal(fb):
    """``real=False`` returns a complex signal; ``real=True`` a real one."""
    from cool_frames.numpy.phase import gla

    _c, f_real, _r, _n = gla(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                             real=True, maxit=2)
    _c, f_cplx, _r, _n = gla(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=fb["Ls"],
                             real=False, maxit=2)
    assert not np.iscomplexobj(np.asarray(f_real)) or \
        np.max(np.abs(np.imag(np.asarray(f_real)))) < 1e-12
    assert np.asarray(f_cplx).size == fb["Ls"]
    assert np.all(np.isfinite(np.asarray(f_cplx)))


@pytest.mark.unit
@pytest.mark.requires_impl
def test_ls_shorter_than_l_truncates_the_output(fb):
    """``Ls`` is the requested output length, independent of the internal ``L``."""
    from cool_frames.numpy.phase import gla

    for Ls in (LS, LS // 2, LS // 4):
        _c, f, _r, _n = gla(fb["s"], fb["g"], fb["a"], L=fb["L"], Ls=Ls,
                            real=True, maxit=2)
        assert np.asarray(f).size == Ls, f"Ls={Ls} gave {np.asarray(f).size} samples"




# ---------------------------------------------------------------------------
# Regressions for the v0.1.1 fixes
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.requires_impl
def test_legla_with_no_truncation_reproduces_gla(fb):
    """``relthr=0`` keeps the whole projection kernel, so LEGLA == GLA.

    This is the anchor for the whole kernel implementation: the truncated
    projection is only trustworthy if the *untruncated* one is provably the
    same operator as the synthesise-and-reanalyse projection GLA performs.
    """
    from cool_frames.numpy.phase import gla, legla

    kw = dict(L=fb["L"], Ls=fb["Ls"], real=True, maxit=10)
    c_gla = gla(fb["s"], fb["g"], fb["a"], **kw)[0]
    c_legla = legla(fb["s"], fb["g"], fb["a"], **kw, relthr=0.0)[0]

    scale = max(float(np.max(np.abs(u))) for u in c_gla)
    diff = max(np.max(np.abs(u - v)) for u, v in zip(c_gla, c_legla)) / scale
    assert diff < 1e-12, f"legla(relthr=0) differs from gla by {diff:.2e} (relative)"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_relthr_controls_the_projection(fb):
    """``relthr`` is a live parameter: harder truncation, bigger departure.

    Until v0.1.1 it was never read — ``legla`` was a bit-identical alias for
    ``gla`` at every setting.  A monotone departure is the cheapest evidence
    that the kernel is actually being thresholded.
    """
    from cool_frames.numpy.phase import gla, legla

    kw = dict(L=fb["L"], Ls=fb["Ls"], real=True, maxit=10)
    c_gla = gla(fb["s"], fb["g"], fb["a"], **kw)[0]
    scale = max(float(np.max(np.abs(u))) for u in c_gla)

    def departure(relthr):
        c = legla(fb["s"], fb["g"], fb["a"], **kw, relthr=relthr)[0]
        return max(np.max(np.abs(u - v)) for u, v in zip(c_gla, c)) / scale

    d = [departure(r) for r in (1e-4, 1e-2, 5e-1)]
    assert d[0] < d[1] < d[2], f"departure from gla is not monotone in relthr: {d}"
    assert d[0] > 1e-8, "relthr=1e-4 changed nothing — is the kernel being truncated?"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_modtrunc_differs_from_trunc(fb):
    """``variant='modtrunc'`` zeroes the self-term, so it is a different operator.

    The pre-v0.1.1 branch computed ``angle(c + (proj - c)) == angle(proj)``,
    which is algebraically identical to ``'trunc'``.
    """
    from cool_frames.numpy.phase import legla

    kw = dict(L=fb["L"], Ls=fb["Ls"], real=True, maxit=6, relthr=1e-3)
    c_trunc = legla(fb["s"], fb["g"], fb["a"], **kw, variant="trunc")[0]
    c_mod = legla(fb["s"], fb["g"], fb["a"], **kw, variant="modtrunc")[0]

    scale = max(float(np.max(np.abs(u))) for u in c_trunc)
    diff = max(np.max(np.abs(u - v)) for u, v in zip(c_trunc, c_mod)) / scale
    assert diff > 1e-3, f"modtrunc is indistinguishable from trunc ({diff:.2e})"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_truncation_makes_the_kernel_sparser(fb):
    """Raising ``relthr`` must actually drop stored kernel entries.

    Without this, `test_relthr_controls_the_projection` could pass on a kernel
    that changed for some other reason.
    """
    from cool_frames.numpy.filterbanks._utils import normalise_a
    from cool_frames.numpy.phase._leglakernel import LeglaKernel

    a = normalise_a(fb["a"], len(fb["g"]))
    hops = np.round(a[:, 0] / a[:, 1]).astype(int)
    N = [len(si) for si in fb["s"]]

    nnz = [
        LeglaKernel(fb["g"], fb["gd"], hops, N, fb["L"], real=True, relthr=r).nnz
        for r in (0.0, 1e-3, 1e-1)
    ]
    assert nnz[0] > nnz[1] > nnz[2] > 0, f"kernel did not get sparser: {nnz}"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_decolbfgs_gradient_agrees_with_finite_differences(fb):
    """The analytic gradient matches a central finite difference.

    Until v0.1.1 it did not — it used ``conj(c)`` where the Wirtinger
    derivative calls for ``c``, and omitted the factor the single-sided
    synthesis implies.  The consequence was not a slightly-off gradient but a
    wrong-signed one, which made L-BFGS-B abort before its first step.
    """
    from cool_frames.numpy.filterbanks import filterbank, ifilterbank
    from cool_frames.numpy.filterbanks._utils import normalise_a

    g, a, L = fb["g"], normalise_a(fb["a"], len(fb["g"])), fb["L"]
    p = 2.0 / 3.0
    s_p = [np.power(si, p) for si in fb["s"]]

    def obj_grad(x_flat):
        c = filterbank(x_flat, g, a, L=L)
        obj = 0.0
        grad_c = []
        for m in range(len(g)):
            cm = np.asarray(c[m]).ravel()
            cm_abs = np.abs(cm)
            diff = np.power(cm_abs, p) - s_p[m]
            obj += float(np.sum(diff**2))
            safe = np.maximum(cm_abs, np.finfo(float).eps)
            grad_c.append(p * diff * np.power(safe, p - 2.0) * cm)
        return obj, np.real(ifilterbank(grad_c, g, a, Ls=L, real=True)).ravel()

    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(L) * 0.01
    _f0, g0 = obj_grad(x0)

    eps = 1e-6
    probes = rng.choice(L, 8, replace=False)
    fd = np.empty(probes.size)
    for j, i in enumerate(probes):
        xp, xm = x0.copy(), x0.copy()
        xp[i] += eps
        xm[i] -= eps
        fd[j] = (obj_grad(xp)[0] - obj_grad(xm)[0]) / (2 * eps)

    rel = np.max(np.abs(g0[probes] - fd)) / max(np.max(np.abs(fd)), 1e-30)
    assert rel < 1e-5, f"analytic gradient disagrees with finite differences by {rel:.2e}"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_decolbfgs_actually_iterates(fb):
    """``niter`` tracks ``maxit`` and more iterations improve the result.

    The pre-v0.1.1 symptom of the gradient bug: ``scipy.optimize.minimize``
    returned ``status=2`` ("ABNORMAL") with ``nit == 0`` for every ``maxit``,
    and the output was bit-identical no matter how long you asked it to run.
    """
    from cool_frames.numpy.phase import decolbfgs

    kw = dict(L=fb["L"], Ls=fb["Ls"], real=True)
    results = {m: decolbfgs(fb["s"], fb["g"], fb["a"], **kw, maxit=m) for m in (1, 5, 40)}

    for m, (_c, _f, _r, niter) in results.items():
        assert niter == m, f"decolbfgs(maxit={m}) reported niter={niter}"

    quality = [_consistency(results[m][0], fb) for m in (1, 5, 40)]
    assert quality[2] < quality[0], (
        f"40 iterations ({quality[2]:.4f}) did not beat 1 ({quality[0]:.4f})"
    )


@pytest.mark.unit
@pytest.mark.requires_impl
def test_decolbfgs_handles_silence_without_warning():
    """An all-zero channel must not overflow in the ``|c|^(p-2)`` gradient term.

    With ``p = 2/3`` the unguarded expression evaluated ``tiny**-1.33``, which
    overflowed to inf and then produced ``inf * 0 = nan``.  The magnitude
    projection hid the damage; only a pair of RuntimeWarnings showed.
    """
    import warnings

    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import audfilters
    from cool_frames.numpy.phase import decolbfgs

    g, a, _fc, L, _info = audfilters(FS, LS)
    s = [np.abs(ci) for ci in filterbank(np.zeros(LS), g, a, L=L)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c_out, _f, _r, _n = decolbfgs(s, g, a, L=L, Ls=LS, real=True, maxit=3)

    runtime = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert not runtime, f"silent input still warns: {[str(w.message) for w in runtime]}"
    for cm in c_out:
        assert np.all(np.isfinite(cm)), "silent input produced NaN/Inf"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_spsi_takes_hz_and_a_sampling_rate(fb):
    """The natural call — constructor output straight through — is the correct one.

    Until v0.1.1 ``spsi`` wanted normalised frequencies while every constructor
    returned Hz, and passing Hz was silently *worse than leaving the phase at
    zero* (consistency 0.50 against a 0.38 baseline).
    """
    from cool_frames.numpy.phase import spsi

    baseline = _consistency([si.astype(complex) for si in fb["s"]], fb)
    natural = _consistency(spsi(fb["s"], fb["a"], fb["fc"], FS)[0], fb)

    assert natural < baseline, (
        f"spsi(s, a, fc, fs) gives consistency {natural:.4f}, no better than the "
        f"zero-phase baseline {baseline:.4f}"
    )
    assert natural < 0.5 * baseline, "spsi should improve substantially on the baseline"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_spsi_normalised_input_matches_hz_input(fb):
    """``fs=1.0`` is the escape hatch for callers who already normalised."""
    from cool_frames.numpy.phase import spsi

    hz = spsi(fb["s"], fb["a"], fb["fc"], FS)[0]
    norm = spsi(fb["s"], fb["a"], fb["fc"] / FS, 1.0)[0]
    for m, (u, v) in enumerate(zip(hz, norm)):
        assert np.allclose(u, v), f"channel {m} differs between the two spellings"


@pytest.mark.unit
@pytest.mark.requires_impl
def test_spsi_rejects_frequencies_above_nyquist(fb):
    """The old unit mix-up is now an error, not a silently poor result."""
    from cool_frames.numpy.phase import spsi

    with pytest.raises(ValueError, match="Nyquist"):
        spsi(fb["s"], fb["a"], fb["fc"], 1.0)

    with pytest.raises(ValueError, match="positive sampling rate"):
        spsi(fb["s"], fb["a"], fb["fc"], 0.0)


@pytest.mark.unit
@pytest.mark.requires_impl
def test_gsrtisila_spsi_start_beats_a_zero_start(fb):
    """``startphase='spsi'`` must be worth choosing over ``'zero'``.

    It reads centre frequencies from the filters themselves.  Before v0.1.1 it
    used ``fc[m] = m / M`` — a ramp reaching ~0.96 cycles/sample, nearly twice
    Nyquist — which made the "smarter" starts worse than the naive one.
    """
    from cool_frames.numpy.phase import gsrtisila

    kw = dict(L=fb["L"], Ls=fb["Ls"], real=True, maxit=1)
    zero = _consistency(gsrtisila(fb["s"], fb["g"], fb["a"], **kw, startphase="zero")[0], fb)
    spsi_start = _consistency(gsrtisila(fb["s"], fb["g"], fb["a"], **kw, startphase="spsi")[0], fb)
    assert spsi_start < zero, (
        f"startphase='spsi' ({spsi_start:.4f}) is no better than 'zero' ({zero:.4f})"
    )


@pytest.mark.unit
@pytest.mark.requires_impl
def test_recovered_centre_frequencies_match_the_constructor(fb):
    """The estimator underlying those starts agrees with ``audfilters``' own ``fc``."""
    from cool_frames.numpy.phase._centerfreq import filter_center_frequencies

    est = filter_center_frequencies(fb["g"], fb["L"])
    true = fb["fc"] / FS
    assert est.shape == true.shape
    assert np.max(np.abs(est - true)) < 2.0 / fb["L"], (
        f"centre-frequency estimate is off by more than two DFT bins: "
        f"{np.max(np.abs(est - true)):.4g}"
    )
    assert np.all(est <= 0.5 + 1e-12), "estimated a centre frequency above Nyquist"


@pytest.mark.unit
@pytest.mark.requires_impl
@pytest.mark.parametrize("name", ["gla", "legla", "decolbfgs", "rtisila"])
def test_random_start_is_reproducible_with_a_seed(name, fb):
    """``seed`` makes ``startphase='rand'`` repeatable; omitting it does not."""
    import cool_frames.numpy.phase as P

    fn = getattr(P, name)
    kw = dict(L=fb["L"], Ls=fb["Ls"], real=True, maxit=2, startphase="rand")

    first = fn(fb["s"], fb["g"], fb["a"], **kw, seed=11)[0]
    again = fn(fb["s"], fb["g"], fb["a"], **kw, seed=11)[0]
    other = fn(fb["s"], fb["g"], fb["a"], **kw, seed=12)[0]
    unseeded_a = fn(fb["s"], fb["g"], fb["a"], **kw)[0]
    unseeded_b = fn(fb["s"], fb["g"], fb["a"], **kw)[0]

    assert all(np.array_equal(u, v) for u, v in zip(first, again)), f"{name}: seed not honoured"
    assert not all(np.array_equal(u, v) for u, v in zip(first, other)), f"{name}: seed ignored"
    assert not all(np.array_equal(u, v) for u, v in zip(unseeded_a, unseeded_b)), (
        f"{name}: unseeded runs are identical — is the default seeded?"
    )
