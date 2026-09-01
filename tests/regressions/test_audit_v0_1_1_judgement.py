"""
test_audit_v0_1_1_judgement.py
==============================
Regression tests for the three items the v0.1.1 audit left as judgement calls
rather than closing outright, and which were subsequently closed properly:

1. **PGHI from magnitude.**  ``filterbankconstphase``'s magnitude path measured
   no better than zero phase, so ``reconstruct(method='pghi')`` was made to
   raise ``NotImplementedError``.  The cause turned out to be a single missing
   term, not a design limit — see the tests below.
2. **Hardcoded ``real=True``** in the ``magnitude_to_audio`` recipe: correct
   for every bank the auditory/CQT designers produce, silently wrong for a
   genuinely two-sided one.  Now derived from the filters.
3. **Lint scope.**  The linted path set lived inline in two CI steps, so
   nothing declared it in one place and a repo-root ``ruff --fix`` quietly
   reformatted files outside it.  Now pinned in ``pyproject.toml``.

Why the PGHI tests are written as *comparisons* rather than thresholds
---------------------------------------------------------------------
Phase retrieval quality is signal-dependent, and magnitude error is vacuous
here (it is ~1e-16 for any method, since every method returns exactly the
magnitudes it was given).  The meaningful metric is **consistency**:
resynthesise, re-analyse, and compare the magnitudes you get back with the
ones you asked for.  A wrong-but-plausible phase estimate shows up as poor
consistency; a *broken* one shows up as consistency no better than zero phase,
which is precisely the failure these tests are here to catch.

So each test pins a *relationship* — better than zero phase, comparable to the
integrator fed the true gradients — rather than a magic constant that would
have to be re-tuned whenever the fixture changes.
"""

from __future__ import annotations

import pytest

import numpy as np

FS = 4000
LS = 512


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def erb():
    """The flagship ERB bank, plus its dual, at a size that keeps tests quick."""
    from cool_frames.numpy.filterbanks import filterbankdual
    from cool_frames.numpy.filters import audfilters
    from cool_frames.numpy.filters._tfr import compute_tfr_from_filters

    g, a, fc, L, _info = audfilters(FS, LS)
    a_arr = np.atleast_1d(np.asarray(a))
    a_int = np.array(
        [int(a_arr[m]) if a_arr.ndim == 1 else int(a_arr[m, 0]) for m in range(len(g))],
        dtype=int,
    )
    return dict(
        g=g,
        a=a,
        a_int=a_int,
        fc=fc,
        L=L,
        Ls=LS,
        gd=filterbankdual(g, a, L),
        sqtfr=np.sqrt(np.asarray(compute_tfr_from_filters(g, L), dtype=float)),
    )


def _signals():
    """Two fixtures that stress the estimator in opposite ways.

    The chirp has a large, smooth, well-resolved instantaneous-frequency
    deviation from each channel's centre frequency — it is what the tgrad
    correction exists for.  The stationary pair has essentially none, so it
    exercises the centre-frequency term almost alone.  A regression in either
    term shows up in at least one of them.
    """
    t = np.arange(LS) / FS
    T = LS / FS
    w = np.hanning(LS)
    return {
        "chirp": np.sin(2 * np.pi * (200 * t + 300 * t**2 / T)) * w,
        "two_sines": (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1320 * t)) * w,
    }


def _consistency(erb, s_target, c_phase):
    """Resynthesise with the given phase, re-analyse, compare magnitudes."""
    from cool_frames.numpy.filterbanks import filterbank, ifilterbank
    from cool_frames.numpy.phase import magnitudeerr

    f = np.real(ifilterbank(c_phase, erb["gd"], erb["a"], Ls=erb["Ls"], real=True))
    c2 = filterbank(f, erb["g"], erb["a"], L=erb["L"])
    return float(magnitudeerr(s_target, [np.abs(u) for u in c2]))


def _analyse(erb, x):
    from cool_frames.numpy.filterbanks import filterbank

    return [np.abs(u) for u in filterbank(x, erb["g"], erb["a"], L=erb["L"])]


# ---------------------------------------------------------------------------
# 1. PGHI from magnitude alone
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("name", ["chirp", "two_sines"])
def test_pghi_from_magnitude_beats_zero_phase(erb, name):
    """The magnitude path must actually use the magnitudes.

    This is the headline regression.  In v0.1.0 the ``'pghi'`` branch of the
    recipe drew ``torch.rand_like`` phase; when that was removed, the
    underlying ``filterbankconstphase`` magnitude path measured 0.385 against a
    zero-phase baseline of 0.385 — i.e. the gradients it estimated carried no
    information at all, so PGHI degenerated to "return the magnitudes with
    whatever phase the integrator happened to accumulate".

    The cause was that ``comp_filterbankphasegradfrommag`` returned the
    instantaneous-frequency *deviation* from each channel's centre frequency,
    while the heap integrator consumes the *absolute* normalised instantaneous
    frequency (the convention ``filterbankphasegrad`` returns).  The
    centre-frequency term is an order of magnitude larger than the deviation it
    was carrying, so omitting it did not merely degrade the estimate — it
    replaced it.

    Asserting merely "better than zero phase" is deliberately loose: it is the
    weakest statement that the broken version cannot satisfy.
    """
    from cool_frames.numpy.phase import filterbankconstphase

    x = _signals()[name]
    s = _analyse(erb, x)

    baseline = _consistency(erb, s, [u.astype(complex) for u in s])
    res = filterbankconstphase(s, erb["a_int"], erb["fc"], sqtfr=erb["sqtfr"], fs=FS)
    got = _consistency(erb, s, res[0])

    assert got < baseline, (
        f"PGHI from magnitude ({got:.4f}) is no better than zero phase "
        f"({baseline:.4f}) on {name!r} — the estimated phase gradients carry "
        f"no information."
    )
    # A real margin, not a coin flip.  Measured at v0.1.1: chirp 16x,
    # two_sines 2.3x.  The 1.5x floor leaves room for fixture drift while
    # still failing hard if the centre-frequency term goes missing again.
    assert baseline / got > 1.5, (
        f"PGHI on {name!r} improves on zero phase by only "
        f"{baseline / got:.2f}x ({got:.4f} vs {baseline:.4f})."
    )


@pytest.mark.requires_impl
def test_pghi_magnitude_path_approaches_the_true_gradient_path_on_a_chirp(erb):
    """On a signal whose frequency really moves, estimated ≈ true gradients.

    The heap integrator was never the broken part: fed the gradients that
    ``filterbankphasegrad`` computes from the *signal*, it reconstructs to
    0.0047 on this bank.  That number is the integrator's own ceiling, and it
    is the right thing to measure the magnitude-only estimate against — a
    threshold on the absolute consistency would conflate a regression in the
    estimator with one in the integrator.

    On the chirp the two land within a factor of two of each other, which says
    the magnitude-derived gradients are close to the true ones rather than
    merely better than nothing.
    """
    from cool_frames.numpy.phase import filterbankconstphase

    x = _signals()["chirp"]
    s = _analyse(erb, x)

    from_signal = _consistency(
        erb, s, filterbankconstphase(x, erb["g"], erb["a"], L=erb["L"], fc=erb["fc"])[0]
    )
    res = filterbankconstphase(s, erb["a_int"], erb["fc"], sqtfr=erb["sqtfr"], fs=FS)
    from_mag = _consistency(erb, s, res[0])

    assert from_mag < 2.0 * from_signal + 1e-3, (
        f"magnitude-only PGHI ({from_mag:.4f}) is far worse than the same "
        f"integrator fed the true gradients ({from_signal:.4f})."
    )


@pytest.mark.requires_impl
def test_phasegradfrommag_returns_absolute_instantaneous_frequency(erb):
    """The two gradient estimators must agree on what they are returning.

    ``filterbankphasegrad`` (from the signal) and
    ``comp_filterbankphasegradfrommag`` (from the magnitudes) feed the same
    integrator, so they must share a convention.  They did not: the latter
    returned a deviation.  The bug is invisible in isolation — both return
    plausible float arrays — and only shows up when you ask whether channel m's
    values sit near channel m's centre frequency.

    For a pure tone the true instantaneous frequency is the tone's frequency in
    every channel that sees it, so the loudest cell's tgrad pins the convention
    directly.
    """
    from cool_frames.numpy.phase._fbphasegradfrommag import (
        comp_filterbankneighbors,
        comp_filterbankphasegradfrommag,
    )

    f0 = 660.0
    t = np.arange(LS) / FS
    s = _analyse(erb, np.sin(2 * np.pi * f0 * t) * np.hanning(LS))

    M = len(erb["g"])
    N = np.array([len(u) for u in s], dtype=int)
    abss = np.concatenate(s)
    NEIGH, posInfo = comp_filterbankneighbors(erb["a_int"], M, N, do_real=True)
    fc_norm = erb["fc"] / FS * 2.0  # the [0, 2] convention: 2 == fs

    tgrad, _fgrad, _logs = comp_filterbankphasegradfrommag(
        abss, N, erb["a_int"], M, erb["sqtfr"], fc_norm, NEIGH, posInfo
    )

    expected = f0 / FS * 2.0
    loudest = int(np.argmax(abss))
    assert abs(tgrad[loudest] - expected) < 0.15, (
        f"tgrad at the loudest cell is {tgrad[loudest]:.4f}, expected about "
        f"{expected:.4f} (the tone's normalised frequency). A value near zero "
        f"means the estimator is returning a deviation from the centre "
        f"frequency instead of the absolute instantaneous frequency."
    )


@pytest.mark.requires_impl
def test_constphase_infers_fs_from_hz_fc_but_says_so(erb):
    """Inferring the sampling rate is fine; inferring it silently was not.

    The magnitude path normalises an ``fc`` that looks like Hz by assuming the
    top channel sits at Nyquist.  Checked against the designers, that
    assumption is *exact* for every single-sided bank the package ships —
    ``audfilters``, ``cqtfilters`` (which appends a Nyquist channel whatever
    ``fmax`` is) and ``gabfilters(real=True)`` — and wrong by about a factor of
    two for a two-sided bank.

    So the inference earns its place and this test pins two things about it:
    that it reproduces the explicit-``fs`` answer *exactly* on a bank where the
    assumption holds, and that it is not silent.  An earlier v0.1.1 build
    raised here instead, which broke every existing caller to buy nothing on
    the common path.
    """
    from cool_frames.numpy.phase import filterbankconstphase

    x = _signals()["chirp"]
    s = _analyse(erb, x)

    def _first(res):
        return res[0]

    # Seeded, so the below-threshold random phase does not mask the
    # comparison — see test_constphase_uses_a_local_generator below.
    with pytest.warns(UserWarning, match=r"Nyquist"):
        inferred = _first(
            filterbankconstphase(s, erb["a_int"], erb["fc"], sqtfr=erb["sqtfr"], rng=0)
        )
    explicit = _first(
        filterbankconstphase(s, erb["a_int"], erb["fc"], sqtfr=erb["sqtfr"], fs=FS, rng=0)
    )

    assert np.allclose(inferred[0], explicit[0], rtol=1e-12, atol=0.0), (
        "the inferred sampling rate does not reproduce the explicit-fs result "
        "on audfilters, where the top channel is exactly at Nyquist"
    )
    assert _consistency(erb, s, inferred) == pytest.approx(
        _consistency(erb, s, explicit), rel=1e-12
    )


@pytest.mark.requires_impl
def test_constphase_does_not_warn_when_fc_is_already_normalised(erb):
    """Normalised input is unambiguous, so it must pass without a warning.

    Guards the other side of the branch: a caller who has already normalised
    ``fc`` to [0, 2] should not be told the sampling rate is being guessed,
    and a mismatched ``fc`` length should still be a hard error rather than
    something the inference papers over.
    """
    import warnings as _warnings

    from cool_frames.numpy.phase import filterbankconstphase

    s = _analyse(erb, _signals()["chirp"])
    fc_norm = erb["fc"] / FS * 2.0

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", UserWarning)
        filterbankconstphase(s, erb["a_int"], fc_norm, sqtfr=erb["sqtfr"])

    with pytest.raises(ValueError, match=r"entries but"):
        filterbankconstphase(s, erb["a_int"], fc_norm[:-1], sqtfr=erb["sqtfr"])


@pytest.mark.requires_impl
def test_phasegradfrommag_edge_channels_are_on_the_interior_scale(erb):
    """Channel 0 and channel M-1 must not be a factor of two out.

    Interior channels average a one-sided difference quotient from each
    neighbour; edge channels have only one neighbour.  The original code summed
    the two sides with the missing side's denominator defaulting to ``1.0``,
    which left the edges both half-scaled and divided by the wrong frequency
    spacing.  With a smooth broadband input the estimated deviation should not
    jump discontinuously at the boundary.
    """
    from cool_frames.numpy.phase._fbphasegradfrommag import (
        comp_filterbankneighbors,
        comp_filterbankphasegradfrommag,
    )

    rng = np.random.default_rng(3)
    s = _analyse(erb, rng.standard_normal(LS) * np.hanning(LS))

    M = len(erb["g"])
    N = np.array([len(u) for u in s], dtype=int)
    abss = np.concatenate(s)
    NEIGH, posInfo = comp_filterbankneighbors(erb["a_int"], M, N, do_real=True)
    fc_norm = erb["fc"] / FS * 2.0

    tgrad, _f, _l = comp_filterbankphasegradfrommag(
        abss, N, erb["a_int"], M, erb["sqtfr"], fc_norm, NEIGH, posInfo
    )

    # Per-channel RMS deviation from the channel's own centre frequency.
    dev = []
    off = 0
    for m in range(M):
        n = int(N[m])
        dev.append(float(np.sqrt(np.mean((tgrad[off : off + n] - fc_norm[m]) ** 2))))
        off += n
    dev = np.asarray(dev)

    interior = np.median(dev[1:-1])
    for m in (0, M - 1):
        assert dev[m] < 6.0 * interior + 1e-12, (
            f"edge channel {m} has RMS deviation {dev[m]:.4g} against an "
            f"interior median of {interior:.4g} — the edge channels are not "
            f"on the same scale as the interior."
        )


# ---------------------------------------------------------------------------
# 2. real= derived from the filters, not assumed
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_filterbank_is_real_separates_the_two_families():
    """The single-sided/two-sided question has an answer; read it off the filters.

    ``reconstruct`` hardcoded ``real=True``.  That is right for every bank the
    auditory and constant-Q designers produce and wrong for a genuinely
    two-sided one, where folding the spectrum double-counts — a ~30 dB error
    that no exception marks.  The detector already existed inside
    ``ifilterbank`` (as the mismatch warning); it is now public so callers can
    ask instead of assume.
    """
    from cool_frames.numpy.filterbanks import filterbank_is_real
    from cool_frames.numpy.filters import audfilters, gabfilters

    g, a, _fc, L, _info = audfilters(FS, LS)
    assert filterbank_is_real(g, a, L) is True, "the ERB bank is single-sided"

    # A painless Gabor lattice (a <= L/M), so the comparison is not muddied by
    # the approximate-dual warning.  Same window, same lattice, both
    # conventions — the only thing that differs is single- vs two-sided.
    g_r, a_r = gabfilters(FS, LS, M=32, a=8, real=True)[:2]
    assert filterbank_is_real(g_r, a_r, LS) is True, "gabfilters(real=True) is single-sided"

    g_c, a_c = gabfilters(FS, LS, M=32, a=8, real=False)[:2]
    assert filterbank_is_real(g_c, a_c, LS) is False, "gabfilters(real=False) is two-sided"


@pytest.mark.requires_torch_impl
def test_reconstruct_derives_real_and_honours_an_override(erb):
    """``reconstruct`` must not assume the convention, and must let you set it.

    Two claims in one test because they are the same claim: the default is
    *derived* (so it still equals True on this single-sided bank, but for a
    reason rather than by luck), and an explicit ``real=`` overrides the
    detection rather than being ignored.  The override is what makes the
    derivation safe to trust — if the detector is ever wrong about an exotic
    bank, the caller has a way out that does not involve editing the library.
    """
    torch = pytest.importorskip("torch")
    from torch_additions.recipes.magnitude_to_audio import reconstruct

    from cool_frames.torch.filterbanks import filterbank as tfilterbank

    x = torch.as_tensor(_signals()["two_sines"], dtype=torch.float64)
    s_mag = [u.abs() for u in tfilterbank(x, erb["g"], erb["a"], L=erb["L"])]

    auto, _info = reconstruct(s_mag, erb["g"], erb["a"], erb["L"], LS, method="spsi")
    explicit_true, _ = reconstruct(
        s_mag, erb["g"], erb["a"], erb["L"], LS, method="spsi", real=True
    )
    explicit_false, _ = reconstruct(
        s_mag, erb["g"], erb["a"], erb["L"], LS, method="spsi", real=False
    )

    # Detection agrees with the correct value for this (single-sided) bank...
    assert np.allclose(np.asarray(auto), np.asarray(explicit_true)), (
        "the derived real= disagrees with real=True on a single-sided bank"
    )
    # ...and the override is actually wired through, not silently dropped.
    assert not np.allclose(np.asarray(auto), np.asarray(explicit_false)), (
        "passing real=False changed nothing — the override is being ignored"
    )


@pytest.mark.requires_torch_impl
@pytest.mark.parametrize("method", ["pghi", "gla", "fgla", "legla", "spsi"])
def test_reconstruct_every_method_beats_zero_phase(erb, method):
    """Every advertised method must be a phase-retrieval method.

    In v0.1.0 ``'pghi'`` and ``'spsi'`` were the same block of
    ``torch.rand_like`` phase, reporting ``converged: True``.  A method that
    silently returns noise under a real algorithm's name is the failure mode
    this whole file exists to prevent, so the check is applied to all five
    rather than only to the two that were broken.
    """
    torch = pytest.importorskip("torch")
    from torch_additions.recipes.magnitude_to_audio import reconstruct

    from cool_frames.torch.filterbanks import filterbank as tfilterbank

    x_np = _signals()["chirp"]
    x = torch.as_tensor(x_np, dtype=torch.float64)
    s_mag = [u.abs() for u in tfilterbank(x, erb["g"], erb["a"], L=erb["L"])]
    s_ref = [np.asarray(u) for u in s_mag]

    baseline = _consistency(erb, s_ref, [u.astype(complex) for u in s_ref])

    f, info = reconstruct(
        s_mag, erb["g"], erb["a"], erb["L"], LS, method=method, fc=erb["fc"], fs=FS
    )
    assert info["method"] == method

    from cool_frames.numpy.filterbanks import filterbank as nfilterbank
    from cool_frames.numpy.phase import magnitudeerr

    c2 = nfilterbank(np.real(np.asarray(f)), erb["g"], erb["a"], L=erb["L"])
    got = float(magnitudeerr(s_ref, [np.abs(u) for u in c2]))

    assert got < baseline, (
        f"reconstruct(method={method!r}) gives consistency {got:.4f}, no better "
        f"than zero phase ({baseline:.4f})."
    )


@pytest.mark.requires_torch_impl
def test_reconstruct_pghi_is_no_longer_unavailable(erb):
    """The NotImplementedError placeholder must be gone, not merely unreachable.

    Pinned separately from the quality checks above because it is a distinct
    regression: an earlier v0.1.1 build raised here deliberately, and a revert
    of the gradient fix that also restored the raise would leave the quality
    tests erroring rather than failing, which reads very differently in CI.
    """
    torch = pytest.importorskip("torch")
    from torch_additions.recipes.magnitude_to_audio import reconstruct

    from cool_frames.torch.filterbanks import filterbank as tfilterbank

    x = torch.as_tensor(_signals()["chirp"], dtype=torch.float64)
    s_mag = [u.abs() for u in tfilterbank(x, erb["g"], erb["a"], L=erb["L"])]

    f, info = reconstruct(s_mag, erb["g"], erb["a"], erb["L"], LS, method="pghi")
    assert info == {"method": "pghi", "n_iters": 0, "converged": True}
    assert np.asarray(f).size == LS
    assert np.all(np.isfinite(np.real(np.asarray(f))))


@pytest.mark.requires_torch_impl
def test_reconstruct_rejects_an_unknown_method(erb):
    """An unknown method name must raise, and name the alternatives."""
    torch = pytest.importorskip("torch")
    from torch_additions.recipes.magnitude_to_audio import reconstruct

    from cool_frames.torch.filterbanks import filterbank as tfilterbank

    x = torch.as_tensor(_signals()["chirp"], dtype=torch.float64)
    s_mag = [u.abs() for u in tfilterbank(x, erb["g"], erb["a"], L=erb["L"])]

    with pytest.raises(ValueError, match=r"legla"):
        reconstruct(s_mag, erb["g"], erb["a"], erb["L"], LS, method="nope")


# ---------------------------------------------------------------------------
# 3. The lint scope is declared in exactly one place
# ---------------------------------------------------------------------------


def test_lint_scope_is_pinned_in_pyproject_not_inline_in_ci():
    """CI must inherit the linted paths from ``pyproject.toml``.

    Not a test of the code's behaviour but of the thing that let a
    whole-tree ``ruff --fix`` slip ~14 unrelated files into a behavioural
    patch: the path set was written out inline in two separate CI steps, so
    there was no single declaration of what "in scope" meant and no way for a
    local ruff invocation to match CI.  With ``[tool.ruff] include`` pinned, a
    bare ``ruff check`` is the CI check.

    Guarding it with a test is proportionate because the failure is silent and
    the fix is one careless edit away from being undone.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    pyproject = root / "pyproject.toml"
    required = ("cool_frames/torch/**/*.py", "torch_additions/**/*.py")

    # `tomllib` is 3.11+, and this package supports 3.10 — parse properly where
    # the parser exists and fall back to scanning the section text where it does
    # not, rather than skipping the check on the oldest supported interpreter.
    # (Written the other way round first, with a bare `import tomllib`, which
    # failed CI's 3.10 job and nowhere else.)
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        tomllib = None  # type: ignore[assignment]

    if tomllib is not None:
        with open(pyproject, "rb") as fh:
            cfg = tomllib.load(fh)
        include = cfg.get("tool", {}).get("ruff", {}).get("include")
        assert include, "[tool.ruff] include is not set — the lint scope is unpinned"
        for entry in required:
            assert entry in include, f"{entry} dropped out of the lint scope"
    else:
        text = pyproject.read_text()
        start = text.index("[tool.ruff]")
        nxt = text.find("\n[", start + 1)
        section = text[start : nxt if nxt != -1 else len(text)]
        assert "include = [" in section, (
            "[tool.ruff] include is not set — the lint scope is unpinned"
        )
        for entry in required:
            assert f'"{entry}"' in section, f"{entry} dropped out of the lint scope"

    ci = (root / ".github" / "workflows" / "ci.yml").read_text()
    for line in ci.splitlines():
        stripped = line.strip()
        if stripped.startswith("run:") and "ruff" in stripped:
            assert "cool_frames/" not in stripped, (
                "ci.yml passes explicit paths to ruff again: "
                f"{stripped!r}. The scope belongs in pyproject.toml so that a "
                "bare `ruff check` matches CI."
            )


# ---------------------------------------------------------------------------
# Found while updating the callers for the fs change
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_explicit_gradients_must_be_passed_by_keyword_and_are_honoured(erb):
    """``filterbankconstphase(s, a, fc, [tgrad, fgrad])`` silently ignores them.

    Three tests in the existing suite passed their precomputed gradients as the
    *fourth positional* argument, which is ``L`` — unused on the magnitude
    path.  So ``test_with_explicit_gradient`` and its two siblings were
    asserting nothing about explicit gradients at all: they were exercising the
    estimate-from-magnitude path and passing because magnitude is preserved by
    construction either way.

    This pins the distinction the callers were missing: gradients handed in by
    keyword change the answer, and the positional form does not reach them.
    """
    from cool_frames.numpy.phase import filterbankconstphase, filterbankphasegrad

    x = _signals()["chirp"]
    s = _analyse(erb, x)
    tgrad, fgrad, _s, _c = filterbankphasegrad(x, erb["g"], erb["a"], erb["L"])

    def _first(res):
        return res[0]

    estimated = _first(filterbankconstphase(s, erb["a_int"], erb["fc"], sqtfr=erb["sqtfr"], fs=FS))
    explicit = _first(
        filterbankconstphase(s, erb["a_int"], erb["fc"], fs=FS, tgrad=tgrad, fgrad=fgrad)
    )

    # The keyword has to change the answer; if it did not, it is inert and the
    # callers who passed gradients positionally were right to think nothing
    # happened.  This is the assertion the test exists for.
    assert not np.allclose(np.angle(estimated[0]), np.angle(explicit[0])), (
        "explicit tgrad/fgrad changed nothing — the keywords are being ignored"
    )

    c_explicit = _consistency(erb, s, explicit)
    c_estimated = _consistency(erb, s, estimated)
    c_zero = _consistency(erb, s, [np.asarray(sm, dtype=complex) for sm in s])

    # Both paths have to be doing real work, which is what separates "the
    # gradients are honoured" from "the gradients are honoured and garbage".
    assert c_explicit < 0.2 * c_zero, (c_explicit, c_zero)
    assert c_estimated < 0.2 * c_zero, (c_estimated, c_zero)

    # This used to assert `explicit < estimated`: the true derivative-filter
    # gradients must beat the ones estimated from magnitude alone.  That held
    # by 0.4 % (0.043738 vs 0.044112) and stopped holding when the sign error
    # on the `below` branch of comp_filterbankphasegradfrommag was fixed --
    # the estimate improved to 0.042236 and now edges *past* the true
    # gradients on this probe.  The explicit number is unchanged at 0.043738,
    # as it must be: that path never touches the estimator.
    #
    # So the ordering was an artefact of the defect, and reinstating it would
    # mean pinning the bug.  What is left is the same signal-path /
    # magnitude-path disagreement documented on `filterbankconstphase`: the
    # two do not agree on interior channels under any gamma, and neither is
    # reliably ahead.  Pin that they stay comparable, which is falsifiable in
    # both directions, and leave which one wins to the probe.
    assert 0.5 < c_explicit / c_estimated < 2.0, (
        f"the two gradient paths have diverged: explicit {c_explicit:.6f} vs "
        f"estimated {c_estimated:.6f}"
    )


@pytest.mark.requires_impl
def test_constphase_is_reproducible_and_leaves_the_global_rng_alone(erb):
    """Found while asserting that two equivalent calls agree — they did not.

    ``filterbankconstphase`` assigns random phase to coefficients below the
    magnitude threshold.  That part is deliberate and correct: the phase of a
    coefficient at the noise floor carries no information, and integrating
    through it would propagate that noise outward.

    The defect was *which* generator supplied it.  ``np.random.uniform`` draws
    from NumPy's global state, so four identical calls returned four different
    answers (1.1e-4 absolute against a scale of 60), and every call silently
    advanced the caller's global random stream — action at a distance from a
    function that is nominally a transform.  A caller seeding ``np.random`` for
    their own experiment had it perturbed by an unrelated library call.

    Both halves are pinned here because they fail independently: a local
    generator with no seed argument would fix the side effect but not
    reproducibility, and a seeded global generator would do the reverse.
    """
    from cool_frames.numpy.phase import filterbankconstphase

    s = _analyse(erb, _signals()["chirp"])

    def _run(**kw):
        res = filterbankconstphase(s, erb["a_int"], erb["fc"], sqtfr=erb["sqtfr"], fs=FS, **kw)
        return res[0]

    # Reproducible when seeded.
    a, b = _run(rng=0), _run(rng=0)
    for m, (u, v) in enumerate(zip(a, b)):
        assert np.array_equal(np.asarray(u), np.asarray(v)), (
            f"two seeded calls disagree in channel {m} — output is not reproducible"
        )

    # Different seeds must actually differ, or the seed is being ignored.
    assert not np.array_equal(np.asarray(_run(rng=0)[0]), np.asarray(_run(rng=1)[0])), (
        "rng=0 and rng=1 gave identical output — the seed is not reaching the draw"
    )

    # And the global stream is untouched.
    np.random.seed(20260821)
    expected = np.random.uniform()
    np.random.seed(20260821)
    _run()
    assert np.random.uniform() == expected, (
        "filterbankconstphase advanced NumPy's global random state; it must "
        "draw from a local Generator"
    )
