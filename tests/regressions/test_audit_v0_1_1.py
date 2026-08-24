"""
test_audit_v0_1_1.py
====================
Regression tests for the defects found by the v0.1.1 subsystem audit.

Every test here corresponds to a specific bug that shipped in v0.1.0 and
silently produced a wrong answer.  Each asserts the *corrected* behaviour and
carries a note describing the original failure — so a reintroduction is
recognisable from the test output, not merely red.

The audit covered the filter designers, the filterbank core and frame theory,
the operators, the diagnostics, and the PyTorch backend.  ``DEFECT_REGISTER.md``
at the repository root lists all of them, including the ones still open.

Why these particular checks
---------------------------
The recurring shape of the bugs was *silence*: a fallback that returned zeros,
a parameter that was read into a variable and never used, a residual measured
on a different object than the one returned, a documented keyword that changed
nothing.  None of them raised.  So the assertions here are mostly structural
invariants that a silent failure cannot satisfy — adjointness, a frame's
condition number, agreement between two backends, a round trip — rather than
comparisons against stored reference values.
"""

from __future__ import annotations

import math
import warnings

import pytest

import numpy as np

FS = 4000
LS = 512


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def erb():
    """A small band-limited ERB bank — the package's flagship design."""
    from cool_frames.numpy.filterbanks import filterbankdual
    from cool_frames.numpy.filters import audfilters

    g, a, fc, L, _info = audfilters(FS, LS)
    return dict(g=g, a=a, fc=fc, L=L, Ls=LS, gd=filterbankdual(g, a, L))


@pytest.fixture(scope="module")
def fir_bank():
    """A well-conditioned *time-domain* (FIR) bank.

    The FIR path was where several of the worst defects lived: it is the code
    the band-limited designers never exercise.
    """
    from cool_frames.numpy.filters.lowlevel import firfilter

    g = [firfilter("hann", 32, fc=f) for f in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]
    return dict(g=g, a=4, L=256)


# ---------------------------------------------------------------------------
# The FIR path: dual, adjointness, coefficient shape
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_painless_dual_refuses_fir_filters_instead_of_returning_zeros(fir_bank):
    """A diagonal dual cannot exist for a time-limited filter; say so.

    ``painlessfilterbank`` used to fall through to ``H = np.zeros(0)`` for any
    channel without an ``'H'`` key — i.e. every FIR channel — so
    ``filterbankdual``/``filterbanktight`` returned an *all-zero bank* and
    ``ifilterbank`` reconstructed exactly 0.0.  Nothing raised, and
    ``filterbankbounds`` still reported a valid frame for the same bank.

    Computing a diagonal dual anyway would only trade a visible failure for a
    plausible-looking wrong answer: the painless construction needs each
    filter's *frequency* support to be at most L/a, and an FIR filter is
    full-band by construction.
    """
    from cool_frames.numpy.filterbanks import filterbankdual, filterbanktight

    for fn in (filterbankdual, filterbanktight):
        with pytest.raises(ValueError, match=r"time-domain|painless"):
            fn(fir_bank["g"], fir_bank["a"], fir_bank["L"])


@pytest.mark.requires_impl
def test_fir_synthesis_is_the_adjoint_of_fir_analysis(fir_bank):
    """``<Af, y> == <f, A*y>`` for a bank with non-zero filter offsets.

    The frequency-domain synthesis leg read the filter's ``offset`` into a
    local named ``skip`` and then never used it, while the analysis leg applied
    it.  The two were therefore not an adjoint pair: the "frame operator" came
    out non-symmetric (symmetry error 1.33) with negative eigenvalues, and CG
    on it diverged to relres 4.6e12.
    """
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filterbanks._utils import comp_ifilterbank, normalise_a

    g, a, L = fir_bank["g"], fir_bank["a"], fir_bank["L"]
    a_n = normalise_a(a, len(g))
    rng = np.random.default_rng(0)

    f = rng.standard_normal(L)
    c = filterbank(f, g, a_n, L=L)
    y = [rng.standard_normal(len(cm)) + 1j * rng.standard_normal(len(cm)) for cm in c]

    lhs = sum(np.vdot(np.asarray(yc).ravel(), np.asarray(ac).ravel()) for ac, yc in zip(c, y))
    rhs = np.vdot(np.fft.ifft(comp_ifilterbank(y, g, a_n, L)[:, 0]), f)

    rel = abs(lhs - rhs) / max(abs(lhs), 1e-30)
    assert rel < 1e-10, f"synthesis is not the adjoint of analysis (rel. error {rel:.2e})"


@pytest.mark.requires_impl
def test_fir_frame_operator_is_symmetric_and_positive(fir_bank):
    """The direct consequence of the adjointness fix, stated as spectra.

    A frame operator ``F*F`` is symmetric positive semi-definite by
    construction.  With the offset dropped it was neither.
    """
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filterbanks._utils import comp_ifilterbank, normalise_a

    g, a, L = fir_bank["g"], fir_bank["a"], fir_bank["L"]
    a_n = normalise_a(a, len(g))

    op = np.zeros((L, L))
    for k in range(L):
        e = np.zeros(L)
        e[k] = 1.0
        c = filterbank(e, g, a_n, L=L)
        op[:, k] = np.real(np.fft.ifft(comp_ifilterbank(c, g, a_n, L)[:, 0]))

    asym = np.max(np.abs(op - op.T)) / max(np.max(np.abs(op)), 1e-30)
    assert asym < 1e-10, f"frame operator is not symmetric (||F - F^T||/||F|| = {asym:.2e})"

    eigmin = float(np.min(np.linalg.eigvalsh(0.5 * (op + op.T))))
    assert eigmin > -1e-10, f"frame operator has a negative eigenvalue ({eigmin:.3e})"


@pytest.mark.requires_impl
def test_comp_ifilterbank_treats_1d_and_2d_coefficients_alike(fir_bank):
    """``np.atleast_2d`` made a 1-D ``(N,)`` array a ``(1, N)`` *row*.

    ``cm[:, w]`` was then a length-1 slice whose single value broadcast across
    every bin.  The two frequency-domain branches of the same function reshape
    to a column; only the FIR branch did not.
    """
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filterbanks._utils import comp_ifilterbank, normalise_a

    g, a, L = fir_bank["g"], fir_bank["a"], fir_bank["L"]
    a_n = normalise_a(a, len(g))
    rng = np.random.default_rng(1)

    c = filterbank(rng.standard_normal(L), g, a_n, L=L)
    c1 = [np.asarray(cm).ravel().astype(complex) for cm in c]
    c2 = [cm.reshape(-1, 1) for cm in c1]

    diff = np.max(np.abs(comp_ifilterbank(c1, g, a_n, L) - comp_ifilterbank(c2, g, a_n, L)))
    assert diff < 1e-12, f"1-D and 2-D coefficient input disagree by {diff:.3e}"


@pytest.mark.requires_impl
def test_iterative_inverse_converges_on_an_fir_bank(fir_bank):
    """CG needs a symmetric positive operator; before the fix it diverged."""
    from cool_frames.numpy.filterbanks import filterbank, filterbankbounds, ifilterbankiter

    g, a, L = fir_bank["g"], fir_bank["a"], fir_bank["L"]
    rng = np.random.default_rng(2)
    x = rng.standard_normal(L)

    A, _B = filterbankbounds(g, a, L, real=False)
    assert A > 0, "fixture bank is not a frame; pick a better one"

    xr, _relres, _n = ifilterbankiter(
        filterbank(x, g, a, L=L), g, a, L, alg="cg", real=False, maxit=300, tol=1e-10
    )
    err = np.linalg.norm(np.real(xr) - x) / np.linalg.norm(x)
    assert err < 1e-2, f"CG did not converge on an FIR bank (rel. error {err:.3e})"


# ---------------------------------------------------------------------------
# Iterative solvers: honest residuals, real preconditioning
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_ifilterbankiter_residual_describes_the_returned_signal(erb):
    """The reported ``relres`` must be the residual of what comes back.

    It used to be measured on the complex CG iterate while ``np.real(x)`` was
    returned — reporting a converged 3.2e-07 for a signal whose true residual
    was 0.226.
    """
    from cool_frames.numpy.filterbanks import filterbank, ifilterbankiter
    from cool_frames.numpy.phase import magnitudeerr

    rng = np.random.default_rng(3)
    x = rng.standard_normal(erb["Ls"])
    c = filterbank(x, erb["g"], erb["a"], L=erb["L"])

    xr, relres, _n = ifilterbankiter(c, erb["g"], erb["a"], erb["L"], real=True, maxit=60)
    xr = np.real(np.asarray(xr)).ravel()

    c2 = filterbank(xr, erb["g"], erb["a"], L=erb["L"])
    true = magnitudeerr([np.abs(u) for u in c], [np.abs(u) for u in c2])

    assert true < 10 * max(relres, 1e-12) + 1e-6, (
        f"reported relres {relres:.3e} but the returned signal's true residual is {true:.3e}"
    )


@pytest.mark.requires_impl
def test_pcg_preconditioner_does_not_slow_convergence(erb):
    """A preconditioner that makes CG slower is not a preconditioner.

    The frame-operator diagonal is indexed by *DFT bin*, but the CG vectors are
    in the time domain.  Multiplying a time-domain residual by a
    frequency-domain diagonal is an arbitrary window; it consistently cost
    iterations (9 -> 15 on a gabfilters bank where the correct one takes 4).
    """
    from cool_frames.numpy.filterbanks import filterbank, ifilterbankiter

    rng = np.random.default_rng(4)
    c = filterbank(rng.standard_normal(erb["Ls"]), erb["g"], erb["a"], L=erb["L"])

    kw = dict(real=True, maxit=100, tol=1e-10)
    _x1, _r1, n_plain = ifilterbankiter(c, erb["g"], erb["a"], erb["L"], alg="cg", **kw)
    _x2, _r2, n_pre = ifilterbankiter(c, erb["g"], erb["a"], erb["L"], alg="pcg", **kw)

    assert n_pre <= n_plain, f"'pcg' took {n_pre} iterations against plain CG's {n_plain}"


# ---------------------------------------------------------------------------
# filterbankwin: the complex variants must be reachable
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_filterbankwin_dual_and_realdual_are_different(erb):
    """All four string branches used to call the same function with real=True.

    ``'dual'`` was a silent synonym for ``'realdual'`` and ``'tight'`` for
    ``'realtight'``, so the complex (two-sided) constructions were unreachable
    through this interface.
    """
    from cool_frames.numpy.filterbanks import filterbankdual
    from cool_frames.numpy.filterbanks._utils import filterbankwin

    g, a, L = erb["g"], erb["a"], erb["L"]
    d_complex = filterbankwin(["dual", g], a, L)[0]
    d_real = filterbankwin(["realdual", g], a, L)[0]

    def _maxdiff(u, v):
        return max(np.max(np.abs(np.asarray(p["H"]) - np.asarray(q["H"]))) for p, q in zip(u, v))

    assert _maxdiff(d_complex, d_real) > 1e-6, "'dual' and 'realdual' are still identical"
    assert _maxdiff(d_complex, filterbankdual(g, a, L, real=False)) < 1e-12
    assert _maxdiff(d_real, filterbankdual(g, a, L, real=True)) < 1e-12


# ---------------------------------------------------------------------------
# gabfilters
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("fs,Ls", [(4000, 512), (8000, 4096)])
def test_gabfilters_is_a_tight_frame(fs, Ls):
    """A 4x-overlap Hann DGT is exactly tight; it used to read kappa = 1.667.

    The DC and Nyquist channels have no conjugate partner, so the
    ``2*real(ifft)`` fold in ``ifilterbank(real=True)`` double-counts them
    unless they carry a 1/sqrt(2).  Every other designer applies it.
    """
    from cool_frames.numpy.filterbanks import filterbankbounds
    from cool_frames.numpy.filters import gabfilters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, _fc, L, _info = gabfilters(fs, Ls)

    A, B = filterbankbounds(g, a, L)
    assert A > 0
    kappa = B / A
    assert abs(kappa - 1.0) < 1e-3, f"expected a tight frame, got kappa = {kappa:.5f}"


@pytest.mark.requires_impl
def test_gabfilters_edge_channels_carry_the_fold_correction():
    """The mechanism behind the test above, asserted directly."""
    from cool_frames.numpy.filters import gabfilters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, _a, _fc, _L, _info = gabfilters(4000, 512, M=64, a=16)

    interior = np.asarray(g[1]["H"])
    np.testing.assert_allclose(
        np.asarray(g[0]["H"]) * math.sqrt(2.0), interior, rtol=1e-12, atol=1e-12
    )
    # M = 64 is even, so the top channel is the Nyquist bin.
    np.testing.assert_allclose(
        np.asarray(g[-1]["H"]) * math.sqrt(2.0), interior, rtol=1e-12, atol=1e-12
    )


@pytest.mark.requires_impl
def test_gabfilters_warns_when_the_lattice_is_not_painless():
    """It used to claim painlessness in a comment and warn only on redundancy < 1.

    Painlessness needs ``M**2 <= 4L``, which the default lattice never
    satisfies, so ``filterbankdual`` returns an approximate dual.  Every other
    designer warns in this situation.
    """
    from cool_frames.numpy.filters import gabfilters

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gabfilters(16000, 16000)

    assert any("painless" in str(w.message) for w in caught), (
        "no painless warning for a lattice that violates the condition"
    )


# ---------------------------------------------------------------------------
# Filter designers: biquad endpoints, butterworth bandwidth
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
@pytest.mark.parametrize("fc", [0.0, 1000.0, 8000.0])
def test_biquad_ml_parameters_round_trip(fc):
    """``rho``/``phi`` must decode back to the ``r``/``theta`` they encode.

    At DC, Nyquist and ``r <= 0`` the logit was undefined and the code fell
    back to 0.0 — the sigmoid's *midpoint*, not its limit.  Rebuilding a DC
    resonator from its own stored parameters moved it to fs/4.
    """
    from cool_frames.numpy.filters import biquadfilter

    out = biquadfilter(fc, 500.0, fs=16000)
    gm = out[0] if isinstance(out, list) else out

    theta_decoded = math.pi / (1.0 + math.exp(-gm["phi"]))
    r_decoded = 1.0 / (1.0 + math.exp(-gm["rho"]))

    assert abs(theta_decoded - gm["theta"]) < 1e-9, (
        f"theta {gm['theta']:.6f} encoded to phi {gm['phi']:.3f}, decoded to {theta_decoded:.6f}"
    )
    assert abs(r_decoded - gm["r"]) < 1e-9


@pytest.mark.requires_impl
def test_freqwin_windows_share_one_bandwidth_convention():
    """``bw`` means the width at ``bwrelheight`` for every window in ``freqwin``.

    ``'butterworth'`` used the half-power convention instead, making it 14 %
    wider than requested while the other three were accurate to 0.4 %.
    """
    from cool_frames.numpy.filters._freqwin import freqwin

    L, fs, bw_hz = 4096, 8000.0, 200.0
    target = 10 ** (-3.0 / 10.0)
    step = fs / L
    idx = int(round(bw_hz / 2 / step))

    values = {}
    for name in ("gauss", "roex", "gammatone", "butterworth"):
        H = np.abs(np.fft.fftshift(np.asarray(freqwin(name, L, bw_hz, fs=fs))))
        H = H / H.max()
        values[name] = H[L // 2 + idx]

    for name, v in values.items():
        assert abs(v - target) < 0.02, (
            f"{name}: |H(bw/2)| = {v:.4f}, expected ~{target:.4f} "
            f"(all of {sorted(values)} must share one convention)"
        )


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_framemuleigs_handles_a_complex_symbol():
    """A multiplier is self-adjoint only for a real symbol and matching frames.

    Both code paths assumed it unconditionally — one symmetrised ``mat``
    outright, the other used ``eigsh`` — so a complex symbol gave a leading
    eigenvalue 10 % off, and genuinely complex eigenvalues were reported as
    real numbers of the wrong sign.
    """
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import audfilters
    from cool_frames.numpy.operators import framemul, framemuleigs

    g, a, _fc, L, _info = audfilters(FS, 64)
    rng = np.random.default_rng(5)
    c = filterbank(rng.standard_normal(L), g, a, L=L)
    sigma = [rng.standard_normal(cm.shape) + 1j * rng.standard_normal(cm.shape) for cm in c]

    mat = np.zeros((L, L), dtype=complex)
    for k in range(L):
        e = np.zeros(L)
        e[k] = 1.0
        mat[:, k] = framemul(e, g, g, a, sigma, L)
    expected = np.linalg.eigvals(mat)
    expected = expected[np.argsort(-np.abs(expected))][:4]

    got = np.asarray(framemuleigs(g, g, a, sigma, L, K=4))
    rel = np.max(np.abs(expected - got)) / np.max(np.abs(expected))
    assert rel < 1e-8, f"eigenvalues differ from the true spectrum by {rel:.3e}"


@pytest.mark.requires_impl
def test_framemulinv_accepts_zero_iterations():
    """``maxit=0`` raised UnboundLocalError from the function's own return."""
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import audfilters
    from cool_frames.numpy.operators import framemulinv

    g, a, _fc, L, _info = audfilters(FS, 64)
    rng = np.random.default_rng(6)
    f = rng.standard_normal(L)
    sigma = [np.ones_like(np.abs(cm)) for cm in filterbank(f, g, a, L=L)]

    _x, info = framemulinv(f, g, g, a, sigma, L, maxit=0)
    assert info["iter"] == 0
    assert not info["converged"]


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_reassigned_spectrogram_returns_real_phase_gradients():
    """Both documented outputs used to be identically zero for every input.

    ``filterbankphasegrad`` was called as ``(c, a, fc)`` — coefficients as the
    signal, hop sizes as the filters, centre frequencies as the hops — and its
    4-tuple return unpacked into two names.  A bare ``except Exception`` turned
    the guaranteed failure into zeros.
    """
    from cool_frames.numpy.diagnostics.spectrogram import reassigned_spectrogram

    t = np.arange(2048) / FS
    x = np.sin(2 * np.pi * 500 * t) + 0.5 * np.sin(2 * np.pi * 1200 * t)
    spec = reassigned_spectrogram(x, FS)

    ifd = np.asarray(spec["instfreq_deviation"])
    gd = np.asarray(spec["groupdelay_shift"])

    assert not np.allclose(ifd, 0.0), "instfreq_deviation is identically zero"
    assert not np.allclose(gd, 0.0), "groupdelay_shift is identically zero"
    assert np.all(np.isfinite(np.real(ifd))) and np.all(np.isfinite(np.real(gd)))


@pytest.mark.requires_impl
@pytest.mark.parametrize("fn_name", ["filterbank_spectrogram", "reassigned_spectrogram"])
def test_spectrogram_db_image_is_scale_equivariant(fn_name):
    """Scaling the input by k must shift the dB image by 20*log10(k), no more.

    Channels are decimated by different hops, so most of the stacked image is
    padding.  Padding at 0.0 dB (magnitude 1.0) and then taking ``peak_db``
    over the padded array meant that for a quiet input the *padding* was the
    peak: the dynamic-range window sat ~30 dB too high and most real
    coefficients fell below the floor.
    """
    import cool_frames.numpy.diagnostics.spectrogram as S

    fn = getattr(S, fn_name)
    t = np.arange(2048) / FS
    base = np.sin(2 * np.pi * 500 * t)

    k = 1e-4
    loud = np.asarray(fn(base, FS)["coeff_db"])
    quiet = np.asarray(fn(k * base, FS)["coeff_db"])

    shift = 20 * np.log10(k)
    err = np.max(np.abs((quiet - shift) - loud))
    # 1e-3 dB, not 0: the dB conversion uses `20*log10(mag + 1e-10)`, and that
    # additive guard is not scale-equivariant, contributing ~9e-5 dB here.  The
    # bug being guarded against moved the image by ~30 dB, so this leaves four
    # orders of headroom.
    assert err < 1e-3, (
        f"a {k:g}x gain changed the dB image by more than a constant (max deviation {err:.4f} dB)"
    )


@pytest.mark.requires_impl
def test_recommend_filterbank_cqt_resolution_responds_to_f0():
    """The "bins from f0" expression contained no ``f0``.

    ``max(12, min(96, ceil(8*log(2)*2)))`` is 12 for every input, so detecting
    an f0 made the resolution *coarser* than the no-f0 default of 48, and the
    ``f0 > 500`` branch below it was dead (``min(12, 24) == 12``).
    """
    from cool_frames.numpy.diagnostics.recommend_filterbank import recommend_filterbank

    fs = 16000
    t = np.arange(fs) / fs
    rng = np.random.default_rng(7)

    def _bins(f0):
        x = sum(np.sin(2 * np.pi * f0 * n * t) / n for n in range(1, 12))
        x = x + 0.05 * rng.standard_normal(t.size)
        rec = recommend_filterbank(x, fs)
        return rec.params.get("bins")

    low, high = _bins(80.0), _bins(600.0)
    assert low is not None and high is not None, "no CQT recommendation to compare"
    assert low > high, (
        f"a low f0 ({low} bins) should get finer resolution than a high one ({high} bins)"
    )


# ---------------------------------------------------------------------------
# Second tranche: the defects the first pass recorded as open
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_framemulappr_is_optimal_in_complex_mode():
    """``framemul`` took ``np.real(...)`` unconditionally.

    So the operator was only R-linear even with ``real=False``, while
    ``_appr_system``'s ``real=False`` branch models a C-linear generator.  On
    an operator built as an exact multiplier — where a zero-error symbol
    provably exists — the returned symbol was 44.5 % off in Hilbert-Schmidt
    norm.  ``real=True`` was always optimal, so the defect was one-sided.
    """
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import gabfilters
    from cool_frames.numpy.operators import framemul, framemulappr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, _fc, L, _info = gabfilters(FS, 128, real=False, M=16)

    rng = np.random.default_rng(20)
    c = filterbank(rng.standard_normal(L), g, a, L=L)
    sigma = [rng.standard_normal(cm.shape) for cm in c]

    def _as_matrix(sym):
        out = np.zeros((L, L), dtype=complex)
        for k in range(L):
            e = np.zeros(L)
            e[k] = 1.0
            out[:, k] = framemul(e, g, g, a, sym, L, real=False)
        return out

    T = _as_matrix(sigma)
    recovered = framemulappr(T, g, g, a, L, real=False, method="full")
    err = np.linalg.norm(T - _as_matrix(recovered)) / np.linalg.norm(T)
    assert err < 1e-10, f"HS round trip through the recovered symbol is {err:.3e}"


@pytest.mark.requires_impl
def test_waveletfilters_redtar_controls_redundancy():
    """``redtar`` was inert on every non-uniform sampling mode.

    The block computed ``N_new`` and then used ``N_old``, re-encoding the
    original hops in fractional form — ``redtar=0.5`` and ``redtar=50`` gave
    bit-identical filters.  On ``sampling='uniform'`` it did act, but an
    unguarded ``np.floor`` produced ``a = 0``, after which ``filterbank``
    raised "negative dimensions are not allowed".
    """
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import waveletfilters

    for sampling in ("regsampling", "uniform"):
        seen = []
        for redtar in (0.5, 2.0, 20.0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                g, a, _fc, L, _info = waveletfilters(FS, LS, redtar=redtar, sampling=sampling)
            af = np.asarray(a)
            af = af[:, 0] / af[:, 1] if af.ndim == 2 else af.astype(float)
            assert np.all(af >= 1), f"{sampling}: redtar={redtar} produced a hop of 0"
            filterbank(np.zeros(LS), g, a, L=L)  # must not raise
            seen.append(round(float(np.sum(1.0 / af)), 4))

        assert len(set(seen)) == len(seen), (
            f"{sampling}: redundancy {seen} did not respond to redtar"
        )
        assert seen[0] < seen[1] < seen[2], f"{sampling}: redundancy is not monotone: {seen}"


@pytest.mark.requires_impl
@pytest.mark.parametrize("sampling", ["regsampling", "uniform", "fractional"])
def test_warpedfilters_complex_range_runs(sampling):
    """``np.vstack`` on a 1-D ``a`` crashed on the default sampling modes.

    ``waveletfilters`` has the identical construct written correctly.
    """
    from cool_frames.numpy.filters._audscale import erbtofreq, freqtoerb
    from cool_frames.numpy.filters._warpedfilters_design import warpedfilters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, _a, fc, L, _info = warpedfilters(
            freqtoerb,
            erbtofreq,
            FS,
            50,
            1900,
            4,
            LS,
            sampling=sampling,
            freqrange="complex",
        )
    assert len(g) > 0 and L > 0 and len(fc) == len(g)


@pytest.mark.requires_impl
def test_analyze_filterbank_uses_the_requested_length():
    """Every internal call omitted ``L``, so redundancy was inflated.

    ``filterbanklength`` silently rounds up to the lcm-multiple, so the
    coefficient counts were computed at one length while the frame section used
    another: 3.63 reported against a true 2.15, plus a *false* "not painless"
    note explaining the resulting mismatch on a bank that is painless.
    """
    from cool_frames.numpy.filterbanks._analysis import analyze_filterbank
    from cool_frames.numpy.filters import audfilters

    g, a, _fc, _L, _info = audfilters(FS, LS)
    report = analyze_filterbank(g, a, LS)

    af = np.asarray(a)
    af = af[:, 0] / af[:, 1] if af.ndim == 2 else af.astype(float)
    expected = float(np.sum(np.ceil(LS / af)) / LS)

    got = report["filterbank"]["redundancy"]
    assert abs(got - expected) / expected < 0.05, (
        f"redundancy {got:.4f} at L={LS} against an expected ~{expected:.4f}"
    )


@pytest.mark.requires_impl
def test_frame_bounds_are_never_negative():
    """``filterbankbounds_svd`` returned the raw eigenvalue extreme.

    A rank-deficient bank reported ``A = -9.7e-16``, which downstream became
    "condition numbers" like -3.2e+15.  A frame bound is non-negative by
    definition; a tiny negative eigenvalue means "not a frame", i.e. zero.
    """
    from cool_frames.numpy.filterbanks._frame import filterbankbounds_svd, filterbanktight
    from cool_frames.numpy.filters import audfilters, waveletfilters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gw, aw, _fc, Lw, _i = waveletfilters(FS, LS)
        gwt = filterbanktight(gw, aw, Lw)
        A_w, B_w = filterbankbounds_svd(gwt, aw, Lw)

        g, a, _fc, L, _i = audfilters(FS, LS)
        A_a, _B_a = filterbankbounds_svd(g, a, L, real=False)

    assert A_w >= 0.0 and B_w >= A_w, f"wavelet tight bounds ({A_w}, {B_w})"
    assert A_a >= 0.0, f"audfilters real=False lower bound {A_a}"


@pytest.mark.requires_impl
def test_diagonal_dual_warns_when_the_bank_is_not_painless():
    """``filterbankwin`` computed ``info['ispainless']`` and nothing read it.

    So a non-painless bank got a silently approximate dual: on
    ``waveletfilters`` the returned "tight" frame is rank-deficient and loses
    72 % of the signal while ``filterbankbounds`` prints kappa = 1.000000.

    ``waveletfilters(FS, LS)`` is no longer such a bank: ``painless`` now
    defaults to ``True``, precisely because that default was indefensible, so
    the non-painless case has to be asked for explicitly.  The subject of this
    test is ``filterbanktight``'s warning, not the designer's default -- that
    default is pinned by ``TestReconstruction`` in
    ``tests/layer1_filters/unit/test_waveletfilters.py``.
    """
    from cool_frames.numpy.filterbanks import filterbanktight
    from cool_frames.numpy.filters import audfilters, waveletfilters

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g, a, _fc, L, _i = waveletfilters(FS, LS, painless=False)
        filterbanktight(g, a, L)
    assert any("painless" in str(w.message) for w in caught), (
        "no warning for a bank that badly violates the painless condition"
    )

    # ...and stays quiet for one that satisfies it.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        g, a, _fc, L, _i = audfilters(FS, LS)
        filterbanktight(g, a, L)
    assert not any("painless" in str(w.message) for w in caught), (
        "false painless warning on audfilters"
    )


@pytest.mark.requires_impl
def test_convention_mismatch_is_detected_for_audfilters():
    """The detector used to measure the *reconstructed spectrum*, not the filters.

    On that measure a single-sided ERB bank scores 0.383 and two-sided banks
    0.28-1.0 — overlapping — so ``audfilters``, the bank in almost every
    docstring in the package, slipped past the 0.3 threshold and reconstructed
    with 46 % error in silence.
    """
    from cool_frames.numpy.filterbanks import filterbank, filterbankdual, ifilterbank
    from cool_frames.numpy.filters import audfilters, cqtfilters

    for designer in (audfilters, cqtfilters):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, a, _fc, L, _i = designer(FS, LS)
            gd = filterbankdual(g, a, L, real=True)  # single-sided dual

        x = np.random.default_rng(21).standard_normal(LS)
        c = filterbank(x, g, a, L=L)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ifilterbank(c, gd, a, Ls=LS, real=False)  # wrong mode for this dual
        assert any("single-sided" in str(w.message) for w in caught), (
            f"{designer.__name__}: mismatch not detected"
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ifilterbank(c, gd, a, Ls=LS, real=True)  # correct mode
        assert not any("appear" in str(w.message) for w in caught), (
            f"{designer.__name__}: false mismatch warning on the correct pairing"
        )


@pytest.mark.requires_impl
def test_window_width_measurement_depends_on_window_shape():
    """``_winwidthatheight`` assumed the peak sat at index 0.

    The designers store peak-*centred* responses, for which the first sample is
    already below threshold, so the crossing indices were pinned and ``w/gl``
    collapsed to 1.  A Hann, a rectangle, a triangle and a three-bin needle all
    returned gamma = 11597.8.
    """
    from cool_frames.numpy.filters._gabfilters import _comp_tfrfromwin

    n = 101
    hann = np.hanning(n)
    rect = np.ones(n)
    needle = np.zeros(n)
    needle[n // 2 - 1 : n // 2 + 2] = [0.5, 1.0, 0.5]

    g_hann, g_rect, g_needle = (_comp_tfrfromwin(w) for w in (hann, rect, needle))

    assert g_needle < g_hann < g_rect, (
        f"gamma does not order by window width: needle {g_needle:.1f}, "
        f"hann {g_hann:.1f}, rect {g_rect:.1f}"
    )
    # A DFT-ordered window must give the same answer as its centred form.
    assert abs(_comp_tfrfromwin(np.fft.ifftshift(hann)) - g_hann) < 1e-9


@pytest.mark.requires_impl
def test_analyze_coefficients_reads_the_stacked_axis_correctly():
    """``filterbank(stack=True)`` returns ``(N, M)``; this read it as ``(M, N)``.

    Every per-channel statistic was computed over time slices instead of
    channels, and a 23-channel bank reported M = 16.
    """
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filterbanks._analysis import analyze_coefficients
    from cool_frames.numpy.filters import audfilters

    g, _a, _fc, L, _info = audfilters(FS, LS)
    x = np.random.default_rng(22).standard_normal(LS)
    stacked = filterbank(x, g, np.full(len(g), 32), L=L, stack=True)

    assert stacked.shape[1] == len(g), "fixture assumption: stacked is (N, M)"
    assert analyze_coefficients(stacked)["shape"]["M"] == len(g)


@pytest.mark.requires_impl
def test_waveletfilters_delay_reaches_every_channel():
    """The delay loop stopped at ``lp_num``, excluding the appended highpass.

    ``delay=5`` produced descriptor delays ``[5, 5, ..., 5, 0]``.
    """
    from cool_frames.numpy.filters import waveletfilters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, _a, _fc, _L, _i = waveletfilters(FS, LS, delay=5)

    delays = [gm.get("delay") for gm in g]
    assert all(d == 5 for d in delays), f"not every channel was delayed: {delays[-4:]}"


@pytest.mark.requires_impl
def test_filterbankfreqz_hop_argument_is_optional():
    """``a`` was documented as a parameter and never read.

    A transfer function does not depend on the hop size.  It is now explicitly
    optional and documented as ignored, rather than inviting callers to believe
    it does something.
    """
    from cool_frames.numpy.filterbanks import filterbankfreqz
    from cool_frames.numpy.filters import audfilters

    g, a, _fc, L, _info = audfilters(FS, LS)
    assert np.array_equal(filterbankfreqz(g, a, L), filterbankfreqz(g, None, L))
