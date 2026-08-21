"""
test_phase_retrieval_parity.py
==============================
NumPy/PyTorch parity for the phase-retrieval family, and the differentiability
that is the torch backend's reason to exist.

``test_phase_retrieval.py`` already covers ``gla`` and the ADMM family.  This
file covers the rest — ``legla``/``flegla``, ``rtisila``, ``lertisila``,
``gsrtisila``, ``decolbfgs``, ``spsi`` — along the three axes that matter for a
second backend:

    1. **Parity.** The torch implementation is the *same algorithm*; on identical
       float64 inputs it must reproduce the NumPy reference to near machine
       precision.  A backend that merely gets close is a second algorithm with a
       second set of bugs.
    2. **Differentiability.** Gradients must flow from a loss on the output back
       to the input magnitudes.  This is the only thing torch buys over NumPy;
       an op that detaches silently makes a training loop optimise nothing.
    3. **Dtype fidelity.** A float32 input gives a float32/complex64 result,
       computed in single precision — not upcast to double and cast back.

The regression section at the bottom covers the v0.1.1 torch fixes: ``legla``
reported ``|Re(c)|`` as its residual, ``decolbfgs`` could not be placed in an
autograd graph at all, and the whole backend upcast float32 to float64.

Cost note: the RTISI family costs seconds per call even on this deliberately
tiny fixture (``FS=4000``, ``Ls=512``, 23 channels), and each parity test runs
both backends — about 15 s for the three of them.  They are not marked ``slow``
on purpose: CI runs ``-m "not slow"``, and marking them would leave those
modules untested.
"""

from __future__ import annotations

import pytest

import numpy as np

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.requires_torch_impl

FS = 4000
LS = 512

# Tolerance for "same algorithm, different array library" in float64.
PARITY_ATOL = 1e-10


@pytest.fixture(scope="module")
def fbp():
    """Tiny ERB filterbank plus target magnitudes in both array libraries."""
    from cool_frames.numpy.filterbanks import filterbank, filterbankdual
    from cool_frames.numpy.filters import audfilters

    t = np.arange(LS) / FS
    x = (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1320 * t)) * np.hanning(LS)

    g, a, fc, L, _info = audfilters(FS, LS)
    gd = filterbankdual(g, a, L)
    c = filterbank(x, g, a, L=L)
    s = [np.abs(ci) for ci in c]
    s_t = [torch.from_numpy(np.ascontiguousarray(si)).to(torch.float64) for si in s]
    return dict(g=g, gd=gd, a=a, fc=fc, L=L, Ls=LS, s=s, s_t=s_t, c=c)


def _max_abs_diff(np_list, torch_list):
    return max(
        float(np.max(np.abs(np.asarray(u) - v.detach().cpu().numpy())))
        for u, v in zip(np_list, torch_list)
    )


def _scale(np_list):
    return max(float(np.max(np.abs(np.asarray(u)))) for u in np_list)


# ---------------------------------------------------------------------------
# 1. Parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["legla", "flegla"])
def test_legla_coefficients_match_numpy(method, fbp):
    """The torch LEGLA iteration reproduces the NumPy one to float64 precision."""
    from cool_frames.numpy.phase import legla as legla_np
    from cool_frames.torch.phase import legla as legla_t

    kw = dict(L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=6, method=method)
    c_np = legla_np(fbp["s"], fbp["g"], fbp["a"], **kw)[0]
    c_t = legla_t(fbp["s_t"], fbp["g"], fbp["a"], **kw)[0]

    rel = _max_abs_diff(c_np, c_t) / _scale(c_np)
    assert rel < PARITY_ATOL, f"legla({method}): backends disagree by {rel:.2e} (relative)"


@pytest.mark.parametrize("method", ["legla", "flegla"])
def test_legla_reconstructed_signal_matches_numpy(method, fbp):
    from cool_frames.numpy.phase import legla as legla_np
    from cool_frames.torch.phase import legla as legla_t

    kw = dict(L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=6, method=method)
    f_np = np.asarray(legla_np(fbp["s"], fbp["g"], fbp["a"], **kw)[1]).ravel()
    f_t = legla_t(fbp["s_t"], fbp["g"], fbp["a"], **kw)[1].detach().cpu().numpy().ravel()

    assert f_t.size == f_np.size == LS
    rel = float(np.max(np.abs(f_np - f_t))) / max(float(np.max(np.abs(f_np))), 1e-30)
    assert rel < PARITY_ATOL, f"legla({method}): signals disagree by {rel:.2e}"


@pytest.mark.parametrize("name", ["rtisila", "lertisila", "gsrtisila"])
def test_rtisi_family_matches_numpy(name, fbp):
    """The real-time family is bit-comparable across backends.

    These are the most intricate members — per-frame loops with lookahead — and
    the most likely to drift between two hand-written implementations, so parity
    here is worth the seconds it costs.
    """
    import cool_frames.numpy.phase as NP
    import cool_frames.torch.phase as TP

    kw = dict(L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=1)
    c_np = getattr(NP, name)(fbp["s"], fbp["g"], fbp["a"], **kw)[0]
    c_t = getattr(TP, name)(fbp["s_t"], fbp["g"], fbp["a"], **kw)[0]

    rel = _max_abs_diff(c_np, c_t) / _scale(c_np)
    assert rel < PARITY_ATOL, f"{name}: backends disagree by {rel:.2e} (relative)"


def test_spsi_matches_numpy_to_single_precision(fbp):
    """``spsi`` agrees across backends, but only to ~1e-4 relative.

    Every other member of the family agrees to ~1e-14.  ``spsi`` accumulates a
    phase by repeated addition over frames, and the two implementations round
    that accumulation differently, so the tolerance here is deliberately three
    orders looser than ``PARITY_ATOL``.  If this ever tightens, tighten the
    assertion with it; if it loosens, something has diverged.
    """
    from cool_frames.numpy.phase import spsi as spsi_np
    from cool_frames.torch.phase import spsi as spsi_t

    c_np = spsi_np(fbp["s"], fbp["a"], fbp["fc"], FS)[0]
    c_t = spsi_t(fbp["s_t"], torch.as_tensor(fbp["a"]), torch.as_tensor(fbp["fc"]), FS)[0]

    rel = _max_abs_diff(c_np, c_t) / _scale(c_np)
    assert rel < 1e-4, f"spsi: backends disagree by {rel:.2e} (relative)"


@pytest.mark.parametrize("name", ["legla", "decolbfgs"])
def test_return_arity_matches_the_numpy_backend(name, fbp):
    """Both backends return ``(c, f, relres, niter)`` with the same shapes."""
    import cool_frames.numpy.phase as NP
    import cool_frames.torch.phase as TP

    kw = dict(L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=3)
    out_np = getattr(NP, name)(fbp["s"], fbp["g"], fbp["a"], **kw)
    out_t = getattr(TP, name)(fbp["s_t"], fbp["g"], fbp["a"], **kw)

    assert len(out_t) == len(out_np) == 4
    assert len(out_t[0]) == len(out_np[0]) == len(fbp["s"])
    assert [tuple(t.shape) for t in out_t[0]] == [u.shape for u in out_np[0]]
    assert out_t[1].numel() == np.asarray(out_np[1]).size == LS
    assert int(out_t[3]) >= 0


@pytest.mark.parametrize("name", ["legla"])
def test_magnitude_constraint_holds_in_torch(name, fbp):
    """``|c_out| == s`` in the torch backend too — same contract as NumPy."""
    import cool_frames.torch.phase as TP

    c_t = getattr(TP, name)(
        fbp["s_t"], fbp["g"], fbp["a"], L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=5
    )[0]
    for m, (cm, sm) in enumerate(zip(c_t, fbp["s_t"])):
        err = float(torch.max(torch.abs(torch.abs(cm) - sm)))
        scale = max(float(torch.max(sm)), 1e-30)
        assert err / scale < 1e-12, f"{name}: channel {m} magnitude drifted by {err / scale:.2e}"


# ---------------------------------------------------------------------------
# 2. Differentiability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,kw",
    [
        ("gla", dict(maxit=3)),
        ("gla", dict(maxit=3, method="fgla")),
        ("legla", dict(maxit=3)),
        ("legla", dict(maxit=3, method="flegla")),
    ],
)
def test_gradients_reach_the_input_magnitudes(name, kw, fbp):
    """A loss on the output must produce a finite, non-zero gradient on ``s``.

    This is the whole point of the torch backend.  A ``.detach()`` anywhere in
    the iteration would leave the gradient at ``None`` (or zero) and a training
    loop built on it would quietly optimise nothing at all.
    """
    import cool_frames.torch.phase as TP

    s_var = [si.clone().requires_grad_(True) for si in fbp["s_t"]]
    c_out = getattr(TP, name)(
        s_var, fbp["g"], fbp["a"], L=fbp["L"], Ls=fbp["Ls"], real=True, **kw
    )[0]
    loss = sum(torch.sum(torch.abs(cm) ** 2) for cm in c_out)
    loss.backward()

    for m, si in enumerate(s_var):
        assert si.grad is not None, f"{name}: channel {m} received no gradient"
        assert torch.all(torch.isfinite(si.grad)), f"{name}: channel {m} gradient has NaN/Inf"
    total = sum(float(torch.linalg.vector_norm(si.grad)) for si in s_var)
    assert total > 0, f"{name}: all gradients are exactly zero"


def test_gradient_through_the_reconstructed_signal(fbp):
    """Gradients also flow through the returned *signal*, not only ``c``.

    Callers train on waveform losses as often as on coefficient losses.
    """
    from cool_frames.torch.phase import gla

    s_var = [si.clone().requires_grad_(True) for si in fbp["s_t"]]
    _c, f, _relres, _niter = gla(
        s_var, fbp["g"], fbp["a"], L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=3
    )
    torch.sum(f.real**2).backward()

    total = sum(
        0.0 if si.grad is None else float(torch.linalg.vector_norm(si.grad)) for si in s_var
    )
    assert total > 0, "no gradient reached s through the reconstructed signal"


def test_gradient_is_not_constant_in_the_input(fbp):
    """The gradient must actually depend on ``s``.

    A constant gradient would pass the flow test above while still meaning the
    iteration had been replaced by something input-independent.
    """
    from cool_frames.torch.phase import gla

    def grad_norm(scale):
        s_var = [(scale * si).clone().detach().requires_grad_(True) for si in fbp["s_t"]]
        c = gla(s_var, fbp["g"], fbp["a"], L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=3)[0]
        sum(torch.sum(torch.abs(cm) ** 2) for cm in c).backward()
        return sum(float(torch.linalg.vector_norm(si.grad)) for si in s_var)

    small, large = grad_norm(1.0), grad_norm(4.0)
    assert large > 2 * small, (
        f"gradient barely responded to a 4x input scale ({small:.3e} -> {large:.3e})"
    )


# ---------------------------------------------------------------------------
# 4. Regressions for the v0.1.1 torch fixes
# ---------------------------------------------------------------------------


def test_torch_legla_reports_the_same_residual_as_numpy(fbp):
    """``relres`` matches the NumPy backend iteration for iteration.

    ``_legla.py`` used to build the residual as
    ``torch.abs(torch.as_tensor(cp, dtype=dtype))`` where ``dtype`` is *real*.
    The cast ran before ``abs``, discarding the imaginary part, so the reported
    residual was ``|Re(c)|``: 0.185 at iteration 6 where NumPy reported 0.070.
    The coefficients were unaffected, but ``relres`` is what a caller plots and
    what ``tol`` is compared against, so the method also stopped at the wrong
    time.
    """
    from cool_frames.numpy.phase import legla as legla_np
    from cool_frames.torch.phase import legla as legla_t

    kw = dict(L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=6)
    r_np = np.asarray(legla_np(fbp["s"], fbp["g"], fbp["a"], **kw)[2], dtype=float)
    r_t = np.asarray([float(v) for v in legla_t(fbp["s_t"], fbp["g"], fbp["a"], **kw)[2]])

    assert r_t.shape == r_np.shape
    assert np.allclose(r_t, r_np, rtol=1e-10, atol=1e-12), (
        f"residual sequences differ:\n  numpy {r_np}\n  torch {r_t}"
    )


def test_torch_legla_does_not_cast_complex_to_real(fbp):
    """No "discards the imaginary part" warning from the residual computation."""
    import warnings

    from cool_frames.torch.phase import legla as legla_t

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legla_t(fbp["s_t"], fbp["g"], fbp["a"], L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=3)

    offending = [w for w in caught if "imaginary part" in str(w.message)]
    assert not offending, f"complex->real cast is back: {[str(w.message) for w in offending]}"


def test_torch_decolbfgs_is_differentiable(fbp):
    """``decolbfgs`` accepts inputs that require grad and passes gradients back.

    It used to build its optimisation variable as
    ``x0_flat.clone().requires_grad_(True)``.  When ``s_list`` already required
    grad, ``x0_flat`` was a non-leaf node of the caller's graph and
    ``torch.optim.LBFGS`` refused it outright with "can't optimize a non-leaf
    Tensor" — so the one member of the family whose torch port exists to *be* an
    optimiser could not go inside an autograd graph at all.

    Gradients now reach ``s`` through the final magnitude projection.  They
    deliberately do *not* flow through the L-BFGS trajectory itself, which is
    neither meaningful nor affordable to differentiate.
    """
    from cool_frames.torch.phase import decolbfgs

    s_var = [si.clone().requires_grad_(True) for si in fbp["s_t"]]
    c_out = decolbfgs(s_var, fbp["g"], fbp["a"], L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=4)[0]
    sum(torch.sum(torch.abs(cm) ** 2) for cm in c_out).backward()

    for m, si in enumerate(s_var):
        assert si.grad is not None, f"channel {m} received no gradient"
        assert torch.all(torch.isfinite(si.grad)), f"channel {m} gradient has NaN/Inf"
    assert sum(float(torch.linalg.vector_norm(si.grad)) for si in s_var) > 0


def test_torch_decolbfgs_niter_matches_numpy(fbp):
    """Both backends report *iterations*, not closure evaluations.

    ``torch.optim.LBFGS`` calls its closure several times per iteration during
    the line search; counting those made ``niter`` incomparable between the two
    backends (4, 10, 50 against 1, 5, 40).
    """
    from cool_frames.numpy.phase import decolbfgs as dec_np
    from cool_frames.torch.phase import decolbfgs as dec_t

    for maxit in (1, 5, 20):
        kw = dict(L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=maxit)
        n_np = dec_np(fbp["s"], fbp["g"], fbp["a"], **kw)[3]
        n_t = int(dec_t(fbp["s_t"], fbp["g"], fbp["a"], **kw)[3])
        assert n_np == n_t == maxit, f"maxit={maxit}: numpy {n_np}, torch {n_t}"


def test_torch_decolbfgs_broadly_agrees_with_numpy(fbp):
    """The two ``decolbfgs`` implementations now land in the same place.

    They disagreed by 34 % while the NumPy gradient was wrong and its optimiser
    aborted at iteration zero.  Exact parity is not expected even now — one uses
    ``scipy``'s L-BFGS-B, the other ``torch.optim.LBFGS`` with a different line
    search, on a non-convex objective — so this asserts agreement to a few parts
    per hundred rather than to machine precision.
    """
    from cool_frames.numpy.phase import decolbfgs as dec_np
    from cool_frames.torch.phase import decolbfgs as dec_t

    kw = dict(L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=6)
    c_np = dec_np(fbp["s"], fbp["g"], fbp["a"], **kw)[0]
    c_t = dec_t(fbp["s_t"], fbp["g"], fbp["a"], **kw)[0]

    rel = _max_abs_diff(c_np, c_t) / _scale(c_np)
    assert rel < 5e-2, f"the backends disagree by {rel:.2e} (relative)"


@pytest.mark.parametrize("relthr", [0.0, 1e-3, 1e-1])
def test_legla_kernel_parity_across_relthr(relthr, fbp):
    """The shared truncated kernel keeps the backends identical at every ``relthr``.

    Both now build the kernel with the same NumPy code and differ only in how
    they apply it (sparse matmul against sparse matvec), so parity here is
    tighter than it was when torch ran its own full projection.
    """
    from cool_frames.numpy.phase import legla as legla_np
    from cool_frames.torch.phase import legla as legla_t

    kw = dict(L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=5, relthr=relthr)
    c_np = legla_np(fbp["s"], fbp["g"], fbp["a"], **kw)[0]
    c_t = legla_t(fbp["s_t"], fbp["g"], fbp["a"], **kw)[0]

    rel = _max_abs_diff(c_np, c_t) / _scale(c_np)
    assert rel < 1e-12, f"relthr={relthr}: backends disagree by {rel:.2e}"


@pytest.mark.parametrize(
    "in_dtype,want_c,want_f",
    [
        (torch.float32, torch.complex64, torch.float32),
        (torch.float64, torch.complex128, torch.float64),
    ],
)
@pytest.mark.parametrize("name", ["gla", "legla", "decolbfgs", "rtisila"])
def test_input_dtype_is_preserved(name, in_dtype, want_c, want_f, fbp):
    """The caller's dtype wins: float32 in, complex64/float32 out.

    The backend used to hard-code ``torch.complex128`` throughout, so a float32
    magnitude list came back as complex128 computed in double precision.  For
    the intended use — these ops inside a network, on a GPU, in mixed precision
    — that silently doubled memory and mismatched the rest of the model's dtype.
    """
    import cool_frames.torch.phase as TP

    s = [si.to(in_dtype) for si in fbp["s_t"]]
    c, f, _relres, _niter = getattr(TP, name)(
        s, fbp["g"], fbp["a"], L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=2
    )

    assert c[0].dtype == want_c, f"{name}: coefficients are {c[0].dtype}, want {want_c}"
    assert f.dtype == want_f, f"{name}: signal is {f.dtype}, want {want_f}"


@pytest.mark.parametrize(
    "in_dtype,want_c,want_f",
    [
        (torch.float32, torch.complex64, torch.float32),
        (torch.float64, torch.complex128, torch.float64),
    ],
)
def test_analysis_and_synthesis_preserve_dtype(in_dtype, want_c, want_f, fbp):
    """The contract holds in the filterbank core, which is where it is decided.

    Filter dicts come from the NumPy side and are always float64, so the
    *signal* has to drive the choice — otherwise every call upcasts straight
    back to double.
    """
    from cool_frames.torch.filterbanks import filterbank, ifilterbank

    x = torch.randn(fbp["Ls"], dtype=in_dtype)
    c = filterbank(x, fbp["g"], fbp["a"], L=fbp["L"])
    assert c[0].dtype == want_c

    f = ifilterbank(c, fbp["gd"], fbp["a"], Ls=fbp["Ls"], real=True)
    assert f.dtype == want_f


def test_float32_actually_computes_in_single_precision(fbp):
    """Not merely cast at the boundary.

    A float64 computation cast down at the end would agree with the float64
    result to ~1e-16; genuine single-precision arithmetic lands near float32
    epsilon.  Anything much *below* 1e-9 here would mean the cast is cosmetic
    and none of the memory or speed benefit is real.
    """
    from cool_frames.torch.phase import gla

    s32 = [si.to(torch.float32) for si in fbp["s_t"]]
    kw = dict(L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=20)
    c32 = gla(s32, fbp["g"], fbp["a"], **kw)[0]
    c64 = gla(fbp["s_t"], fbp["g"], fbp["a"], **kw)[0]

    scale = max(float(torch.max(torch.abs(v))) for v in c64)
    diff = (
        max(float(torch.max(torch.abs(u.to(torch.complex128) - v))) for u, v in zip(c32, c64))
        / scale
    )
    assert 1e-9 < diff < 1e-3, (
        f"float32 vs float64 differ by {diff:.2e}; expected single-precision error "
        f"(~1e-7), not double ({diff:.0e} suggests the float32 path is a cast)"
    )


def test_float32_halves_coefficient_memory(fbp):
    """The point of the exercise, stated as a number."""
    from cool_frames.torch.filterbanks import filterbank

    def nbytes(dtype):
        x = torch.randn(fbp["Ls"], dtype=dtype)
        c = filterbank(x, fbp["g"], fbp["a"], L=fbp["L"])
        return sum(cm.numel() * cm.element_size() for cm in c)

    assert nbytes(torch.float32) * 2 == nbytes(torch.float64)


def test_mixed_precision_inputs_use_the_wider_dtype(fbp):
    """A float32 and a float64 channel together compute in double.

    Silently discarding the precision of the wider input would be the worse
    failure mode of the two.
    """
    from cool_frames.torch.phase import gla

    mixed = [si.to(torch.float32) for si in fbp["s_t"]]
    mixed[0] = mixed[0].to(torch.float64)

    c = gla(mixed, fbp["g"], fbp["a"], L=fbp["L"], Ls=fbp["Ls"], real=True, maxit=2)[0]
    assert c[0].dtype == torch.complex128
