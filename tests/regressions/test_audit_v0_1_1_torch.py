"""
test_audit_v0_1_1_torch.py
==========================
Regression tests for the PyTorch half of the v0.1.1 subsystem audit.

Separate from ``test_audit_v0_1_1.py`` because this module skips wholesale when
torch is absent, and the NumPy job in CI runs without it.  See that file's
docstring for what the audit was and why the assertions take the shape they do.

The torch backend exists to be (a) numerically identical to NumPy in float64
and (b) differentiable.  Every defect below broke one of those two promises
without raising.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

torch = pytest.importorskip("torch")

FS = 4000
LS = 512


def _np_fir_bank():
    from cool_frames.numpy.filters.lowlevel import blfilter, firfilter

    return {
        "pure_fir": [firfilter("hann", 9, fc=f) for f in (0.1, 0.3, 0.6)],
        "mixed": [
            firfilter("hann", 9, fc=0.1),
            blfilter("hann", 0.2, fc=0.3),
            firfilter("hann", 9, fc=0.6),
            blfilter("hann", 0.2, fc=0.8),
        ],
    }


@pytest.mark.requires_torch_impl
@pytest.mark.parametrize("case", ["pure_fir", "mixed"])
def test_torch_fir_analysis_matches_numpy(case):
    """Coefficient count and values, against the NumPy reference.

    ``comp_filterbank_td`` truncated the convolution to ``L`` *before* the
    offset slice, so any filter with a non-zero offset returned too few
    coefficients — 59 where NumPy gives 64 at ``a=1``.
    """
    import cool_frames.torch.filterbanks as tfb
    from cool_frames.numpy.filterbanks import filterbank as nfb

    g = _np_fir_bank()[case]
    L, a = 64, 2
    x = np.random.default_rng(8).standard_normal(L)

    c_np = nfb(x, g, a, L=L)
    c_t = tfb.filterbank(torch.from_numpy(x), g, a, L)

    assert [len(u) for u in c_np] == [len(v) for v in c_t], (
        f"coefficient counts differ: numpy {[len(u) for u in c_np]}, torch {[len(v) for v in c_t]}"
    )
    rel = max(
        float(np.max(np.abs(np.asarray(u).ravel() - v.detach().numpy().ravel())))
        for u, v in zip(c_np, c_t)
    ) / max(float(np.max(np.abs(np.asarray(u)))) for u in c_np)
    assert rel < 1e-12, f"{case}: analysis differs from NumPy by {rel:.3e}"


@pytest.mark.requires_torch_impl
@pytest.mark.parametrize("case", ["pure_fir", "mixed"])
def test_torch_fir_synthesis_matches_numpy(case):
    """``comp_ifilterbank_td`` used zero padding where NumPy extends periodically.

    A different boundary condition put the two backends 31.5 % apart on
    identical input.
    """
    import cool_frames.torch.filterbanks as tfb
    from cool_frames.numpy.filterbanks import filterbank as nfb
    from cool_frames.numpy.filterbanks import ifilterbank as nifb

    g = _np_fir_bank()[case]
    L, a = 64, 2
    x = np.random.default_rng(9).standard_normal(L)
    c = [np.asarray(u).ravel().astype(complex) for u in nfb(x, g, a, L=L)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o_np = np.asarray(nifb(c, g, a, Ls=L, real=False)).ravel()
        o_t = (
            tfb.ifilterbank([torch.from_numpy(u) for u in c], g, a, Ls=L, real=False)
            .detach()
            .numpy()
            .ravel()
        )

    n = min(o_np.size, o_t.size)
    rel = np.max(np.abs(o_np[:n] - o_t[:n])) / max(np.max(np.abs(o_np)), 1e-30)
    assert rel < 1e-12, f"{case}: synthesis differs from NumPy by {rel:.3e}"


@pytest.mark.requires_torch_impl
def test_torch_ifilterbank_uses_every_channel_of_a_mixed_bank():
    """It used to *return early* whenever any FIR channel was present.

    Every band-limited and full-length channel was discarded, and the real-mode
    fold skipped — 125 % error against NumPy.  Zeroing the band-limited
    coefficients left the output unchanged, which is the sharpest symptom.
    """
    import cool_frames.torch.filterbanks as tfb
    from cool_frames.numpy.filterbanks import filterbank as nfb

    g = _np_fir_bank()["mixed"]
    L, a = 64, 2
    x = np.random.default_rng(10).standard_normal(L)
    c = [torch.from_numpy(np.asarray(u).ravel().astype(complex)) for u in nfb(x, g, a, L=L)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = tfb.ifilterbank(c, g, a, Ls=L, real=False)
        muted = [z.clone() for z in c]
        muted[1] *= 0
        muted[3] *= 0
        without_bl = tfb.ifilterbank(muted, g, a, Ls=L, real=False)

    assert not torch.allclose(full, without_bl), (
        "zeroing the band-limited channels did not change the output — they are being ignored"
    )


@pytest.mark.requires_torch_impl
def test_torch_fir_filters_accept_a_real_signal():
    """``conv1d`` rejects mixed real/complex operands.

    The filters are held in the complex working dtype, so every real-signal
    call raised "expected scalar type Double but found ComplexDouble" — i.e.
    FIR filters were unusable in the torch backend.
    """
    import cool_frames.torch.filterbanks as tfb

    g = _np_fir_bank()["pure_fir"]
    for dtype in (torch.float32, torch.float64):
        c = tfb.filterbank(torch.randn(64, dtype=dtype), g, 2, 64)
        assert len(c) == len(g)
        assert all(torch.all(torch.isfinite(torch.abs(cm))) for cm in c)


@pytest.mark.requires_torch_impl
def test_torch_framemul_matches_numpy_in_float64():
    """Every entry point in ``torch.operators`` forced float32.

    A float64 call therefore matched NumPy only to ~1.8e-07 and
    ``torch.autograd.gradcheck`` failed outright.
    """
    import cool_frames.numpy.operators as NO
    import cool_frames.torch.operators as TO
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import audfilters

    g, a, _fc, L, _info = audfilters(FS, LS)
    rng = np.random.default_rng(11)
    f = rng.standard_normal(L)
    sigma = [rng.uniform(0.5, 2.0, size=cm.shape) for cm in filterbank(f, g, a, L=L)]

    m_np = np.asarray(NO.framemul(f, g, g, a, sigma, L)).ravel()
    m_t = TO.framemul(torch.from_numpy(f), g, g, a, [torch.from_numpy(s) for s in sigma], L)

    assert m_t.dtype == torch.float64, f"forced dtype {m_t.dtype}"
    rel = np.max(np.abs(m_np - m_t.detach().numpy().ravel())) / np.max(np.abs(m_np))
    assert rel < 1e-12, f"framemul differs from NumPy by {rel:.3e}"


@pytest.mark.requires_torch_impl
def test_torch_framemul_gradients_flow():
    import cool_frames.torch.operators as TO
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import audfilters

    g, a, _fc, L, _info = audfilters(FS, LS)
    rng = np.random.default_rng(12)
    f = rng.standard_normal(L)
    sigma = [
        torch.from_numpy(rng.uniform(0.5, 2.0, size=cm.shape)) for cm in filterbank(f, g, a, L=L)
    ]

    x = torch.from_numpy(f).clone().requires_grad_(True)
    TO.framemul(x, g, g, a, sigma, L).sum().backward()

    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    assert float(torch.linalg.vector_norm(x.grad)) > 0


@pytest.mark.requires_torch_impl
def test_torch_framemulinv_reports_real_iteration_counts():
    """It always reported ``'iter': 1`` — the loop bound ``_k``, the return read ``k``."""
    import cool_frames.torch.operators as TO
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import audfilters

    g, a, _fc, L, _info = audfilters(FS, LS)
    rng = np.random.default_rng(13)
    f = rng.standard_normal(L)
    sigma = [torch.from_numpy(np.ones_like(np.abs(cm))) for cm in filterbank(f, g, a, L=L)]

    _x, info = TO.framemulinv(torch.from_numpy(f), g, g, a, sigma, L, maxit=40, tol=1e-14)
    assert info["iter"] > 1, f"reported {info['iter']} iterations for a 40-iteration solve"


@pytest.mark.requires_torch_impl
def test_torch_denoise_round_trips_without_thresholding():
    """``denoise`` synthesised with the tight frame after analysing with ``g``.

    That is not a round trip: with thresholding effectively disabled the
    residual was 28.7 *relative*, where the canonical dual gives ~1e-16.
    """
    torch_additions = pytest.importorskip("torch_additions.recipes.audio_denoising")

    t = np.arange(2048) / FS
    x = torch.from_numpy(np.sin(2 * np.pi * 440 * t))

    out = torch_additions.denoise(x, FS, threshold_db=-300)
    y = out[0] if isinstance(out, tuple) else out
    y = torch.as_tensor(y).detach().real

    n = min(len(x), len(y))
    rel = float(torch.linalg.vector_norm(y[:n] - x[:n])) / float(torch.linalg.vector_norm(x))
    assert rel < 1e-6, f"no-op denoise changed the signal by {rel:.3e}"


# ---------------------------------------------------------------------------
# Second tranche
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch_impl
def test_torch_framemulappr_matches_numpy():
    """It was a different, wrong algorithm.

    It built its synthesis matrix by *analysing* with ``g_synthesis`` instead
    of measuring the synthesis atoms, only ever did the diagonal
    approximation, and never read ``real``.  On an operator constructed as an
    exact multiplier, NumPy returned 1.05e-15 Hilbert-Schmidt error and this
    returned 1.033 — worse than a zero symbol.
    """
    import cool_frames.numpy.operators as NO
    import cool_frames.torch.operators as TO
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import audfilters

    g, a, _fc, L, _info = audfilters(FS, 64)
    rng = np.random.default_rng(30)
    c = filterbank(rng.standard_normal(L), g, a, L=L)
    sigma = [rng.standard_normal(cm.shape) for cm in c]

    T = np.zeros((L, L))
    for k in range(L):
        e = np.zeros(L)
        e[k] = 1.0
        T[:, k] = NO.framemul(e, g, g, a, sigma, L)

    s_np = NO.framemulappr(T, g, g, a, L, method="full")
    s_t = TO.framemulappr(torch.from_numpy(T), g, g, a, L, method="full")

    diff = max(
        float(np.max(np.abs(np.asarray(u) - v.detach().cpu().numpy()))) for u, v in zip(s_np, s_t)
    )
    assert diff < 1e-12, f"torch symbol differs from NumPy by {diff:.3e}"


@pytest.mark.requires_torch_impl
def test_torch_framemulappr_real_flag_is_live():
    """``real`` was never read: ``real=False`` output was bitwise identical."""
    import cool_frames.numpy.operators as NO
    import cool_frames.torch.operators as TO
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import audfilters

    g, a, _fc, L, _info = audfilters(FS, 64)
    rng = np.random.default_rng(31)
    c = filterbank(rng.standard_normal(L), g, a, L=L)
    sigma = [rng.standard_normal(cm.shape) for cm in c]

    T = np.zeros((L, L))
    for k in range(L):
        e = np.zeros(L)
        e[k] = 1.0
        T[:, k] = NO.framemul(e, g, g, a, sigma, L)
    T_t = torch.from_numpy(T)

    s_real = TO.framemulappr(T_t, g, g, a, L, real=True, method="full")
    s_cplx = TO.framemulappr(T_t, g, g, a, L, real=False, method="full")

    diff = max(float(torch.max(torch.abs(u - v))) for u, v in zip(s_real, s_cplx))
    assert diff > 1e-9, "real=True and real=False still give identical symbols"


@pytest.mark.requires_torch_impl
@pytest.mark.parametrize("Ls", [256, 432])
def test_torch_filterbankiter_matches_numpy(Ls):
    """It sliced the CG iterate to ``Ls``, making the map a projection.

    With ``Ls < L`` it diverged — relres 399.7 after 60 iterations against
    NumPy's 8.5e-11 in 17.  With ``Ls == L`` the slice was a no-op, which is
    why it went unnoticed.
    """
    import cool_frames.numpy.filterbanks as NF
    import cool_frames.torch.filterbanks as TF
    from cool_frames.numpy.filters import audfilters

    g, a, _fc, L, _info = audfilters(FS, Ls)
    x = np.random.default_rng(32).standard_normal(Ls)

    kw = dict(alg="cg", real=True, maxit=60, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c_np, _r_np, _n = NF.filterbankiter(x, g, a, L, **kw)
        c_t, r_t, _n = TF.filterbankiter(torch.from_numpy(x), g, a, L, **kw)

    assert r_t < 1e-6, f"torch did not converge (relres {r_t:.3e})"
    rel = max(
        float(np.max(np.abs(np.asarray(u).ravel() - v.detach().cpu().numpy().ravel())))
        for u, v in zip(c_np, c_t)
    ) / max(float(np.max(np.abs(np.asarray(u)))) for u in c_np)
    assert rel < 1e-10, f"Ls={Ls}: backends disagree by {rel:.3e}"


@pytest.mark.requires_torch_impl
def test_torch_full_length_filters_with_fractional_hops():
    """torch tested only ``len(H) == L``; NumPy also requires an integer hop.

    A fractional-hop full-length channel went down the uniform kernel and
    raised "shape [...] is invalid for input of size ...".
    """
    import cool_frames.numpy.filterbanks as NF
    import cool_frames.torch.filterbanks as TF

    L = 24
    g = [
        {"H": np.ones(L, dtype=complex), "foff": 0, "realonly": 0, "delay": 0, "fs": None}
        for _ in range(3)
    ]
    a = np.array([[4, 2]] * 3)
    x = np.random.default_rng(33).standard_normal(L)

    c_np = NF.filterbank(x, g, a, L=L)
    c_t = TF.filterbank(torch.from_numpy(x), g, a, L)

    assert [len(u) for u in c_np] == [len(v) for v in c_t]
    rel = max(
        float(np.max(np.abs(np.asarray(u).ravel() - v.detach().cpu().numpy().ravel())))
        for u, v in zip(c_np, c_t)
    ) / max(float(np.max(np.abs(np.asarray(u)))) for u in c_np)
    assert rel < 1e-12


@pytest.mark.requires_torch_impl
def test_reconstruct_methods_are_distinct_and_real():
    """``'pghi'`` and ``'spsi'`` were both random phase, bitwise identical.

    Both reported ``converged: True`` unconditionally, and the GLA calls took
    the library default ``real=False`` on a single-sided bank — a convention
    mismatch costing ~30 dB.  ``'pghi'`` is now the real algorithm (the missing
    centre-frequency term in its gradient estimator is fixed — see
    DEFECT_REGISTER.md J1), so it is checked here alongside the rest: distinct
    from ``'spsi'``, and better than zero phase.
    """
    recipes = pytest.importorskip("torch_additions.recipes.magnitude_to_audio")

    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import audfilters
    from cool_frames.numpy.phase import magnitudeerr

    g, a, _fc, L, _info = audfilters(FS, LS)
    t = np.arange(LS) / FS
    x = (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 1320 * t)) * np.hanning(LS)
    s = [np.abs(u) for u in filterbank(x, g, a, L=L)]
    s_t = [torch.from_numpy(np.ascontiguousarray(u)) for u in s]

    def _mag_err(y):
        y = np.real(np.asarray(torch.as_tensor(y).detach()))[:LS]
        return magnitudeerr(s, [np.abs(u) for u in filterbank(y, g, a, L=L)])

    errs = {}
    for method in ("pghi", "gla", "fgla", "legla", "spsi"):
        y, info = recipes.reconstruct(s_t, g, a, L, LS, method=method)
        errs[method] = _mag_err(y)
        assert info["method"] == method

    # Every method must beat the zero-phase baseline...
    baseline = magnitudeerr(
        s,
        [
            np.abs(u)
            for u in filterbank(
                np.real(
                    __import__(
                        "cool_frames.numpy.filterbanks", fromlist=["ifilterbank"]
                    ).ifilterbank(
                        [u.astype(complex) for u in s],
                        __import__(
                            "cool_frames.numpy.filterbanks", fromlist=["filterbankdual"]
                        ).filterbankdual(g, a, L),
                        a,
                        Ls=LS,
                        real=True,
                    )
                ),
                g,
                a,
                L=L,
            )
        ],
    )
    for method, err in errs.items():
        assert err < baseline, (
            f"{method} ({err:.4f}) is no better than zero phase ({baseline:.4f})"
        )

    # ...and 'spsi' must not be a clone of the iterative ones.
    assert abs(errs["spsi"] - errs["gla"]) > 1e-6, "'spsi' looks like a copy of 'gla'"
