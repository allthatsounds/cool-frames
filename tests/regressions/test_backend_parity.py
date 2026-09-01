"""
test_backend_parity.py
======================
Tests that hold the two backends to the same answers.

The v0.1.1 audit fixed the same defect four times in four places, because the
NumPy and torch implementations of the phase family are *separate* code rather
than one wrapping the other, and nothing compared them.  Each fix went in where
the failure was noticed and nowhere else:

* ``real=False`` as a default: fixed in NumPy ``filterbankiter``, then torch
  ``filterbankiter``, then NumPy ``ifilterbankiter``, and finally torch
  ``ifilterbankiter`` — which had gone on reconstructing the flagship bank with
  23 % error through all three of the others.
* The missing centre-frequency term in the phase-gradient estimator: fixed in
  NumPy, and left in torch for a whole release.

Both survived because the torch copies have no internal callers — the torch
``filterbankconstphase`` delegates to NumPy — so the test suite stayed green
while a publicly exported, publicly documented function returned the wrong
numbers.

The lesson these tests encode is that "fix it where it broke" does not work for
a two-backend library.  What works is asserting the backends agree, so a fix
applied to one and not the other fails immediately rather than in a release or
two.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

FS = 4000
LS = 512


def _fixture():
    """An ERB bank plus the arrays the gradient estimator wants."""
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.filters import audfilters
    from cool_frames.numpy.filters._tfr import compute_tfr_from_filters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, fc, L, _info = audfilters(FS, LS)
        t = np.arange(LS) / FS
        # A pure tone, so the true instantaneous frequency is known exactly and
        # the absolute-vs-deviation question has a checkable answer.
        x = np.sin(2 * np.pi * 660.0 * t) * np.hanning(LS)
        s = [np.abs(u) for u in filterbank(x, g, a, L=L)]

    M = len(g)
    a_arr = np.atleast_1d(np.asarray(a))
    a_int = np.array(
        [int(a_arr[m]) if a_arr.ndim == 1 else int(a_arr[m, 0]) for m in range(M)],
        dtype=int,
    )
    return dict(
        M=M,
        L=L,
        N=np.array([len(u) for u in s], dtype=int),
        a_int=a_int,
        abss=np.concatenate(s),
        sqtfr=np.sqrt(np.asarray(compute_tfr_from_filters(g, L), dtype=float)),
        # The [0, 2] convention the integrator uses: 2 == fs.
        fc_norm=np.asarray(fc, dtype=float) / FS * 2.0,
        f0=660.0,
    )


@pytest.mark.requires_torch_impl
def test_phase_gradient_estimators_agree_between_backends():
    """The torch estimator kept the missing centre-frequency term for a release.

    ``comp_filterbankphasegradfrommag`` exists twice — a NumPy version and an
    independent 346-line torch version — and only the NumPy one was fixed when
    the defect was found.  On a 660 Hz tone at fs = 4000, where the true
    normalised instantaneous frequency is 0.33, torch returned **-0.0509**: the
    deviation from the channel's centre frequency rather than the absolute
    value.

    Nothing internal calls the torch copy, so no existing test could notice.
    This one compares the two directly, which is the only check that would have.
    """
    torch = pytest.importorskip("torch")
    from cool_frames.numpy.phase._fbphasegradfrommag import (
        comp_filterbankneighbors as np_neighbors,
    )
    from cool_frames.numpy.phase._fbphasegradfrommag import (
        comp_filterbankphasegradfrommag as np_grad,
    )
    from cool_frames.torch.phase import comp_filterbankphasegradfrommag as t_grad
    from cool_frames.torch.phase._fbphasegradfrommag import (
        comp_filterbankneighbors as t_neighbors,
    )

    f = _fixture()
    NEIGH, pos = np_neighbors(f["a_int"], f["M"], f["N"], do_real=True)
    tg_np, fg_np, _logs = np_grad(
        f["abss"], f["N"], f["a_int"], f["M"], f["sqtfr"], f["fc_norm"], NEIGH, pos
    )

    NEIGH_t, pos_t = t_neighbors(
        torch.as_tensor(f["a_int"]), f["M"], torch.as_tensor(f["N"]), do_real=True
    )
    tg_t, fg_t, _l = t_grad(
        torch.as_tensor(f["abss"]),
        torch.as_tensor(f["N"]),
        torch.as_tensor(f["a_int"]),
        f["M"],
        torch.as_tensor(f["sqtfr"]),
        torch.as_tensor(f["fc_norm"]),
        NEIGH_t,
        pos_t,
    )
    tg_t = np.asarray(tg_t.detach())
    fg_t = np.asarray(fg_t.detach())

    assert np.allclose(tg_np, tg_t, rtol=1e-7, atol=1e-9), (
        f"tgrad differs between backends: max |diff| = {np.max(np.abs(tg_np - tg_t)):.3e}"
    )
    assert np.allclose(fg_np, fg_t, rtol=1e-7, atol=1e-9), (
        f"fgrad differs between backends: max |diff| = {np.max(np.abs(fg_np - fg_t)):.3e}"
    )

    # ...and both are the *absolute* instantaneous frequency, not a deviation.
    # This is the half that a pure parity check would miss: two backends can
    # agree and both be wrong.
    expected = f["f0"] / FS * 2.0
    loudest = int(np.argmax(f["abss"]))
    for name, tg in (("numpy", tg_np), ("torch", tg_t)):
        assert abs(tg[loudest] - expected) < 0.15, (
            f"{name} tgrad at the loudest cell is {tg[loudest]:.4f}, expected about "
            f"{expected:.4f}. A value near zero means the centre-frequency term "
            f"is missing again."
        )


@pytest.mark.requires_torch_impl
def test_phase_gradient_estimator_is_actually_differentiable():
    """The torch copy exists *only* to be differentiable, so pin that.

    Its docstring says "fully differentiable", and that claim is the entire
    reason for maintaining a second implementation instead of wrapping NumPy
    like the rest of the torch phase module does.  An untested claim of this
    kind is how the module came to be 4 % covered while carrying a wrong answer.
    """
    torch = pytest.importorskip("torch")
    from cool_frames.torch.phase import comp_filterbankphasegradfrommag as t_grad
    from cool_frames.torch.phase._fbphasegradfrommag import (
        comp_filterbankneighbors as t_neighbors,
    )

    f = _fixture()
    NEIGH_t, pos_t = t_neighbors(
        torch.as_tensor(f["a_int"]), f["M"], torch.as_tensor(f["N"]), do_real=True
    )
    abss = torch.as_tensor(f["abss"], dtype=torch.float64).clone().requires_grad_(True)

    tgrad, fgrad, _logs = t_grad(
        abss,
        torch.as_tensor(f["N"]),
        torch.as_tensor(f["a_int"]),
        f["M"],
        torch.as_tensor(f["sqtfr"]),
        torch.as_tensor(f["fc_norm"]),
        NEIGH_t,
        pos_t,
    )

    assert tgrad.requires_grad, "tgrad is detached from the graph"
    assert fgrad.requires_grad, "fgrad is detached from the graph"

    (tgrad.pow(2).sum() + fgrad.pow(2).sum()).backward()

    assert abss.grad is not None, "no gradient reached the magnitudes"
    assert torch.all(torch.isfinite(abss.grad)), "the gradient contains inf or nan"
    assert float(torch.linalg.norm(abss.grad)) > 0, "the gradient is identically zero"


@pytest.mark.requires_torch_impl
def test_iterative_routines_agree_between_backends():
    """The ``real=False`` default defect lived in four places, one per backend.

    Each was found separately and three were fixed before the fourth was even
    noticed.  Comparing the pairs is what makes the next such divergence fail on
    the first run rather than the fourth.
    """
    torch = pytest.importorskip("torch")
    from cool_frames.numpy.filterbanks import filterbank as np_fb
    from cool_frames.numpy.filterbanks import filterbankiter as np_iter
    from cool_frames.numpy.filterbanks import ifilterbankiter as np_iiter
    from cool_frames.numpy.filters import audfilters
    from cool_frames.torch.filterbanks import filterbank as t_fb
    from cool_frames.torch.filterbanks import filterbankiter as t_iter
    from cool_frames.torch.filterbanks import ifilterbankiter as t_iiter

    x = np.random.default_rng(0).standard_normal(LS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, _fc, L, _info = audfilters(FS, LS)

        # Analysis: same iteration count and residual from the derived default.
        _cn, rn, itn = np_iter(x, g, a)
        _ct, rt, itt = t_iter(torch.as_tensor(x, dtype=torch.float64), g, a)

        # Synthesis: same reconstruction.
        cn = np_fb(x, g, a, L=L)
        xr_np = np.real(np.asarray(np_iiter(cn, g, a, Ls=LS)[0]))[:LS]
        ct = t_fb(torch.as_tensor(x, dtype=torch.float64), g, a, L=L)
        xr_t = np.real(np.asarray(t_iiter(ct, g, a, Ls=LS)[0].detach()))[:LS]

    assert itn == itt, f"filterbankiter iteration counts differ: numpy {itn}, torch {itt}"
    assert float(np.atleast_1d(rn)[-1]) == pytest.approx(float(rt), rel=1e-6)

    for name, xr in (("numpy", xr_np), ("torch", xr_t)):
        err = np.linalg.norm(xr - x) / np.linalg.norm(x)
        assert err < 1e-10, f"{name} ifilterbankiter round-trip error {err:.4g}"
    assert np.allclose(xr_np, xr_t, rtol=1e-8, atol=1e-10), (
        "the two backends' iterative synthesis disagree"
    )


# ---------------------------------------------------------------------------
# Reassignment: exported, and until now essentially untested
# ---------------------------------------------------------------------------


def _chirp_bank():
    from cool_frames.numpy.filters import audfilters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, fc, L, _info = audfilters(FS, LS)
    t = np.arange(LS) / FS
    T = LS / FS
    # A chirp: its energy lies along a sloping ridge, which is the case
    # reassignment exists to sharpen and the case where a wrong sign or a
    # transposed axis would be obvious.
    x = np.sin(2 * np.pi * (200 * t + 300 * t**2 / T)) * np.hanning(LS)
    return g, a, fc, L, x


def _flat(result):
    first = result[0] if isinstance(result, tuple) else result
    return np.concatenate(
        [np.asarray(u.detach() if hasattr(u, "detach") else u).ravel() for u in first]
    )


@pytest.mark.requires_impl
def test_reassignment_conserves_energy_and_concentrates_it():
    """``filterbankreassign`` was 71 % covered in NumPy and 5 % in torch.

    Three exported functions with no meaningful test between them.  These are
    the two properties that define the operation, so a version that moved
    energy to the wrong place would fail one or the other:

    * **Energy is conserved.** Reassignment relocates energy; it does not create
      or destroy it. The totals must match to floating-point.
    * **Energy concentrates.** That is the entire point — the reassigned
      distribution needs fewer cells to hold the same fraction of the total.
      Measured at 21 cells for 90 % of the energy against 26 for the raw
      spectrogram.
    """
    from cool_frames.numpy.filterbanks import filterbank
    from cool_frames.numpy.phase import filterbankreassign

    g, a, fc, L, x = _chirp_bank()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = np.concatenate([np.abs(u).ravel() ** 2 for u in filterbank(x, g, a, L=L)])
        reassigned = _flat(filterbankreassign(x, g, a, L, fc))

    assert reassigned.sum() == pytest.approx(raw.sum(), rel=1e-9), (
        "reassignment changed the total energy; it may only move it"
    )
    assert np.all(reassigned >= 0), "reassigned energy must be non-negative"

    def _cells_for(v, frac=0.9):
        v = np.sort(v)[::-1]
        c = np.cumsum(v)
        return int(np.searchsorted(c, frac * c[-1])) + 1

    assert _cells_for(reassigned) < _cells_for(raw), (
        f"reassignment did not concentrate the energy: "
        f"{_cells_for(reassigned)} cells against {_cells_for(raw)} raw"
    )


@pytest.mark.requires_impl
def test_synchrosqueezing_differs_from_full_reassignment():
    """The two are not the same operation, and must not return the same thing.

    Reassignment moves energy in both time and frequency; synchrosqueezing moves
    it in frequency only, preserving the time spread so the transform stays
    invertible.  A synchrosqueeze that simply delegated to reassign would look
    entirely plausible — same shapes, same energy — which is why this asserts the
    *difference* rather than just that each runs.
    """
    from cool_frames.numpy.phase import filterbankreassign, filterbanksynchrosqueeze

    g, a, fc, L, x = _chirp_bank()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reassigned = _flat(filterbankreassign(x, g, a, L, fc))
        squeezed = _flat(filterbanksynchrosqueeze(x, g, a, L, fc))

    assert squeezed.sum() == pytest.approx(reassigned.sum(), rel=1e-9), (
        "both operations conserve energy, so their totals should match"
    )
    assert not np.allclose(reassigned, squeezed), (
        "synchrosqueezing returned the same distribution as full reassignment"
    )
    # Frequency-only reassignment leaves the time spread intact, so it occupies
    # strictly more cells than the two-dimensional version.
    assert np.count_nonzero(squeezed) > np.count_nonzero(reassigned), (
        f"synchrosqueezing occupies {np.count_nonzero(squeezed)} cells against "
        f"reassignment's {np.count_nonzero(reassigned)}; frequency-only "
        f"reassignment should be the less concentrated of the two"
    )


@pytest.mark.requires_torch_impl
def test_reassignment_agrees_between_backends():
    """``torch/phase/_reassign.py`` is 180 statements at 5 % coverage.

    An independent torch implementation of three exported functions, with no
    test comparing it to the NumPy original — the same shape as the two defects
    at the top of this file, and the reason to check it before assuming it is
    fine.  It is: the two agree to 2.7e-12.  Pinned so it stays that way.
    """
    torch = pytest.importorskip("torch")
    from cool_frames.numpy.phase import filterbankreassign as np_ra
    from cool_frames.numpy.phase import filterbanksynchrosqueeze as np_ss
    from cool_frames.torch.phase import filterbankreassign as t_ra
    from cool_frames.torch.phase import filterbanksynchrosqueeze as t_ss

    g, a, fc, L, x = _chirp_bank()
    xt = torch.as_tensor(x, dtype=torch.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, np_fn, t_fn in (
            ("filterbankreassign", np_ra, t_ra),
            ("filterbanksynchrosqueeze", np_ss, t_ss),
        ):
            dn = _flat(np_fn(x, g, a, L, fc))
            dt = _flat(t_fn(xt, g, a, L, fc))
            assert dn.shape == dt.shape, f"{name}: shapes differ, {dn.shape} vs {dt.shape}"
            assert np.allclose(dn, dt, rtol=1e-7, atol=1e-9), (
                f"{name} differs between backends: max |diff| = {np.max(np.abs(dn - dt)):.3e}"
            )
