"""Dense differentiable frame algebra — checked against cool-frames' own references.

Every property here is asserted against something already in the package (the
``filterbank``/``ifilterbank`` kernels, or the O(L^2) ``filterbankbounds_svd``
ground truth), not against a hand-computed expectation. The point of the module
under test is that it is the *same* algebra, differentiably; if it disagrees
with the reference path, it is wrong.
"""

from __future__ import annotations

import pytest

import numpy as np

torch = pytest.importorskip("torch")

from cool_frames.numpy.filterbanks import filterbankbounds_svd  # noqa: E402
from cool_frames.torch.filterbanks import filterbank, ifilterbank  # noqa: E402
from cool_frames.torch.filterbanks._dense import (  # noqa: E402
    dense_analyse,
    dense_dual,
    dense_synthesise,
    frame_bounds,
    frame_condition,
    frame_gram,
    frame_response,
    frame_retract,
    mirror_bank,
    retraction_gain,
)

DT = torch.complex128


def _random_bank(M: int, L: int, seed: int = 0, single_sided: bool = False) -> torch.Tensor:
    """A random but *covering* bank, as a dense ``(M, L)`` response tensor.

    Overlapping raised-cosine bumps with centres spanning ``[0, L/2]``
    inclusive, so every frequency — DC and Nyquist included — is carried by at
    least two channels. Coverage is deliberate: an uncovered frequency makes
    ``A = 0`` no matter what else is true, which is a property of the *design*,
    not of the code under test, and a test bank that failed to cover would be
    testing the wrong thing.
    """
    g = torch.Generator().manual_seed(seed)
    half = L // 2
    H = torch.zeros(M, L, dtype=DT)
    step = half / (M - 1)
    for m in range(M):
        centre = m * step
        k = torch.arange(0, half + 1, dtype=torch.float64)
        u = (k - centre) / (2.0 * step)
        bump = torch.where(u.abs() < 1.0, torch.cos(0.5 * torch.pi * u) ** 2, torch.zeros_like(u))
        phase = torch.rand(half + 1, generator=g, dtype=torch.float64) * 0.2
        H[m, : half + 1] = (bump * torch.exp(1j * phase)).to(DT)
        if not single_sided:
            H[m, half + 1 :] = torch.flip(H[m, 1:half].conj(), dims=(0,))
    return H


def _dicts(H: torch.Tensor) -> list[dict]:
    """Full-length response dicts, i.e. the ``m_fft`` path of ``filterbank``."""
    return [{"H": H[m].clone(), "foff": 0, "realonly": 0} for m in range(H.shape[0])]


# ---------------------------------------------------------------------------
# the dense kernels ARE the package's kernels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [1, 2, 4, 8])
def test_dense_analyse_matches_filterbank(a):
    L, M = 64, 6
    H = _random_bank(M, L, seed=1)
    x = torch.randn(L, dtype=torch.float64)

    ref = filterbank(x.to(DT), _dicts(H), a, L=L, stack=True)  # (K, M)
    got = dense_analyse(x, H, a)  # (M, K)

    assert torch.allclose(got, ref.T, atol=1e-12), (got - ref.T).abs().max()


@pytest.mark.parametrize("a", [1, 2, 4])
def test_dense_synthesise_is_the_adjoint(a):
    """<A x, c> == <x, A* c> to machine precision."""
    L, M = 64, 5
    H = _random_bank(M, L, seed=2)
    x = torch.randn(L, dtype=torch.float64).to(DT)
    c = torch.randn(M, L // a, dtype=torch.float64).to(DT)

    lhs = torch.vdot(dense_analyse(x, H, a).reshape(-1), c.reshape(-1))
    rhs = torch.vdot(x, dense_synthesise(c, H, a))
    assert abs(lhs - rhs) < 1e-12 * max(1.0, abs(lhs))


# ---------------------------------------------------------------------------
# bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [1, 2, 4])
def test_bounds_match_the_svd_ground_truth(a):
    """Against the O(L^2) operator eigenvalues, which assume nothing at all."""
    L, M = 32, 7
    H = _random_bank(M, L, seed=3)
    A, B = frame_bounds(H, a)

    g_np = [{"H": H[m].numpy(), "foff": 0, "realonly": 0} for m in range(M)]
    A_ref, B_ref = filterbankbounds_svd(g_np, a, L, real=False)

    assert A.item() == pytest.approx(A_ref, rel=1e-9, abs=1e-12)
    assert B.item() == pytest.approx(B_ref, rel=1e-9, abs=1e-12)


def test_response_is_the_gram_diagonal():
    """The painless response is the diagonal of G, folded the same way."""
    L, M, a = 64, 6, 4
    H = _random_bank(M, L, seed=4)
    G = frame_gram(H, a)  # (K, a, a)
    diag = torch.diagonal(G, dim1=-2, dim2=-1).real  # (K, a)
    resp = frame_response(H, a).reshape(a, L // a).T  # (K, a)
    assert torch.allclose(diag, resp, atol=1e-12)


def test_flat_response_does_not_imply_a_frame():
    """The defect the response cannot see: flat to machine precision, A = 0.

    Two channels whose responses coincide give a rank-1 Gram block: the
    diagonal is perfectly flat and the lower bound is nevertheless zero. A
    flatness penalty on the response is therefore not a frame constraint, which
    is why :func:`frame_bounds` exists.
    """
    L, a = 16, 2
    H = torch.zeros(2, L, dtype=DT)
    H[0] = 1.0
    H[1] = 1.0
    resp = frame_response(H, a)
    A, _ = frame_bounds(H, a)
    assert resp.std().item() < 1e-15  # perfectly flat
    assert A.item() < 1e-12  # and not a frame


# ---------------------------------------------------------------------------
# retraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [1, 2, 4, 8])
@pytest.mark.parametrize("real", [False, True])
def test_retraction_gives_a_parseval_frame_exactly(a, real):
    L, M = 64, 12
    H = _random_bank(M, L, seed=5, single_sided=real)
    Ht = frame_retract(H, a, kappa_target=1.0, real=real)
    A, B = frame_bounds(Ht, a, real=real)
    assert A.item() == pytest.approx(1.0, abs=1e-10)
    assert B.item() == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("kappa", [1.5, 4.0, 20.0])
def test_retraction_hits_its_condition_target(kappa):
    L, M, a = 64, 12, 4
    H = _random_bank(M, L, seed=6)
    Ht = frame_retract(H, a, kappa_target=kappa)
    A, B = frame_bounds(Ht, a)
    assert B.item() == pytest.approx(1.0, abs=1e-10)
    assert A.item() >= 1.0 / kappa - 1e-10
    assert frame_condition(Ht, a).item() <= kappa + 1e-8


def test_retraction_cannot_create_coverage():
    """The lemma: retraction normalises conditioning, it does not create a frame.

    A bank with a hole in its coverage has ``A = 0``; rescaling the energy that
    is present cannot put energy where there is none, so the retraction refuses
    rather than returning a bank whose bounds are not what was asked for.
    """
    L, M, a = 64, 6, 1
    H = _random_bank(M, L, seed=30)
    H[:, 10:20] = 0.0  # punch a hole in the covered band
    A, _ = frame_bounds(H, a)
    assert A.item() < 1e-12

    with pytest.raises(ValueError, match="not a frame"):
        frame_retract(H, a, strict=True)

    A2, _ = frame_bounds(frame_retract(H, a, strict=False), a)
    assert A2.item() < 1e-8  # and permissively, it is still not a frame


def test_retraction_gain_reports_the_unretracted_conditioning():
    L, M, a = 64, 10, 4
    H = _random_bank(M, L, seed=31)
    gain = retraction_gain(H, a)
    assert gain.item() == pytest.approx(frame_condition(H, a).item())
    assert frame_condition(frame_retract(H, a), a).item() == pytest.approx(1.0, abs=1e-8)
    assert gain.item() > 1.0  # the constant post-retraction kappa hides this


def test_retraction_is_idempotent():
    L, M, a = 64, 10, 4
    H = _random_bank(M, L, seed=7)
    H1 = frame_retract(H, a)
    H2 = frame_retract(H1, a)
    assert torch.allclose(H1, H2, atol=1e-10)


def test_retraction_preserves_single_sidedness():
    """The claim the real=True path rests on: retraction commutes with mirroring.

    If it did not, the retracted 'single-sided' bank would have negative-frequency
    energy and the mirrored set used for the bounds would no longer be the set
    the synthesis actually uses.
    """
    L, M, a = 64, 8, 4
    H = _random_bank(M, L, seed=8, single_sided=True)
    assert H[:, L // 2 + 1 :].abs().max() < 1e-15
    Ht = frame_retract(H, a, real=True)
    assert Ht[:, L // 2 + 1 :].abs().max() < 1e-10

    # and the mirror of the retracted bank is the retraction of the mirrored one
    full = frame_retract(mirror_bank(H), a, real=False)
    assert torch.allclose(mirror_bank(Ht), full, atol=1e-9)


# ---------------------------------------------------------------------------
# inversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a", [1, 2, 4])
def test_dual_inverts_analysis_exactly(a):
    L, M = 64, 10
    H = _random_bank(M, L, seed=9)
    x = torch.randn(L, dtype=torch.float64).to(DT)
    c = dense_analyse(x, H, a)
    xr = dense_synthesise(c, dense_dual(H, a), a)
    assert torch.allclose(xr, x, atol=1e-9), (xr - x).abs().max()


def test_parseval_bank_is_its_own_dual():
    L, M, a = 64, 12, 4
    H = frame_retract(_random_bank(M, L, seed=10), a)
    x = torch.randn(L, dtype=torch.float64).to(DT)
    c = dense_analyse(x, H, a)
    assert torch.allclose(dense_synthesise(c, H, a), x, atol=1e-10)
    assert torch.allclose(dense_dual(H, a), H, atol=1e-8)


def test_real_roundtrip_through_the_mirrored_set():
    """A real signal, a single-sided Parseval bank, exact reconstruction."""
    L, M, a = 128, 24, 4
    H = frame_retract(_random_bank(M, L, seed=11, single_sided=True), a, real=True)
    x = torch.randn(L, dtype=torch.float64)
    c = dense_analyse(x, H, a)
    xr = dense_synthesise(c, H, a, real=True)
    assert torch.allclose(xr, x, atol=1e-9), (xr - x).abs().max()


def test_parseval_preserves_energy_on_real_signals():
    L, M, a = 128, 24, 4
    H = frame_retract(_random_bank(M, L, seed=12, single_sided=True), a, real=True)
    x = torch.randn(L, dtype=torch.float64)
    c = dense_analyse(x, H, a)
    # the mirrored partner carries the conjugate, hence the factor 2
    assert (2 * c.abs().pow(2).sum()).item() == pytest.approx(x.pow(2).sum().item(), rel=1e-9)


def test_reference_ifilterbank_agrees_on_the_dual_path():
    """Cross-check the whole inversion against the package's own synthesis."""
    L, M, a = 64, 8, 4
    H = _random_bank(M, L, seed=13)
    x = torch.randn(L, dtype=torch.float64).to(DT)
    c = dense_analyse(x, H, a)
    Hd = dense_dual(H, a)
    ref = ifilterbank([c[m] for m in range(M)], _dicts(Hd), a, Ls=L, real=False)
    got = dense_synthesise(c, Hd, a)
    assert torch.allclose(got, ref, atol=1e-10)


# ---------------------------------------------------------------------------
# the reason the module exists: gradients
# ---------------------------------------------------------------------------


def test_bounds_carry_a_gradient():
    L, M, a = 64, 10, 4
    H = _random_bank(M, L, seed=14).requires_grad_(True)
    A, B = frame_bounds(H, a)
    (B - A).backward()
    assert H.grad is not None
    assert torch.isfinite(H.grad).all()
    assert H.grad.abs().max() > 0


def test_retraction_carries_a_gradient_to_the_free_parameter():
    L, M, a = 64, 10, 4
    H = _random_bank(M, L, seed=15).requires_grad_(True)
    x = torch.randn(L, dtype=torch.float64)
    c = dense_analyse(x, frame_retract(H, a), a)
    c.abs().pow(2).sum().backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()
    assert H.grad.abs().max() > 0


def test_dual_carries_a_gradient():
    L, M, a = 32, 8, 4
    H = _random_bank(M, L, seed=16).requires_grad_(True)
    dense_dual(H, a).abs().sum().backward()
    assert H.grad is not None and torch.isfinite(H.grad).all()


def test_gradcheck_bounds():
    """Finite differences against autograd on a small bank."""
    L, M, a = 16, 4, 2
    H = _random_bank(M, L, seed=17).requires_grad_(True)

    def f(h):
        A, B = frame_bounds(h, a)
        return A + B

    assert torch.autograd.gradcheck(f, (H,), eps=1e-6, atol=1e-5)


def test_batched_banks():
    L, M, a, Nb = 64, 8, 4, 3
    H = torch.stack([_random_bank(M, L, seed=s) for s in range(Nb)])
    A, B = frame_bounds(H, a)
    assert A.shape == (Nb,) and B.shape == (Nb,)
    Ht = frame_retract(H, a)
    assert Ht.shape == H.shape
    A2, B2 = frame_bounds(Ht, a)
    assert torch.allclose(A2, torch.ones(Nb, dtype=A2.dtype), atol=1e-10)
    assert torch.allclose(B2, torch.ones(Nb, dtype=B2.dtype), atol=1e-10)


def test_analysis_is_batched_over_leading_dims():
    L, M, a = 64, 6, 4
    H = _random_bank(M, L, seed=18)
    x = torch.randn(5, 3, L, dtype=torch.float64)
    c = dense_analyse(x, H, a)
    assert c.shape == (5, 3, M, L // a)
    assert torch.allclose(c[2, 1], dense_analyse(x[2, 1], H, a), atol=1e-13)


def test_hop_must_divide_the_length():
    H = _random_bank(4, 64, seed=19)
    with pytest.raises(ValueError, match="must divide"):
        frame_bounds(H, 5)


def test_redundancy_below_one_is_not_a_frame():
    """M < a cannot be a frame: the Gram blocks are rank-deficient by shape."""
    L, M, a = 64, 2, 4
    H = _random_bank(M, L, seed=20)
    A, _ = frame_bounds(H, a)
    assert A.item() < 1e-12


def test_dtype_and_device_are_preserved():
    L, M, a = 64, 6, 4
    H = _random_bank(M, L, seed=21).to(torch.complex64)
    x = torch.randn(L, dtype=torch.float32)
    c = dense_analyse(x, H, a)
    assert c.dtype == torch.complex64
    assert frame_retract(H, a).dtype == torch.complex64
    assert np.isfinite(frame_bounds(H, a)[0].item())
