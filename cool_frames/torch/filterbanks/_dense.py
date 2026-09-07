r"""Dense, batched, **differentiable** frame algebra for uniform-hop filterbanks.

Why this module exists
----------------------
:mod:`cool_frames.torch.filterbanks._frame` reaches the NumPy solvers for
bounds, duals and tight frames.  Those wrappers call ``.detach().cpu().numpy()``
on the filter responses, so none of them can carry a gradient — which makes the
one thing the torch backend advertises ("differentiable, for training a
learnable front end") unavailable for the frame properties themselves.  A
learnable filterbank could be *measured* but not *constrained*.

This module supplies the missing path for the case a trainable front end
actually uses: a **uniform integer hop** and filters held as a dense
``(M, L)`` response tensor.  Everything here is a pure torch expression, so
bounds, condition numbers, duals and the Parseval retraction are all
differentiable with respect to ``H``.

Conventions (identical to the rest of cool-frames, verified in the tests)
------------------------------------------------------------------------
For a signal ``x`` of length ``L``, responses ``H`` of shape ``(M, L)`` and a
hop ``a`` dividing ``L`` with ``K = L // a``, analysis is

.. math::
   c_m[n] \;=\; \bigl(\mathrm{ifft}(\mathrm{fft}(x)\,H_m)\bigr)[na],
   \qquad n = 0,\dots,K-1,

which is exactly what :func:`cool_frames.torch.core.comp_filterbank_fft`
computes.  Downsampling by ``a`` aliases frequency ``k`` onto ``k + pK``, so
with

.. math::
   V[k]_{p,m} \;=\; H_m[k + pK], \qquad G[k] \;=\; \tfrac1a\,V[k]V[k]^{H},

the frame operator is **block diagonal** with ``K`` Hermitian ``a x a`` blocks
and therefore, with no painless assumption anywhere,

.. math::
   A = \min_k \lambda_{\min}(G[k]), \qquad B = \max_k \lambda_{\max}(G[k]).

``torch.linalg.eigvalsh`` makes both differentiable in ``H``.  In the painless
case every ``G[k]`` is diagonal and this collapses to the familiar
``sum_m |H_m|^2 / a`` of :func:`filterbankresponse`.

Real signals
------------
``real=True`` means the bank is **single-sided** (analytic: ``H_m`` supported on
the positive frequencies) and the signal is real.  The frame is then the bank
*together with its conjugate mirrors*, and every quantity here is computed on
that mirrored channel set — analysis, synthesis, bounds and retraction alike.
Computing the flatness on one channel set and the bounds on another is a real
defect that has been shipped before; here there is one channel set and one
code path.

Nothing in this module assumes the painless condition, and nothing in it
allocates an ``L x L`` operator: the cost is ``O(K a^3 + M L)``.
"""

from __future__ import annotations

import torch

__all__ = [
    "alias_blocks",
    "dense_analyse",
    "dense_dual",
    "dense_synthesise",
    "frame_bounds",
    "frame_condition",
    "frame_gram",
    "frame_response",
    "frame_retract",
    "is_frame",
    "mirror_bank",
    "retraction_gain",
]

_EIG_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# channel-set construction
# ---------------------------------------------------------------------------


def mirror_bank(H: torch.Tensor) -> torch.Tensor:
    """Append the conjugate-mirror channels of a single-sided bank.

    For a real signal, a filter with response ``H_m`` and its mirror
    ``H_m[-j]^*`` together produce conjugate coefficient pairs.  The frame the
    real signal actually sees is the union of the two, so bounds, duals and
    retraction must all be computed on the union.

    Parameters
    ----------
    H : ``(..., M, L)`` complex tensor.

    Returns
    -------
    ``(..., 2M, L)`` complex tensor: the original channels followed by their
    mirrors.
    """
    Hm = torch.flip(H, dims=(-1,)).roll(1, dims=-1).conj()
    return torch.cat([H, Hm], dim=-2)


def alias_blocks(H: torch.Tensor, a: int) -> torch.Tensor:
    """Reshape responses into the ``K`` aliasing blocks ``V[k] (a x M)``.

    Parameters
    ----------
    H : ``(..., M, L)`` complex tensor.
    a : hop size; must divide ``L``.

    Returns
    -------
    ``(..., K, a, M)`` complex tensor with ``V[k][p, m] = H[m, k + pK]``.
    """
    *batch, M, L = H.shape
    if L % a:
        raise ValueError(f"hop a={a} must divide the response length L={L}")
    K = L // a
    V = H.reshape(*batch, M, a, K)
    return V.permute(*range(len(batch)), len(batch) + 2, len(batch) + 1, len(batch))


def frame_gram(H: torch.Tensor, a: int, real: bool = False) -> torch.Tensor:
    """Block-diagonal frame operator ``G[k] = V[k] V[k]^H / a``.

    Returns ``(..., K, a, a)``, Hermitian by construction (and explicitly
    symmetrised against round-off, which matters because ``eigvalsh`` reads
    only one triangle and a silently non-Hermitian input gives a wrong,
    plausible answer).
    """
    if real:
        H = mirror_bank(H)
    V = alias_blocks(H, a)
    G = V @ V.conj().transpose(-1, -2) / a
    return 0.5 * (G + G.conj().transpose(-1, -2))


def frame_response(H: torch.Tensor, a: int, real: bool = False) -> torch.Tensor:
    """Diagonal of the frame operator, ``sum_m |H_m|^2 / a``, as a length-``L`` tensor.

    This is the painless-case response and it is what
    :func:`cool_frames.torch.filterbanks.filterbankresponse` returns — but
    differentiable, and folded over the mirrored channel set when ``real=True``
    so that it agrees with :func:`frame_bounds` on the same bank.

    Flatness of this response is **not** the frame property: a bank can be flat
    to machine precision and still have ``A = 0``, i.e. not be a frame at all
    (the response only sees the diagonal of ``G[k]``).  Use :func:`frame_bounds`
    for the property and this for the picture.
    """
    if real:
        H = mirror_bank(H)
    return H.abs().pow(2).sum(dim=-2) / a


def frame_bounds(H: torch.Tensor, a: int, real: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact frame bounds ``(A, B)``, differentiable with respect to ``H``.

    No painless assumption: the bounds come from the eigenvalues of the
    aliasing blocks, so they are correct for aliased and non-painless banks
    too.

    Returns
    -------
    ``(A, B)`` — zero-dimensional tensors (or batch-shaped, if ``H`` was
    batched).  ``A`` is clamped at zero: a frame bound is non-negative by
    definition, and a tiny negative eigenvalue means "not a frame", not a
    negative condition number.
    """
    ev = torch.linalg.eigvalsh(frame_gram(H, a, real=real))
    A = ev.amin(dim=(-2, -1)).clamp_min(0.0)
    B = ev.amax(dim=(-2, -1))
    return A, B


def frame_condition(H: torch.Tensor, a: int, real: bool = False) -> torch.Tensor:
    """Condition number ``kappa = B / A``, differentiable; ``inf`` when ``A = 0``."""
    A, B = frame_bounds(H, a, real=real)
    return torch.where(A > 0, B / A.clamp_min(_EIG_FLOOR), torch.full_like(B, float("inf")))


# ---------------------------------------------------------------------------
# retraction — the frame property as a manifold, not a penalty
# ---------------------------------------------------------------------------


def frame_retract(
    H: torch.Tensor,
    a: int,
    kappa_target: float = 1.0,
    real: bool = False,
    eps: float = 1e-10,
    strict: bool = True,
    stable_grad: bool = True,
) -> torch.Tensor:
    """Project a bank onto ``{kappa <= kappa_target, B = 1}``, exactly and differentiably.

    .. admonition:: What retraction can and cannot do

       Retraction rescales each aliasing block by an invertible transform, so it
       **normalises conditioning but cannot create coverage**.  ``A = B = 1``
       is reached if and only if every block ``G[k]`` already has full rank
       ``a`` — that is, if and only if the bank was a frame to begin with.  A
       bank that leaves a frequency (or an aliasing direction) uncovered has
       ``A = 0``, and no transform of its channels can fix that: the energy
       simply is not there.  ``strict=True`` raises rather than returning a
       bank whose bounds silently are not what the caller asked for.

       The useful consequence for training: because the transform is
       invertible wherever the rank is full, the frame property, once
       initialised, is preserved for every parameter value — and the *size* of
       the transform (:func:`retraction_gain`) is a live diagnostic of the free
       parameters drifting toward rank deficiency, which a penalty never gave
       you.

    Each aliasing block is transformed by
    ``T[k] = U diag(sqrt(lambda_target / lambda)) U^H`` where ``lambda`` are the
    eigenvalues of ``G[k]`` and ``lambda_target`` is ``lambda`` rescaled so the
    global maximum is 1 and then clamped from below at ``1 / kappa_target``.
    The retracted bank satisfies ``B = 1`` and ``A >= 1 / kappa_target``
    **for every value of the free parameter, at every step and between steps** —
    there is no schedule, no weight to tune, and nothing that can drift between
    optimiser steps.  ``kappa_target = 1`` gives a Parseval frame, ``A = B = 1``.

    Why a retraction rather than a penalty: a penalty can only pull, and it
    competes with the task loss for the same parameters.  Worse, near a
    well-conditioned initialisation the classification gradient is very nearly
    orthogonal to the condition-number direction, so the penalty has almost
    nothing to act on and the unconstrained bank barely drifts anyway — the
    penalty ends up measuring nothing.  A retraction makes the property hold by
    construction and leaves the loss to do only the task.

    ``real=True`` retracts the mirrored channel set.  The retraction commutes
    with conjugate mirroring (``T[K-k] = P T[k]^* P^T`` for the alias-flip
    permutation ``P``), so the retracted bank is still single-sided and its
    mirrors are still the mirrors of its channels; the test suite asserts this
    rather than assuming it.

    Gradients flow through :func:`torch.linalg.eigh`.  That derivative has a
    ``1 / (lambda_i - lambda_j)`` factor, so it is ill-conditioned at exactly
    degenerate eigenvalues; ``eps`` floors the eigenvalues used in the
    transform, ``stable_grad`` (default) freezes the eigenbasis in the backward
    pass so that clustered eigenvalues cannot produce an infinite gradient --
    the forward stays exact, only the gradient is approximated -- and the
    ``a = 1`` fast path avoids the issue entirely.
    """
    if kappa_target < 1.0:
        raise ValueError(f"kappa_target must be >= 1, got {kappa_target}")

    H_full = mirror_bank(H) if real else H
    V = alias_blocks(H_full, a)  # (..., K, a, M)
    G = V @ V.conj().transpose(-1, -2) / a
    G = 0.5 * (G + G.conj().transpose(-1, -2))

    if a == 1:
        # G is 1x1: the whole thing is the painless response, and the transform
        # is a scalar per frequency.  No eigendecomposition, no degeneracy.
        lam_raw = G[..., 0, 0].real  # (..., K)
        lam = lam_raw.clamp_min(eps)
        lam_max = lam.amax(dim=-1, keepdim=True)
        target = (lam / lam_max).clamp_min(1.0 / kappa_target)
        scale = (target / lam).sqrt()  # (..., K)
        Vt = V * scale[..., None, None]
    else:
        lam_raw, U = torch.linalg.eigh(G)  # (..., K, a), (..., K, a, a)
        if stable_grad:
            # The eigh backward carries a 1 / (lambda_i - lambda_j) factor, so a
            # block with clustered eigenvalues -- exactly what a bank with
            # weakly covered aliasing directions has -- produces an infinite
            # gradient and NaNs the whole model within one step. Freeze the
            # eigenbasis and recompute the eigenvalues differentiably inside it:
            # U still diagonalises G, so the forward value is unchanged and the
            # frame property stays exact; the backward loses only the
            # basis-rotation term, which is the ill-conditioned part. This is
            # the usual differential-of-a-retraction compromise, and it is a
            # compromise in the gradient alone, never in the constraint.
            U = U.detach()
            lam_raw = torch.diagonal(U.conj().transpose(-1, -2) @ G @ U, dim1=-2, dim2=-1).real
        lam = lam_raw.clamp_min(eps)
        lam_max = lam.amax(dim=(-2, -1), keepdim=True)
        target = (lam / lam_max).clamp_min(1.0 / kappa_target)
        scale = (target / lam).sqrt().to(U.dtype)  # (..., K, a)
        T = (U * scale[..., None, :]) @ U.conj().transpose(-1, -2)
        Vt = T @ V

    if strict:
        rel = lam_raw / lam_raw.amax().clamp_min(_EIG_FLOOR)
        if bool((rel < eps).any()):
            n_dead = int((rel < eps).sum())
            raise ValueError(
                f"frame_retract: {n_dead} of {rel.numel()} aliasing eigenvalues are "
                f"zero to within eps={eps:g}, so this bank is not a frame (A = 0) and "
                "retraction cannot make it one — it rescales the energy that is "
                "there, it does not create coverage. Fix the design (more channels, "
                "wider filters, or a smaller hop), or pass strict=False if you are "
                "deliberately retracting a rank-deficient bank and will check "
                "frame_bounds yourself."
            )

    *batch, K, _, M = Vt.shape
    Ht = Vt.permute(*range(len(batch)), len(batch) + 2, len(batch) + 1, len(batch))
    Ht = Ht.reshape(*batch, M, K * a)
    if real:
        Ht = Ht[..., : M // 2, :]
    return Ht


def retraction_gain(H: torch.Tensor, a: int, real: bool = False) -> torch.Tensor:
    """How hard the retraction has to work: the condition number it removes.

    This is just ``frame_condition`` of the **un**retracted bank, but it is the
    quantity worth logging during training. The retracted bank's ``kappa`` is
    constant by construction and therefore tells you nothing; this one rises as
    the free parameters drift toward a rank-deficient bank, and it is the early
    warning that the design (channel count, bandwidths, hop) is too tight for
    where training is going. Diverging gain with a constant post-retraction
    ``kappa`` is the signature to watch for.
    """
    return frame_condition(H, a, real=real)


def is_frame(H: torch.Tensor, a: int, real: bool = False, tol: float = 1e-10) -> bool:
    """Whether the bank has a strictly positive lower frame bound."""
    A, B = frame_bounds(H, a, real=real)
    return bool(tol * B.clamp_min(_EIG_FLOOR) < A)


# ---------------------------------------------------------------------------
# dual frame
# ---------------------------------------------------------------------------


def dense_dual(H: torch.Tensor, a: int, real: bool = False, eps: float = 0.0) -> torch.Tensor:
    """Canonical dual bank, in closed form and differentiable.

    ``G[k]^{-1}`` applied blockwise, i.e. ``V_dual[k] = G[k]^{-1} V[k]``.  With
    the resulting responses, :func:`dense_synthesise` inverts
    :func:`dense_analyse` exactly for **any** bank with ``A > 0``, not only for
    painless or tight ones — and it is a genuine least-squares left inverse, so
    on an *edited* coefficient tensor it returns the signal whose analysis is
    closest to the edit.

    ``eps`` defaults to **zero** on purpose. A Tikhonov term here is not a
    harmless guard: it biases the dual by roughly ``eps / A``, which for a
    merely well-conditioned bank is already visible far above float64 noise
    (``eps = 1e-10`` against ``A = 1e-3`` costs seven digits of reconstruction).
    A singular ``G`` should raise, because that bank is not a frame and its
    "dual" would be a fiction. Pass ``eps > 0`` only for a deliberate
    regularised pseudo-dual, and expect the reconstruction error that buys.
    """
    H_full = mirror_bank(H) if real else H
    V = alias_blocks(H_full, a)
    G = V @ V.conj().transpose(-1, -2) / a
    G = 0.5 * (G + G.conj().transpose(-1, -2))
    if eps:
        a_dim = G.shape[-1]
        eye = torch.eye(a_dim, dtype=G.dtype, device=G.device).expand_as(G)
        G = G + eps * eye
    Vd = torch.linalg.solve(G, V)

    *batch, K, _, M = Vd.shape
    Hd = Vd.permute(*range(len(batch)), len(batch) + 2, len(batch) + 1, len(batch))
    Hd = Hd.reshape(*batch, M, K * a)
    if real:
        Hd = Hd[..., : M // 2, :]
    return Hd  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# analysis / synthesis
# ---------------------------------------------------------------------------


def dense_analyse(x: torch.Tensor, H: torch.Tensor, a: int) -> torch.Tensor:
    """Uniform-hop filterbank analysis, batched over any leading dimensions.

    Parameters
    ----------
    x : ``(..., L)`` real or complex tensor.
    H : ``(M, L)`` complex responses.
    a : hop; must divide ``L``.

    Returns
    -------
    ``(..., M, L // a)`` complex coefficients, numerically identical to
    :func:`cool_frames.torch.filterbanks.filterbank` on the same bank (asserted
    in the tests, not assumed).
    """
    _, L = H.shape
    if x.shape[-1] != L:
        raise ValueError(f"signal length {x.shape[-1]} does not match response length {L}")
    if L % a:
        raise ValueError(f"hop a={a} must divide L={L}")
    K = L // a
    X = torch.fft.fft(x, dim=-1)
    FG = X.unsqueeze(-2) * H  # (..., M, L)
    folded = FG.reshape(*FG.shape[:-1], a, K).sum(dim=-2)
    return torch.fft.ifft(folded, dim=-1) / a  # type: ignore[no-any-return]


def dense_synthesise(c: torch.Tensor, H: torch.Tensor, a: int, real: bool = False) -> torch.Tensor:
    """Synthesis with the bank ``H``; the adjoint of :func:`dense_analyse`.

    Pass the **dual** bank (:func:`dense_dual`) to invert an analysis, or the
    bank itself when it has been retracted to Parseval, where the dual is the
    bank.

    ``real=True`` adds the conjugate-mirror channels' contribution, which for a
    real signal is the conjugate of the single-sided one — so the result is
    ``2 Re(...)``, exactly and not approximately, provided the bank is strictly
    single-sided (no channel is its own mirror).  A bank with energy at DC or
    Nyquist violates that and is double-counted there; :func:`dense_analyse`
    cannot detect it, so it is the caller's contract.
    """
    M, L = H.shape
    K = L // a
    if c.shape[-2:] != (M, K):
        raise ValueError(f"coefficients {tuple(c.shape[-2:])} do not match bank ({M}, {K})")
    C = torch.fft.fft(c, dim=-1)  # (..., M, K)
    tiled = C.unsqueeze(-2).expand(*C.shape[:-1], a, K).reshape(*C.shape[:-1], L)
    F = (tiled * H.conj()).sum(dim=-2)  # (..., L)
    x = torch.fft.ifft(F, dim=-1)
    return 2.0 * x.real if real else x  # type: ignore[no-any-return]
