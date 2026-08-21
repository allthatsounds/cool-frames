"""Differentiable second-order IIR (biquad) resonator — torch backend.

This is the torch counterpart of ``cool_frames.numpy.filters._filters.comp_biquad`` /
``biquadfilter``.  Unlike the other entries in :mod:`cool_frames.torch.filters`
(which are thin wrappers around the NumPy filter-design code), the biquad is
implemented *natively* in torch so that gradients flow to the pole
parameters.  That is the whole point of having it on the torch side: an
allpole resonator parameterised by the unconstrained ML pair ``(rho, phi)``
with

    r     = sigmoid(rho)        in (0, 1)        -- always stable
    theta = pi * sigmoid(phi)   in (0, pi)

is trainable by gradient descent and is guaranteed BIBO-stable by
construction.

The frequency response materialised here is the *full-length DFT response*
``H(k) = 1 / D(z^{-1})`` with ``D = 1 - 2 r cos(theta) z^{-1} + r^2 z^{-2}``,
exactly matching the NumPy reference, so a biquad descriptor produced here
runs through the ordinary full-length-FFT analysis path.  (A genuinely
recursive ``sosfilt`` computation path is a separate, future addition; see the
companion document.)
"""

from __future__ import annotations

import math

import torch

__all__ = ["biquad_response", "biquadfilter", "comp_biquad"]


def _complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """Map a real dtype to its complex partner (default to complex128)."""
    if dtype in (torch.float64, torch.complex128):
        return torch.complex128
    if dtype in (torch.float32, torch.complex64):
        return torch.complex64
    return torch.complex128


def comp_biquad(
    r: float | torch.Tensor,
    theta: float | torch.Tensor,
    L: int,
    norm: str = "energy",
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Full-length DFT response of a second-order allpole resonator.

    Differentiable in ``r`` and ``theta`` (pass them as tensors that
    ``requires_grad`` to back-propagate).  Mirrors
    :func:`cool_frames.numpy.filters.comp_biquad` numerically.

    Parameters
    ----------
    r : float or Tensor
        Pole radius, ``0 < r < 1`` (``r >= 1`` is unstable).
    theta : float or Tensor
        Pole angle in radians, ``0 <= theta <= pi``.
    L : int
        Transform length; output has length *L*.
    norm : str
        ``'energy'``/``'2'`` (default), ``'inf'``/``'peak'``, ``'1'``/``'area'``,
        or ``'none'``.  Same conventions as the NumPy reference.
    device, dtype : optional
        Real dtype (default ``float64`` for gradcheck-grade precision); the
        returned tensor is the matching complex dtype.

    Returns
    -------
    H : complex Tensor, shape ``(L,)``
    """
    cdtype = _complex_dtype(dtype)
    r_t = torch.as_tensor(r, dtype=dtype, device=device)
    theta_t = torch.as_tensor(theta, dtype=dtype, device=device)
    dev = r_t.device

    k = torch.arange(L, device=dev, dtype=dtype)
    omega = (2.0 * math.pi / L) * k
    # z_inv = exp(-j omega), unit modulus -> polar(1, -omega)
    z_inv = torch.polar(torch.ones_like(omega), -omega).to(cdtype)

    a1 = (-2.0 * r_t * torch.cos(theta_t)).to(cdtype)
    a2 = (r_t * r_t).to(cdtype)

    D = 1.0 + a1 * z_inv + a2 * z_inv * z_inv
    H = 1.0 / D

    if norm in ("energy", "2"):
        H = H * (math.sqrt(L) / torch.linalg.vector_norm(H))
    elif norm in ("inf", "peak"):
        H = H / torch.max(torch.abs(H))
    elif norm in ("1", "area"):
        H = H * (L / torch.sum(torch.abs(H)))
    # else: 'none' / unknown -> no normalisation

    return H


def biquad_response(
    rho: float | torch.Tensor,
    phi: float | torch.Tensor,
    L: int,
    norm: str = "energy",
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """ML-friendly biquad response in unconstrained pole parameters.

    ``r = sigmoid(rho)``, ``theta = pi * sigmoid(phi)`` — stable for any real
    ``rho, phi``.  Differentiable in ``rho`` and ``phi``.
    """
    rho_t = torch.as_tensor(rho, dtype=dtype, device=device)
    phi_t = torch.as_tensor(phi, dtype=dtype, device=device)
    r = torch.sigmoid(rho_t)
    theta = math.pi * torch.sigmoid(phi_t)
    return comp_biquad(r, theta, L, norm, device=device, dtype=dtype)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def biquadfilter(
    fc,
    bw,
    *,
    fs: float | None = None,
    norm: str = "energy",
    delay: int = 0,
    scal: float = 1.0,
    realonly: bool = False,
    rho=None,
    phi=None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> dict | list[dict]:
    """Construct a differentiable biquad filter descriptor (torch).

    Mirrors :func:`cool_frames.numpy.filters.biquadfilter`: maps centre frequency and
    bandwidth (or explicit ``rho``/``phi``) to conjugate poles and returns a
    descriptor dict with a differentiable ``H(L)`` callable plus the physical
    (``r``, ``theta``) and ML (``rho``, ``phi``) parameters.

    Pass ``rho``/``phi`` as tensors with ``requires_grad=True`` to train the
    pole locations; the returned ``H(L)`` then back-propagates to them.

    Normalised-frequency convention: ``fc, bw`` in ``[0, 2]`` (1 = Nyquist),
    unless ``fs`` is given (then in Hz).
    """
    fc_arr = torch.atleast_1d(torch.as_tensor(fc, dtype=dtype, device=device))
    bw_arr = torch.atleast_1d(torch.as_tensor(bw, dtype=dtype, device=device))

    N = max(fc_arr.numel(), bw_arr.numel())
    if fc_arr.numel() == 1 and N > 1:
        fc_arr = fc_arr.expand(N)
    if bw_arr.numel() == 1 and N > 1:
        bw_arr = bw_arr.expand(N)

    if fs is not None:
        fc_arr = fc_arr / fs * 2.0
        bw_arr = bw_arr / fs * 2.0

    # centre-wrap onto [-1, 1] (period 2), matching numpy modcent(f, 2)
    fc_arr = fc_arr - 2.0 * torch.round(fc_arr / 2.0)

    rho_list = (
        None if rho is None else torch.atleast_1d(torch.as_tensor(rho, dtype=dtype, device=device))
    )
    phi_list = (
        None if phi is None else torch.atleast_1d(torch.as_tensor(phi, dtype=dtype, device=device))
    )

    gout: list[dict] = []
    for ii in range(N):
        fc_ii = fc_arr[ii]
        bw_ii = bw_arr[ii]

        # pole angle
        if phi_list is not None:
            phi_ii = phi_list[ii] if phi_list.numel() > 1 else phi_list[0]
            theta_ii = math.pi * torch.sigmoid(phi_ii)
        else:
            theta_ii = math.pi * torch.abs(fc_ii)
            frac = (theta_ii / math.pi).clamp(1e-12, 1 - 1e-12)
            phi_ii = torch.log(frac) - torch.log(1.0 - frac)

        # pole radius
        if rho_list is not None:
            rho_ii = rho_list[ii] if rho_list.numel() > 1 else rho_list[0]
            r_ii = torch.sigmoid(rho_ii)
        else:
            r_ii = (1.0 - math.pi * bw_ii / 2.0).clamp(0.0, 1.0 - 1e-6)
            frac_r = r_ii.clamp(1e-12, 1 - 1e-12)
            rho_ii = torch.log(frac_r) - torch.log(1.0 - frac_r)

        def _make_H(r_v, th_v, sc, nm):
            def H(L: int) -> torch.Tensor:
                return comp_biquad(r_v, th_v, L, nm, device=device, dtype=dtype) * sc

            return H

        gout.append(
            {
                "H": _make_H(r_ii, theta_ii, scal, norm),
                "foff": (lambda L: 0),
                "realonly": 1 if realonly else 0,
                "delay": int(delay),
                "fs": fs,
                "fc": fc_ii,
                "bw": bw_ii,
                "r": r_ii,
                "theta": theta_ii,
                "rho": rho_ii,
                "phi": phi_ii,
            }
        )

    if N == 1 and torch.as_tensor(fc).ndim == 0:
        return gout[0]
    return gout
