"""
torch/phase/_phasederiv.py
==========================
Higher-order phase derivatives for non-uniform filterbanks (PyTorch).

Ports the second-order derivative computation from:
  numpy/phase/_phasederiv.py

to fully differentiable torch operations. Enables second-order phase
derivatives (tt, ff, tf) to flow gradients during training.

Theory
------
For a filterbank coefficient c_m[n] = <f, T_{na_m} g_m> with window g_m:

  d_t   = -Im(c_d · conj(c) / |c|²)          [instantaneous frequency]
  d_f   = -Re(c_h · conj(c) / |c|²)          [group delay]
  d_tt  =  Im(c_d2 · conj(c) / |c|² - 2π·(c_d · conj(c) / |c|²)²) / L
  d_ff  =  Im(-c_h2 · conj(c) / |c|² + (c_h · conj(c) / |c|²)²) · 2π/L
  d_tf  =  Re(c_hd · conj(c) / |c|² - (1/L)·c_h · c_d · (conj(c)/|c|²)²) · 2π

where:
  c_d  = <f, T_{na} g'/(2π)>        (time-derivative of window)
  c_h  = <f, T_{na} (n·g)>          (frequency-weighted window)
  c_d2 = <f, T_{na} g''/(2π)>       (second time-derivative)
  c_h2 = <f, T_{na} (n²·g)>         (second frequency-weight)
  c_hd = <f, T_{na} (n/L)·g'/(2π)>  (cross-weighted window)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from ...numpy.filterbanks._utils import normalise_a
from ...numpy.filters._design import filterbanklength
from ...numpy.phase._phasegrad import comp_phasegradfilters
from ..filterbanks._core import filterbank

# ---------------------------------------------------------------------------
# Build second-order derivative filters (setup-time, uses numpy)
# ---------------------------------------------------------------------------


def comp_phasederivfilters_2nd(
    g: list[dict],
    a,
    L: int,
    ch: list[dict] | None = None,
    cd: list[dict] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    """Build first- and second-order derivative filter banks.

    Parameters
    ----------
    g  : list of M filter dicts (the analysis filters)
    a  : hop sizes
    L  : signal length
    ch, cd : optional pre-computed first-order derivative filters
             (from comp_phasegradfilters); will be computed if None.

    Returns
    -------
    ch   : list of M filter dicts (frequency-weighted, for d_f)
    cd   : list of M filter dicts (time-derivative, for d_t)
    cd2  : list of M filter dicts (second time-derivative, for d_tt)
    ch2  : list of M filter dicts (second frequency-weight, for d_ff)
    chd  : list of M filter dicts (cross freq-time weight, for d_tf)
    """
    M = len(g)

    # First-order filters
    if ch is None or cd is None:
        ch, cd = comp_phasegradfilters(g, a, L)

    cd2 = []  # second time-derivative: g'' / (2π)²
    ch2 = []  # second frequency-weight: n² · g
    chd = []  # cross: (n/L) · g' / (2π)

    for m in range(M):
        gm = g[m]

        if "H" in gm:
            # Band-limited filter: work in frequency domain
            H_callable = gm["H"]
            fo_callable = gm["foff"]

            H_ref = H_callable
            fo_ref = fo_callable

            # cd2: second derivative in time = multiply by (-j2πk/L)²
            def _make_cd2(H_c, L_=L):
                H_vals = np.asarray(H_c(L_), dtype=complex)
                n_h = len(H_vals)
                k = np.arange(n_h)
                fmul = (-1j * 2 * np.pi * k / L_) ** 2
                return H_vals * fmul

            cd2_m = dict(gm)
            cd2_m["H"] = lambda L_, _H=H_ref: _make_cd2(_H, L_)
            cd2_m["foff"] = fo_ref

            # ch2: second frequency-weight = multiply by (jn)²  = -n²
            def _make_ch2(H_c, fo_c, L_=L):
                H_vals = np.asarray(H_c(L_), dtype=complex)
                n_h = len(H_vals)
                fo_v = int(fo_c(L_)) if callable(fo_c) else int(fo_c)
                k_abs = (np.arange(fo_v, fo_v + n_h) % L_).astype(float)
                k_abs[k_abs > L_ / 2] -= L_
                tmul = -(k_abs**2)  # (j·k)² = -k²
                return H_vals * tmul

            ch2_m = dict(gm)
            ch2_m["H"] = lambda L_, _H=H_ref, _fo=fo_ref: _make_ch2(_H, _fo, L_)
            ch2_m["foff"] = fo_ref

            # chd: cross = (n/L) · dg/(2π) in freq domain
            def _make_chd(H_c, fo_c, L_=L):
                H_vals = np.asarray(H_c(L_), dtype=complex)
                n_h = len(H_vals)
                fo_v = int(fo_c(L_)) if callable(fo_c) else int(fo_c)

                # Build full-length response for dg
                H_full = np.zeros(L_, dtype=complex)
                idx = (np.arange(n_h) + fo_v) % L_
                H_full[idx] = H_vals

                # dg in frequency domain: H * (-j*2π*k/L) / (2π) = H * (-j*k/L)
                k_full = np.arange(L_, dtype=float)
                k_full[k_full > L_ / 2] -= L_
                Hdg_full = H_full * (-1j * k_full / L_)

                # Convert to time domain
                dg_time = np.fft.ifft(Hdg_full)

                # Multiply by n/L (time ramp)
                n_ramp = np.arange(L_, dtype=float)
                n_ramp[n_ramp > L_ / 2] -= L_
                hdg_time = (n_ramp / L_) * dg_time

                # Back to frequency domain, extract band-limited part
                Hchd_full = np.fft.fft(hdg_time)
                return Hchd_full[idx]

            chd_m = dict(gm)
            chd_m["H"] = lambda L_, _H=H_ref, _fo=fo_ref: _make_chd(_H, _fo, L_)
            chd_m["foff"] = fo_ref

        else:
            # FIR filter: work in time domain
            h = np.asarray(gm.get("h", np.array([])))
            Lh = len(h)
            n = np.arange(Lh, dtype=float)

            # cd2: second time-derivative
            cd2_m = dict(gm)
            cd2_m["h"] = h * (-1j * 2 * np.pi / L * n) ** 2

            # ch2: second frequency-weight
            ch2_m = dict(gm)
            ch2_m["h"] = h * (1j * n) ** 2  # = -h * n²

            # chd: cross (n/L) · dg/(2π)
            chd_m = dict(gm)
            chd_m["h"] = h * (-1j * n**2 / L**2)

        cd2.append(cd2_m)
        ch2.append(ch2_m)
        chd.append(chd_m)

    return ch, cd, cd2, ch2, chd


# ---------------------------------------------------------------------------
# Core computation (fully torch-based)
# ---------------------------------------------------------------------------


def comp_filterbankphasederiv(
    c: list[torch.Tensor],
    cd: list[torch.Tensor],
    ch: list[torch.Tensor],
    cd2: list[torch.Tensor],
    ch2: list[torch.Tensor],
    chd: list[torch.Tensor],
    L: int,
    minlvl: float = 1e-6,
    derivs: Sequence[str] = ("tt", "ff", "tf"),
) -> dict[str, list[torch.Tensor]]:
    """Compute second-order phase derivatives from filterbank coefficients.

    All operations are differentiable.

    Parameters
    ----------
    c, cd, ch, cd2, ch2, chd : lists of M complex torch tensors
        Analysis and derivative-filter coefficients.
    L : int
        Signal length.
    minlvl : float
        Relative floor for the spectrogram denominator.
    derivs : sequence of str
        Which derivatives to compute: 'tt', 'ff', 'tf' (any subset).

    Returns
    -------
    result : dict mapping derivative name to list of M real tensors
    """
    M = len(c)

    # Global floor
    all_abs = torch.cat([torch.abs(cm).ravel() for cm in c])
    lvl = minlvl * torch.max(all_abs**2).item() if all_abs.numel() > 0 else minlvl

    result: dict[str, list[torch.Tensor]] = {d: [] for d in derivs}

    for m in range(M):
        cm = c[m]
        sm = torch.clamp(torch.abs(cm) ** 2, min=lvl)
        conj_cm = torch.conj(cm)
        ratio = conj_cm / sm  # conj(c) / |c|²

        if "tt" in derivs:
            cdm = cd[m]
            cd2m = cd2[m]
            # d_tt = Im(cd2·conj(c)/s - 2π·(cd·conj(c)/s)²) / L
            term1 = cd2m * ratio
            term2 = (cdm * ratio) ** 2
            d_tt = torch.imag(term1 - 2 * np.pi * term2) / L
            result["tt"].append(d_tt.real if d_tt.is_complex() else d_tt)

        if "ff" in derivs:
            chm = ch[m]
            ch2m = ch2[m]
            # d_ff = Im(-ch2·conj(c)/s + (ch·conj(c)/s)²) · 2π/L
            term1 = -ch2m * ratio
            term2 = (chm * ratio) ** 2
            d_ff = torch.imag(term1 + term2) * 2 * np.pi / L
            result["ff"].append(d_ff.real if d_ff.is_complex() else d_ff)

        if "tf" in derivs:
            cdm = cd[m]
            chm = ch[m]
            chdm = chd[m]
            # d_tf = Re(chd·conj(c)/s - (1/L)·ch·cd·(conj(c)/s)²) · 2π
            term1 = chdm * ratio
            term2 = (1.0 / L) * chm * cdm * ratio**2
            d_tf = torch.real(term1 - term2) * 2 * np.pi
            result["tf"].append(d_tf.real if d_tf.is_complex() else d_tf)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filterbankphasederiv(
    f: torch.Tensor,
    g: list[dict],
    a,
    derivs: str | Sequence[str] = ("tt", "ff", "tf"),
    L: int | None = None,
    minlvl: float = 1e-6,
) -> tuple[dict[str, list[torch.Tensor]], list[torch.Tensor]]:
    """Compute second-order phase derivatives for a filterbank.

    This extends :func:`filterbankphasegrad` (which computes first-order
    derivatives t and f) to second-order derivatives tt, ff, tf.

    Parameters
    ----------
    f : torch.Tensor, shape (Ls,)
        Input signal.
    g : list of M filter dicts
        Analysis filters.
    a : array_like
        Hop sizes.
    derivs : str or sequence of str
        Which derivatives to compute.  Can be a single string ('tt')
        or a sequence ('tt', 'ff', 'tf').  Valid values: 'tt', 'ff', 'tf'.
    L : int, optional
        DFT length.  Inferred from signal and hop sizes if omitted.
    minlvl : float
        Relative spectrogram floor for numerical stability.

    Returns
    -------
    result : dict mapping derivative name → list of M real tensors
        Each tensor has the same length as the corresponding filterbank
        coefficient tensor.
    c : list of M complex torch tensors
        The filterbank coefficients (always computed as a byproduct).

    Examples
    --------
    >>> import torch
    >>> from cool_frames.torch.phase import filterbankphasederiv
    >>> signal = torch.randn(4096)
    >>> result, c = filterbankphasederiv(signal, g, a, derivs=['tt', 'tf'])
    >>> chirp_rate = result['tt']  # list of M tensors
    >>> mixed_deriv = result['tf']
    """
    M = len(g)
    a_norm = normalise_a(a, M)

    if isinstance(derivs, str):
        derivs = [derivs]
    derivs = [d.lower() for d in derivs]
    for d in derivs:
        if d not in ("tt", "ff", "tf"):
            raise ValueError(f"Unknown derivative '{d}'. Valid: 'tt', 'ff', 'tf'.")

    if L is None:
        L = filterbanklength(len(f), a_norm)

    # Build all derivative filters (setup-time, uses numpy)
    ch, cd, cd2, ch2, chd = comp_phasederivfilters_2nd(g, a_norm, L)

    # Run filterbanks — we need c plus whichever derivative coefficients
    c = filterbank(f, g, a_norm, L=L)

    # Only compute the filterbank outputs we actually need
    cd_c: list[torch.Tensor] = [None] * M  # type: ignore[assignment,list-item]
    ch_c: list[torch.Tensor] = [None] * M  # type: ignore[assignment,list-item]
    cd2_c: list[torch.Tensor] = [None] * M  # type: ignore[assignment,list-item]
    ch2_c: list[torch.Tensor] = [None] * M  # type: ignore[assignment,list-item]
    chd_c: list[torch.Tensor] = [None] * M  # type: ignore[assignment,list-item]

    need_cd = "tt" in derivs or "tf" in derivs
    need_ch = "ff" in derivs or "tf" in derivs
    need_cd2 = "tt" in derivs
    need_ch2 = "ff" in derivs
    need_chd = "tf" in derivs

    if need_cd:
        cd_c = filterbank(f, cd, a_norm, L=L)
    if need_ch:
        ch_c = filterbank(f, ch, a_norm, L=L)
    if need_cd2:
        cd2_c = filterbank(f, cd2, a_norm, L=L)
    if need_ch2:
        ch2_c = filterbank(f, ch2, a_norm, L=L)
    if need_chd:
        chd_c = filterbank(f, chd, a_norm, L=L)

    result = comp_filterbankphasederiv(
        c,
        cd_c,
        ch_c,
        cd2_c,
        ch2_c,
        chd_c,
        L,
        minlvl,
        derivs,
    )

    return result, c
