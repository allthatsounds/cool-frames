"""
torch/filterbanks/_core.py
==========================
High-level filterbank analysis and synthesis — PyTorch implementation.

Mirrors ``numpy.filterbanks._core`` (filterbank / ifilterbank)
but operates on ``torch.Tensor``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch

from ..core._core import (
    _normalise_a,
    comp_filterbank_fft,
    comp_filterbank_fftbl,
    comp_filterbank_td,
    comp_ifilterbank_fft,
    comp_ifilterbank_fftbl,
    comp_ifilterbank_td,
    filterbanklength,
)

# ---------------------------------------------------------------------------
# Filter preparation — convert filter dicts to concrete tensors
# ---------------------------------------------------------------------------


def _prepare_filters(
    g: list[dict],
    a_norm: np.ndarray,
    L: int,
    device: torch.device,
) -> tuple[list[dict], list[int], list[int], list[int]]:
    """Materialise filter dicts and classify into td / fft / fftbl groups.

    This helper processes a list of filter specification dicts, converting
    any callable H (frequency response), h (impulse response), or foff
    (frequency offset) into concrete tensors or scalars. Filters are then
    classified by type:
    - Time-domain (FIR): has "h" key (impulse response)
    - Full-length FFT: has "H" key with length L
    - Band-limited FFT: has "H" key with length < L

    Parameters
    ----------
    g : list of dict
        Filter specifications (from filters.audfilters etc.)
    a_norm : (M, 2) ndarray
        Normalized hop size pairs
    L : int
        Target DFT length
    device : torch.device
        Target device for tensors

    Returns
    -------
    g_ready : list of dict
        Processed filter dicts with concrete tensor H/h and scalar foff/offset
    m_td : list of int
        Indices of time-domain (FIR) filters (have "h" key)
    m_fft : list of int
        Indices of full-length FFT filters (have "H" with len = L)
    m_fftbl : list of int
        Indices of band-limited FFT filters (have "H" with len < L)
    """
    M = len(g)
    g_ready: list[dict[str, Any] | None] = [None] * M
    m_td, m_fft, m_fftbl = [], [], []

    for m in range(M):
        gm = dict(g[m])

        # Check for time-domain FIR filter (has "h" key)
        h = gm.get("h")
        if h is not None:
            # Time-domain (FIR) filter
            if callable(h):
                h_arr = h(L)
                if isinstance(h_arr, np.ndarray) or not isinstance(h_arr, torch.Tensor):
                    h_arr = torch.tensor(h_arr, dtype=torch.complex128, device=device)
                gm["h"] = h_arr
            elif not isinstance(h, torch.Tensor):
                gm["h"] = torch.tensor(
                    np.asarray(h, dtype=np.complex128), dtype=torch.complex128, device=device
                )

            # Ensure offset is materialized if callable
            offset = gm.get("offset")
            if callable(offset):
                gm["offset"] = int(offset(L))

            m_td.append(m)
            g_ready[m] = gm  # type: ignore[assignment]
            continue

        # Materialise callable H (frequency response)
        H = gm.get("H")
        if callable(H):
            H_arr = H(L)
            if isinstance(H_arr, np.ndarray):
                H_arr = torch.tensor(H_arr, dtype=torch.complex128, device=device)
            gm["H"] = H_arr

        # Materialise callable foff
        foff = gm.get("foff")
        if callable(foff):
            gm["foff"] = int(foff(L))

        # Ensure H is a tensor
        H = gm.get("H")
        if H is not None and not isinstance(H, torch.Tensor):
            gm["H"] = torch.tensor(
                np.asarray(H, dtype=np.complex128), dtype=torch.complex128, device=device
            )

        H = gm.get("H")
        if H is None:
            raise ValueError(
                f"Filter {m}: must have either 'h' (impulse response) or 'H' (frequency response)"
            )
        elif len(H) == L:
            m_fft.append(m)
        else:
            m_fftbl.append(m)

        g_ready[m] = gm  # type: ignore[assignment]

    return g_ready, m_td, m_fft, m_fftbl  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# filterbank (analysis)
# ---------------------------------------------------------------------------


def filterbank(
    f: torch.Tensor,
    g: list[dict],
    a,
    L: int | None = None,
    stack: bool = False,
):
    """Apply a (non-uniform) filterbank to signal *f*.

    Parameters
    ----------
    f : (Ls,) or (Ls, W) real or complex tensor
    g : list of M filter dicts (from torch.filters.audfilters etc.)
    a : hop sizes — int, (M,), or (M, 2)
    L : DFT length (computed from Ls and a if omitted)
    stack : bool, default False
        If True, stack the per-channel coefficients into a single tensor
        instead of returning a ragged list. This requires a *uniform*
        filterbank (every channel the same length, i.e. a scalar hop ``a``);
        a ``ValueError`` is raised otherwise. Subsumes the former
        ``ufilterbank``.

    Returns
    -------
    c : list of M tensors (``stack=False``, default)
        Each (N_m,) for mono or (N_m, W) for multi-channel.
    c : Tensor (``stack=True``)
        Shape (N, M) for mono or (N, M, W) for multi-channel.

    Examples
    --------
    >>> import torch
    >>> from cool_frames.torch.filterbanks import filterbank
    >>> from cool_frames.torch.filters import audfilters
    >>> # Create a real signal
    >>> x = torch.randn(8000)
    >>> # Load auditory filters at 16 kHz (M=28 channels)
    >>> g = audfilters(16000)
    >>> # Analyze with hop size 64
    >>> c = filterbank(x, g, a=64)
    >>> len(c)
    28
    >>> c[0].shape[0] > 0
    True
    """
    mono = f.dim() == 1
    if mono:
        f = f.unsqueeze(1)
    Ls, W = f.shape
    device = f.device

    M = len(g)
    a_norm = _normalise_a(a, M)

    if L is None:
        L = filterbanklength(Ls, a_norm)

    # Zero-pad to length L
    if Ls < L:
        f_pad = torch.nn.functional.pad(f, (0, 0, 0, L - Ls))
    else:
        f_pad = f[:L]

    # Prepare filters and classify
    g_ready, m_td, m_fft, m_fftbl = _prepare_filters(g, a_norm, L, device)

    c: list[torch.Tensor | None] = [None] * M

    # Time-domain (FIR) path
    if m_td:
        g_td_list = [g_ready[m]["h"] for m in m_td]
        offset_list = np.array([g_ready[m].get("offset", 0) for m in m_td], dtype=int)
        a_td = a_norm[m_td, :]
        c_td = comp_filterbank_td(f_pad, g_td_list, a_td, offset_list)
        for k, m in enumerate(m_td):
            c[m] = c_td[k]  # type: ignore[assignment]

    # Full-length FFT path
    F = None
    if m_fft or m_fftbl:
        F = torch.fft.fft(f_pad, dim=0)

    if m_fft:
        G_fft = [g_ready[m]["H"] for m in m_fft]
        a_fft = a_norm[m_fft, :]
        c_fft = comp_filterbank_fft(F, G_fft, a_fft)  # type: ignore[arg-type]
        for k, m in enumerate(m_fft):
            c[m] = c_fft[k]  # type: ignore[assignment]

    # Band-limited FFT path
    if m_fftbl:
        G_bl = [g_ready[m]["H"] for m in m_fftbl]
        fo_bl = np.array([g_ready[m]["foff"] for m in m_fftbl], dtype=int)
        a_bl = a_norm[m_fftbl, :]
        ro_bl = np.array([g_ready[m].get("realonly", 0) for m in m_fftbl])

        # Skip zero-length filters
        valid = [k for k in range(len(m_fftbl)) if len(G_bl[k]) > 0]
        zero_k = [k for k in range(len(m_fftbl)) if len(G_bl[k]) == 0]

        if valid:
            G_v = [G_bl[k] for k in valid]
            fo_v = fo_bl[[k for k in valid]]
            a_v = a_bl[[k for k in valid], :]
            ro_v = ro_bl[[k for k in valid]]
            c_v = comp_filterbank_fftbl(F, G_v, fo_v, a_v, ro_v)  # type: ignore[arg-type]
            for i, k in enumerate(valid):
                c[m_fftbl[k]] = c_v[i]  # type: ignore[assignment]

        for k in zero_k:
            m = m_fftbl[k]
            Nm = max(1, round(L / (a_norm[m, 0] / a_norm[m, 1])))
            c[m] = torch.zeros(Nm, W, dtype=torch.complex128, device=device)  # type: ignore[assignment]

    # Squeeze mono
    if mono:
        c = [
            cm.squeeze(1) if cm is not None and cm.dim() == 2 and cm.shape[1] == 1 else cm
            for cm in c
        ]  # type: ignore[union-attr]

    if stack:
        lengths = {cm.shape[0] for cm in c}  # type: ignore[union-attr]
        if len(lengths) != 1:
            raise ValueError(
                "filterbank(..., stack=True) needs a uniform filterbank "
                "(all channels the same length); got channel lengths "
                f"{sorted(lengths)}. Pass a scalar hop `a`, or leave stack=False."
            )
        return torch.stack(c, dim=1)  # (N, M) or (N, M, W)  # type: ignore[arg-type]

    return c  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ifilterbank (synthesis)
# ---------------------------------------------------------------------------


def ifilterbank(
    c: list[torch.Tensor],
    g: list[dict],
    a,
    Ls: int | None = None,
    real: bool = True,
) -> torch.Tensor:
    """Synthesise a signal from filterbank coefficients.

    Parameters
    ----------
    c    : list of M tensors (N_m,) or (N_m, W)
    g    : list of M filter dicts
    a    : hop sizes
    Ls   : output length (trimmed from L)
    real : default ``True`` --- the correct mode for **real audio**, where the
           synthesis filters (canonical dual/tight of a single-sided real
           filterbank) cover only the positive frequencies, so the one-sided
           spectrum is mirrored via ``2 * real(ifft(F))``.  Pass ``real=False``
           **for complex / two-sided frames**.  The default matches
           ``filterbankdual``/``filterbanktight`` (also ``real=True``), so the
           common pipeline reconstructs exactly without passing any flag.  A
           convention mismatch (single-sided filters with ``real=False``, or
           two-sided with ``real=True``) is detected and warned about, since it
           otherwise reconstructs silently wrong.

    Returns
    -------
    f : (Ls,) or (Ls, W) reconstructed tensor

    Examples
    --------
    >>> import torch
    >>> from cool_frames.torch.filterbanks import filterbank, ifilterbank, filterbankdual
    >>> from cool_frames.torch.filters import audfilters
    >>> # Analyze a signal
    >>> x = torch.randn(8000)
    >>> g, a, _, L, _ = audfilters(16000, 8000)
    >>> c = filterbank(x, g, a, L)
    >>> # Reconstruct using dual window (real=True default)
    >>> x_recon = ifilterbank(c, filterbankdual(g, a, L), a, 8000)
    >>> x_recon.shape
    torch.Size([8000])
    """
    M = len(c)
    a_norm = _normalise_a(a, M)
    device = c[0].device

    # Infer L from first coefficient
    afrac = a_norm[:, 0] / a_norm[:, 1]
    c0 = c[0]
    N0 = c0.shape[0]
    L = int(round(N0 * afrac[0]))

    # Ensure 2D
    c2d = [cm.unsqueeze(1) if cm.dim() == 1 else cm for cm in c]
    W = c2d[0].shape[1]

    # Prepare filters
    g_ready, m_td, m_fft, m_fftbl = _prepare_filters(g, a_norm, L, device)

    # Time-domain (FIR) synthesis path
    if m_td:
        g_td_list = [g_ready[m]["h"] for m in m_td]
        offset_list = np.array([g_ready[m].get("offset", 0) for m in m_td], dtype=int)
        a_td = a_norm[m_td, :]
        c_sub = [c2d[m] for m in m_td]
        out_td = comp_ifilterbank_td(
            c_sub, g_td_list, a_td, Ls if Ls is not None else L, offset_list
        )
        if out_td.dim() == 1:
            out_td = out_td.unsqueeze(1)
        if W == 1:
            out_td = out_td.squeeze(1)
        return out_td

    F = torch.zeros(L, W, dtype=c2d[0].dtype, device=device)

    # Full-length FFT synthesis
    if m_fft:
        G_fft = [g_ready[m]["H"] for m in m_fft]
        a_fft = a_norm[m_fft, :]
        c_sub = [c2d[m] for m in m_fft]
        F_fft = comp_ifilterbank_fft(c_sub, G_fft, a_fft, L)
        if F_fft.dim() == 1:
            F_fft = F_fft.unsqueeze(1)
        F = F + F_fft

    # Band-limited FFT synthesis
    if m_fftbl:
        G_bl = [g_ready[m]["H"] for m in m_fftbl]
        fo_bl = np.array([g_ready[m]["foff"] for m in m_fftbl], dtype=int)
        a_bl = a_norm[m_fftbl, :]
        ro_bl = np.array([g_ready[m].get("realonly", 0) for m in m_fftbl])

        valid = [k for k in range(len(m_fftbl)) if len(G_bl[k]) > 0]
        if valid:
            c_sub = [c2d[m_fftbl[k]] for k in valid]
            G_v = [G_bl[k] for k in valid]
            fo_v = fo_bl[[k for k in valid]]
            a_v = a_bl[[k for k in valid], :]
            ro_v = ro_bl[[k for k in valid]]
            F_bl = comp_ifilterbank_fftbl(c_sub, G_v, fo_v, a_v, ro_v, L)
            if F_bl.dim() == 1:
                F_bl = F_bl.unsqueeze(1)
            F = F + F_bl

    # Detect an obvious analysis/synthesis convention mismatch (cheap, on F).
    # A single-sided (real-audio) frame has energy only on the positive-frequency
    # half; a two-sided (complex) frame fills both halves. Warn when ``real`` does
    # not match the apparent frame -- the mismatch reconstructs silently wrong.
    if L >= 8:
        half = L // 2
        F_chk = F.detach()
        f_pos = float(torch.linalg.vector_norm(F_chk[1:half]))
        f_neg = float(torch.linalg.vector_norm(F_chk[half + 1 :]))
        # single-sided (real-audio) frames carry little negative-half energy
        # (empirically f_neg/f_pos ~ 0.1); two-sided (complex) frames carry
        # comparable energy on both halves (~1.0). Thresholds 0.3 / 0.7 separate
        # them with a wide margin, so the warning fires only on a clear mismatch.
        if real and f_neg > 0.7 * f_pos:
            warnings.warn(
                "ifilterbank(real=True) but the synthesis filters appear two-sided "
                "(comparable negative-frequency energy); folding will double-count. "
                "Pass real=False for complex/two-sided frames.",
                stacklevel=2,
            )
        elif (not real) and f_pos > 0.0 and f_neg < 0.3 * f_pos:
            warnings.warn(
                "ifilterbank(real=False) but the synthesis filters appear single-sided "
                "(little negative-frequency energy); this will not reconstruct a real "
                "signal. Pass real=True (the default) for real-audio frames.",
                stacklevel=2,
            )

    # Inverse FFT
    if real:
        out = 2.0 * torch.fft.ifft(F, dim=0).real
    else:
        out = torch.fft.ifft(F, dim=0)

    # Trim to Ls
    if Ls is not None and Ls <= L:
        out = out[:Ls]

    # Squeeze mono
    if W == 1:
        out = out.squeeze(1)

    return out  # type: ignore[no-any-return]
