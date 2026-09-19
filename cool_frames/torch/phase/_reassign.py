"""
torch/phase/_reassign.py
========================
Spectral reassignment and synchrosqueezing for non-uniform filterbanks (PyTorch).

Ports `comp_filterbankreassign`, `filterbankreassign`, and `filterbanksynchrosqueeze`
from the NumPy backend. All operations are differentiable with respect to the input
magnitudes/energies.

MATLAB originals
----------------
  layer3/reassignment/comp_filterbankreassign.m
  (synchrosqueezing is a frequency-only reassignment variant)
"""

from __future__ import annotations

from typing import Any

import torch

from ...numpy.filterbanks._utils import normalise_a
from ...numpy.filters._design import filterbanklength
from ._phasegrad import filterbankphasegrad


def _centre_frequencies(g, a_norm, L: int, device) -> torch.Tensor:
    """Normalised centre frequency of every channel, in [0, 2) with 2 = fs.

    The circular mean of the DFT frequency weighted by ``|H_m|``, as LTFAT's
    ``cent_freqs`` computes it, in float64.  The arithmetic mean over [0, 2)
    that this replaces put any channel whose support wraps around 0 -- the
    DC complement, with bins near 0 *and* near L -- at about 1, i.e. at
    Nyquist, so its energy was reassigned across the whole band.

    The kernel expects the channels in ascending order with the wrap-around
    after the last one.  A DC complement whose centroid lands a hair below 0
    (just under 2 once wrapped) would put that wrap between channels 0 and 1
    and send its energy to the Nyquist channel, so such a channel is set to 0.
    """
    import math

    from ..filterbanks._frame import filterbankfreqz

    H = filterbankfreqz(g, a_norm, L).detach().to(device="cpu", dtype=torch.complex128)
    k = torch.arange(L, dtype=torch.float64) / L * 2.0
    z = torch.exp(1j * math.pi * k).to(torch.complex128) @ H.abs().to(torch.complex128)
    fc = torch.remainder(torch.angle(z) / math.pi, 2.0)
    if fc.numel() > 1 and fc[0] > fc[1] and 2.0 - fc[0] < fc[1]:
        fc[0] = 0.0
    return fc.to(device=device)


def _length_from_coefficients(s, a_norm) -> int:
    """Transform length implied by the subband lengths and hops (L = N_m a_m)."""
    afrac = [float(a_norm[m, 0]) / float(a_norm[m, 1]) for m in range(len(s))]
    lengths = [int(torch.as_tensor(sm).numel()) * am for sm, am in zip(s, afrac)]
    L = int(round(lengths[0]))
    if any(abs(Lm - L) > 0.5 for Lm in lengths):
        raise ValueError(
            "filterbankreassign: the subband lengths and hops do not agree on one "
            f"transform length (N_m * a_m ranges over {min(lengths):g}..{max(lengths):g}); "
            "pass the normalised centre frequencies instead of the filters"
        )
    return L


def comp_filterbankreassign(
    s: list[torch.Tensor],
    tgrad: list[torch.Tensor],
    fgrad: list[torch.Tensor],
    a,
    cfreq: torch.Tensor,
    return_repos: bool = False,
):
    """Port of ``comp_filterbankreassign.m``.

    Each coefficient in subband m, time index n is accumulated into the
    subband whose normalised centre frequency is closest to
    ``tgrad[m][n]`` (the absolute normalised instantaneous frequency, wrap
    around 2), and placed at the time index closest to
    ``fgrad[m][n] + a[m]*n`` (mod N[target]).

    All operations are differentiable with respect to the magnitudes in ``s``.

    Parameters
    ----------
    s       : list of M energy tensors (``|c[m]|^2`` or similar)
    tgrad   : list of M instantaneous-frequency tensors
    fgrad   : list of M group-delay tensors
    a       : hop sizes (M,) or (M,2) — will be normalized
    cfreq   : (M,) normalised centre frequencies in [0, 2), as tensor or array
    return_repos : if True also return a ``repos`` list

    Returns
    -------
    sr      : list of M reassigned tensors (differentiable)
    repos   : list of lists of source indices (only if return_repos=True)
    Lc      : list of M subband lengths
    """
    M = len(s)
    a_norm = normalise_a(a, M)
    afrac = a_norm[:, 0].astype(float) / a_norm[:, 1].astype(float)
    Lc = [s[m].numel() for m in range(M)]

    # Convert cfreq to tensor if needed
    if not isinstance(cfreq, torch.Tensor):
        cfreq = torch.tensor(cfreq, dtype=torch.float32, device=s[0].device)
    else:
        cfreq = cfreq.to(device=s[0].device, dtype=torch.float32)

    # Wrap cfreq to [0, 2)
    cfreq2 = torch.fmod(cfreq, 2.0)

    # Initialize output tensors with same dtype/device as input
    device = s[0].device
    dtype = s[0].dtype
    sr = [torch.zeros(Lc[m], dtype=dtype, device=device) for m in range(M)]

    if return_repos:
        chan_pos = [0]
        for m in range(M):
            chan_pos.append(chan_pos[-1] + Lc[m])
        repos: list[list[int]] | None = [[] for _ in range(chan_pos[-1])]
    else:
        repos = None

    # Process channels in reverse order
    for mm in range(M - 1, -1, -1):
        sm_arr = s[mm].reshape(-1)
        tg_arr = tgrad[mm].reshape(-1)
        fg_arr = fgrad[mm].reshape(-1)
        cfreqm = float(cfreq2[mm])
        am = afrac[mm]

        for jj in range(Lc[mm]):
            # ``tgrad`` is the absolute normalised instantaneous frequency
            # (as filterbankphasegrad returns it); LTFAT's kernel expects the
            # deviation from the channel centre, wrapped to [-1, 1).  Adding
            # ``cfreqm`` to the absolute value, as this used to, reassigned
            # every coefficient to about twice its frequency.
            dev = float((float(tg_arr[jj]) - cfreqm + 1.0) % 2.0 - 1.0)
            tgradmjj = cfreqm + dev
            oldtgrad = 10.0

            if dev > 0:
                pos = mm
                for ii in range(mm, M):
                    pos = ii
                    tmptgrad = float(cfreq2[ii]) - tgradmjj
                    if tmptgrad >= 0:
                        tgradIdx = pos if abs(tmptgrad) < abs(oldtgrad) else pos - 1
                        break
                    oldtgrad = tmptgrad
                else:
                    # Wrapped around
                    for ii in range(0, mm + 1):
                        pos = ii
                        tmptgrad = float(cfreq2[ii]) - tgradmjj + 2.0
                        if tmptgrad >= 0:
                            tgradIdx = pos if abs(tmptgrad) < abs(oldtgrad) else pos - 1
                            break
                        oldtgrad = tmptgrad
                    else:
                        tgradIdx = mm

                if tgradIdx < 0:
                    tgradIdx = M - 1
            else:
                pos = mm
                for ii in range(mm, -1, -1):
                    pos = ii
                    tmptgrad = float(cfreq2[ii]) - tgradmjj
                    if tmptgrad <= 0:
                        tgradIdx = pos if abs(tmptgrad) < abs(oldtgrad) else pos + 1
                        break
                    oldtgrad = tmptgrad
                else:
                    for ii in range(M - 1, mm - 1, -1):
                        pos = ii
                        tmptgrad = float(cfreq2[ii]) - tgradmjj - 2.0
                        if tmptgrad <= 0:
                            tgradIdx = pos if abs(tmptgrad) < abs(oldtgrad) else pos + 1
                            break
                        oldtgrad = tmptgrad
                    else:
                        tgradIdx = mm

                if tgradIdx >= M:
                    tgradIdx = 0

            # Clamp
            tgradIdx = max(0, min(M - 1, tgradIdx))
            at_idx = afrac[tgradIdx]
            Lt_idx = Lc[tgradIdx]

            # Compute fgradIdx with bounds checking
            try:
                fgradIdx_raw = int(
                    torch.fmod(
                        torch.tensor(round(float((fg_arr[jj] + am * jj) / at_idx))),
                        torch.tensor(Lt_idx),
                    ).item()
                )
            except (IndexError, ValueError):
                fgradIdx_raw = 0

            # Ensure fgradIdx is within bounds [0, Lt_idx)
            if fgradIdx_raw >= Lt_idx or fgradIdx_raw < 0:
                fgradIdx_raw = int(fgradIdx_raw % Lt_idx)
            fgradIdx = max(0, min(Lt_idx - 1, fgradIdx_raw))

            # Differentiable accumulation
            sr[tgradIdx][fgradIdx] = sr[tgradIdx][fgradIdx] + sm_arr[jj]

            if return_repos:
                assert repos is not None
                src_flat = int(chan_pos[mm]) + jj
                dst_flat = int(chan_pos[tgradIdx]) + fgradIdx
                repos[dst_flat].append(src_flat)

    if return_repos:
        assert repos is not None
        # Flatten repos into a single concatenated tensor
        repos_flat = torch.cat(
            [
                torch.tensor(r, dtype=torch.long, device=device)
                if len(r) > 0
                else torch.tensor([], dtype=torch.long, device=device)
                for r in repos
            ]
        )
        return sr, repos_flat, Lc
    return sr, Lc


def filterbankreassign(
    f, g, a=None, L: int | None = None, fc: torch.Tensor | None = None, return_repos: bool = False
):
    """Spectral reassignment of filterbank coefficients.

    Can be called in two ways:
    1. filterbankreassign(signal, g_filters, a_hops, L, fc, return_repos)
    2. filterbankreassign(magnitudes, tgrad, fgrad, a_hops, fc) – always returns (sr, repos, Lc)

    Parameters
    ----------
    f          : signal (Ls,) or list of magnitude tensors
    g          : list of M filter dicts, OR tgrad list (if f is magnitudes)
    a          : hop sizes (if g is filters) or fgrad (if g is tgrad)
    L          : DFT length or a_hops (if f is magnitudes with pre-computed gradients)
    fc         : (M,) normalised centre frequencies or filter cell
    return_repos : whether to return repositioning table (signal mode only)

    Returns
    -------
    When called with pre-computed magnitudes:
        sr    : list of M reassigned energy tensors
        repos : repositioning tensor
        Lc    : list of M subband lengths

    When called with signal:
        sr    : list of M reassigned energy tensors
        repos : repositioning tensor (only if return_repos=True)
        Lc    : list of M subband lengths (only if return_repos=True)
    """
    # Detect if f is a list (pre-computed magnitudes) vs signal tensor
    f_is_list = isinstance(f, (list, tuple))

    if f_is_list:
        # Pre-computed case: f=magnitudes, g=tgrad, a=fgrad, L=a_hops, fc=fc_or_g
        s = [torch.as_tensor(fi) for fi in f]
        tgrad = [torch.as_tensor(gi) for gi in g]
        fgrad = [torch.as_tensor(ai) for ai in a]
        # Deliberately dynamic: in this call convention ``L`` carries the hop
        # sizes, and the signal branch below rebinds this to the (M, 2) array
        # from ``normalise_a``. ``comp_filterbankreassign`` takes ``a`` untyped
        # for the same reason.
        a_hops: Any = L
        M = len(s)

        # fc can be frequencies or filter cell
        if isinstance(fc, (list, tuple)) and len(fc) > 0 and isinstance(fc[0], dict):
            # fc is the filter cell: take each channel's centre from its
            # response, at the length the subbands and hops imply.  (This
            # used to fall back to M evenly spaced frequencies, which is
            # right for no non-uniform bank.)
            a_norm_pc = normalise_a(a_hops, M)
            fc_arr = _centre_frequencies(
                list(fc), a_norm_pc, _length_from_coefficients(s, a_norm_pc), s[0].device
            )
        elif fc is None:
            # No fc provided; default to evenly spaced
            import numpy as np

            fc_arr = torch.tensor(np.arange(M, dtype=float) / M * 2.0, device=s[0].device)
        else:
            fc_arr = torch.as_tensor(fc, device=s[0].device)

        # Pre-computed path ALWAYS returns (sr, repos, Lc)
        return_repos = True
    else:
        # Signal case: compute gradients
        f = torch.as_tensor(f)
        M = len(g)
        if a is None:
            raise ValueError("filterbankreassign: a (hop sizes) is required when f is a signal")

        a_norm = normalise_a(a, M)

        if L is None:
            L = filterbanklength(len(f), a_norm)  # type: ignore[assignment]

        tgrad, fgrad, s, _c = filterbankphasegrad(f, g, a_norm, L)
        a_hops = a_norm

        # Compute normalised centre frequencies if not provided
        if fc is None:
            fc_arr = _centre_frequencies(g, a_norm, int(L), f.device)
        elif isinstance(fc, (list, tuple)) and len(fc) > 0 and isinstance(fc[0], dict):
            # fc is actually a filter cell g; compute from filters
            fc_arr = _centre_frequencies(list(fc), a_norm, int(L), f.device)
        else:
            fc_arr = torch.as_tensor(fc, device=f.device)

    result = comp_filterbankreassign(s, tgrad, fgrad, a_hops, fc_arr, return_repos=return_repos)
    if return_repos:
        return result  # (sr, repos, Lc)
    else:
        return result[0]  # Just sr


def filterbanksynchrosqueeze(
    f, g, a=None, L: int | None = None, fc: torch.Tensor | None = None, return_repos: bool = False
):
    """Synchrosqueezing (frequency-only reassignment).

    .. deprecated::
        ``filterbanksynchrosqueeze`` is deprecated and will be removed in a
        future release.  It is equivalent to :func:`filterbankreassign` with
        the group delay (``fgrad``) zeroed out, so coefficients move in
        frequency only.  Use :func:`filterbankreassign` directly
        for all reassignment tasks.

    Can be called in two ways:
    1. filterbanksynchrosqueeze(signal, g_filters, a_hops, L, fc, return_repos)
    2. filterbanksynchrosqueeze(coefficients, tgrad, fgrad, a_hops, fc, return_repos)

    Parameters
    ----------
    Same as :func:`filterbankreassign`.

    Returns
    -------
    sr    : list of M reassigned energy tensors
    repos : repositioning tensor (only if return_repos=True)
    Lc    : list of M subband lengths (only if return_repos=True)
    """
    import warnings

    warnings.warn(
        "filterbanksynchrosqueeze is deprecated and will be removed in a "
        "future release. Use filterbankreassign instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Detect if f is a list (pre-computed data) vs signal tensor
    f_is_list = isinstance(f, (list, tuple))

    if f_is_list:
        # Pre-computed case: f=coefficients, g=tgrad, a=fgrad, L=a_hops, fc=fc_or_g
        c = [torch.as_tensor(fi) for fi in f]
        tgrad = [torch.as_tensor(gi) for gi in g]
        fgrad = [torch.as_tensor(ai) for ai in a]
        # Deliberately dynamic: in this call convention ``L`` carries the hop
        # sizes, and the signal branch below rebinds this to the (M, 2) array
        # from ``normalise_a``. ``comp_filterbankreassign`` takes ``a`` untyped
        # for the same reason.
        a_hops: Any = L
        M = len(c)

        # fc can be frequencies or filter cell
        if isinstance(fc, (list, tuple)) and len(fc) > 0 and isinstance(fc[0], dict):
            # fc is the filter cell: take each channel's centre from its
            # response, at the length the subbands and hops imply.  (This
            # used to fall back to M evenly spaced frequencies, which is
            # right for no non-uniform bank.)
            a_norm_pc = normalise_a(a_hops, M)
            fc_arr = _centre_frequencies(
                list(fc), a_norm_pc, _length_from_coefficients(c, a_norm_pc), c[0].device
            )
        elif fc is None:
            # No fc provided; default to evenly spaced
            import numpy as np

            fc_arr = torch.tensor(np.arange(M, dtype=float) / M * 2.0, device=c[0].device)
        else:
            fc_arr = torch.as_tensor(fc, device=c[0].device)

        # Compute energy
        s = [torch.abs(torch.as_tensor(ci)) ** 2 for ci in c]
        # Pre-computed path ALWAYS returns (sr, repos, Lc)
        return_repos = True
    else:
        # Signal case: compute gradients
        f = torch.as_tensor(f)
        M = len(g)
        if a is None:
            raise ValueError(
                "filterbanksynchrosqueeze: a (hop sizes) is required when f is a signal"
            )

        a_norm = normalise_a(a, M)

        if L is None:
            L = filterbanklength(len(f), a_norm)  # type: ignore[assignment]

        tgrad, fgrad, s, c = filterbankphasegrad(f, g, a_norm, L)
        a_hops = a_norm

        if fc is None:
            fc_arr = _centre_frequencies(g, a_norm, int(L), f.device)
        elif isinstance(fc, (list, tuple)) and len(fc) > 0 and isinstance(fc[0], dict):
            fc_arr = _centre_frequencies(list(fc), a_norm, int(L), f.device)
        else:
            fc_arr = torch.as_tensor(fc, device=f.device)

    # Synchrosqueezing is reassignment in frequency only: keep each
    # coefficient's time position (zero group delay ``fgrad``) and move it by
    # its instantaneous frequency ``tgrad``.  This used to zero ``tgrad``
    # instead, which gave time-only reassignment.
    fgrad_zero = [torch.zeros_like(fg) for fg in fgrad]

    result = comp_filterbankreassign(
        s, tgrad, fgrad_zero, a_hops, fc_arr, return_repos=return_repos
    )
    if return_repos:
        return result  # (sr, repos, Lc)
    else:
        return result[0]  # Just sr
