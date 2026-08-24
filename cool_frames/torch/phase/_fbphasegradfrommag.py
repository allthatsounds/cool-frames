"""
torch/phase/_fbphasegradfrommag.py
==================================
Phase gradient estimation from magnitude for **non-uniform** filterbanks (PyTorch).

Ports ``numpy/phase/_fbphasegradfrommag.py`` to fully differentiable torch operations.

For non-uniform filterbanks the channels may have different lengths and hop sizes,
so a neighbor-lookup structure (NEIGH, posInfo) is precomputed to map each
coefficient to its frequency neighbors.
"""

from __future__ import annotations

import numpy as np
import torch

# ---------------------------------------------------------------------------
# pderiv – periodic discrete derivative (order 2 central differences)
# ---------------------------------------------------------------------------


def _pderiv(f: torch.Tensor, difforder: int = 2) -> torch.Tensor:
    """Periodic derivative of a 1-D tensor using central differences.

    Port of ``pderiv.m`` (1-D case only).
    The result is scaled by *L* (the length of *f*), matching MATLAB
    where ``pderiv(f, 1, difforder)`` returns ``L * (shift-based diff)``.

    Parameters
    ----------
    f : torch.Tensor, shape (L,)
    difforder : int, {2, 4}

    Returns
    -------
    df : torch.Tensor, shape (L,)
    """
    L = f.shape[0]
    if difforder == 2:
        return L * (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / 2.0
    elif difforder == 4:
        return (
            L
            * (
                -torch.roll(f, -2, dims=0)
                + 8 * torch.roll(f, -1, dims=0)
                - 8 * torch.roll(f, 1, dims=0)
                + torch.roll(f, 2, dims=0)
            )
            / 12.0
        )
    else:
        raise ValueError(f"pderiv: difforder must be 2 or 4, got {difforder}")


# ---------------------------------------------------------------------------
# comp_filterbankneighbors – neighbour lookup for non-uniform filterbanks
# ---------------------------------------------------------------------------


def comp_filterbankneighbors(
    a: np.ndarray | torch.Tensor,
    M: int,
    N: np.ndarray | torch.Tensor,
    do_real: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the neighbour structure for a non-uniform filterbank.

    Port of ``comp_filterbankneighbors.m``.

    Parameters
    ----------
    a : (M,) int array/tensor — hop sizes per channel
    M : int — number of channels
    N : (M,) int array/tensor — number of time frames per channel
    do_real : bool — True for real (non-cyclic) filterbanks

    Returns
    -------
    NEIGH : (6, Nsum) int tensor
        Neighbour indices (0-based).  Rows:
        row 0 — left time neighbour (same channel),
        row 1 — right time neighbour (same channel, unused in MATLAB but reserved),
        rows 2–3 — below (lower channel) neighbour range [low, high],
        rows 4–5 — above (upper channel) neighbour range [low, high].
        A value of -1 means "no neighbour".

    posInfo : (2, Nsum) float tensor
        Row 0: channel index (0-based).
        Row 1: time position (sample offset = frame_index * a[m]).

    Note
    ----
    The MATLAB version uses 1-based indices.  We use **0-based** and encode
    "no neighbour" as -1 (MATLAB uses 0 after the ``NEIGH = NEIGH - 1``
    adjustment that happens in ``filterbankconstphase.m``).
    """
    # Convert to numpy for computation, then convert back to torch
    a_np = np.asarray(a, dtype=int).ravel()
    N_np = np.asarray(N, dtype=int).ravel()
    assert len(a_np) == M and len(N_np) == M

    Nsum = int(np.sum(N_np))
    chanStart = np.concatenate([[0], np.cumsum(N_np)])  # length M+1

    # Initialise to -1 (no neighbour)
    NEIGH = -np.ones((6, Nsum), dtype=int)

    LIM = 0.8  # time-distance limit for vertical neighbours

    # --- Horizontal (time) neighbours within each channel ---
    for kk in range(M):
        cs = chanStart[kk]
        Nk = N_np[kk]
        if Nk >= 2:
            # left neighbour (row 0): first sample wraps? MATLAB sets to idx+1
            NEIGH[0, cs] = cs + 1
            NEIGH[0, cs + Nk - 1] = cs + Nk - 2
            for nn in range(1, Nk - 1):
                NEIGH[0, cs + nn] = cs + nn - 1  # left
                NEIGH[1, cs + nn] = cs + nn + 1  # right

    # --- Vertical (frequency) neighbours: one channel above ---
    for kk in range(M - 1):
        cs = chanStart[kk]
        cs_above = chanStart[kk + 1]
        aTemp = a_np[kk] / a_np[kk + 1]
        n_idx = np.arange(N_np[kk])
        POSlow = cs_above + np.clip(
            np.ceil((n_idx - LIM) * aTemp).astype(int), 0, N_np[kk + 1] - 1
        )
        POShigh = cs_above + np.clip(
            np.floor((n_idx + LIM) * aTemp).astype(int), 0, N_np[kk + 1] - 1
        )
        NEIGH[4, cs : cs + N_np[kk]] = POSlow
        NEIGH[5, cs : cs + N_np[kk]] = POShigh

    # Wrap-around for complex (non-real) filterbanks
    if not do_real:
        cs_last = chanStart[M - 1]
        aTemp = a_np[M - 1] / a_np[0]
        n_idx = np.arange(N_np[M - 1])
        POSlow = chanStart[0] + np.clip(np.ceil((n_idx - LIM) * aTemp).astype(int), 0, N_np[0] - 1)
        POShigh = chanStart[0] + np.clip(
            np.floor((n_idx + LIM) * aTemp).astype(int), 0, N_np[0] - 1
        )
        NEIGH[4, cs_last : cs_last + N_np[M - 1]] = POSlow
        NEIGH[5, cs_last : cs_last + N_np[M - 1]] = POShigh

    # Where low == high, clear the high entry (single neighbour only)
    mask_eq = NEIGH[5, :] == NEIGH[4, :]
    NEIGH[5, mask_eq] = -1

    # --- Vertical (frequency) neighbours: one channel below ---
    for kk in range(1, M):
        cs = chanStart[kk]
        cs_below = chanStart[kk - 1]
        aTemp = a_np[kk] / a_np[kk - 1]
        n_idx = np.arange(N_np[kk])
        POSlow = cs_below + np.clip(
            np.ceil((n_idx - LIM) * aTemp).astype(int), 0, N_np[kk - 1] - 1
        )
        POShigh = cs_below + np.clip(
            np.floor((n_idx + LIM) * aTemp).astype(int), 0, N_np[kk - 1] - 1
        )
        NEIGH[2, cs : cs + N_np[kk]] = POSlow
        NEIGH[3, cs : cs + N_np[kk]] = POShigh

    if not do_real:
        cs_first = chanStart[0]
        aTemp = a_np[0] / a_np[M - 1]
        n_idx = np.arange(N_np[0])
        POSlow = chanStart[M - 1] + np.clip(
            np.ceil((n_idx - LIM) * aTemp).astype(int), 0, N_np[M - 1] - 1
        )
        POShigh = chanStart[M - 1] + np.clip(
            np.floor((n_idx + LIM) * aTemp).astype(int), 0, N_np[M - 1] - 1
        )
        NEIGH[2, cs_first : cs_first + N_np[0]] = POSlow
        NEIGH[3, cs_first : cs_first + N_np[0]] = POShigh

    mask_eq2 = NEIGH[3, :] == NEIGH[2, :]
    NEIGH[3, mask_eq2] = -1

    # --- posInfo ---
    posInfo = np.zeros((2, Nsum))
    for kk in range(M):
        cs = chanStart[kk]
        posInfo[0, cs : cs + N_np[kk]] = kk
        posInfo[1, cs : cs + N_np[kk]] = np.arange(N_np[kk]) * a_np[kk]

    # Convert to torch
    NEIGH_t = torch.from_numpy(NEIGH).to(torch.int64)
    posInfo_t = torch.from_numpy(posInfo).to(torch.float32)

    return NEIGH_t, posInfo_t


# ---------------------------------------------------------------------------
# comp_filterbankphasegradfrommag – main algorithm (fully torch-based)
# ---------------------------------------------------------------------------


def comp_filterbankphasegradfrommag(
    abss: torch.Tensor,
    N: np.ndarray | torch.Tensor,
    a: np.ndarray | torch.Tensor,
    M: int,
    sqtfr: np.ndarray | torch.Tensor,
    fc: np.ndarray | torch.Tensor,
    NEIGH: torch.Tensor,
    posInfo: torch.Tensor,
    gderivweight: float = 0.5,
    do_tfrdiff: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Phase gradient estimation from magnitude for a non-uniform filterbank.

    Port of ``comp_filterbankphasegradfrommag.m`` (fully differentiable).

    Parameters
    ----------
    abss : torch.Tensor, shape (Nsum,)
        Magnitude vector (all channels concatenated).
    N : (M,) int array/tensor
        Number of frames per channel.
    a : (M,) hop sizes per channel (array or tensor).
    M : int
        Number of channels.
    sqtfr : (M,) sqrt of time-frequency ratios (array or tensor).
    fc : (M,) normalised centre frequencies (array or tensor).
    NEIGH : (6, Nsum) int tensor
        Neighbour indices (0-based, -1 = no neighbour).
    posInfo : (2, Nsum) float tensor
        Position info [channel; time_position].
    gderivweight : float
        Weight for tfr-difference correction (default 0.5).
    do_tfrdiff : bool
        Whether to include sqtfr-difference correction.

    Returns
    -------
    tgrad : torch.Tensor, shape (Nsum,)
        Time-direction phase gradient.
    fgrad : torch.Tensor, shape (Nsum,)
        Frequency-direction phase gradient.
    logs : torch.Tensor, shape (Nsum,)
        Log-magnitude.
    """
    abss = abss.ravel()
    N_np = np.asarray(N, dtype=int).ravel()
    a_np = np.asarray(a, dtype=float).ravel()
    sqtfr_t = torch.as_tensor(sqtfr, dtype=abss.dtype, device=abss.device).ravel()
    fc_t = torch.as_tensor(fc, dtype=abss.dtype, device=abss.device).ravel()

    Nsum = int(np.sum(N_np))
    assert abss.shape[0] == Nsum
    assert NEIGH.shape == (6, Nsum)
    assert posInfo.shape == (2, Nsum)

    L = int(a_np[0] * N_np[0])  # transform length
    difforder = 2
    fac = gderivweight

    logs = torch.log(abss + torch.finfo(abss.dtype).tiny)

    # ----- fgrad: frequency-direction phase gradient -----
    # Per-channel periodic derivative of log-magnitude along time
    fgrad = torch.zeros_like(abss)
    chanStart = 0
    for m in range(M):
        Nm = N_np[m]
        idx = slice(chanStart, chanStart + Nm)
        fgrad[idx] = _pderiv(logs[idx], difforder) / Nm
        chanStart += Nm

    # ----- tgrad: time-direction phase gradient -----
    # Uses frequency neighbours to estimate the frequency derivative of
    # log-magnitude, weighted by channel distance and inverse tfr.
    tgrad = torch.zeros_like(abss)

    chanStart = 0
    for m in range(M):
        Nm = N_np[m]
        denom = sqtfr_t[m] ** 2 * (np.pi * L)

        # Precompute above/below nom/denom
        aboveNom: torch.Tensor | float = 0.0
        aboveDenom = 1.0
        belowNom: torch.Tensor | float = 0.0
        belowDenom = 1.0

        if m < M - 1:
            if do_tfrdiff:
                aboveNom = fac * (sqtfr_t[m + 1] - sqtfr_t[m]) / sqtfr_t[m]
            aboveDenom = float((fc_t[m + 1] - fc_t[m]).item())

        if m > 0:
            if do_tfrdiff:
                belowNom = fac * (sqtfr_t[m] - sqtfr_t[m - 1]) / sqtfr_t[m]
            belowDenom = float((fc_t[m] - fc_t[m - 1]).item())

        temp = torch.zeros(Nm, dtype=abss.dtype, device=abss.device)
        for n in range(Nm):
            w = chanStart + n

            # --- Above neighbours (rows 4, 5 of NEIGH) ---
            tempValAbove = torch.tensor(0.0, dtype=abss.dtype, device=abss.device)
            numNeighAbove = 0
            for jj in [4, 5]:
                neigh = int(NEIGH[jj, w].item())
                if neigh >= 0:
                    numNeighAbove += 1
                    dist = (posInfo[1, neigh] - posInfo[1, w]) / a_np[m]  # type: ignore[index]
                    tempValAbove += logs[neigh] - logs[w] - dist * fgrad[w]  # type: ignore[index]
            if numNeighAbove > 0:
                tempValAbove /= numNeighAbove

            # --- Below neighbours (rows 2, 3 of NEIGH) ---
            tempValBelow = torch.tensor(0.0, dtype=abss.dtype, device=abss.device)
            numNeighBelow = 0
            for jj in [2, 3]:
                neigh = int(NEIGH[jj, w].item())
                if neigh >= 0:
                    numNeighBelow += 1
                    dist = (posInfo[1, neigh] - posInfo[1, w]) / a_np[m]  # type: ignore[index]
                    # Sign: both branches adjust the neighbour to the centre
                    # coefficient's time instant, but they form the difference
                    # in opposite orders.  The above branch computes
                    # ``logs[neigh] - logs[w]`` and subtracts ``dist*fgrad[w]``;
                    # here the difference runs the other way, so the same
                    # correction has to be ADDED.
                    #
                    # This backend kept the wrong sign after the NumPy copy was
                    # fixed -- the same asymmetry that left the centre-frequency
                    # term behind for a whole release, and for the same reason:
                    # nothing internal calls this function.  It cost 1.53e-02 in
                    # tgrad on *interior* channels, which is what
                    # test_phase_gradient_estimators_agree_between_backends
                    # exists to catch.
                    tempValBelow += logs[w] - logs[neigh] + dist * fgrad[w]  # type: ignore[index]
            if numNeighBelow > 0:
                tempValBelow /= numNeighBelow

            # Each side is a one-sided difference quotient estimating the same
            # frequency-derivative of the log-magnitude.  Interior channels have
            # both; the DC and Nyquist complements have only one, and summing
            # the available sides leaves those two at half the interior
            # scaling.  That is what LTFAT does, and measuring both against the
            # exact signal path showed it is also the more accurate choice on
            # the DC channel (1.4-2.0x lower error, five of five probes across
            # three designers).  This backend has no `edge_mode` switch: it
            # tracks the NumPy default, which is now 'ltfat'.  See
            # ``cool_frames/numpy/phase/_fbphasegradfrommag.py`` for the table.
            sides = []
            if m < M - 1:
                sides.append((tempValAbove + aboveNom) / aboveDenom)
            if m > 0:
                sides.append((tempValBelow + belowNom) / belowDenom)
            if sides:
                acc = sides[0]
                for extra in sides[1:]:
                    acc = acc + extra
                temp[n] = acc
            else:
                temp[n] = 0.0

        # ``fc`` is a *normalised* centre frequency in [0, 2] (2 == fs).  The
        # difference quotient above estimates the *deviation* of the
        # instantaneous frequency from the channel's centre frequency, while the
        # heap integrator consumes the absolute value.  Omitting this term left
        # tgrad off by the centre frequency itself.
        #
        # This backend kept the defect for a whole release after the NumPy one
        # was fixed, because nothing internal calls it — the torch
        # ``filterbankconstphase`` delegates to NumPy — so no test went red.  On
        # a 660 Hz tone at fs = 4000 it returned -0.0509 where the truth is
        # 0.3300.  The parity test now pins the two backends together.
        tgrad[chanStart : chanStart + Nm] = temp / denom + fc_t[m]
        chanStart += Nm

    # ----- Scale fgrad by tfr² / (2π) * N(m) -----
    chanStart = 0
    for m in range(M):
        Nm = N_np[m]
        idx = slice(chanStart, chanStart + Nm)
        fgrad[idx] = fgrad[idx] * sqtfr_t[m] ** 2 / (2 * np.pi) * Nm
        chanStart += Nm

    return tgrad, fgrad, logs
