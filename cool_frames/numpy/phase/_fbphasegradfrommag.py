"""
numpy/layer3/_fbphasegradfrommag.py
====================================
Phase gradient estimation from magnitude for **non-uniform** filterbanks.

This is the general-case companion to ``comp_ufilterbankphasegradfrommag``
(which handles the uniform case).  For non-uniform filterbanks the channels
may have different lengths and hop sizes, so a neighbor-lookup structure
(NEIGH, posInfo) is precomputed to map each coefficient to its frequency
neighbors.

MATLAB originals
-----------------
  layer3/phase_processing/comp_filterbankphasegradfrommag.m
  layer3/phase_processing/comp_filterbankneighbors.m
  layer0/math_utils/pderiv.m
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# pderiv – periodic discrete derivative (order 2 central differences)
# ---------------------------------------------------------------------------


def _pderiv(f: np.ndarray, difforder: int = 2) -> np.ndarray:
    """Periodic derivative of a 1-D array using central differences.

    Port of ``pderiv.m`` (1-D case only).
    The result is scaled by *L* (the length of *f*), matching MATLAB
    where ``pderiv(f, 1, difforder)`` returns ``L * (shift-based diff)``.
    """
    L = len(f)
    if difforder == 2:
        return L * (np.roll(f, -1) - np.roll(f, 1)) / 2.0  # type: ignore[no-any-return]
    elif difforder == 4:
        return (  # type: ignore[no-any-return]
            L * (-np.roll(f, -2) + 8 * np.roll(f, -1) - 8 * np.roll(f, 1) + np.roll(f, 2)) / 12.0
        )
    else:
        raise ValueError(f"pderiv: difforder must be 2 or 4, got {difforder}")


# ---------------------------------------------------------------------------
# comp_filterbankneighbors – neighbour lookup for non-uniform filterbanks
# ---------------------------------------------------------------------------


def comp_filterbankneighbors(
    a: np.ndarray,
    M: int,
    N: np.ndarray,
    do_real: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the neighbour structure for a non-uniform filterbank.

    Port of ``comp_filterbankneighbors.m``.

    Parameters
    ----------
    a : (M,) int array — hop sizes per channel
    M : int — number of channels
    N : (M,) int array — number of time frames per channel
    do_real : bool — True for real (non-cyclic) filterbanks

    Returns
    -------
    NEIGH : (6, Nsum) int array
        Neighbour indices (0-based).  Rows:
        row 0 — left time neighbour (same channel),
        row 1 — right time neighbour (same channel, unused in MATLAB but reserved),
        rows 2–3 — below (lower channel) neighbour range [low, high],
        rows 4–5 — above (upper channel) neighbour range [low, high].
        A value of -1 means "no neighbour".

    posInfo : (2, Nsum) float array
        Row 0: channel index (0-based).
        Row 1: time position (sample offset = frame_index * a[m]).

    Note
    ----
    The MATLAB version uses 1-based indices.  We use **0-based** and encode
    "no neighbour" as -1 (MATLAB uses 0 after the ``NEIGH = NEIGH - 1``
    adjustment that happens in ``filterbankconstphase.m``).
    """
    a = np.asarray(a, dtype=int).ravel()
    N = np.asarray(N, dtype=int).ravel()
    assert len(a) == M and len(N) == M

    Nsum = int(np.sum(N))
    chanStart = np.concatenate([[0], np.cumsum(N)])  # length M+1

    # Initialise to -1 (no neighbour)
    NEIGH = -np.ones((6, Nsum), dtype=int)

    LIM = 0.8  # time-distance limit for vertical neighbours

    # --- Horizontal (time) neighbours within each channel ---
    for kk in range(M):
        cs = chanStart[kk]
        Nk = N[kk]
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
        aTemp = a[kk] / a[kk + 1]
        n_idx = np.arange(N[kk])
        POSlow = cs_above + np.clip(np.ceil((n_idx - LIM) * aTemp).astype(int), 0, N[kk + 1] - 1)
        POShigh = cs_above + np.clip(np.floor((n_idx + LIM) * aTemp).astype(int), 0, N[kk + 1] - 1)
        NEIGH[4, cs : cs + N[kk]] = POSlow
        NEIGH[5, cs : cs + N[kk]] = POShigh

    # Wrap-around for complex (non-real) filterbanks
    if not do_real:
        cs_last = chanStart[M - 1]
        aTemp = a[M - 1] / a[0]
        n_idx = np.arange(N[M - 1])
        POSlow = chanStart[0] + np.clip(np.ceil((n_idx - LIM) * aTemp).astype(int), 0, N[0] - 1)
        POShigh = chanStart[0] + np.clip(np.floor((n_idx + LIM) * aTemp).astype(int), 0, N[0] - 1)
        NEIGH[4, cs_last : cs_last + N[M - 1]] = POSlow
        NEIGH[5, cs_last : cs_last + N[M - 1]] = POShigh

    # Where low == high, clear the high entry (single neighbour only)
    mask_eq = NEIGH[5, :] == NEIGH[4, :]
    NEIGH[5, mask_eq] = -1

    # --- Vertical (frequency) neighbours: one channel below ---
    for kk in range(1, M):
        cs = chanStart[kk]
        cs_below = chanStart[kk - 1]
        aTemp = a[kk] / a[kk - 1]
        n_idx = np.arange(N[kk])
        POSlow = cs_below + np.clip(np.ceil((n_idx - LIM) * aTemp).astype(int), 0, N[kk - 1] - 1)
        POShigh = cs_below + np.clip(np.floor((n_idx + LIM) * aTemp).astype(int), 0, N[kk - 1] - 1)
        NEIGH[2, cs : cs + N[kk]] = POSlow
        NEIGH[3, cs : cs + N[kk]] = POShigh

    if not do_real:
        cs_first = chanStart[0]
        aTemp = a[0] / a[M - 1]
        n_idx = np.arange(N[0])
        POSlow = chanStart[M - 1] + np.clip(
            np.ceil((n_idx - LIM) * aTemp).astype(int), 0, N[M - 1] - 1
        )
        POShigh = chanStart[M - 1] + np.clip(
            np.floor((n_idx + LIM) * aTemp).astype(int), 0, N[M - 1] - 1
        )
        NEIGH[2, cs_first : cs_first + N[0]] = POSlow
        NEIGH[3, cs_first : cs_first + N[0]] = POShigh

    mask_eq2 = NEIGH[3, :] == NEIGH[2, :]
    NEIGH[3, mask_eq2] = -1

    # --- posInfo ---
    posInfo = np.zeros((2, Nsum))
    for kk in range(M):
        cs = chanStart[kk]
        posInfo[0, cs : cs + N[kk]] = kk
        posInfo[1, cs : cs + N[kk]] = np.arange(N[kk]) * a[kk]

    return NEIGH, posInfo


# ---------------------------------------------------------------------------
# comp_filterbankphasegradfrommag – main algorithm
# ---------------------------------------------------------------------------


def comp_filterbankphasegradfrommag(
    abss: np.ndarray,
    N: np.ndarray,
    a: np.ndarray,
    M: int,
    sqtfr: np.ndarray,
    fc: np.ndarray,
    NEIGH: np.ndarray,
    posInfo: np.ndarray,
    gderivweight: float = 0.5,
    do_tfrdiff: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Phase gradient estimation from magnitude for a non-uniform filterbank.

    Port of ``comp_filterbankphasegradfrommag.m``.

    Parameters
    ----------
    abss : (Nsum,) magnitude vector (all channels concatenated)
    N : (M,) number of frames per channel
    a : (M,) hop sizes per channel
    M : number of channels
    sqtfr : (M,) sqrt of time-frequency ratios
    fc : (M,) normalised centre frequencies
    NEIGH : (6, Nsum) neighbour indices (0-based, -1 = no neighbour)
    posInfo : (2, Nsum) position info [channel; time_position]
    gderivweight : weight for tfr-difference correction (default 0.5)
    do_tfrdiff : whether to include sqtfr-difference correction

    Returns
    -------
    tgrad : (Nsum,) time-direction phase gradient
    fgrad : (Nsum,) frequency-direction phase gradient
    logs  : (Nsum,) log-magnitude
    """
    abss = np.asarray(abss, dtype=float).ravel()
    N = np.asarray(N, dtype=int).ravel()
    a = np.asarray(a, dtype=float).ravel()
    sqtfr = np.asarray(sqtfr, dtype=float).ravel()
    fc = np.asarray(fc, dtype=float).ravel()

    Nsum = int(np.sum(N))
    assert len(abss) == Nsum
    assert NEIGH.shape == (6, Nsum)
    assert posInfo.shape == (2, Nsum)

    L = a[0] * N[0]  # transform length
    difforder = 2
    fac = gderivweight

    logs = np.log(abss + np.finfo(float).tiny)

    # ----- fgrad: frequency-direction phase gradient -----
    # Per-channel periodic derivative of log-magnitude along time
    fgrad = np.zeros(Nsum)
    chanStart = 0
    for m in range(M):
        Nm = N[m]
        idx = slice(chanStart, chanStart + Nm)
        fgrad[idx] = _pderiv(logs[idx], difforder) / Nm
        chanStart += Nm

    # ----- tgrad: time-direction phase gradient -----
    # Uses frequency neighbours to estimate the frequency derivative of
    # log-magnitude, weighted by channel distance and inverse tfr.
    tgrad = np.zeros(Nsum)

    chanStart = 0
    for m in range(M):
        Nm = N[m]
        denom = sqtfr[m] ** 2 * (np.pi * L)

        # Precompute above/below nom/denom
        aboveNom = 0.0
        aboveDenom = 1.0
        belowNom = 0.0
        belowDenom = 1.0

        if m < M - 1:
            if do_tfrdiff:
                aboveNom = fac * (sqtfr[m + 1] - sqtfr[m]) / sqtfr[m]
            aboveDenom = fc[m + 1] - fc[m]

        if m > 0:
            if do_tfrdiff:
                belowNom = fac * (sqtfr[m] - sqtfr[m - 1]) / sqtfr[m]
            belowDenom = fc[m] - fc[m - 1]

        temp = np.zeros(Nm)
        for n in range(Nm):
            w = chanStart + n

            # --- Above neighbours (rows 4, 5 of NEIGH) ---
            tempValAbove = 0.0
            numNeighAbove = 0
            for jj in [4, 5]:
                neigh = NEIGH[jj, w]
                if neigh >= 0:
                    numNeighAbove += 1
                    dist = (posInfo[1, neigh] - posInfo[1, w]) / a[m]
                    tempValAbove += logs[neigh] - logs[w] - dist * fgrad[w]
            if numNeighAbove > 0:
                tempValAbove /= numNeighAbove

            # --- Below neighbours (rows 2, 3 of NEIGH) ---
            tempValBelow = 0.0
            numNeighBelow = 0
            for jj in [2, 3]:
                neigh = NEIGH[jj, w]
                if neigh >= 0:
                    numNeighBelow += 1
                    dist = (posInfo[1, neigh] - posInfo[1, w]) / a[m]
                    tempValBelow += logs[w] - logs[neigh] - dist * fgrad[w]
            if numNeighBelow > 0:
                tempValBelow /= numNeighBelow

            temp[n] = (tempValAbove + aboveNom) / aboveDenom + (
                tempValBelow + belowNom
            ) / belowDenom

        tgrad[chanStart : chanStart + Nm] = temp / denom
        chanStart += Nm

    # ----- Scale fgrad by tfr² / (2π) * N(m) -----
    chanStart = 0
    for m in range(M):
        Nm = N[m]
        idx = slice(chanStart, chanStart + Nm)
        fgrad[idx] = fgrad[idx] * sqtfr[m] ** 2 / (2 * np.pi) * Nm
        chanStart += Nm

    return tgrad, fgrad, logs
