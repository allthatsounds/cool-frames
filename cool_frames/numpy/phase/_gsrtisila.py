"""
numpy/phase/_gsrtisila.py
=========================
Gnann and Spiertz's RTISI-LA (GSRTISI-LA).

Port of PHASERET's ``gabor/gsrtisila.m`` [gsrtisila-gs08]_ [gsrtisila-gs10]_,
on a Gabor frame (``gsrtisila(s, g, a, M)``) or on any filter bank.
"""

from __future__ import annotations

from typing import Literal

from ._rtisila import _bank_only, _gabor_only, _is_window


def gsrtisila(
    s_list,
    g,
    a,
    M: int | None = None,
    *,
    L: int | None = None,
    Ls: int | None = None,
    real: bool | None = None,
    maxit: int = 5,
    lookahead: int | None = None,
    frame_hop: int | None = None,
    startphase: Literal["zhu", "zeros", "zero", "input", "unwrap", "spsi"] = "zhu",
    unwrappar: float = 0.3,
    phase: Literal["freqinv", "timeinv"] = "freqinv",
):
    """Gnann and Spiertz's Real-Time Iterative Spectrogram Inversion.

    RTISI-LA (see :func:`rtisila`) with two changes by Gnann and Spiertz:
    each look-ahead frame is analysed with the window divided by the overlap
    of the analysis-synthesis window products of the frames present around
    it [gsrtisila-gs08]_, and the newest frame can start from an estimated
    phase rather than from nothing [gsrtisila-gs10]_.

    ``gsrtisila(s, g, a, M)`` (a Gabor window) is PHASERET's ``gsrtisila``,
    checked against PHASERET, with the initial phase of the newest frame
    used as in PHASERET's C library (its MATLAB fallback never reads it).
    On a filter bank (``g`` a list of filters) the frames are as in
    :func:`rtisila`.  The normalised windows divide by the overlap of the
    window products of the frames present, which for a filter bank is not a
    function of time alone; the newest frame is analysed with
    :func:`rtisila`'s windows instead, so what differs is the
    initialisation, and with ``startphase='zhu'`` the result is
    :func:`rtisila`'s.

    Parameters
    ----------
    s_list, g, a, M, L, Ls, real, maxit, lookahead, frame_hop, phase :
        As for :func:`rtisila`.
    startphase : what the newest frame starts from.

        - ``'zhu'`` (default; ``'zeros'`` is PHASERET's name) : nothing, the
          phase coming from the frames it overlaps;
        - ``'zero'`` (filter bank only) : its magnitude, zero phase;
        - ``'input'`` : the phase of the complex ``s``;
        - ``'unwrap'`` : phase-vocoder unwrapping from the two frames
          before, magnitude scaled by ``unwrappar``;
        - ``'spsi'`` : one step of single-pass spectrogram inversion from
          the refined phase of the frame before, as in PHASERET (on a filter
          bank, :func:`spsi`'s step at each time instant of the frame, from
          each channel's previous coefficient).

        PHASERET's ``'rtpghi'`` is not ported.
    unwrappar : float, default 0.3

    Returns
    -------
    c, f, relres, niter : as for :func:`rtisila`.

    References
    ----------
    .. [gsrtisila-gs08] V. Gnann and M. Spiertz, "Comb-filter free audio
           mixing using STFT magnitude spectra and phase estimation," Proc.
           11th Int. Conf. on Digital Audio Effects (DAFx-08), 2008.
    .. [gsrtisila-gs10] V. Gnann and M. Spiertz, "Improving RTISI phase
           estimation with energy order and phase unwrapping," Proc. 13th
           Int. Conf. on Digital Audio Effects (DAFx-10), 2010.
    """
    if _is_window(g):
        if M is None:
            raise TypeError(
                "gsrtisila: a window needs the number of channels M "
                "(gsrtisila(s, g, a, M)); pass a list of filters for a filter bank"
            )
        _gabor_only("gsrtisila", L=(L, None), real=(real, None), frame_hop=(frame_hop, None))
        from ._rtisila_gabor import gsrtisila_gabor

        return gsrtisila_gabor(
            s_list,
            g,
            a,
            M,
            Ls=Ls,
            maxit=maxit,
            lookahead=lookahead,
            startphase=startphase,
            unwrappar=unwrappar,
            phase=phase,
        )
    if M is not None:
        raise TypeError(
            "gsrtisila: M is the number of channels of a Gabor window; a filter bank has its own"
        )
    _bank_only("gsrtisila", phase=(phase, "freqinv"))
    from ._rtisila_fb import gsrtisila_fb

    return gsrtisila_fb(
        list(s_list),
        g,
        a,
        L=L,
        Ls=Ls,
        real=bool(real),
        maxit=maxit,
        lookahead=lookahead,
        frame_hop=frame_hop,
        startphase=startphase,
        unwrappar=unwrappar,
    )
