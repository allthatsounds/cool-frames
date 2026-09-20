"""
numpy/phase/_lertisila.py
=========================
RTISI-LA with Le Roux's truncated projection (TF-RTISI-LA).

Port of PHASERET's ``gabor/lertisila.m`` [lertisila-leroux]_, on a Gabor
frame (``lertisila(s, g, a, M)``) or on any filter bank.
"""

from __future__ import annotations

from typing import Literal

from ._rtisila import _bank_only, _gabor_only, _is_window


def lertisila(
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
    startphase: Literal["zhu", "zero", "rand", "input", "unwrap"] = "zhu",
    seed: int | None = None,
    variant: Literal["trunc", "modtrunc"] = "trunc",
    energy_order: bool = False,
    onthefly: bool = False,
    unwrappar: float = 0.3,
    freqneighs: int | None = None,
    asymwin: bool | None = None,
    relthr: float = 1e-3,
    phase: Literal["freqinv", "timeinv"] = "freqinv",
):
    """RTISI-LA with Le Roux's truncated projection (TF-RTISI-LA).

    The schedule of :func:`rtisila`, with each frame's re-analysis done by
    Le Roux's truncated projection kernel [lertisila-leroux]_ -- the
    analysis of the synthesis of a single coefficient, cut to a small
    neighbourhood -- applied to the coefficients in the buffer, instead of a
    synthesis and analysis of the partial reconstruction.

    ``lertisila(s, g, a, M)`` (a Gabor window) is PHASERET's ``lertisila``,
    checked against PHASERET: a ``(2 freqneighs + 1) x (2 lookback + 1)``
    kernel, the asymmetric kernels for the newest frame (``asymwin``), and
    PHASERET's buffer, which starts with the last frames of the signal.  On
    a filter bank (``g`` a list of filters) the kernel is
    :func:`legla`'s, truncated at ``relthr`` of its peak, over the frames of
    :func:`rtisila`; with fractional hops there is no kernel and the exact
    re-analysis of :func:`rtisila` is used.

    Parameters
    ----------
    s_list, g, a, M, L, Ls, real, maxit, lookahead, frame_hop, phase :
        As for :func:`rtisila`.
    startphase : what the newest frame starts from: ``'zhu'`` (default)
        nothing; ``'zero'`` its magnitude with zero phase; ``'rand'`` a
        random phase (``seed``); ``'input'`` the phase of the complex ``s``;
        ``'unwrap'`` a phase-vocoder step from the two frames before,
        magnitude scaled by ``unwrappar`` (on a Gabor frame, as PHASERET,
        from the *initial* coefficients, which have zero phase unless ``s``
        is complex).
    variant : ``'trunc'`` (default) or ``'modtrunc'``, the kernel's centre
        set to zero (integer hops).
    energy_order : after the first pass, update the window's frames in
        order of decreasing energy (PHASERET's ``'energy'``).
    onthefly : update coefficient by coefficient, each seeing the ones
        before it (PHASERET's ``'onthefly'``; integer hops).
    freqneighs : Gabor only -- the kernel's half-height (default the
        look-back, as PHASERET).
    asymwin : Gabor only -- the asymmetric kernels for the newest frame
        (PHASERET's default, ``True``).
    relthr : filter bank only -- kernel entries below ``relthr`` times its
        peak are dropped (:func:`legla`'s ``relthr``); ``0`` keeps the
        whole kernel, which makes the projection exact.

    Returns
    -------
    c, f, relres, niter : as for :func:`rtisila`.

    References
    ----------
    .. [lertisila-leroux] J. Le Roux, H. Kameoka, N. Ono, and S. Sagayama,
           "Phase initialization schemes for faster
           spectrogram-consistency-based signal reconstruction," Proc.
           Acoustical Society of Japan Autumn Meeting, 2010.
    """
    if _is_window(g):
        if M is None:
            raise TypeError(
                "lertisila: a window needs the number of channels M "
                "(lertisila(s, g, a, M)); pass a list of filters for a filter bank"
            )
        _gabor_only(
            "lertisila",
            L=(L, None),
            real=(real, None),
            frame_hop=(frame_hop, None),
            relthr=(relthr, 1e-3),
        )
        from ._rtisila_gabor import lertisila_gabor

        return lertisila_gabor(
            s_list,
            g,
            a,
            M,
            Ls=Ls,
            maxit=maxit,
            lookahead=lookahead,
            freqneighs=freqneighs,
            startphase=startphase,
            seed=seed,
            variant=variant,
            energy_order=energy_order,
            asymwin=True if asymwin is None else bool(asymwin),
            onthefly=onthefly,
            unwrappar=unwrappar,
            phase=phase,
        )
    if M is not None:
        raise TypeError(
            "lertisila: M is the number of channels of a Gabor window; a filter bank has its own"
        )
    _bank_only("lertisila", phase=(phase, "freqinv"), freqneighs=(freqneighs, None))
    if asymwin:
        raise ValueError(
            "lertisila: asymwin applies to a Gabor window with M channels, not "
            "to a filter bank (its asymmetric kernels are time-domain windows)"
        )
    from ._rtisila_fb import lertisila_fb

    return lertisila_fb(
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
        seed=seed,
        variant=variant,
        energy_order=energy_order,
        onthefly=onthefly,
        relthr=relthr,
        unwrappar=unwrappar,
    )
