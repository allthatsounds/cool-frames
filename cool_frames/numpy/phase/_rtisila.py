"""
numpy/phase/_rtisila.py
=======================
Real-Time Iterative Spectrogram Inversion with Look-Ahead (RTISI-LA).

Port of PHASERET's ``gabor/rtisila.m`` [rtisila-zhu]_, on a Gabor frame
(``rtisila(s, g, a, M)``, PHASERET's own calling convention) or on any
filter bank (``rtisila(s, g, a)`` with ``g`` a list of filters).
"""

from __future__ import annotations

from typing import Literal


def _is_window(g) -> bool:
    """A Gabor window (array or window name) rather than a list of filters."""
    if isinstance(g, str):
        return True
    if isinstance(g, (list, tuple)):
        return len(g) > 0 and not isinstance(g[0], dict)
    return True


def _gabor_only(fn: str, **given) -> None:
    bad = [k for k, (v, default) in given.items() if v is not default and v != default]
    if bad:
        raise ValueError(
            f"{fn}: {', '.join(bad)} apply to a filter bank, not to a Gabor window with M channels"
        )


def _bank_only(fn: str, **given) -> None:
    bad = [k for k, (v, default) in given.items() if v is not default and v != default]
    if bad:
        raise ValueError(
            f"{fn}: {', '.join(bad)} apply to a Gabor window with M channels, not to a filter bank"
        )


def rtisila(
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
    startphase: Literal["zhu", "zero", "rand"] = "zhu",
    seed: int | None = None,
    phase: Literal["freqinv", "timeinv"] = "freqinv",
):
    """Real-Time Iterative Spectrogram Inversion with Look-Ahead.

    Zhu, Beauregard and Wyse's RTISI-LA [rtisila-zhu]_: frames of
    coefficients enter one at a time; the ``lookahead + 1`` newest are
    updated ``maxit`` times, newest first, each update re-analysing the
    partial reconstruction -- the frames already committed and those in the
    look-ahead, never a later one -- and imposing the target magnitude; then
    the oldest is committed.  An offline simulation of the real-time
    algorithm.

    Two forms:

    ``rtisila(s, g, a, M)``, with ``g`` a window (array or ``firwin`` name,
        at most ``M`` samples) and ``s`` the ``(M // 2 + 1, N)`` magnitudes of
        ``gabor.dgtreal(f, g, a, M)``: PHASERET's ``rtisila``, including the
        modified analysis windows of the newest frame, checked against
        PHASERET.  Returns ``c`` as an array.
    ``rtisila(s_list, g, a)``, with ``g`` a list of filters: the same
        algorithm on a filter bank, where a frame is a block of ``frame_hop``
        samples of time (a coefficient column for a uniform bank).  The
        partial reconstruction is kept and re-analysed exactly, in the
        frequency domain, one frame at a time.  The newest frame is analysed
        with Zhu's windows built from the bank's own atoms: for each channel
        the sum of its synthesis atoms at this and the following hops, cut
        to the main lobe of its analysis atom, which is PHASERET's window
        when the bank is a Gabor frame (and then the result is PHASERET's,
        up to the end of the signal).  Channels with fractional hops are
        analysed with their own filters.

    Parameters
    ----------
    s_list : ``(M // 2 + 1, N)`` array (Gabor), or list of per-channel arrays
        Target magnitudes (the modulus is taken).
    g : window, or list of filter dicts
    a : hop size(s)
    M : int, Gabor form only -- number of channels.
    L : transform length (filter bank; default from ``s_list`` and ``a``).
    Ls : length of the returned signal.
    real : filter bank only.  ``True`` synthesises real signals
        (``ifilterbank(..., real=True)``, for a single-sided bank of a real
        signal); default ``False``.  The Gabor form is always real.
    maxit : int, default 5 -- iterations per step.
    lookahead : int -- frames after the committed one in the window.
        Default: PHASERET's ``ceil(M/a) - 1`` (capped at ``N - 1``); on a
        filter bank, the number of frames an atom reaches beyond its own,
        from the longest atom's duration (the shortest interval holding all
        but 1e-3 of its energy) -- which gives PHASERET's value on a
        Gabor-type bank.
    frame_hop : int, filter bank only -- the frame length in samples.
        Default: the hop of a uniform bank, else the largest hop.  Channels
        containing DC or Nyquist are left out of both defaults: the DC and
        Nyquist complements of auditory and wavelet banks are narrow, with
        atoms and hops that can be as long as the signal.
    startphase : filter bank only.  What a frame enters the window with:
        ``'zhu'`` (default, PHASERET's) nothing, its first phase coming from
        the frames it overlaps; ``'zero'`` its magnitude with zero phase;
        ``'rand'`` its magnitude with a random phase (``seed``).
    phase : Gabor form only.  ``'freqinv'`` (default) returns ``c`` in
        cool-frames' (LTFAT's default) convention, so ``|c|`` and
        ``gabor.dgtreal(f, g, a, M)`` agree; ``'timeinv'`` in PHASERET's.

    Returns
    -------
    c : coefficients with the reconstructed phase (array, or list per channel)
    f : reconstructed signal (``Ls`` samples if given, else ``L``)
    relres : float -- ``|| |T f| - s || / ||s||`` for the full-length signal
        the coefficients synthesise, ``T`` the analysis.  Until v0.1.1 this
        was ``|| |c| - s || / ||s||``, which is zero by construction; PHASERET's
        own ``rtisila`` takes the modulus of the complex difference instead,
        which measures the phase.
    niter : int -- updates a frame receives while it crosses the window,
        ``maxit * (lookahead + 1)`` (fewer for the first ``lookahead``
        frames).  PHASERET documents this and returns ``maxit * lookahead``.

    Notes
    -----
    Near the end of the signal the window shrinks; PHASERET's simulation
    wraps round and reads the first frames again as look-ahead.  The Gabor
    form reproduces PHASERET, wrap included.

    References
    ----------
    .. [rtisila-zhu] X. Zhu, G. T. Beauregard, and L. L. Wyse, "Real-time
           signal estimation from modified short-time Fourier transform
           magnitude spectra," IEEE Trans. Audio, Speech, Lang. Process.,
           vol. 15, no. 5, pp. 1645-1653, 2007.
    """
    if _is_window(g):
        if M is None:
            raise TypeError(
                "rtisila: a window needs the number of channels M "
                "(rtisila(s, g, a, M)); pass a list of filters for a filter bank"
            )
        _gabor_only(
            "rtisila",
            L=(L, None),
            real=(real, None),
            frame_hop=(frame_hop, None),
            startphase=(startphase, "zhu"),
            seed=(seed, None),
        )
        from ._rtisila_gabor import rtisila_gabor

        return rtisila_gabor(s_list, g, a, M, Ls=Ls, maxit=maxit, lookahead=lookahead, phase=phase)
    if M is not None:
        raise TypeError(
            "rtisila: M is the number of channels of a Gabor window; a filter bank has its own"
        )
    _bank_only("rtisila", phase=(phase, "freqinv"))
    from ._rtisila_fb import rtisila_fb

    return rtisila_fb(
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
    )
