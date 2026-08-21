"""
numpy/layer3/_phasegrad.py
==========================
Phase-gradient computation for non-uniform filterbanks.

MATLAB originals
----------------
  layer3/phase_processing/comp_filterbankphasegrad.m
  layer3/phase_processing/comp_phasegradfilters.m
"""

from __future__ import annotations

import numpy as np

from ..filterbanks._core import filterbank
from ..filterbanks._utils import normalise_a
from ..filters._design import filterbanklength

# ---------------------------------------------------------------------------
# comp_phasegradfilters – build derivative filters ch and cd
# ---------------------------------------------------------------------------


def comp_phasegradfilters(g: list[dict], a, L: int) -> tuple[list[dict], list[dict]]:
    """Build the time-derivative (cd) and frequency-derivative (ch) filter banks.

    For a band-limited filter with transfer function H_m:
      - cd[m] = H_m * (-j * 2*pi * k / L)    (time derivative, multiply by freq)
      - ch[m] = H_m * (-j * n)                (freq derivative, multiply by time-ramp)

    Both operate in the frequency domain by pointwise multiplication of
    the filter's frequency response.

    Parameters
    ----------
    g : list of M filter dicts
    a : hop sizes
    L : DFT length

    Returns
    -------
    ch : list of M filter dicts (group-delay derivative)
    cd : list of M filter dicts (instantaneous-frequency derivative)

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.phase._phasegrad import comp_phasegradfilters
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> ch, cd = comp_phasegradfilters(g, a, L)
    >>> len(ch) == len(g) and len(cd) == len(g)
    True
    """
    len(g)
    ch = []
    cd = []

    for _m, gm in enumerate(g):
        if "H" in gm:
            # Filter dicts from different design functions store H either as
            # a callable (lazy: audfilters, cqtfilters) or as a precomputed
            # ndarray (eager: gabfilters). Normalise to a callable so the
            # derivative-filter constructors below can treat them uniformly.
            _H_raw = gm["H"]
            if callable(_H_raw):
                H_callable = _H_raw
            else:
                _H_arr_static = np.asarray(_H_raw)

                def H_callable(L_=None, _Hc=_H_arr_static):
                    return _Hc

            _fo_raw = gm["foff"]
            if callable(_fo_raw):
                fo_callable = _fo_raw
            else:

                def fo_callable(L_=None, _fc=int(_fo_raw)):
                    return _fc

            # --- gd: frequency-weighted (for instantaneous frequency / tgrad) ---
            # MATLAB: gd.H = fftindex(tempind) .* g.H
            # Multiply compact filter by centered DFT indices (Nyquist set to 0).
            def _make_cd(H_c, fo_c, L_):
                """Create frequency-weighted derivative filter (instantaneous frequency).

                Multiplies the filter response by centered DFT indices, implementing
                the time-domain derivative in the frequency domain.
                """
                H_vals = np.asarray(H_c(L_), dtype=complex)
                n_h = len(H_vals)
                fo_v = int(fo_c(L_)) if callable(fo_c) else int(fo_c)
                # Centered DFT indices (fftindex with Nyquist=0)
                k_abs = (np.arange(fo_v, fo_v + n_h) % L_).astype(float)
                k_cent = k_abs.copy()
                nyq = L_ // 2
                k_cent[k_abs > nyq] -= L_
                k_cent[k_abs == nyq] = 0.0  # Nyquist convention
                return k_cent * H_vals

            # --- gh: time-weighted (for group delay / fgrad) ---
            # MATLAB: gh.H = L/Lg * real(fftshift(ifft(1j*n*fft(fftshift(H)))))
            # Compute the periodic time-derivative of the compact filter.
            def _make_ch(H_c, fo_c, L_):
                """Create time-weighted derivative filter (group delay).

                Computes the periodic time-domain derivative of the compact filter
                using FFT roundtrip, scaling by L/n_h for proper normalization.
                """
                H_vals = np.asarray(H_c(L_), dtype=complex)
                n_h = len(H_vals)
                # fftindex(Lg, 0): centered indices of length n_h, Nyquist=0
                n_idx = np.arange(n_h, dtype=float)
                nyq_n = n_h // 2
                n_idx[n_idx > nyq_n] -= n_h
                if n_h % 2 == 0:
                    n_idx[int(nyq_n)] = 0.0  # Nyquist to 0
                # Time-domain derivative via FFT roundtrip
                H_cent = np.fft.fftshift(H_vals)
                H_fft = np.fft.fft(H_cent)
                H_deriv = np.fft.ifft(1j * n_idx * H_fft)
                return (L_ / n_h) * np.real(np.fft.fftshift(H_deriv))

            H_ref = H_callable
            fo_ref = fo_callable

            cd_m = dict(gm)
            cd_m["H"] = lambda L_, _H=H_ref, _fo=fo_ref: _make_cd(_H, _fo, L_)
            cd_m["foff"] = fo_ref

            ch_m = dict(gm)
            ch_m["H"] = lambda L_, _H=H_ref, _fo=fo_ref: _make_ch(_H, _fo, L_)
            ch_m["foff"] = fo_ref

        else:
            # FIR filter: derive time-domain via differentiation
            h = np.asarray(gm.get("h", np.array([])))
            n = np.arange(len(h))
            cd_m = dict(gm)
            cd_m["h"] = h * (-1j * 2 * np.pi / L * n)
            ch_m = dict(gm)
            ch_m["h"] = h * (-1j * n)

        cd.append(cd_m)
        ch.append(ch_m)

    return ch, cd


# ---------------------------------------------------------------------------
# comp_filterbankphasegrad – core computation
# ---------------------------------------------------------------------------


def comp_filterbankphasegrad(
    c: list[np.ndarray], ch: list[np.ndarray], cd: list[np.ndarray], L: int, minlvl: float = 1e-6
) -> tuple:
    """Compute phase gradients from filterbank coefficients.

    Ports ``comp_filterbankphasegrad.m``. cool_frames deviation: the returned
    spectrogram is the *true* ``s[m] = abs(c[m])^2`` (so ``s`` equals the
    coefficient power to float precision), while the ``minlvl`` floor
    ``sden[m] = max(abs(c[m])^2, minlvl*max(abs(c))^2)`` is applied ONLY as the
    division denominator for the gradients (avoiding divide-by-zero). MATLAB
    returns the floored value as ``s``, which leaks the floor into the
    spectrogram (and into reassignment, which uses ``s`` as an energy weight).
      s[m]     = abs(c[m])^2
      tgrad[m] = real(cd[m] * conj(c[m]) / sden[m]) / L * 2,  clipped to [-2,2]
      fgrad[m] = imag(ch[m] * conj(c[m]) / sden[m])

    Parameters
    ----------
    c, ch, cd : lists of M coefficient arrays (phase-gradient filter outputs)
    L         : signal length (DFT length)
    minlvl    : relative floor for the spectrogram denominator

    Returns
    -------
    tgrad : list of M real arrays
        Normalized instantaneous frequency.
    fgrad : list of M real arrays
        Group delay in samples.
    s     : list of M non-negative arrays
        Spectrogram (squared magnitude).

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.phase._phasegrad import (
    ...     comp_filterbankphasegrad
    ... )
    >>> c = [np.random.randn(10, 2) + 1j*np.random.randn(10, 2) for _ in range(3)]
    >>> ch = [np.random.randn(10, 2) + 1j*np.random.randn(10, 2) for _ in range(3)]
    >>> cd = [np.random.randn(10, 2) + 1j*np.random.randn(10, 2) for _ in range(3)]
    >>> tgrad, fgrad, s = comp_filterbankphasegrad(c, ch, cd, 8000)
    >>> len(tgrad) == 3
    True
    """
    # Global maximum for the floor
    all_abs = np.concatenate([np.abs(np.asarray(cm)).ravel() for cm in c])
    lvl = minlvl * float(np.max(all_abs**2)) if all_abs.size > 0 else minlvl

    tgrad_out = []
    fgrad_out = []
    s_out = []

    for m in range(len(c)):
        cm = np.asarray(c[m])
        chm = np.asarray(ch[m])
        cdm = np.asarray(cd[m])

        sm = np.abs(cm) ** 2  # true spectrogram |c|^2
        sm_den = np.maximum(sm, lvl)  # floored ONLY as the gradient
        # denominator (avoid divide-by-zero)

        tg = np.real(cdm * np.conj(cm) / sm_den) / L * 2
        # Clip to [-2, 2] (as in MATLAB)
        tg = tg * (np.abs(tg) <= 2)

        fg = np.imag(chm * np.conj(cm) / sm_den)

        tgrad_out.append(tg.real)
        fgrad_out.append(fg.real)
        s_out.append(sm.real)  # return the true |c|^2, not the floor

    return tgrad_out, fgrad_out, s_out


# ---------------------------------------------------------------------------
# filterbankphasegrad – public API
# ---------------------------------------------------------------------------


def filterbankphasegrad(f, g: list[dict], a, L: int | None = None, minlvl: float = 1e-6) -> tuple:
    """Compute phase gradients for a filterbank.

    Parameters
    ----------
    f     : signal (Ls,)
    g     : list of M filter dicts
    a     : hop sizes
    L     : DFT length (computed from Ls and a if omitted)
    minlvl: relative spectrogram floor

    Returns
    -------
    tgrad : list of M real arrays  (normalised instantaneous frequency)
    fgrad : list of M real arrays  (group delay in samples)
    s     : list of M non-negative arrays  (spectrogram)
    c     : list of M complex arrays  (filterbank coefficients)

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.phase import filterbankphasegrad
    >>> x = np.random.randn(8000)
    >>> g, a, fc, L, _info = audfilters(8000, len(x))
    >>> tgrad, fgrad, s, c = filterbankphasegrad(x, g, a, L)
    >>> len(tgrad) == len(g)  # same number of channels
    True
    >>> all(tg.shape[0] > 0 for tg in tgrad)  # all non-empty
    True
    """
    f = np.asarray(f)
    M = len(g)
    a_norm = normalise_a(a, M)

    if L is None:
        L = filterbanklength(len(f), a_norm)

    # Build derivative filters
    ch, cd = comp_phasegradfilters(g, a_norm, L)

    # Run all three filterbanks
    c = filterbank(f, g, a_norm, L=L)
    ch_c = filterbank(f, ch, a_norm, L=L)
    cd_c = filterbank(f, cd, a_norm, L=L)

    tgrad, fgrad, s = comp_filterbankphasegrad(c, ch_c, cd_c, L, minlvl)
    return tgrad, fgrad, s, c
