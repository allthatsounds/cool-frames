"""
numpy/phase/_phasederiv.py
==========================
Higher-order phase derivatives for non-uniform filterbanks.

Ports the second-order derivative computation from:
  gabor/gabphasederiv.m  (the 'dgt' method)
  gabor/gabphasederivreal.m

to the filterbank domain using the Auger-Flandrin approach:
  - tt (chirp rate): uses the second time-derivative of the window
  - ff (group delay dispersion): uses the second frequency-weight of the window
  - tf (mixed derivative): uses the cross time-frequency weighted window

The existing `filterbankphasegrad` (in _phasegrad.py) computes first-order
derivatives (t, f).  This module extends it to second order.

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

from ..filterbanks._core import filterbank
from ..filterbanks._utils import normalise_a
from ..filters._design import filterbanklength
from ..filters._hval import eval_H as _eval_H
from ._phasegrad import comp_phasegradfilters

# ---------------------------------------------------------------------------
# Build second-order derivative filters
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
                H_vals = np.asarray(_eval_H(H_c, L_), dtype=complex)
                n_h = len(H_vals)
                k = np.arange(n_h)
                fmul = (-1j * 2 * np.pi * k / L_) ** 2
                return H_vals * fmul

            cd2_m = dict(gm)
            cd2_m["H"] = lambda L_, _H=H_ref: _make_cd2(_H, L_)
            cd2_m["foff"] = fo_ref

            # ch2: second frequency-weight = multiply by (jn)²  = -n²
            def _make_ch2(H_c, fo_c, L_=L):
                H_vals = np.asarray(_eval_H(H_c, L_), dtype=complex)
                n_h = len(H_vals)
                fo_v = int(fo_c(L_)) if callable(fo_c) else int(fo_c)
                # Contiguous signed indices -- see the long note in
                # ``_phasegrad.py``.  Reducing mod L and centring each index
                # independently splits any support that crosses L/2, which is
                # every bank's Nyquist complement.  Here the index is squared
                # so the split nearly cancels, but the seam is still wrong and
                # there is no reason to keep two different conventions.
                k_abs = np.arange(fo_v, fo_v + n_h).astype(float)
                k_abs -= L_ * np.round(0.5 * (k_abs[0] + k_abs[-1]) / L_)
                tmul = -(k_abs**2)  # (j·k)² = -k²
                return H_vals * tmul

            ch2_m = dict(gm)
            ch2_m["H"] = lambda L_, _H=H_ref, _fo=fo_ref: _make_ch2(_H, _fo, L_)
            ch2_m["foff"] = fo_ref

            # chd: cross = (n/L) · dg/(2π) in freq domain
            # = H · (j·k_abs / L) · (-j·2π·k_shift / L) / (2π)
            # = H · (k_abs / L) · (2π·k_shift / L) / (2π)
            # = H · k_abs · k_shift / L²
            # Wait, let's be more careful. From MATLAB:
            #   hdg = (fftindex(size(dg,1))/L) .* dg
            # where dg = pderiv(g,[],Inf)/(2*pi)
            # In frequency domain: dg has response H * (-j*2π*k/L) / (2π) = H * (-j*k/L)
            # Then hdg = (n/L) * dg  in time domain
            # In freq domain: convolution... but for band-limited:
            # hdg has freq response = (j*k_abs/L) * (H * (-j*k_shift/L))
            # Hmm, the MATLAB code works in time domain for this.
            # For band-limited case, we need:
            #   chd_m = filterbank(f, hdg_filter)
            # where hdg is the product of time-ramp (k_abs/L) and time-derivative filter
            #
            # In the frequency domain, multiplying by k_abs in time is a
            # frequency-domain convolution. For bandlimited filters, we can
            # approximate: if the filter is narrow-band centered at fc,
            # then the time-weight n ≈ fc for the dominant frequency components.
            #
            # But for correctness we build the full response:
            # hdg[n] = (n/L) · dg[n]  where dg = IFFT(H · (-j2πk/L))
            # So: chd_filter = FFT(n/L · IFFT(H · (-j2πk/L)))
            def _make_chd(H_c, fo_c, L_=L):
                H_vals = np.asarray(_eval_H(H_c, L_), dtype=complex)
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
            # dg = h * (-j*2π*n/L) / (2π) = h * (-j*n/L)
            # hdg = (n/L) * dg = h * (-j*n²/L²)
            # Wait, the n in time-ramp and the n in derivative are the same
            # index, so: hdg[n] = (n/L) * h[n] * (-j*n/L) = h[n] * (-j*n²/L²)
            chd_m = dict(gm)
            chd_m["h"] = h * (-1j * n**2 / L**2)

        cd2.append(cd2_m)
        ch2.append(ch2_m)
        chd.append(chd_m)

    return ch, cd, cd2, ch2, chd


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def comp_filterbankphasederiv(
    c: list[np.ndarray],
    cd: list[np.ndarray],
    ch: list[np.ndarray],
    cd2: list[np.ndarray],
    ch2: list[np.ndarray],
    chd: list[np.ndarray],
    L: int,
    minlvl: float = 1e-6,
    derivs: Sequence[str] = ("tt", "ff", "tf"),
) -> dict[str, list[np.ndarray]]:
    """Compute second-order phase derivatives from filterbank coefficients.

    Parameters
    ----------
    c, cd, ch, cd2, ch2, chd : lists of M coefficient arrays
        Analysis and derivative-filter coefficients.
    L : int
        Signal length.
    minlvl : float
        Relative floor for the spectrogram denominator.
    derivs : sequence of str
        Which derivatives to compute: 'tt', 'ff', 'tf' (any subset).

    Returns
    -------
    result : dict mapping derivative name to list of M arrays
    """
    M = len(c)

    # Global floor
    all_abs = np.concatenate([np.abs(np.asarray(cm)).ravel() for cm in c])
    lvl = minlvl * float(np.max(all_abs**2)) if all_abs.size > 0 else minlvl

    result: dict[str, list[np.ndarray]] = {d: [] for d in derivs}

    for m in range(M):
        cm = np.asarray(c[m], dtype=complex)
        sm = np.maximum(np.abs(cm) ** 2, lvl)
        conj_cm = np.conj(cm)
        ratio = conj_cm / sm  # conj(c) / |c|²

        if "tt" in derivs:
            cdm = np.asarray(cd[m], dtype=complex)
            cd2m = np.asarray(cd2[m], dtype=complex)
            # d_tt = Im(cd2·conj(c)/s - 2π·(cd·conj(c)/s)²) / L
            term1 = cd2m * ratio
            term2 = (cdm * ratio) ** 2
            d_tt = np.imag(term1 - 2 * np.pi * term2) / L
            result["tt"].append(d_tt.real)

        if "ff" in derivs:
            chm = np.asarray(ch[m], dtype=complex)
            ch2m = np.asarray(ch2[m], dtype=complex)
            # d_ff = Im(-ch2·conj(c)/s + (ch·conj(c)/s)²) · 2π/L
            term1 = -ch2m * ratio
            term2 = (chm * ratio) ** 2
            d_ff = np.imag(term1 + term2) * 2 * np.pi / L
            result["ff"].append(d_ff.real)

        if "tf" in derivs:
            cdm = np.asarray(cd[m], dtype=complex)
            chm = np.asarray(ch[m], dtype=complex)
            chdm = np.asarray(chd[m], dtype=complex)
            # d_tf = Re(chd·conj(c)/s - (1/L)·ch·cd·(conj(c)/s)²) · 2π
            term1 = chdm * ratio
            term2 = (1.0 / L) * chm * cdm * ratio**2
            d_tf = np.real(term1 - term2) * 2 * np.pi
            result["tf"].append(d_tf.real)

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def filterbankphasederiv(
    f,
    g: list[dict],
    a,
    derivs: str | Sequence[str] = ("tt", "ff", "tf"),
    L: int | None = None,
    minlvl: float = 1e-6,
) -> tuple[dict[str, list[np.ndarray]], list[np.ndarray]]:
    """Compute second-order phase derivatives for a filterbank.

    This extends :func:`filterbankphasegrad` (which computes first-order
    derivatives t and f) to second-order derivatives tt, ff, tf.

    Parameters
    ----------
    f : array_like, shape (Ls,)
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
    result : dict mapping derivative name → list of M arrays
        Each array has the same length as the corresponding filterbank
        coefficient array.
    c : list of M complex arrays
        The filterbank coefficients (always computed as a byproduct).

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.phase import filterbankphasederiv
    >>> g, a, fc, L, _info = audfilters(8000, 2048)
    >>> signal = np.random.default_rng(0).standard_normal(2048)
    >>> result, c = filterbankphasederiv(signal, g, a, L=L, derivs=['tt', 'tf'])
    >>> sorted(result)
    ['tf', 'tt']
    >>> len(result['tt']) == len(g)
    True
    """
    f = np.asarray(f)
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

    # Build all derivative filters
    ch, cd, cd2, ch2, chd = comp_phasederivfilters_2nd(g, a_norm, L)

    # Run filterbanks — we need c plus whichever derivative coefficients
    c = filterbank(f, g, a_norm, L=L)

    # Only compute the filterbank outputs we actually need
    cd_c: list[np.ndarray | None] = [None] * M
    ch_c: list[np.ndarray | None] = [None] * M
    cd2_c: list[np.ndarray | None] = [None] * M
    ch2_c: list[np.ndarray | None] = [None] * M
    chd_c: list[np.ndarray | None] = [None] * M

    need_cd = "tt" in derivs or "tf" in derivs
    need_ch = "ff" in derivs or "tf" in derivs
    need_cd2 = "tt" in derivs
    need_ch2 = "ff" in derivs
    need_chd = "tf" in derivs

    if need_cd:
        cd_c = filterbank(f, cd, a_norm, L=L)  # type: ignore[assignment]
    if need_ch:
        ch_c = filterbank(f, ch, a_norm, L=L)  # type: ignore[assignment]
    if need_cd2:
        cd2_c = filterbank(f, cd2, a_norm, L=L)  # type: ignore[assignment]
    if need_ch2:
        ch2_c = filterbank(f, ch2, a_norm, L=L)  # type: ignore[assignment]
    if need_chd:
        chd_c = filterbank(f, chd, a_norm, L=L)  # type: ignore[assignment]

    result = comp_filterbankphasederiv(
        c,
        cd_c,  # type: ignore[arg-type]
        ch_c,  # type: ignore[arg-type]
        cd2_c,  # type: ignore[arg-type]
        ch2_c,  # type: ignore[arg-type]
        chd_c,  # type: ignore[arg-type]
        L,
        minlvl,
        derivs,
    )

    return result, c
