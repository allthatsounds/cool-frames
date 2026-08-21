"""
numpy/core/_core.py
===================
Low-level FFT-domain filterbank kernels and signal-processing math utilities.

Shared primitives (involute, modcent, fftindex, floor23, postpad,
middlepad) are vendored in ``_math.py`` and ``_fourier.py``.

MATLAB originals
----------------
  comp_filterbank_fft.m    – full-length FFT analysis
  comp_filterbank_fftbl.m  – band-limited FFT analysis
  comp_ifilterbank_fft.m   – full-length FFT synthesis
  comp_ifilterbank_fftbl.m – band-limited FFT synthesis
  comp_filterbank_td.m     – time-domain (FIR) analysis
"""

from __future__ import annotations

import math

import numpy as np

from ._fourier import middlepad  # noqa: F401

# Shared primitives — vendored from the former ltfat_core package
from ._math import floor23, involute, modcent, postpad  # noqa: F401


def filterbanklength(Ls: int, a) -> int:
    """Return the smallest L >= Ls that is divisible by lcm(a).

    Parameters
    ----------
    Ls : int – signal length
    a  : int, scalar, 1-D array (M,), or 2-D array (M,2) fractional

    Returns
    -------
    L : int – smallest valid transform length

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.core import filterbanklength
    >>> L = filterbanklength(1000, 128)  # scalar hop size
    >>> L >= 1000 and L % 128 == 0
    True
    >>> L2 = filterbanklength(1000, [128, 256])  # multiple hop sizes
    >>> L2 >= 1000 and L2 % 128 == 0 and L2 % 256 == 0
    True
    """
    Ls = int(Ls)
    a = np.asarray(a)

    if a.ndim == 0:
        a_int = np.array([int(a)])
    elif a.ndim == 1:
        a_int = a.astype(int)
    else:
        a_int = a[:, 0].astype(int)

    lcm_a = int(a_int[0])
    for v in a_int[1:]:
        lcm_a = math.lcm(lcm_a, int(v))

    if lcm_a == 0:
        return Ls
    return math.ceil(Ls / lcm_a) * lcm_a


# ---------------------------------------------------------------------------
# Full-length FFT analysis kernel
# ---------------------------------------------------------------------------


def comp_filterbank_fft(F: np.ndarray, G: list[np.ndarray], a: np.ndarray) -> list[np.ndarray]:
    """Full-length FFT filterbank analysis.

    Parameters
    ----------
    F : (L, W) complex array  – DFT of the input signal(s)
    G : list of M arrays, each of length L  – full-length transfer functions
    a : (M,) or (M, 2) integer array – hop sizes (integer or fractional)

    Returns
    -------
    c : list of M arrays, each of shape (N_m, W)
    """
    squeezed = F.ndim == 1
    if squeezed:
        F = F[:, np.newaxis]
    L, W = F.shape
    M = len(G)
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])

    afrac = a[:, 0] / a[:, 1]  # rational hop sizes
    N = np.round(L / afrac).astype(int)

    c = []
    for m in range(M):
        Gm = np.asarray(G[m])  # (L,)
        Nm = N[m]
        am = int(a[m, 0])
        out = np.zeros((Nm, W), dtype=np.result_type(F, Gm))
        for w in range(W):
            tmp = (F[:, w] * Gm).reshape(am, Nm).sum(axis=0)
            out[:, w] = np.fft.ifft(tmp) / am
        c.append(out)
    if squeezed:
        c = [cm.ravel() for cm in c]
    return c


# ---------------------------------------------------------------------------
# Full-length FFT synthesis kernel
# ---------------------------------------------------------------------------


def comp_ifilterbank_fft(
    c: list[np.ndarray], G: list[np.ndarray], a: np.ndarray, L: int | None = None
) -> np.ndarray:
    """Full-length FFT filterbank synthesis.

    Parameters
    ----------
    c : list of M arrays, each (N_m, W)
    G : list of M arrays, each of length L
    a : (M,) or (M, 2) integer array
    L : output signal length (if None, inferred as N[0] * a[0])

    Returns
    -------
    F : (L, W) complex array, or (L,) if input was 1D
    """
    # Track if input was 1D (single-channel) before converting to 2D
    # Only squeeze if ALL inputs were genuinely 1D (not already 2D with W=1)
    squeezed = all(cm.ndim == 1 for cm in c)

    # Convert c to 2D (N_m, W) if necessary, ensuring proper column format
    c = [cm.reshape(-1, 1) if cm.ndim == 1 else cm for cm in c]
    M = len(c)
    W = c[0].shape[1] if c[0].ndim > 1 else 1
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])

    # Infer L from filter length if not provided
    if L is None:
        L = len(G[0])

    F = np.zeros((L, W), dtype=np.result_type(c[0], G[0]))
    for m in range(M):
        Gm = np.asarray(G[m])
        cm = c[m]  # already 2D from above
        am = int(a[m, 0])
        for w in range(W):
            Cm = np.fft.fft(cm[:, w])  # (N_m,)
            rep = np.tile(Cm, am)  # (L,)
            F[:, w] += rep * Gm.conj()

    # Squeeze output if input was single-channel (all 1D or all 2D with W=1)
    if squeezed:
        return F.ravel()
    return F


# ---------------------------------------------------------------------------
# Band-limited FFT analysis kernel
# ---------------------------------------------------------------------------


def comp_filterbank_fftbl(
    F: np.ndarray, G: list[np.ndarray], foff: np.ndarray, a: np.ndarray, realonly: np.ndarray
) -> list[np.ndarray]:
    """Band-limited FFT filterbank analysis.

    Each filter G[m] has length ``len(G[m]) << L``.

    Parameters
    ----------
    F        : (L, W) complex DFT of input
    G        : list of M short frequency-response arrays
    foff     : (M,) integer array – frequency offset of each G[m]
    a        : (M,) or (M, 2) integer array
    realonly : (M,) integer/bool array – 1 if filter is real-valued

    Returns
    -------
    c : list of M arrays (N_m, W)
    """
    if F.ndim == 1:
        F = F[:, np.newaxis]
    L, W = F.shape
    M = len(G)
    foff = np.asarray(foff, dtype=int)
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])

    afrac = a[:, 0] / a[:, 1]
    N = np.round(L / afrac).astype(int)

    # Frequency support indices for each filter (mod L, 0-based)
    fsupp_range = [np.mod(np.arange(foff[m], foff[m] + len(G[m])), L) for m in range(M)]

    c = []
    for m in range(M):
        Nm = int(N[m])
        Gm = np.asarray(G[m])
        idx = fsupp_range[m]
        out = np.zeros((Nm, W), dtype=np.result_type(F, Gm))
        for w in range(W):
            Ftmp = F[idx, w] * Gm
            # Zero-pad to a multiple of Nm, then reshape and sum (overlap-add)
            postpad_L = math.ceil(max(Nm, len(Gm)) / Nm) * Nm
            Ftmp = postpad(Ftmp, postpad_L)
            Ftmp = Ftmp.reshape(-1, Nm).sum(axis=0)
            Ftmp = np.roll(Ftmp, foff[m])
            out[:, w] = np.fft.ifft(Ftmp) / afrac[m]
        c.append(out)

    # NOTE: The original MATLAB comp_filterbank_fftbl had a ``realonly``
    # block here that averaged positive- and conjugate-mirror coefficients
    # via ``c[i] = (c[i] + c_conj[k]) / 2``.  The corresponding inverse
    # operation in comp_ifilterbank_fftbl was broken in MATLAB (wrong
    # argument count) and intentionally omitted here.  Without a matching
    # synthesis mirror, the /2 averaging discards the imaginary part of
    # the coefficients and cannot be undone, causing ~6 dB round-trip SDR
    # for any filterbank with realonly=1 channels (e.g. CQT).
    #
    # For real-signal reconstruction the caller (ifilterbank) applies
    # ``2 * real(ifft(F))`` which already mirrors the one-sided spectrum.
    # Keeping the coefficients complex here preserves full phase
    # information and gives machine-precision round-trip (~300 dB SDR).

    return c


# ---------------------------------------------------------------------------
# Band-limited FFT synthesis kernel
# ---------------------------------------------------------------------------


def comp_ifilterbank_fftbl(
    c: list[np.ndarray],
    G: list[np.ndarray],
    foff: np.ndarray,
    a: np.ndarray,
    realonly: np.ndarray,
    L: int | None = None,
) -> np.ndarray:
    """Band-limited FFT filterbank synthesis.

    Parameters
    ----------
    c        : list of M arrays (N_m, W)
    G        : list of M short frequency-response arrays
    foff     : (M,) integer frequency offsets
    a        : (M,) or (M,2) hop-size array
    realonly : (M,) array (1 = real-valued filter)
    L        : output signal length (if None, inferred as N[0] * a[0])

    Returns
    -------
    F : (L, W) complex output DFT, or (L,) if input was 1D
    """
    # Track if input was 1D (single-channel) before converting to 2D
    squeezed = all(cm.ndim == 1 for cm in c)

    # Convert c to 2D (N_m, W) if necessary, ensuring proper column format
    c = [cm.reshape(-1, 1) if cm.ndim == 1 else cm for cm in c]
    M = len(c)
    W = c[0].shape[1] if c[0].ndim > 1 else 1
    foff = np.asarray(foff, dtype=int)
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])

    # Infer L from max frequency offset + filter length if not provided
    if L is None:
        L = max(foff[m] + len(G[m]) for m in range(M))
    a[:, 0] / a[:, 1]

    N = np.array([c[m].shape[0] if c[m].ndim > 1 else len(c[m]) for m in range(M)])

    fsupp_range = [np.mod(np.arange(foff[m], foff[m] + len(G[m])), L) for m in range(M)]

    F = np.zeros((L, W), dtype=np.result_type(c[0], G[0]))
    for w in range(W):
        for m in range(M):
            Gm = np.asarray(G[m])
            cm = c[m][:, w]  # already 2D from above
            idx = fsupp_range[m]
            # Un-circshift, FFT, periodize to bandwidth of G[m], accumulate
            Ctmp = np.roll(np.fft.fft(cm), -foff[m])
            periods = math.ceil(len(Gm) / N[m])
            Ctmp = postpad(np.tile(Ctmp, periods), len(Gm))
            F[idx, w] += Ctmp * Gm.conj()

    # NOTE: No realonly mirror synthesis is needed here.  The analysis
    # (comp_filterbank_fftbl) keeps coefficients complex for all channels,
    # and ``ifilterbank(..., real=True)`` applies ``2*real(ifft(F))`` which
    # correctly mirrors the one-sided spectrum for reconstruction.

    # Squeeze output if input was all 1D
    if squeezed and W == 1:
        return F.ravel()
    return F


# ---------------------------------------------------------------------------
# Signal extension (boundary handling)
# ---------------------------------------------------------------------------


def comp_extBoundary(f: np.ndarray, extLen: int, mode: str) -> np.ndarray:
    """Extend a 1-D or 2-D array along axis 0 by *extLen* on each side.

    Port of MATLAB ``comp_extBoundary.m``.

    Parameters
    ----------
    f      : (L,) or (L, W) input array
    extLen : non-negative int — number of samples to add on each side
    mode   : boundary extension type, one of:

        ==========  ===================================================
        ``per``     periodic (wrap-around)
        ``ppd``     alias for ``per``
        ``zpd``     zero-padded
        ``zero``    alias for ``zpd``
        ``sym``     half-point symmetric (a.k.a. ``even``)
        ``even``    alias for ``sym``
        ``symw``    whole-point symmetric (excludes boundary sample)
        ``asym``    half-point antisymmetric (a.k.a. ``odd``)
        ``odd``     alias for ``asym``
        ``asymw``   whole-point antisymmetric
        ``sp0``     constant extension (replicate first/last sample)
        ==========  ===================================================

    Returns
    -------
    fout : array of shape ``(L + 2*extLen, ...)``
    """
    f = np.asarray(f)
    squeezed = f.ndim == 1
    if squeezed:
        f = f[:, np.newaxis]
    L = f.shape[0]

    fout = np.zeros((L + 2 * extLen, f.shape[1]), dtype=f.dtype)
    fout[extLen : extLen + L, :] = f

    if extLen == 0:
        return fout.ravel() if squeezed else fout

    legal = min(L, extLen)

    if mode in ("per", "ppd"):
        # For extLen > L, tile full copies then handle remainder
        times = extLen // L
        mod_ = extLen % L
        if times > 0:
            tile = np.tile(f, (times, 1))
            fout[extLen - times * L : extLen, :] = tile
            fout[L + extLen : L + extLen + times * L, :] = tile
        if mod_ > 0:
            fout[:mod_, :] = f[L - mod_ :, :]
            fout[L + extLen + times * L :, :] = f[:mod_, :]
    elif mode in ("zpd", "zero", "valid"):
        pass  # already zero
    elif mode in ("sym", "even"):
        fout[extLen - legal : extLen, :] = f[:legal][::-1, :]
        fout[L + extLen : L + extLen + legal, :] = f[L - legal :][::-1, :]
    elif mode == "symw":
        lw = min(L - 1, extLen)
        fout[extLen - lw : extLen, :] = f[1 : lw + 1][::-1, :]
        fout[L + extLen : L + extLen + lw, :] = f[L - lw - 1 : L - 1][::-1, :]
    elif mode in ("asym", "odd"):
        fout[extLen - legal : extLen, :] = -f[:legal][::-1, :]
        fout[L + extLen : L + extLen + legal, :] = -f[L - legal :][::-1, :]
    elif mode == "asymw":
        lw = min(L - 1, extLen)
        fout[extLen - lw : extLen, :] = -f[1 : lw + 1][::-1, :]
        fout[L + extLen : L + extLen + lw, :] = -f[L - lw - 1 : L - 1][::-1, :]
    elif mode == "sp0":
        fout[:extLen, :] = f[0:1, :]
        fout[L + extLen :, :] = f[L - 1 : L, :]
    else:
        raise ValueError(f"comp_extBoundary: unsupported mode '{mode}'")

    return fout.ravel() if squeezed else fout


# ---------------------------------------------------------------------------
# Up/down-sampling helpers
# ---------------------------------------------------------------------------


def _comp_ups(f: np.ndarray, a: int) -> np.ndarray:
    """(Deprecated) Internal version for 2D arrays."""
    N = f.shape[0]
    W = f.shape[1] if f.ndim > 1 else 1
    f2 = f if f.ndim > 1 else f[:, np.newaxis]
    out = np.zeros((N * a, W), dtype=f.dtype)
    out[::a, :] = f2
    return out


def _comp_downs(f: np.ndarray, a: int, skip: int = 0, L: int | None = None) -> np.ndarray:
    """(Deprecated) Internal version for 2D arrays."""
    if L is not None:
        f = f[skip : skip + L, :]
        skip = 0
    return f[skip::a, :]


# ---------------------------------------------------------------------------
# Time-domain FIR analysis kernel
# ---------------------------------------------------------------------------


def comp_filterbank_td(
    f: np.ndarray, g_td: list[np.ndarray], a: np.ndarray, offset: np.ndarray, ext: str = "per"
) -> list[np.ndarray]:
    """Time-domain (FIR) filterbank analysis using conv2 with boundary extension.

    Port of MATLAB ``comp_filterbank_td.m``.

    Parameters
    ----------
    f      : (L,) or (L, W) signal array
    g_td   : list of M impulse response vectors
    a      : (M,) subsampling factors (positive ints)
    offset : (M,) filter offsets (non-positive for causal filters:
             offset=0 means the filter's first sample aligns with the
             signal's first sample)
    ext    : boundary extension mode (default ``'per'``), passed to
             :func:`comp_extBoundary`.

    Returns
    -------
    c : list of M arrays, each (N_m,) or (N_m, W)
    """
    f = np.asarray(f)
    squeezed = f.ndim == 1
    if squeezed:
        f = f[:, np.newaxis]
    L, W = f.shape
    M = len(g_td)
    a = np.asarray(a, dtype=int).ravel()
    offset = np.asarray(offset, dtype=int).ravel()

    filtLen = np.array([len(np.asarray(g).ravel()) for g in g_td], dtype=int)
    skip = -offset

    if np.any(skip >= filtLen) or np.any(skip < 0):
        raise ValueError("comp_filterbank_td: filter zero-index position outside filter support")

    # Determine output lengths
    if ext == "per":
        Lext = L  # type: ignore[assignment]
    elif ext == "valid":
        Lext = L - (filtLen - 1)  # type: ignore[assignment]
    else:
        Lext = L + filtLen - 1  # type: ignore[assignment]

    N = (
        np.ceil(Lext / a).astype(int)
        if np.isscalar(Lext)
        else np.ceil((Lext - skip) / a).astype(int)
    )
    if ext == "per":
        N = np.ceil(np.full(M, L) / a).astype(int)
    elif ext == "valid":
        N = np.ceil((L - (filtLen - 1)) / a).astype(int)
    else:
        N = np.ceil((L + filtLen - 1 - skip) / a).astype(int)

    Lreq = a * (N - 1) + 1

    c = []
    for m in range(M):
        h = np.asarray(g_td[m]).ravel()
        fLen = int(filtLen[m])

        # Extend the input signal
        fext = comp_extBoundary(f, fLen - 1, ext)

        # 2-D linear convolution with 'valid' mode along axis 0
        # This matches MATLAB's conv2(fext, g{m}(:), 'valid')
        conv_out = np.zeros((fext.shape[0] - fLen + 1, W), dtype=np.result_type(f, h))
        for w in range(W):
            conv_out[:, w] = np.convolve(fext[:, w], h, mode="valid")

        # Downsample: take every a[m]-th sample starting at skip[m],
        # using Lreq[m] samples of the convolution output
        sk = int(skip[m])
        am = int(a[m])
        Lr = int(Lreq[m])
        c_m = conv_out[sk : sk + Lr : am, :]

        if squeezed:
            c.append(c_m.ravel())
        else:
            c.append(c_m)  # type: ignore[arg-type]
    return c


# ---------------------------------------------------------------------------
# Time-domain FIR synthesis kernel
# ---------------------------------------------------------------------------


def comp_ifilterbank_td(
    c: list[np.ndarray],
    g_td: list[np.ndarray],
    a: np.ndarray,
    Ls: int,
    offset: np.ndarray,
    ext: str = "per",
) -> np.ndarray:
    """Time-domain (FIR) filterbank synthesis using upsampling + convolution.

    Port of MATLAB ``comp_ifilterbank_td.m``.

    Parameters
    ----------
    c      : list of M coefficient arrays, each (N_m,) or (N_m, W)
    g_td   : list of M impulse response vectors
    a      : (M,) upsampling factors (positive ints)
    Ls     : desired output signal length
    offset : (M,) filter offsets (typically ``-(filtLen-1)`` for synthesis)
    ext    : boundary extension mode (default ``'per'``), passed to
             :func:`comp_extBoundary`.

    Returns
    -------
    f : (Ls,) or (Ls, W) reconstructed signal
    """
    M = len(g_td)
    c0 = np.asarray(c[0])
    squeezed = c0.ndim == 1
    W = 1 if squeezed else c0.shape[1]

    a = np.asarray(a, dtype=int).ravel()
    offset = np.asarray(offset, dtype=int).ravel()

    filtLen = np.array([len(np.asarray(g).ravel()) for g in g_td], dtype=int)

    # MATLAB: skip = -(1 - filtLen - offset)  = filtLen - 1 + offset
    skip = filtLen - 1 + offset
    if np.any(skip >= filtLen) or np.any(skip < 0):
        raise ValueError("comp_ifilterbank_td: filter zero-index position outside filter support")

    # Output allocation
    dtype_out = np.result_type(c[0], g_td[0])
    f = np.zeros((Ls, W), dtype=dtype_out)

    # If not periodic, fall back to zero-padded boundary for synthesis
    if ext != "per":
        ext_synth = "zero"
    else:
        ext_synth = "per"

    # MATLAB: skipOut = a .* (filtLen - 1) + skip
    skipOut = a * (filtLen - 1) + skip

    for m in range(M):
        h = np.asarray(g_td[m]).ravel()
        cm = np.asarray(c[m])
        if cm.ndim == 1:
            cm = cm[:, np.newaxis]
        fLen = int(filtLen[m])
        am = int(a[m])

        # Extend coefficients (boundary extension before upsampling)
        cext = comp_extBoundary(cm, fLen - 1, ext_synth)

        # Upsample
        cup = _comp_ups(cext, am)

        # Convolve with time-reversed conjugate filter
        h_rev = np.conj(h[::-1])
        so = int(skipOut[m])
        for w in range(W):
            conv_out = np.convolve(cup[:, w], h_rev, mode="full")
            # Extract the Ls-length output segment
            f[:, w] += conv_out[so : so + Ls]

    if squeezed:
        return f.ravel()
    return f


# ---------------------------------------------------------------------------
# pderiv – periodic derivative
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# psech – periodized hyperbolic secant
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# setnorm – normalise a signal to a given norm
# (moved here 2026-06-12 so both the filters and filterbanks layers can use
#  it without an upward import; was filterbanks/_sigproc.py)
# ---------------------------------------------------------------------------

_NORM_ALIASES = {
    "area": "1",
    "energy": "2",
    "peak": "inf",
}


def setnorm(
    f, norm: str = "2", *, val: float = 1.0, dim: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Set the norm of *f* to *val* and return both the result and the
    original norm value.

    Parameters
    ----------
    f : array_like
        Input signal (1-D or N-D).
    norm : str
        One of ``'1'`` / ``'area'``, ``'2'`` / ``'energy'``,
        ``'inf'`` / ``'peak'``, ``'rms'``, ``'wav'``, ``'null'``.
    val : float
        Target norm value (default 1).
    dim : int or None
        Axis along which to normalise.  ``None`` → first non-singleton.

    Returns
    -------
    f_out : ndarray
        Normalised signal.
    fnorm : ndarray or float
        Original norm value (before normalisation).
    """
    f = np.asarray(f)
    if f.size == 0:
        return f.copy(), np.float64(0.0)  # type: ignore[return-value]

    norm_key = _NORM_ALIASES.get(norm.lower(), norm.lower())

    if norm_key in ("null", "none", ""):
        return f.copy(), np.float64(0.0)  # type: ignore[return-value]

    if dim is None:
        dim = next((i for i in range(f.ndim) if f.shape[i] > 1), 0)

    # Compute the norm
    if norm_key == "1":
        fnorm = np.sum(np.abs(f), axis=dim, keepdims=True)
    elif norm_key == "2":
        fnorm = np.sqrt(np.sum(np.abs(f) ** 2, axis=dim, keepdims=True))
    elif norm_key == "inf":
        fnorm = np.max(np.abs(f), axis=dim, keepdims=True)
    elif norm_key == "rms":
        N = f.shape[dim]
        fnorm = np.sqrt(np.sum(np.abs(f) ** 2, axis=dim, keepdims=True) / N)
    elif norm_key == "wav":
        fnorm = np.max(np.abs(f), axis=dim, keepdims=True) / 0.99
    else:
        raise ValueError(f"Unknown norm type: {norm!r}")

    # Scale
    safe = np.where(fnorm > 0, fnorm, 1.0)
    f_out = f * (val / safe)
    # Where norm was zero, leave unchanged
    f_out = np.where(fnorm > 0, f_out, f)

    # Return the norm values without the keepdims axis
    fnorm_out = fnorm.squeeze(axis=dim)
    if fnorm_out.ndim == 0:
        fnorm_out = float(fnorm_out)

    return f_out, fnorm_out
