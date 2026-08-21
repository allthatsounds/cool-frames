"""
torch/core/_core.py
===================
Low-level FFT-domain and time-domain filterbank kernels — PyTorch implementation.

Mirrors ``numpy.core._core`` but uses ``torch.fft`` so that every
operation is differentiable and GPU-compatible.

MATLAB originals
----------------
  comp_filterbank_fft.m    – full-length FFT analysis
  comp_ifilterbank_fft.m   – full-length FFT synthesis
  comp_filterbank_fftbl.m  – band-limited FFT analysis
  comp_ifilterbank_fftbl.m – band-limited FFT synthesis
  comp_filterbank_td.m     – time-domain (FIR) analysis
  comp_ifilterbank_td.m    – time-domain (FIR) synthesis
"""

from __future__ import annotations

import math

import numpy as np
import torch


# Re-export the pure-int helper from numpy (no tensor involved)
def filterbanklength(Ls: int, a) -> int:
    """Return the smallest L >= Ls divisible by lcm(a).

    This is a setup-time function — pure Python/NumPy, no gradients.

    Examples
    --------
    >>> from cool_frames.torch.core import filterbanklength
    >>> L = filterbanklength(1000, a=[64, 96])
    >>> L % 64 == 0 and L % 96 == 0
    True
    >>> L >= 1000
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
# Hop-size normalisation (pure numpy, setup-time)
# ---------------------------------------------------------------------------


def _normalise_a(a, M: int) -> np.ndarray:
    """Normalise hop sizes to (M, 2) integer array.

    Parameters
    ----------
    a : scalar, (M,), or (M, 2) array-like
        Hop sizes. If scalar, broadcast to all M channels.
        If (M,), assume downsampling factor only, upsampling = 1.
        If (M, 2), (downsampling, upsampling) pairs.
    M : int
        Number of filterbank channels.

    Returns
    -------
    a : (M, 2) ndarray of int
        Normalized (downsampling, upsampling) hop size pairs.
    """
    a = np.asarray(a)
    if a.ndim == 0:
        a = np.full((M, 1), int(a))
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])
    return a.astype(int)  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Full-length FFT analysis kernel
# ---------------------------------------------------------------------------


def comp_filterbank_fft(
    F: torch.Tensor,
    G: list[torch.Tensor],
    a: np.ndarray,
) -> list[torch.Tensor]:
    """Full-length FFT filterbank analysis.

    Parameters
    ----------
    F : (L,) or (L, W) complex tensor — ``torch.fft.fft(signal)``
    G : list of M tensors, each of length L — transfer functions
    a : (M,) or (M, 2) integer ndarray — hop sizes

    Returns
    -------
    c : list of M tensors, each (N_m,) or (N_m, W)

    Examples
    --------
    >>> import torch
    >>> # Synthesize a signal and its FFT
    >>> x = torch.randn(1024)
    >>> F = torch.fft.fft(x)
    >>> # Define M=3 filters (could be windowed or parametric)
    >>> G = [torch.ones(1024, dtype=F.dtype) / 3 for _ in range(3)]
    >>> a = np.array([64, 64, 64])
    >>> c = comp_filterbank_fft(F, G, a)
    >>> len(c)
    3
    >>> c[0].shape[0]  # N_0 ≈ L / a[0]
    16
    """
    squeezed = F.dim() == 1
    if squeezed:
        F = F.unsqueeze(1)
    L, W = F.shape
    M = len(G)

    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])

    afrac = a[:, 0] / a[:, 1]
    N = np.round(L / afrac).astype(int)

    c = []
    for m in range(M):
        Gm = G[m]  # (L,)
        Nm = int(N[m])
        am = int(a[m, 0])

        # Multiply, reshape, fold-and-sum, then IFFT
        # (F[:, w] * Gm).reshape(am, Nm).sum(0) / am  for each w
        FG = F * Gm.unsqueeze(1)  # (L, W)
        FG = FG.reshape(am, Nm, W).sum(dim=0)  # (Nm, W)
        out = torch.fft.ifft(FG, dim=0) / am  # (Nm, W)
        c.append(out)

    if squeezed:
        c = [cm.squeeze(1) for cm in c]
    return c


# ---------------------------------------------------------------------------
# Full-length FFT synthesis kernel
# ---------------------------------------------------------------------------


def comp_ifilterbank_fft(
    c: list[torch.Tensor],
    G: list[torch.Tensor],
    a: np.ndarray,
    L: int | None = None,
) -> torch.Tensor:
    """Full-length FFT filterbank synthesis.

    Parameters
    ----------
    c : list of M tensors, each (N_m,) or (N_m, W)
    G : list of M tensors, each of length L
    a : (M,) or (M, 2) integer ndarray
    L : output length (inferred from G[0] if None)

    Returns
    -------
    F : (L, W) or (L,) complex tensor

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> # Reconstruct from coefficients
    >>> c = [torch.randn(16, dtype=torch.complex64) for _ in range(3)]
    >>> L = 1024
    >>> G = [torch.ones(L, dtype=torch.complex64) / 3 for _ in range(3)]
    >>> a = np.array([64, 64, 64])
    >>> F = comp_ifilterbank_fft(c, G, a, L=L)
    >>> F.shape
    torch.Size([1024])
    >>> x = torch.fft.ifft(F).real
    >>> x.shape[0]
    1024
    """
    squeezed = all(cm.dim() == 1 for cm in c)
    c = [cm.unsqueeze(1) if cm.dim() == 1 else cm for cm in c]
    M = len(c)
    W = c[0].shape[1]

    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])

    if L is None:
        L = G[0].shape[0]

    F = torch.zeros(L, W, dtype=c[0].dtype, device=c[0].device)
    for m in range(M):
        Gm = G[m]  # (L,)
        cm = c[m]  # (Nm, W)
        am = int(a[m, 0])

        Cm = torch.fft.fft(cm, dim=0)  # (Nm, W)
        rep = Cm.repeat(am, 1)  # (L, W)  — tile Nm → am*Nm = L
        F = F + rep * Gm.conj().unsqueeze(1)

    if squeezed:
        return F.squeeze(1)
    return F


# ---------------------------------------------------------------------------
# Band-limited FFT analysis kernel
# ---------------------------------------------------------------------------


def comp_filterbank_fftbl(
    F: torch.Tensor,
    G: list[torch.Tensor],
    foff: np.ndarray,
    a: np.ndarray,
    realonly: np.ndarray,
) -> list[torch.Tensor]:
    """Band-limited FFT filterbank analysis.

    Parameters
    ----------
    F      : (L,) or (L, W) complex tensor
    G      : list of M short frequency-response tensors
    foff   : (M,) integer offsets
    a      : (M,) or (M, 2) hop sizes
    realonly : (M,) flags (1 = real-valued filter)

    Returns
    -------
    c : list of M tensors (N_m,) or (N_m, W)

    Examples
    --------
    >>> x = torch.randn(512)
    >>> F = torch.fft.fft(x)
    >>> G = [torch.ones(64, dtype=F.dtype) for _ in range(3)]
    >>> foff = np.array([0, 64, 128])
    >>> a = np.array([16, 16, 16])
    >>> realonly = np.array([0, 0, 0])
    >>> c = comp_filterbank_fftbl(F, G, foff, a, realonly)  # doctest: +SKIP
    >>> len(c)
    3
    """
    squeezed = F.dim() == 1
    if squeezed:
        F = F.unsqueeze(1)
    L, W = F.shape
    M = len(G)
    foff = np.asarray(foff, dtype=int)
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])

    afrac = a[:, 0] / a[:, 1]
    N = np.round(L / afrac).astype(int)

    fsupp_range = [np.mod(np.arange(foff[m], foff[m] + len(G[m])), L) for m in range(M)]

    c = []
    for m in range(M):
        Nm = int(N[m])
        Gm = G[m]  # (Lm,)
        idx = torch.tensor(fsupp_range[m], dtype=torch.long, device=F.device)
        Ftmp = F[idx, :] * Gm.unsqueeze(1)  # (Lm, W)

        # Zero-pad to a multiple of Nm, fold, sum
        Lm = Ftmp.shape[0]
        postpad_L = math.ceil(max(Nm, Lm) / Nm) * Nm
        if postpad_L > Lm:
            pad = torch.zeros(postpad_L - Lm, W, dtype=Ftmp.dtype, device=Ftmp.device)
            Ftmp = torch.cat([Ftmp, pad], dim=0)
        Ftmp = Ftmp.reshape(-1, Nm, W).sum(dim=0)  # (Nm, W)
        Ftmp = torch.roll(Ftmp, int(foff[m]), dims=0)
        out = torch.fft.ifft(Ftmp, dim=0) / afrac[m]
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

    if squeezed:
        c = [cm.squeeze(1) if cm.dim() == 2 else cm for cm in c]
    return c


# ---------------------------------------------------------------------------
# Band-limited FFT synthesis kernel
# ---------------------------------------------------------------------------


def comp_ifilterbank_fftbl(
    c: list[torch.Tensor],
    G: list[torch.Tensor],
    foff: np.ndarray,
    a: np.ndarray,
    realonly: np.ndarray,
    L: int | None = None,
) -> torch.Tensor:
    """Band-limited FFT filterbank synthesis.

    Parameters
    ----------
    c       : list of M tensors (N_m,) or (N_m, W)
    G       : list of M short frequency-response tensors
    foff    : (M,) integer offsets
    a       : (M,) or (M, 2) hop sizes
    realonly : (M,) flags
    L       : output length

    Returns
    -------
    F : (L, W) or (L,) complex tensor

    Examples
    --------
    >>> c = [torch.randn(32, dtype=torch.complex64) for _ in range(3)]
    >>> G = [torch.ones(64, dtype=torch.complex64) for _ in range(3)]
    >>> foff = np.array([0, 64, 128])
    >>> a = np.array([16, 16, 16])
    >>> realonly = np.array([0, 0, 0])
    >>> F = comp_ifilterbank_fftbl(c, G, foff, a, realonly, L=512)  # doctest: +SKIP
    >>> F.shape
    torch.Size([512])
    """
    squeezed = all(cm.dim() == 1 for cm in c)
    c = [cm.unsqueeze(1) if cm.dim() == 1 else cm for cm in c]
    M = len(c)
    W = c[0].shape[1]
    foff = np.asarray(foff, dtype=int)
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])

    if L is None:
        L = max(int(foff[m]) + len(G[m]) for m in range(M))

    N = np.array([c[m].shape[0] for m in range(M)])

    fsupp_range = [np.mod(np.arange(foff[m], foff[m] + len(G[m])), L) for m in range(M)]

    device = c[0].device
    F = torch.zeros(L, W, dtype=c[0].dtype, device=device)

    for m in range(M):
        Gm = G[m]
        cm = c[m]  # (Nm, W)
        idx = torch.tensor(fsupp_range[m], dtype=torch.long, device=device)
        Ctmp = torch.roll(torch.fft.fft(cm, dim=0), -int(foff[m]), dims=0)
        periods = math.ceil(len(Gm) / int(N[m]))
        Ctmp = Ctmp.repeat(periods, 1)[: len(Gm), :]
        # Accumulate via index_add for differentiability
        contrib = Ctmp * Gm.conj().unsqueeze(1)
        F.index_add_(0, idx, contrib)

    if squeezed and W == 1:
        return F.squeeze(1)
    return F


# ---------------------------------------------------------------------------
# Time-domain FIR analysis kernel
# ---------------------------------------------------------------------------


def _periodic_extend_1d(x: torch.Tensor, pad_left: int, pad_right: int) -> torch.Tensor:
    """Periodically extend a 1D tensor on both sides.

    Parameters
    ----------
    x : (L,) tensor
    pad_left, pad_right : ints — number of samples to add on each side

    Returns
    -------
    out : (L + pad_left + pad_right,) tensor
    """
    if pad_left == 0 and pad_right == 0:
        return x
    L = x.shape[0]
    out = torch.zeros(L + pad_left + pad_right, dtype=x.dtype, device=x.device)
    out[pad_left : pad_left + L] = x
    if pad_left > 0:
        out[:pad_left] = x[-pad_left:]  # wrap from end
    if pad_right > 0:
        out[pad_left + L :] = x[:pad_right]  # wrap from start
    return out


def comp_filterbank_td(
    f: torch.Tensor,
    g_td: list[torch.Tensor],
    a: np.ndarray,
    offset: np.ndarray,
) -> list[torch.Tensor]:
    """Time-domain (FIR) filterbank analysis using convolution with periodic extension.

    Parameters
    ----------
    f : (L,) or (L, W) real or complex tensor — input signal(s)
    g_td : list of M impulse response tensors — each (filtLen_m,)
    a : (M,) or (M, 2) integer ndarray — hop sizes (downsampling factors)
    offset : (M,) integer ndarray — filter offsets (non-positive for causal;
             indicates the center of the filter)

    Returns
    -------
    c : list of M tensors, each (N_m,) or (N_m, W)

    Notes
    -----
    - Uses periodic boundary extension (circular convolution)
    - Downsamples by ``a[m]`` after convolution at phase ``-offset[m]``
    - Non-causal filters have negative offset (filter extends into past)
    - All operations are differentiable
    """
    squeezed = f.dim() == 1
    if squeezed:
        f = f.unsqueeze(1)
    L, W = f.shape
    M = len(g_td)

    a = np.asarray(a, dtype=int)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])

    offset = np.asarray(offset, dtype=int).ravel()

    # Compute filter lengths and skip amounts
    filtLen = np.array([len(gm) for gm in g_td], dtype=int)
    skip = -offset

    if np.any(skip >= filtLen) or np.any(skip < 0):
        raise ValueError("comp_filterbank_td: filter zero-index position outside filter support")

    # Output lengths (periodic boundary keeps L samples after convolution)
    a[:, 0] / a[:, 1]
    N = np.ceil(np.full(M, L) / a[:, 0]).astype(int)
    Lreq = a[:, 0] * (N - 1) + 1

    c = []
    for m in range(M):
        h = g_td[m]  # (filtLen_m,)
        fLen = int(filtLen[m])
        int(N[m])
        am = int(a[m, 0])

        # Convolve each channel using torch.nn.functional.conv1d
        # Note: torch.nn.functional.conv1d performs correlation, not convolution.
        # To match np.convolve (which does true convolution), we must flip h.
        h_flipped = h.flip(0)  # flip for correlation = convolution
        out_list = []
        for w in range(W):
            # Extract column w and extend periodically
            f_w = f[:, w]  # (L,)
            f_ext_w = _periodic_extend_1d(f_w, fLen - 1, fLen - 1)  # (L + 2*(fLen-1),)

            # Prepare filter for conv1d: (1, 1, fLen)
            h_w = h_flipped[None, None, :]  # (1, 1, fLen)

            # Input for conv1d: (1, 1, L + 2*(fLen-1))
            sig_input = f_ext_w[None, None, :]

            # Perform 1D convolution with mode='valid'
            conv_out = torch.nn.functional.conv1d(sig_input, h_w, padding=0)
            # Output: (1, 1, L + 2*(fLen-1) - fLen + 1) = (1, 1, L + fLen - 2)

            # Keep the whole 'valid' convolution — length L + fLen - 2 — and
            # let the `skip`/`Lreq` slice below pick the samples, exactly as
            # the NumPy kernel does.  Truncating to L here (the pre-v0.1.1
            # behaviour) discarded the tail, so any filter with a non-zero
            # offset returned too few coefficients: `firfilter('hann', 9)`
            # (offset -5) gave 59 samples at a=1 where NumPy gives 64.
            out_list.append(conv_out[0, 0, :])

        # Stack channels
        conv_all = torch.stack(out_list, dim=1)  # (L, W)

        # Downsample: take every am-th sample starting at skip[m]
        sk = int(skip[m])
        Lr = int(Lreq[m])
        c_m = conv_all[sk : sk + Lr : am, :]  # (Nm, W)

        if squeezed:
            c.append(c_m.squeeze(1))
        else:
            c.append(c_m)

    return c


# ---------------------------------------------------------------------------
# Time-domain FIR synthesis kernel
# ---------------------------------------------------------------------------


def comp_ifilterbank_td(
    c: list[torch.Tensor],
    g_td: list[torch.Tensor],
    a: np.ndarray,
    Ls: int,
    offset: np.ndarray,
) -> torch.Tensor:
    """Time-domain (FIR) filterbank synthesis using upsampling + convolution (overlap-add).

    Parameters
    ----------
    c : list of M coefficient tensors, each (N_m,) or (N_m, W)
    g_td : list of M impulse response tensors
    a : (M,) or (M, 2) integer ndarray — hop sizes (upsampling factors)
    Ls : int — desired output signal length
    offset : (M,) integer ndarray — filter offsets (typically negative for synthesis)

    Returns
    -------
    f : (Ls,) or (Ls, W) reconstructed tensor

    Notes
    -----
    - Uses overlap-add (zero-padded boundary extension before upsampling)
    - Upsamples by ``a[m]`` and convolves with conjugate-reversed filter
    - All operations are differentiable
    """
    M = len(c)
    c0 = c[0]
    squeezed = c0.dim() == 1
    W = 1 if squeezed else c0.shape[1]
    device = c0.device

    a = np.asarray(a, dtype=int)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.shape[1] == 1:
        a = np.hstack([a, np.ones((M, 1), dtype=a.dtype)])

    offset = np.asarray(offset, dtype=int).ravel()

    filtLen = np.array([len(gm) for gm in g_td], dtype=int)
    skip = filtLen - 1 + offset

    if np.any(skip >= filtLen) or np.any(skip < 0):
        raise ValueError("comp_ifilterbank_td: filter zero-index position outside filter support")

    # Output allocation
    dtype_out = c[0].dtype
    f = torch.zeros(Ls, W, dtype=dtype_out, device=device)

    # skipOut = a * (filtLen - 1) + skip (starting position after upsampling+convolution)
    skipOut = a[:, 0] * (filtLen - 1) + skip

    for m in range(M):
        h = g_td[m]  # impulse response
        cm = c[m]  # coefficients (N_m,) or (N_m, W)
        if cm.dim() == 1:
            cm = cm.unsqueeze(1)  # (N_m, 1)
        fLen = int(filtLen[m])
        am = int(a[m, 0])
        cm.shape[0]

        # Extend coefficients *periodically* before upsampling, matching the
        # NumPy kernel's ext='per' default.  Zero-padding (the pre-v0.1.1
        # behaviour) is a different boundary condition and put the two backends
        # 31.5 % apart on identical input.
        cext = torch.cat(
            [
                _periodic_extend_1d(cm[:, w], fLen - 1, fLen - 1).unsqueeze(1)
                for w in range(cm.shape[1])
            ],
            dim=1,
        )
        # cext shape: (Nm + 2*(fLen-1), W)

        # Upsample by am: insert am-1 zeros between samples
        cext_L = cext.shape[0]
        cup = torch.zeros(cext_L * am, W, dtype=cm.dtype, device=device)
        cup[::am, :] = cext  # (cext_L * am, W)

        # Convolve with time-reversed conjugate filter
        # For synthesis, numpy uses h_rev = conj(flipped h)
        # numpy.convolve(cup, h_rev) internally flips h_rev before convolving
        # torch.conv1d does correlation: out[n] = sum_k x[n+k] * w[k]
        # To match numpy.convolve, I need to flip w in torch.conv1d
        # So I need: torch.conv1d(x, flip(h_rev)) = numpy.convolve(x, h_rev)
        h_rev = torch.conj(h.flip(0))  # h_rev = conj(flipped h)
        h_rev_flipped = h_rev.flip(0)  # Flip again for torch conv1d

        so = int(skipOut[m])

        for w in range(W):
            # Convolve cup[:, w] with h_rev_flipped in 'full' mode via torch's conv1d
            sig_w = cup[:, w].unsqueeze(0).unsqueeze(0)  # (1, 1, cup_len)
            h_w = h_rev_flipped.unsqueeze(0).unsqueeze(0)  # (1, 1, fLen)
            # Pad for 'full' mode: add (fLen - 1) on both sides
            sig_padded = torch.nn.functional.pad(sig_w, (fLen - 1, fLen - 1))
            # Now convolve: output length = sig_padded_len - fLen + 1
            conv_out = torch.nn.functional.conv1d(sig_padded, h_w, padding=0)
            # conv_out shape: (1, 1, cup_len + fLen - 1)
            conv_len = conv_out.shape[2]

            # Extract segment [so : so+Ls] safely
            end_idx = min(so + Ls, conv_len)
            output_len = end_idx - so
            if output_len > 0:
                f[:output_len, w] += conv_out[0, 0, so:end_idx]

    if squeezed:
        return f.squeeze(1)
    return f


# ---------------------------------------------------------------------------
# comp_extBoundary – boundary extension
# ---------------------------------------------------------------------------


def comp_extBoundary(f: torch.Tensor, extLen: int, mode: str) -> torch.Tensor:
    """Extend a 1-D or 2-D tensor along dim 0 by *extLen* on each side.

    Differentiable port of ``numpy.core.comp_extBoundary``.

    Parameters
    ----------
    f      : (L,) or (L, W) tensor
    extLen : non-negative int
    mode   : boundary type — ``per``, ``ppd``, ``zpd``, ``zero``,
             ``sym``, ``even``, ``symw``, ``asym``, ``odd``,
             ``asymw``, ``sp0``

    Returns
    -------
    fout : (L + 2*extLen, ...) tensor
    """
    squeezed = f.dim() == 1
    if squeezed:
        f = f.unsqueeze(1)
    L = f.shape[0]

    fout = torch.zeros(L + 2 * extLen, f.shape[1], dtype=f.dtype, device=f.device)
    fout[extLen : extLen + L, :] = f

    if extLen == 0:
        return fout.squeeze(1) if squeezed else fout

    legal = min(L, extLen)

    if mode in ("per", "ppd"):
        times = extLen // L
        mod_ = extLen % L
        if times > 0:
            tile = f.repeat(times, 1)
            fout[extLen - times * L : extLen, :] = tile
            fout[L + extLen : L + extLen + times * L, :] = tile
        if mod_ > 0:
            fout[:mod_, :] = f[L - mod_ :, :]
            fout[L + extLen + times * L :, :] = f[:mod_, :]
    elif mode in ("zpd", "zero", "valid"):
        pass  # already zero
    elif mode in ("sym", "even"):
        fout[extLen - legal : extLen, :] = f[:legal].flip(0)
        fout[L + extLen : L + extLen + legal, :] = f[L - legal :].flip(0)
    elif mode == "symw":
        lw = min(L - 1, extLen)
        fout[extLen - lw : extLen, :] = f[1 : lw + 1].flip(0)
        fout[L + extLen : L + extLen + lw, :] = f[L - lw - 1 : L - 1].flip(0)
    elif mode in ("asym", "odd"):
        fout[extLen - legal : extLen, :] = -f[:legal].flip(0)
        fout[L + extLen : L + extLen + legal, :] = -f[L - legal :].flip(0)
    elif mode == "asymw":
        lw = min(L - 1, extLen)
        fout[extLen - lw : extLen, :] = -f[1 : lw + 1].flip(0)
        fout[L + extLen : L + extLen + lw, :] = -f[L - lw - 1 : L - 1].flip(0)
    elif mode == "sp0":
        fout[:extLen, :] = f[0:1, :]
        fout[L + extLen :, :] = f[L - 1 : L, :]
    else:
        raise ValueError(f"comp_extBoundary: unsupported mode '{mode}'")

    return fout.squeeze(1) if squeezed else fout


# ---------------------------------------------------------------------------
# Up/down-sampling helpers
# ---------------------------------------------------------------------------


def comp_ups(f: torch.Tensor, a: int) -> torch.Tensor:
    """Upsample *f* by factor *a* (zero-insertion).

    Differentiable: gradients flow through the non-zero positions.

    Parameters
    ----------
    f : (N,) or (N, W) tensor
    a : positive int

    Returns
    -------
    out : (N*a,) or (N*a, W) tensor
    """
    if f.dim() == 1:
        out = torch.zeros(len(f) * a, dtype=f.dtype, device=f.device)
        out[::a] = f
        return out
    N, W = f.shape
    out = torch.zeros(N * a, W, dtype=f.dtype, device=f.device)
    out[::a, :] = f
    return out


def comp_downs(f: torch.Tensor, a: int, skip: int = 0, L: int | None = None) -> torch.Tensor:
    """Downsample *f*: take every *a*-th sample starting at *skip*.

    Parameters
    ----------
    f    : (Lf,) or (Lf, W) tensor
    a    : positive int
    skip : non-negative int
    L    : if given, slice f[skip:skip+L] first

    Returns
    -------
    out : downsampled tensor
    """
    if L is not None:
        f = f[skip : skip + L]
        skip = 0
    return f[skip::a]


# ---------------------------------------------------------------------------
# pderiv – periodic derivative
# ---------------------------------------------------------------------------


def pderiv(f: torch.Tensor, difforder: float = 4) -> torch.Tensor:
    """Periodic derivative of a signal on [0, 1).

    Differentiable port of ``numpy.core.pderiv``.

    Parameters
    ----------
    f : (L,) tensor
    difforder : 2, 4, or inf (spectral)

    Returns
    -------
    fd : (L,) tensor
    """
    L = f.shape[0]

    if difforder == 2:
        return L * (torch.roll(f, -1) - torch.roll(f, 1)) / 2

    elif difforder == 4:
        return (
            L
            * (
                -torch.roll(f, -2)
                + 8 * torch.roll(f, -1)
                - 8 * torch.roll(f, 1)
                + torch.roll(f, 2)
            )
            / 12
        )

    elif difforder == float("inf"):
        n = torch.arange(L, dtype=torch.float64, device=f.device)
        if L % 2 == 0:
            n = torch.where(n <= L // 2, n, n - L)
            n[L // 2] = 0  # zero Nyquist
        else:
            n = torch.where(n <= L // 2, n, n - L)
        F = torch.fft.fft(f.to(torch.complex128))
        fd = 2 * math.pi * torch.fft.ifft(1j * n * F)
        return fd.real if f.is_floating_point() else fd  # type: ignore[no-any-return]

    else:
        raise ValueError(f"difforder must be 2, 4, or inf, got {difforder}")


# ---------------------------------------------------------------------------
# psech – periodized hyperbolic secant
# ---------------------------------------------------------------------------


def psech(L: int, tfr: float = 1.0, device: torch.device | None = None) -> torch.Tensor:
    """Periodised hyperbolic-secant with unit L2-norm.

    Parameters
    ----------
    L : int — window length
    tfr : float — time-frequency ratio (default 1.0)
    device : torch device

    Returns
    -------
    g : (L,) tensor
    """
    safe = 12
    sqrtl = math.sqrt(float(L))
    nk = int(math.ceil(safe / math.sqrt(L / math.sqrt(tfr))))
    lr = torch.arange(L, dtype=torch.float64, device=device)
    g = torch.zeros(L, dtype=torch.float64, device=device)

    for k in range(-nk, nk + 1):
        g = g + 1.0 / torch.cosh(math.pi * (lr / sqrtl - k * sqrtl) / math.sqrt(tfr))

    g = g * math.sqrt(math.pi / (2 * math.sqrt(L * tfr)))
    return g
