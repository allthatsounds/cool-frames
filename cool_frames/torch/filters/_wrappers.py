"""
torch/filters/_wrappers.py
==========================
Thin wrappers around NumPy filter design functions.

Each wrapper calls the NumPy implementation, then converts the filter
descriptors so that frequency-response arrays (``H``) are
``torch.Tensor``.  Callable ``H`` entries are materialised at a given
``L`` so the torch backend can work with concrete tensors.

This module is **not** differentiable — filter design is a setup-time
operation.  Gradients flow through the analysis/synthesis kernels, not
through filter construction.
"""

from __future__ import annotations

import numpy as np
import torch

# NumPy filter design functions
from ...numpy.filters import (
    audfilters as _np_audfilters,
)
from ...numpy.filters import (
    cqtfilters as _np_cqtfilters,
)
from ...numpy.filters import (
    firwin as _np_firwin,
)
from ...numpy.filters import (
    warpedfilters as _np_warpedfilters,
)
from ...numpy.filters import (
    waveletfilters as _np_waveletfilters,
)


def _try_import_np_func(name: str):
    """Try to import a numpy filter function that might not exist."""
    try:
        mod = __import__("cool_frames.numpy.filters", fromlist=[name])
        return getattr(mod, name, None)
    except (ImportError, AttributeError):
        return None


_np_gabfilters = _try_import_np_func("gabfilters")


# ---------------------------------------------------------------------------
# Core conversion utility
# ---------------------------------------------------------------------------


def numpy_filters_to_torch(
    g_np: list[dict],
    L: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> list[dict]:
    """Convert a list of NumPy filter dicts to torch filter dicts.

    Each filter dict is copied.  The ``H`` field is materialised (if
    callable) at length *L* and converted to a ``torch.Tensor``.
    Integer/scalar fields (``foff``, ``realonly``, ``delay``, ``fs``)
    are kept as-is.

    Parameters
    ----------
    g_np   : list of M filter dicts from the NumPy backend
    L      : DFT length at which to materialise callable H entries
    device : target device
    dtype  : complex dtype for H tensors

    Returns
    -------
    g_torch : list of M dicts with ``H`` as ``torch.Tensor``

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters as _np_audfilters
    >>> g_np, _, _, _, _ = _np_audfilters(16000, 8000)
    >>> g_torch = numpy_filters_to_torch(g_np, 8000, device='cpu')
    >>> isinstance(g_torch[0]['H'], torch.Tensor)
    True
    """
    g_torch = []
    for gm in g_np:
        d = dict(gm)  # shallow copy
        H = d.get("H")
        if H is not None:
            if callable(H):
                H = H(L)
            H_np = np.asarray(H, dtype=np.complex128)
            d["H"] = torch.tensor(H_np, dtype=dtype, device=device)
        # Also convert foff if callable
        foff = d.get("foff")
        if callable(foff):
            d["foff"] = int(foff(L))
        g_torch.append(d)
    return g_torch


# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------


def audfilters(
    fs: float,
    Ls: int,
    *args,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    **kwargs,
) -> tuple[list[dict], np.ndarray, np.ndarray, int, dict]:
    """Design an auditory filterbank, returning torch-compatible filter dicts.

    All positional/keyword arguments are forwarded to the NumPy
    ``audfilters``.

    Returns
    -------
    g : list of M filter dicts with ``H`` as ``torch.Tensor``
    a : (M,) or (M,2) ndarray of hop sizes
    fc : ndarray of centre frequencies (Hz)
    L  : DFT length
    info : dict of design metadata

    Examples
    --------
    >>> g, a, fc, L, _ = audfilters(16000, 32000, device='cpu')
    >>> len(g)  # Number of filters
    35
    >>> fc.shape
    (35,)
    """
    g_np, a, fc, L, info = _np_audfilters(fs, Ls, *args, **kwargs)
    g = numpy_filters_to_torch(g_np, L, device=device, dtype=dtype)
    return g, a, fc, L, info


def cqtfilters(
    fs: float,
    Ls: int,
    *args,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    **kwargs,
) -> tuple[list[dict], np.ndarray, np.ndarray, int, dict]:
    """Design a CQT filterbank, returning torch-compatible filter dicts.

    All positional/keyword arguments are forwarded to the NumPy
    ``cqtfilters``; ``fmin``/``fmax``/``bins`` are keyword-only.

    Examples
    --------
    >>> g, a, fc, L, _ = cqtfilters(16000, 16000, fmin=50, fmax=5000, bins=48, device='cpu')
    >>> len(g)  # Number of CQT filters  # doctest: +SKIP
    48
    >>> fc.min(), fc.max()  # Centre frequency range  # doctest: +SKIP
    (50.0, 5000.0)
    """
    g_np, a, fc, L, info = _np_cqtfilters(fs, Ls, *args, **kwargs)
    g = numpy_filters_to_torch(g_np, L, device=device, dtype=dtype)
    return g, a, fc, L, info


def waveletfilters(
    fs: float,
    Ls: int,
    *args,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    **kwargs,
) -> tuple[list[dict], np.ndarray, np.ndarray, int, dict]:
    """Design a wavelet filterbank, returning torch-compatible filter dicts.

    All positional/keyword arguments are forwarded to the NumPy
    ``waveletfilters``; ``scales``/``wavelet`` are keyword-only.

    Examples
    --------
    >>> g, a, fc, L, _ = waveletfilters(16000, 1024, scales=[1, 2, 4, 8], device='cpu')  # doctest: +SKIP
    >>> len(g)  # One filter per scale  # doctest: +SKIP
    4
    """
    result = _np_waveletfilters(fs, Ls, *args, **kwargs)
    g_np, a, fc, L, info = result
    g = numpy_filters_to_torch(g_np, L, device=device, dtype=dtype)
    return (g, a, fc, L, info)


def warpedfilters(
    *args,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    **kwargs,
):
    """Design a warped filterbank, returning torch-compatible filter dicts.

    Examples
    --------
    >>> tiny = np.finfo(float).tiny  # clamped: this example only shows the call
    >>> f2s = lambda f: np.log(np.maximum(np.asarray(f, dtype=float), tiny))
    >>> s2f = lambda s: np.exp(np.asarray(s, dtype=float))
    >>> g, a, fc, L, _ = warpedfilters(
    ...     f2s, s2f, 16000, 50.0, 8000.0, 4, 8000, device='cpu')
    >>> isinstance(g[0]['H'], torch.Tensor)
    True
    """
    result = _np_warpedfilters(*args, **kwargs)
    g_np, a, fc, L, *rest = result
    g = numpy_filters_to_torch(g_np, L, device=device, dtype=dtype)
    return (g, a, fc, L, *rest)


def gabfilters(
    *args,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
    **kwargs,
):
    """Design a Gabor filterbank, returning torch-compatible filter dicts.

    Examples
    --------
    >>> g, a, fc, L, *extra = gabfilters(16, 4, 4, 32, device='cpu')  # doctest: +SKIP
    >>> len(g)  # Number of Gabor atoms  # doctest: +SKIP
    16
    """
    if _np_gabfilters is None:
        raise NotImplementedError("gabfilters not available in numpy backend")
    result = _np_gabfilters(*args, **kwargs)
    g_np, a, fc, L, *rest = result
    g = numpy_filters_to_torch(g_np, L, device=device, dtype=dtype)
    return (g, a, fc, L, *rest)


def firwin(
    name: str,
    M: int,
    *args,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
    **kwargs,
) -> torch.Tensor:
    """Compute a FIR window, returning a torch.Tensor.

    All positional arguments are forwarded to the NumPy ``firwin``.

    Parameters
    ----------
    device, dtype :
        Where to place the result and what type to give it.  Every other
        wrapper in this module took these and ``firwin`` did not, so the one
        window-building entry point was the one that always returned a CPU
        float64 tensor — a caller assembling a filterbank on the GPU had to
        notice and move it by hand, and a float32 pipeline silently widened.

        ``dtype`` defaults to ``torch.float64`` rather than the
        ``torch.complex128`` used by the filter-designer wrappers, because a
        FIR window is real by construction; passing a complex dtype is allowed
        and simply yields a complex tensor with zero imaginary part.

    Examples
    --------
    >>> w = firwin('hann', 512)
    >>> w.shape
    torch.Size([512])
    >>> w.dtype
    torch.float64
    >>> firwin('hann', 8, dtype=torch.float32).dtype
    torch.float32
    """
    w_np = _np_firwin(name, M, *args, **kwargs)
    return torch.as_tensor(w_np, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Utility pass-throughs
# ---------------------------------------------------------------------------


def filterbanklength(Ls: int, a) -> int:
    """Compute the next valid transform length.

    Delegates to the NumPy implementation (pure integer arithmetic).

    Examples
    --------
    >>> a = [1, 2, 4, 8]
    >>> L = filterbanklength(16000, a)
    >>> L % 4 == 0  # Must be divisible by gcd of a
    True
    """
    from ...numpy.filters import filterbanklength as _np_filterbanklength

    return _np_filterbanklength(Ls, a)


def filter_freqresp(
    g: dict,
    L: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.complex128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the frequency response of a single filter.

    Parameters
    ----------
    g     : filter dict (NumPy or torch — ``H`` may be callable)
    L     : DFT length
    device, dtype : target device and dtype for output tensors

    Returns
    -------
    H : torch.Tensor, shape (L,)
        Full-length frequency response.
    foff : torch.Tensor, scalar
        Frequency offset in DFT bins.

    Examples
    --------
    >>> g, _, _, L, _ = audfilters(16000, 32000, device='cpu')
    >>> H, foff = filter_freqresp(g[0], L, device='cpu')
    >>> H.shape
    torch.Size([41472])
    """
    from ...numpy.filters import filter_freqresp as _np_filter_freqresp

    # If the filter has a torch H, convert to numpy dict first
    g_np = dict(g)
    H = g_np.get("H")
    if isinstance(H, torch.Tensor):
        g_np["H"] = H.detach().cpu().numpy()

    H_np, foff_np = _np_filter_freqresp(g_np, L)
    H_t = torch.tensor(H_np, dtype=dtype, device=device)
    foff_t = torch.tensor(int(foff_np), device=device)
    return H_t, foff_t
