"""
torch/_dtypes.py
================
Dtype resolution for the PyTorch backend.

The backend used to hard-code ``torch.complex128`` everywhere, so a ``float32``
signal came back as ``complex128`` coefficients computed in double precision.
For the backend's intended use — these ops inside a network, on a GPU, in mixed
precision — that silently doubled memory and produced a dtype mismatch against
the rest of the model.

The rule now: **the caller's dtype wins.**  A ``float32``/``complex64`` input
gives ``complex64`` coefficients; ``float64``/``complex128`` gives
``complex128``; half precision is promoted to single, because ``torch.fft`` has
no half-precision CPU kernels and accumulating an FFT in ``float16`` is not
something to do by accident.

Filters are materialised in whatever complex dtype the *signal* implies, not the
other way round: filter dicts come from the NumPy side and are always float64,
so letting them drive the choice would upcast every call straight back to double.
"""

from __future__ import annotations

import torch

# Real dtype -> complex dtype.
_TO_COMPLEX = {
    torch.float16: torch.complex64,
    torch.bfloat16: torch.complex64,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128,
    torch.complex64: torch.complex64,
    torch.complex128: torch.complex128,
}

# Complex dtype -> real dtype.
_TO_REAL = {
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}


def complex_dtype(x) -> torch.dtype:
    """The complex dtype to compute in, given a tensor or dtype.

    Anything narrower than single precision is promoted to ``complex64``.
    """
    dt = x.dtype if isinstance(x, torch.Tensor) else x
    return _TO_COMPLEX.get(dt, torch.complex128)


def real_dtype(x) -> torch.dtype:
    """The real dtype matching :func:`complex_dtype` for the same input."""
    return _TO_REAL[complex_dtype(x)]


def resolve(*tensors) -> tuple[torch.dtype, torch.dtype]:
    """``(real, complex)`` dtypes for a collection of inputs.

    The widest input wins, so mixing a ``float32`` signal with ``float64``
    magnitudes computes in double rather than silently discarding precision.
    ``None`` entries and non-tensors are ignored; if nothing is left, the
    historical default (``float64``/``complex128``) applies.
    """
    seen = [complex_dtype(t) for t in tensors if isinstance(t, torch.Tensor)]
    cdt = torch.complex128 if torch.complex128 in seen or not seen else torch.complex64
    return _TO_REAL[cdt], cdt
