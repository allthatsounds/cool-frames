"""Evaluate a filter descriptor's transfer function, however it is stored.

A filter dict's ``'H'`` (and ``'foff'``) may be either a ``callable(L)`` or an
already-materialised array/int.  Both forms are valid and both occur in
practice: the designers build callables so a bank can be re-evaluated at any
transform length, while ``prepare_filters`` and the torch wrappers materialise
them once for a fixed ``L``.

Several call sites assumed the callable form and did ``gm["H"](L)``
unconditionally, which raised ``TypeError: 'numpy.ndarray' object is not
callable`` (or ``'Tensor' object is not callable``) for every materialised
bank — in effect, for the whole torch backend.  The failures were scattered
across the phase modules and each was found only by tripping over it, so the
check lives in one place now.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def eval_H(H: Any, L: int) -> np.ndarray:
    """Return the transfer function as an array, calling it if it is callable."""
    return np.asarray(H(L) if callable(H) else H)


def eval_foff(foff: Any, L: int) -> int:
    """Return the frequency offset as an int, calling it if it is callable."""
    return int(foff(L) if callable(foff) else (foff or 0))
