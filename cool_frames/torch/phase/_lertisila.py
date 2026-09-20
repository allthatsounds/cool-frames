"""
torch/phase/_lertisila.py
=========================
RTISI-LA with Le Roux's truncated projection for the torch backend.

Runs :func:`cool_frames.numpy.phase.lertisila` (see ``_rtisila_delegate.py``)
and returns tensors on the caller's device, in the caller's precision.
Same arguments and results as the NumPy function, whose docstring is
reproduced below.
"""

from __future__ import annotations

import inspect

from ...numpy.phase._lertisila import lertisila as _np_lertisila
from ._rtisila_delegate import delegate


def lertisila(s_list, g, a, M=None, **kwargs):
    return delegate(_np_lertisila, s_list, g, a, M, kwargs)


lertisila.__doc__ = (
    "RTISI-LA with Le Roux's truncated projection (torch backend: runs the NumPy "
    "implementation and returns tensors; no gradient).\n\n"
    + inspect.cleandoc(_np_lertisila.__doc__ or "")
)
