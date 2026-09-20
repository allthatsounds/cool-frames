"""
torch/phase/_gsrtisila.py
=========================
Gnann and Spiertz's RTISI-LA for the torch backend.

Runs :func:`cool_frames.numpy.phase.gsrtisila` (see ``_rtisila_delegate.py``)
and returns tensors on the caller's device, in the caller's precision.
Same arguments and results as the NumPy function, whose docstring is
reproduced below.
"""

from __future__ import annotations

import inspect

from ...numpy.phase._gsrtisila import gsrtisila as _np_gsrtisila
from ._rtisila_delegate import delegate


def gsrtisila(s_list, g, a, M=None, **kwargs):
    return delegate(_np_gsrtisila, s_list, g, a, M, kwargs)


gsrtisila.__doc__ = (
    "Gnann and Spiertz's RTISI-LA (torch backend: runs the NumPy "
    "implementation and returns tensors; no gradient).\n\n"
    + inspect.cleandoc(_np_gsrtisila.__doc__ or "")
)
