"""
torch/phase/_rtisila.py
=======================
Real-Time Iterative Spectrogram Inversion with Look-Ahead for the torch backend.

Runs :func:`cool_frames.numpy.phase.rtisila` (see ``_rtisila_delegate.py``)
and returns tensors on the caller's device, in the caller's precision.
Same arguments and results as the NumPy function, whose docstring is
reproduced below.
"""

from __future__ import annotations

import inspect

from ...numpy.phase._rtisila import rtisila as _np_rtisila
from ._rtisila_delegate import delegate


def rtisila(s_list, g, a, M=None, **kwargs):
    return delegate(_np_rtisila, s_list, g, a, M, kwargs)


rtisila.__doc__ = (
    "Real-Time Iterative Spectrogram Inversion with Look-Ahead (torch backend: runs the NumPy "
    "implementation and returns tensors; no gradient).\n\n"
    + inspect.cleandoc(_np_rtisila.__doc__ or "")
)
