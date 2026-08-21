"""cool_frames.torch.filterbanks — differentiable analysis, synthesis and frame theory.

Torch port of :mod:`cool_frames.numpy.filterbanks`. Analysis and synthesis are
differentiable with respect to the signal, and the frame-theoretic helpers
(bounds, dual, tight) are available on tensors so a frame constraint can be
used as a training objective.
"""

from ._core import filterbank, ifilterbank
from ._frame import (
    filterbankbounds,
    filterbankdual,
    filterbankfreqz,
    filterbankiter,
    filterbanklengthcoef,
    filterbankresponse,
    filterbankscale,
    filterbanktight,
    ifilterbankiter,
)

__all__ = [
    "filterbank",
    "filterbankbounds",
    "filterbankdual",
    "filterbankfreqz",
    "filterbankiter",
    "filterbanklengthcoef",
    "filterbankresponse",
    "filterbankscale",
    "filterbanktight",
    "ifilterbank",
    "ifilterbankiter",
]
