"""cool_frames.torch.filterbanks — differentiable analysis, synthesis and frame theory.

Torch port of :mod:`cool_frames.numpy.filterbanks`. Analysis and synthesis are
differentiable with respect to the signal, and the frame-theoretic helpers
(bounds, dual, tight) are available on tensors so a frame constraint can be
used as a training objective.
"""

from ._core import filterbank, ifilterbank
from ._dense import (
    alias_blocks,
    dense_analyse,
    dense_dual,
    dense_synthesise,
    frame_bounds,
    frame_condition,
    frame_gram,
    frame_response,
    frame_retract,
    is_frame,
    mirror_bank,
    retraction_gain,
)
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
    "alias_blocks",
    "dense_analyse",
    "dense_dual",
    "dense_synthesise",
    "filterbank",
    "filterbankbounds",
    "filterbankdual",
    "filterbankfreqz",
    "filterbankiter",
    "filterbanklengthcoef",
    "filterbankresponse",
    "filterbankscale",
    "filterbanktight",
    "frame_bounds",
    "frame_condition",
    "frame_gram",
    "frame_response",
    "frame_retract",
    "ifilterbank",
    "ifilterbankiter",
    "is_frame",
    "mirror_bank",
    "retraction_gain",
]
