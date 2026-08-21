"""cool_frames.torch.phase — differentiable phase gradients, reconstruction and retrieval.

Torch port of :mod:`cool_frames.numpy.phase`: PGHI, Griffin-Lim / fast Griffin-Lim,
LeGLA, SPSI, the RTISILA family, reassignment and phase derivatives. Every
algorithm here is differentiable with respect to the input magnitudes, so a
phase-retrieval step can sit inside a training graph.
"""

from ._constphase import filterbankconstphase
from ._decolbfgs import decolbfgs
from ._fbphasegradfrommag import (
    comp_filterbankneighbors,
    comp_filterbankphasegradfrommag,
)
from ._gla import gla
from ._gsrtisila import gsrtisila
from ._legla import legla
from ._lertisila import lertisila
from ._metrics import magnitudeerr, magnitudeerrdb
from ._phasederiv import (
    comp_filterbankphasederiv,
    comp_phasederivfilters_2nd,
    filterbankphasederiv,
)
from ._phasegrad import comp_filterbankphasegrad, filterbankphasegrad
from ._reassign import (
    comp_filterbankreassign,
    filterbankreassign,
    filterbanksynchrosqueeze,
)
from ._rtisila import rtisila
from ._spsi import spsi

__all__ = [
    "comp_filterbankneighbors",
    "comp_filterbankphasederiv",
    "comp_filterbankphasegrad",
    "comp_filterbankphasegradfrommag",
    "comp_filterbankreassign",
    "comp_phasederivfilters_2nd",
    "decolbfgs",
    "filterbankconstphase",
    "filterbankphasederiv",
    "filterbankphasegrad",
    "filterbankreassign",
    "filterbanksynchrosqueeze",
    "gla",
    "gsrtisila",
    "legla",
    "lertisila",
    "magnitudeerr",
    "magnitudeerrdb",
    "rtisila",
    "spsi",
]
