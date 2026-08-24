"""cool_frames.torch.phase — differentiable phase gradients, reconstruction and retrieval.

Torch port of :mod:`cool_frames.numpy.phase`: PGHI, Griffin-Lim / fast Griffin-Lim,
LeGLA, SPSI, the RTISILA family, reassignment and phase derivatives.

Not everything here is differentiable, and the difference matters if you are
putting a phase step inside a training graph:

* :func:`gla` is a native torch port and **is** differentiable with respect to
  the input magnitudes -- it returns the reconstructed waveform alongside the
  coefficients, so it is a single-call magnitude-to-waveform op. This is the
  one to reach for.
* :func:`filterbankconstphase` (PGHI) wraps the NumPy heap integrator and is
  **not** differentiable: the traversal order is a discrete function of the
  magnitudes. Its output can still be frozen as a constant phase, with the
  gradient flowing through the magnitude factor into ``ifilterbank``.
* :func:`decolbfgs` runs its inner LBFGS under ``torch.no_grad()``, and
  :func:`gsrtisila` drops to NumPy internally; neither carries a gradient.

Analysis and synthesis themselves (``cool_frames.torch.filterbanks``) are fully
differentiable, with the exception of ``ifilterbankiter``.
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
