"""numpy.phase – phase gradients, reconstruction, reassignment and retrieval.

Everything needed to go from filterbank magnitudes back to a signal, and to
sharpen a time-frequency picture using the phase gradient:

- phase gradients and derivatives (``filterbankphasegrad``,
  ``filterbankphasederiv``), the basis of both reassignment and PGHI;
- phase construction from magnitude: PGHI (``filterbankconstphase``),
  Griffin-Lim and fast Griffin-Lim (``gla``), Le Roux's GLA (``legla``),
  SPSI (``spsi``), the RTISILA family (``rtisila``, ``lertisila``,
  ``gsrtisila``) and ``decolbfgs``;
- reassignment and synchrosqueezing (``filterbankreassign``);
- convergence metrics (``magnitudeerr``, ``magnitudeerrdb``) and the
  time-frequency-ratio helpers (``pghi_findgamma``, ``wpghi_findgamma``).

Unlike the STFT-only implementations found elsewhere, these operate on
arbitrary (including non-uniform) filterbanks.
"""

# --- phase gradients & reconstruction (LTFAT) ---
from ._constphase import filterbankconstphase
from ._decolbfgs import decolbfgs
from ._fbphasegradfrommag import (
    comp_filterbankneighbors,
    comp_filterbankphasegradfrommag,
)
from ._findgamma import pghi_findgamma, wpghi_findgamma
from ._gla import gla
from ._gsrtisila import gsrtisila
from ._legla import legla
from ._lertisila import lertisila
from ._metrics import magnitudeerr, magnitudeerrdb
from ._phasederiv import filterbankphasederiv
from ._phasegrad import filterbankphasegrad
from ._reassign import filterbankreassign, filterbanksynchrosqueeze
from ._rtisila import rtisila
from ._spsi import spsi

__all__ = [
    "comp_filterbankneighbors",
    "comp_filterbankphasegradfrommag",
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
    "pghi_findgamma",
    "rtisila",
    "spsi",
    "wpghi_findgamma",
]
