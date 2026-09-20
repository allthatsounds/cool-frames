"""numpy.gabor – the discrete Gabor transform (LTFAT-equivalent core).

``dgt`` / ``idgt`` and the real-signal pair ``dgtreal`` / ``idgtreal`` on a
rectangular lattice, the canonical dual and tight windows (``gabdual``,
``gabtight``), the frame bounds and frame-operator diagonal
(``gabframebounds``, ``gabframediag``) and ``dgtlength``.  Definitions,
normalisation and phase convention are LTFAT's defaults, and so is the choice
of algorithm: the filter-bank algorithm for a window shorter than the signal,
Søndergaard's long-window factorisation for a full-length one.

Example
-------
>>> import numpy as np
>>> from cool_frames.filters import pgauss
>>> from cool_frames.gabor import dgt, gabdual, idgt
>>> a, M, L = 8, 16, 144
>>> g = pgauss(L, a * M / L)
>>> f = np.random.default_rng(0).standard_normal(L)
>>> c = dgt(f, g, a, M)
>>> c.shape
(16, 18)
>>> bool(np.allclose(idgt(c, gabdual(g, a, M, L), a).real, f))
True
"""

from ._dgt import (
    dgt,
    dgtlength,
    dgtreal,
    gabdual,
    gabframebounds,
    gabframediag,
    gabtight,
    idgt,
    idgtreal,
)

__all__ = [
    "dgt",
    "dgtlength",
    "dgtreal",
    "gabdual",
    "gabframebounds",
    "gabframediag",
    "gabtight",
    "idgt",
    "idgtreal",
]
