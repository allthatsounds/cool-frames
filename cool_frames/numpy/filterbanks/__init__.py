"""numpy.filterbanks – analysis/synthesis and frame theory (LTFAT-equivalent).

Analysis and synthesis for uniform and non-uniform filterbanks
(``filterbank`` / ``ifilterbank``, direct and iterative), together with the
frame-theoretic machinery that makes them invertible: frame bounds, dual and
tight frames, painless construction, and operator-level analysis.
"""
# Filter design lives in cool_frames.filters (no longer re-exported here -- import it
# from cool_frames.filters directly).
# Coefficient-domain sparsity primitives (thresh, largest) live in cool_frames.sigproc.
from ._firtools import (
    rampsignal,
    resize_fir,
    transferfunction,
)
from ._core import filterbankiter
from ._analysis import (
    analyze_coefficients,
    analyze_filterbank,
    analyze_frame_operator,
    magresp,
    pgrpdelay,
    print_report,
)
from ._core import (
    filterbank,
    filterbank_is_real,
    filterbanklengthcoef,
    ifilterbank,
    ifilterbankiter,
)
from ._frame import (
    painlessfilterbank,
    filterbankbounds,
    filterbankbounds_svd,
    filterbankdual,
    filterbankfreqz,
    filterbankresponse,
    filterbankscale,
    filterbanktight,
)
from ._plot import plotfilterbank
from ._plotutils import plotfft
from ._utils import filterbankwin, pack_coefficients, unpack_coefficients

__all__ = [
    # --- filterbanks own symbols (LTFAT) ---
    "filterbank", "filterbank_is_real", "ifilterbank", "ifilterbankiter",
    "filterbankiter",
    "filterbanklengthcoef", "filterbankwin", "plotfilterbank",
    "pack_coefficients", "unpack_coefficients",
    "plotfft",
    "filterbankbounds",
    "filterbankbounds_svd",
    "filterbankdual",
    "filterbanktight",
    "painlessfilterbank",
    "filterbankscale", "filterbankresponse", "filterbankfreqz",
    # --- analysis functions ---
    "analyze_coefficients", "analyze_filterbank", "analyze_frame_operator", "print_report",
    "pgrpdelay", "magresp",
    # --- FIR helpers ---
    "transferfunction",
    "resize_fir",
    "rampsignal",
]


