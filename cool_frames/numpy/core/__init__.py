"""numpy.core – low-level FFT kernels and math utilities.

Includes vendored primitives from the former ``ltfat_core`` package
(``_math``, ``_fourier``, ``_norm``), so there is no external dependency.
"""

from ._core import (
    comp_extBoundary,
    comp_filterbank_fft,
    comp_filterbank_fftbl,
    comp_filterbank_td,
    comp_ifilterbank_fft,
    comp_ifilterbank_fftbl,
    comp_ifilterbank_td,
    filterbanklength,
    floor23,
    involute,
    middlepad,
    modcent,
    postpad,
    setnorm,
)


def __getattr__(name):
    """Lazy re-export of sigproc utilities to avoid circular imports."""
    if name == "resize_fir":
        from ..filterbanks._firtools import resize_fir

        globals()["resize_fir"] = resize_fir
        return resize_fir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "comp_extBoundary",
    "comp_filterbank_fft",
    "comp_filterbank_fftbl",
    "comp_filterbank_td",
    "comp_ifilterbank_fft",
    "comp_ifilterbank_fftbl",
    "comp_ifilterbank_td",
    "filterbanklength",
    "floor23",
    "involute",
    "middlepad",
    "modcent",
    "postpad",
    "resize_fir",
    "setnorm",
]
