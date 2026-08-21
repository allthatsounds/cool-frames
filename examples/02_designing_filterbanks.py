"""
02 - Designing a filterbank
===========================

cool_frames gives you several ways to build the analysis frame, and one string --
``scale=`` -- that selects the frequency warping. This tutorial walks the
designers and the ten scale families, and shows the one habit worth keeping:
*check the frame bounds before you commit* (companion S1.1, point 6 --
perceptually-matched resolution with guaranteed inversion).

Run::

    python examples/02_designing_filterbanks.py
"""

from __future__ import annotations

import numpy as np
from cool_frames.numpy.filterbanks import filterbank, filterbankbounds, filterbankdual, ifilterbank
from cool_frames.numpy.filters import (
    audfilters,
    cqtfilters,
    filterbanklength,
    greenwoodfilters,
)
from cool_frames.numpy.filters.lowlevel import blfilter


def _pr(x, g, a, L):
    gd = filterbankdual(g, a, L)
    xr = np.real(ifilterbank(filterbank(x, g, a, L), gd, a, L, real=True))[: len(x)]
    return np.linalg.norm(xr - x) / np.linalg.norm(x)


def main() -> None:
    fs = 16_000
    Ls = 8192
    x = np.random.default_rng(0).standard_normal(Ls)

    print("== The string-driven multi-scale designer: audfilters(scale=...) ==")
    # audfilters realises nine of the ten scale families directly. 'greenwood'
    # needs the cochlear A/alpha/k constants, so it has its own designer
    # (greenwoodfilters) -- see below. 'linear' is the degenerate per-bin
    # (STFT-like) extreme.
    for scale in ["erb", "erb83", "bark", "mel", "mel1000", "log", "semitone", "third-octave"]:
        try:
            g, a, _fc, L, _info = audfilters(fs, Ls, scale=scale)
            A, B = filterbankbounds(g, a, L)
            print(
                f"  scale={scale:12s}  M={len(g):4d}  kappa={B / A:6.2f}  PR={_pr(x, g, a, L):.1e}"
            )
        except Exception as exc:
            print(f"  scale={scale:12s}  unavailable: {str(exc)[:40]}")
    print("  (greenwood -> greenwoodfilters; linear -> degenerate per-bin/STFT-like)")

    print("\n== Dedicated designers ==")
    # Constant-Q (log by construction): pass fmin/fmax/bins.
    g, a, _fc, L, _info = cqtfilters(fs, Ls, fmin=50.0, fmax=fs / 2, bins=12)
    A, B = filterbankbounds(g, a, L)
    print(f"  cqtfilters       M={len(g):3d}  kappa={B / A:6.2f}  PR={_pr(x, g, a, L):.1e}")
    # Greenwood cochlear map (its own A/alpha/k).
    g, a, _fc, L, _info = greenwoodfilters(fs, Ls)
    A, B = filterbankbounds(g, a, L)
    print(f"  greenwoodfilters M={len(g):3d}  kappa={B / A:6.2f}  PR={_pr(x, g, a, L):.1e}")

    print("\n== Rolling your own from single band-limited filters ==")
    # A hand-built 1/3-octave-ish bank via blfilter; check it is a frame (A>0).
    L = filterbanklength(Ls, 1)
    centres = np.array([125, 250, 500, 1000, 2000, 4000], dtype=float)
    a_hop = 64
    g = [blfilter("hann", fc * 0.5, fc, fs=fs, realonly=True) for fc in centres]
    a = np.full(len(g), a_hop)
    A, B = filterbankbounds(g, a, L)
    print(
        f"  custom blfilter bank: M={len(g)}  A={A:.3g}  B={B:.3g}  "
        f"(A>0 => it is a frame: {A > 0})"
    )
    print("  (a hand-built bank is not automatically painless -- always check A>0)")


if __name__ == "__main__":
    main()
