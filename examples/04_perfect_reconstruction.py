"""
04 - Perfect reconstruction across every designer
=================================================

The single most important property to verify yourself: every filterbank
designer in cool_frames gives an *exact* inverse (companion S1.1, points 1-3, 8).
This script analyses one signal with each designer, reconstructs with the
canonical dual, and tabulates the round-trip error -- all at the level of
floating-point noise.

Run::

    python examples/04_perfect_reconstruction.py
"""

from __future__ import annotations

import numpy as np
from cool_frames.numpy.filterbanks import filterbank, filterbankbounds, filterbankdual, ifilterbank
from cool_frames.numpy.filters import audfilters, cqtfilters, greenwoodfilters


def _roundtrip(name, g, a, L, x):
    gd = filterbankdual(g, a, L)
    xr = np.real(ifilterbank(filterbank(x, g, a, L), gd, a, L, real=True))[: len(x)]
    err = np.linalg.norm(xr - x) / np.linalg.norm(x)
    A, B = filterbankbounds(g, a, L)
    print(f"  {name:14s}  M={len(g):3d}  kappa={B / A:7.2f}  round-trip={err:.2e}")
    return err


def main() -> None:
    fs = 16_000
    Ls = fs
    x = np.random.default_rng(0).standard_normal(Ls)

    print("Round-trip error, painless designers (perfect reconstruction <=> ~1e-16):")
    errs = []
    g, a, _fc, L, _info = audfilters(fs, Ls, scale="erb")
    errs.append(_roundtrip("audfilters", g, a, L, x))
    g, a, _fc, L, _info = audfilters(fs, Ls, scale="bark")
    errs.append(_roundtrip("audfilters/bark", g, a, L, x))
    g, a, _fc, L, _info = cqtfilters(fs, Ls, fmin=50.0, fmax=fs / 2, bins=12)
    errs.append(_roundtrip("cqtfilters", g, a, L, x))
    g, a, _fc, L, _info = greenwoodfilters(fs, Ls)
    errs.append(_roundtrip("greenwoodfilters", g, a, L, x))

    worst = max(errs)
    print(f"\nworst round-trip error: {worst:.2e}")
    assert worst < 1e-9, "painless designers must reconstruct to machine precision"
    print(
        "OK - the painless auditory/CQT designers are exactly invertible via\n"
        "the closed-form canonical dual."
    )

    # Gabor frames: painless when each filter fits its hop (lenH <= L/a), in
    # which case the direct dual is exact; a well-conditioned non-painless frame
    # is recovered by the iterative inverse (ifilterbankiter / CG).
    from cool_frames.numpy.filterbanks import ifilterbankiter
    from cool_frames.numpy.filters import gabfilters

    print("\nGabor frames (uniform):")
    # painless: small hop a=16 so each length-128 filter fits L/a=125
    g, a, _fc, L, _ = gabfilters(fs, Ls, window="hann", a=16, M=128)
    _roundtrip("gabfilters a=16 (painless, direct dual)", g, a, L, x)
    # non-painless but well-conditioned (short signal): the iterative inverse
    Ls2 = 2048
    x2 = np.random.default_rng(3).standard_normal(Ls2)
    g, a, _fc, L, _ = gabfilters(fs, Ls2, window="hann", a=64, M=128)
    xr, _relres, niter = ifilterbankiter(
        filterbank(x2, g, a, L), g, a, Ls=Ls2, real=True, maxit=100, tol=1e-9
    )
    err = np.linalg.norm(np.real(xr)[:Ls2] - x2) / np.linalg.norm(x2)
    print(f"  gabfilters a=64 (non-painless, iterative): round-trip={err:.2e} in {niter} CG iters")

    # Wavelet bank: with cool_frames's Nyquist highpass complement (highpass='auto')
    # and scales spanning toward Nyquist, the real wavelet bank is invertible;
    # it is non-painless, so reconstruct with the iterative inverse.
    from cool_frames.numpy.filters import waveletfilters

    print("\nWavelet bank (with Nyquist complement):")
    scales = 2.0 ** np.linspace(5, -3, 40)
    g, a, _fc, L, _ = waveletfilters(fs, Ls2, scales=scales)
    xr, _relres, niter = ifilterbankiter(
        filterbank(x2, g, a, L), g, a, Ls=Ls2, real=True, maxit=300, tol=1e-9
    )
    err = np.linalg.norm(np.real(xr)[:Ls2] - x2) / np.linalg.norm(x2)
    print(
        f"  waveletfilters (scales→Nyquist, iterative): round-trip={err:.2e} in {niter} CG iters"
    )
    print(
        "\n(Painless frames invert in one step; the iterative inverse, fixed to "
        "report\n the true residual, recovers non-painless frames "
        "incl. the wavelet bank.)"
    )


if __name__ == "__main__":
    main()
