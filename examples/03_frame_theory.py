"""
03 - Frame theory, hands-on
===========================

The frame operator is the audio analogue of W^T W, and the frame bounds
A <= B (condition number kappa = B/A) are the conditioning knob (companion
S1.1, point 5). A *tight* frame has kappa = 1 -- the orthogonal-init analogue
for a learned front end.

This demo shows:
  * the raw bounds of an auditory bank,
  * the canonical dual and tight frames,
  * ``partial_tighten`` sweeping the frame from raw (alpha=0) to tight (alpha=1),
  * ``filterbankbounds_svd`` -- the exact frame-operator eigenvalues -- used as
    a ground-truth cross-check of the cheap painless-diagonal bounds.

Run::

    python examples/03_frame_theory.py
"""

from __future__ import annotations

import numpy as np
from cool_frames.numpy.filterbanks import (
    filterbank,
    filterbankbounds,
    filterbankbounds_svd,
    filterbanktight,
    ifilterbank,
)
from cool_frames.numpy.filters import audfilters, partial_tighten


def main() -> None:
    fs = 16_000
    # filterbankbounds_svd materialises an L x L frame operator (O(L^2)), so we
    # use a short signal here -- it is a verification tool, not the hot path.
    Ls = 2048
    g, a, _fc, L, _info = audfilters(fs, Ls, scale="erb")

    A, B = filterbankbounds(g, a, L)
    As, Bs = filterbankbounds_svd(g, a, L)
    print("Raw auditory bank")
    print(f"  diagonal-formula bounds : A={A:.4g}  B={B:.4g}  kappa={B / A:.4f}")
    print(f"  SVD (exact) bounds      : A={As:.4g}  B={Bs:.4g}  kappa={Bs / As:.4f}")
    print(f"  painless? (kappa match) : {abs(B / A - Bs / As) < 1e-3 * Bs / As}")

    gt = filterbanktight(g, a, L)
    At, Bt = filterbankbounds(gt, a, L)
    Ats, Bts = filterbankbounds_svd(gt, a, L)
    print("\nCanonical tight frame  (filterbanktight)")
    print(f"  kappa(diagonal)={Bt / At:.6f}   kappa(SVD)={Bts / Ats:.6f}   (tight <=> kappa=1)")

    print("\npartial_tighten sweep  (alpha: 0 = raw, 1 = tight)")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ga = partial_tighten(g, a, L, alpha)
        Aa, Ba = filterbankbounds(ga, a, L)
        print(f"  alpha={alpha:4.2f}  kappa={Ba / Aa:8.4f}")

    # A tight frame is self-dual: synth(analyse(x)) = x (up to the frame constant).
    x = np.random.default_rng(1).standard_normal(Ls)
    xr = np.real(ifilterbank(filterbank(x, gt, a, L), gt, a, L, real=True))[:Ls]
    print(
        f"\ntight-frame self-reconstruction error: "
        f"{np.linalg.norm(xr - x) / np.linalg.norm(x):.2e}"
    )


if __name__ == "__main__":
    main()
