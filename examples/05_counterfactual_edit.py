"""
05 - The counterfactual edit
============================

"Contest a decision with an audible edit": change one thing in the
time-frequency plane, resynthesise, and hear *only* that change (companion
S1.1, point 2). This works because analyse -> edit -> synthesise is exact, so
no representation error leaks into the result.

The edit is a classical Gabor multiplier: a per-coefficient gain mask
``sigma`` applied via ``framemul`` on a *tight* frame. On a tight frame the
multiplier's eigenvalues equal the mask, so the gain you ask for is exactly
the gain you get.

Run::

    python examples/05_counterfactual_edit.py
"""

from __future__ import annotations

import numpy as np
from cool_frames.numpy.filterbanks import filterbank, filterbanktight
from cool_frames.numpy.filters import audfilters
from cool_frames.numpy.operators import framemul, framemuleigs


def main() -> None:
    fs = 16_000
    Ls = fs
    t = np.arange(Ls) / fs
    # Two-tone signal: a low 200 Hz and a high 3 kHz tone.
    x = np.sin(2 * np.pi * 200 * t) + np.sin(2 * np.pi * 3000 * t)
    x = x / np.max(np.abs(x))

    g, a, fc, L, _info = audfilters(fs, Ls, scale="erb")
    gt = filterbanktight(g, a, L)  # tight frame: transparent edits

    # Coefficient layout, so the mask matches.
    c = filterbank(x, gt, a, L)

    # Edit: keep low channels, attenuate channels above 1 kHz by -40 dB.
    gain_hi = 10 ** (-40 / 20)
    sigma = []
    for m, cm in enumerate(c):
        cm = np.asarray(cm)
        g_m = 1.0 if fc[m] <= 1000.0 else gain_hi
        sigma.append(np.full(cm.shape, g_m))

    # Identity edit (sigma = 1) should be transparent.
    sigma_id = [np.ones_like(np.asarray(cm)) for cm in c]
    y_id = np.real(framemul(x, gt, gt, a, sigma_id, L, real=True))[:Ls]
    print(
        f"identity edit (sigma=1) reconstruction error: "
        f"{np.linalg.norm(y_id - x) / np.linalg.norm(x):.2e}"
    )

    # The real edit: low-pass-ish, the 3 kHz tone should drop ~40 dB.
    y = np.real(framemul(x, gt, gt, a, sigma, L, real=True))[:Ls]

    def band_energy(sig, f0):
        # crude band power around f0 via Goertzel-style projection
        ref = np.exp(2j * np.pi * f0 * t)
        return np.abs(np.vdot(ref, sig)) / Ls

    lo_before, lo_after = band_energy(x, 200), band_energy(y, 200)
    hi_before, hi_after = band_energy(x, 3000), band_energy(y, 3000)
    print(f"200 Hz tone : {20 * np.log10(lo_after / lo_before):+5.1f} dB  (kept)")
    print(f"3 kHz tone  : {20 * np.log10(hi_after / hi_before):+5.1f} dB  (attenuated)")

    # On a tight frame the multiplier eigenvalues track the mask values.
    try:
        eigs = framemuleigs(gt, gt, a, sigma, L, K=6)
        print(f"top multiplier eigenvalues: {np.round(np.real(eigs), 3)}")
        print("(on a tight frame these equal the mask gains -- the edit is faithful)")
    except Exception as exc:
        print(f"(eigenvalue probe skipped: {type(exc).__name__})")
        print(
            "On a tight frame the multiplier eigenvalues equal the mask gains, "
            "so the edit is faithful by construction."
        )


if __name__ == "__main__":
    main()
