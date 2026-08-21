"""
08 - Phase retrieval from magnitudes (classical algorithms)
==========================================================

When a pipeline keeps only the magnitude of the coefficients (mel features,
masking, codecs), phase must be rebuilt before anything can be synthesised.
This demo reconstructs a signal from magnitude-only filterbank coefficients
using the *classical* algorithms shipped in ``cool_frames.phase``:

  * Griffin-Lim (GLA)
  * Fast Griffin-Lim (fGLA, Perraudin momentum)
  * SPSI (single-pass spectrogram inversion)

(The differentiable / real-time RTPGHI variants are a separate research line
and are intentionally not used here.)

Quality is measured in the magnitude domain -- "spectral convergence" -- since
the absolute phase is genuinely unrecoverable.

Run::

    python examples/08_phase_retrieval.py
"""

from __future__ import annotations

import numpy as np
from cool_frames.numpy.filterbanks import filterbank
from cool_frames.numpy.filters import audfilters
from cool_frames.numpy.phase import gla, spsi


def _spectral_convergence(s_target, g, a, L):
    """||  |c(x_rec)| - s  || / || s ||  for a reconstructed signal's coeffs."""

    def mag(sig):
        return [np.abs(np.asarray(cm)) for cm in filterbank(sig, g, a, L)]

    def sc(x_rec):
        s2 = mag(x_rec)
        num = sum(np.sum((t - r) ** 2) for t, r in zip(s_target, s2))
        den = sum(np.sum(t**2) for t in s_target)
        return float(np.sqrt(num / den))

    return sc


def main() -> None:
    fs = 16_000
    Ls = fs // 2
    t = np.arange(Ls) / fs
    # A harmonic, slightly vibrato'd tone -- structured, so phase retrieval works well.
    f0 = 330.0
    x = sum(np.sin(2 * np.pi * k * f0 * t + 0.3 * k) / k for k in range(1, 9))
    x = x / np.max(np.abs(x))

    g, a, fc, L, _info = audfilters(fs, Ls, scale="erb")
    c = filterbank(x, g, a, L)
    s = [np.abs(np.asarray(cm)) for cm in c]  # magnitude-only -- phase discarded
    sc = _spectral_convergence(s, g, a, L)

    print("Reconstructing from magnitude-only coefficients:")
    _rec_g, x_g, _, it_g = gla(s, g, a, L=L, Ls=Ls, real=True, maxit=100, method="gla")
    print(f"  GLA   ({it_g:3d} it): spectral convergence = {sc(np.real(x_g)[:Ls]):.4f}")

    _rec_f, x_f, _, it_f = gla(s, g, a, L=L, Ls=Ls, real=True, maxit=100, method="fgla")
    print(f"  fGLA  ({it_f:3d} it): spectral convergence = {sc(np.real(x_f)[:Ls]):.4f}")

    try:
        _, phase = spsi(s, a, fc, fs)  # fc in Hz, fs required
        c_spsi = [sm * np.exp(1j * ph) for sm, ph in zip(s, phase)]
        from cool_frames.numpy.filterbanks import filterbankdual, ifilterbank

        gd = filterbankdual(g, a, L)
        x_s = np.real(ifilterbank(c_spsi, gd, a, L, real=True))[:Ls]
        print(f"  SPSI  (1-pass): spectral convergence = {sc(x_s):.4f}")
    except Exception as exc:  # pragma: no cover
        print(f"  SPSI skipped: {exc}")

    print("\n(lower is better; fGLA usually beats GLA at equal iterations.)")
    print("PGHI is also available as cool_frames.numpy.phase.filterbankconstphase.")


if __name__ == "__main__":
    main()
