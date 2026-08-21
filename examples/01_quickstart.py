"""
01 - Quickstart: analyse -> (look) -> synthesise
================================================

The thirty-second tour of cool_frames: turn a signal into time-frequency
coefficients with an auditory filterbank, look at them, and turn them
back into the *exact* same signal.

This is the headline guarantee of the library (companion S1.1, points 1-3
and 8): the analysis is perfectly invertible, so the round-trip error is at
the level of floating-point noise (~1e-16). Everything else cool_frames does --
auditable models, counterfactual edits, falsifiable restoration -- rests on
this property.

Run::

    python examples/01_quickstart.py

No audio I/O required; a synthetic signal is generated.
"""

from __future__ import annotations

import os

import numpy as np
from cool_frames.numpy.filterbanks import filterbank, filterbankdual, ifilterbank
from cool_frames.numpy.filters import audfilters


def main() -> None:
    fs = 16_000
    Ls = fs  # 1 second

    # A simple harmonic tone with vibrato -- something with structure to look at.
    t = np.arange(Ls) / fs
    f0 = 220.0
    vib = 1.0 + 0.01 * np.sin(2 * np.pi * 5.0 * t)
    x = sum(np.sin(2 * np.pi * k * f0 * vib * t) / k for k in range(1, 8))
    x = x / np.max(np.abs(x))

    # 1. Design an ERB-spaced auditory filterbank.
    g, a, fc, L, _info = audfilters(fs, Ls, scale="erb")
    print(f"filterbank: {len(g)} channels, fs={fs} Hz, L={L}")

    # 2. Analyse: signal -> non-uniform time-frequency coefficients.
    c = filterbank(x, g, a, L)
    print(
        f"coefficients: {len(c)} subbands, "
        f"lengths {min(len(np.asarray(cm)) for cm in c)}..{max(len(np.asarray(cm)) for cm in c)}"
    )

    # 3. Synthesise: coefficients -> signal, using the canonical dual frame.
    gd = filterbankdual(g, a, L)  # canonical dual (real=True)
    x_rec = np.real(ifilterbank(c, gd, a, L, real=True))[:Ls]

    rel_err = np.linalg.norm(x_rec - x) / np.linalg.norm(x)
    print(f"round-trip relative error: {rel_err:.2e}   (perfect reconstruction <=> ~1e-16)")
    assert rel_err < 1e-10, "perfect reconstruction should hold"

    # 4. (optional) look at the coefficients as an auditory spectrogram.
    _try_plot(c, a, fc, fs)


def _try_plot(c, a, fc, fs) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from cool_frames.numpy.filterbanks import plotfilterbank
    except Exception as exc:  # pragma: no cover - plotting is optional
        print(f"(plot skipped: {exc})")
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    plotfilterbank(c, a, fc=fc, fs=fs, ax=ax)
    ax.set_title("ERB filterbank coefficients (auditory spectrogram)")
    out = _outpath("01_quickstart_spectrogram.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


def _outpath(name: str) -> str:
    d = os.path.join(os.path.dirname(__file__), "_output")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


if __name__ == "__main__":
    main()
