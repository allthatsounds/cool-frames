"""
06 - Auditory spectrogram gallery
=================================

Perception needs non-uniform frequency resolution *and* exact inversion at
once (companion S1.1, point 6). This demo renders the same signal through
several auditory scales side by side, then tours the ten ``scale=`` families
of ``audfilters`` -- all from invertible frames.

Run::

    python examples/06_auditory_gallery.py

Saves PNGs to examples/_output/ when matplotlib is available; otherwise it
prints the coefficient grid sizes.
"""

from __future__ import annotations

import os

import numpy as np
from cool_frames.numpy.filterbanks import filterbank
from cool_frames.numpy.filters import audfilters, cqtfilters


def _signal(fs, Ls):
    t = np.arange(Ls) / fs
    # rising chirp + a couple of steady harmonics, so the warping is visible
    chirp = np.sin(2 * np.pi * (200 + (3000 - 200) * t / t[-1]) * t)
    tones = 0.5 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 1760 * t))
    x = chirp + tones
    return x / np.max(np.abs(x))


def main() -> None:
    fs = 16_000
    Ls = fs
    x = _signal(fs, Ls)

    banks = {
        "erb": audfilters(fs, Ls, scale="erb")[:4],
        "bark": audfilters(fs, Ls, scale="bark")[:4],
        "mel": audfilters(fs, Ls, scale="mel")[:4],
        "cqt": cqtfilters(fs, Ls, fmin=50.0, fmax=fs / 2, bins=12)[:4],
    }

    coeffs = {}
    for name, (g, a, fc, L) in banks.items():
        c = filterbank(x, g, a, L)
        coeffs[name] = (c, a, fc)
        print(f"{name:5s}: {len(c)} channels")

    _try_gallery(coeffs, fs)

    print("\nScale tour (audfilters scale=...):")
    for scale in [
        "erb",
        "erb83",
        "bark",
        "mel",
        "mel1000",
        "log",
        "semitone",
        "third-octave",
        "linear",
    ]:
        try:
            g, a, fc, L, _info = audfilters(fs, Ls, scale=scale)
            print(f"  {scale:12s}  M={len(g):4d}  fc[0]={fc[0]:7.1f} Hz  fc[-1]={fc[-1]:7.1f} Hz")
        except Exception as exc:
            print(f"  {scale:12s}  unavailable: {str(exc)[:40]}")
    print("  (greenwood has its own designer: greenwoodfilters)")


def _try_gallery(coeffs, fs) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from cool_frames.numpy.filterbanks import plotfilterbank
    except Exception as exc:  # pragma: no cover
        print(f"(gallery plot skipped: {exc})")
        return
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (name, (c, a, fc)) in zip(axes.ravel(), coeffs.items()):
        plotfilterbank(c, a, fc=fc, fs=fs, ax=ax, colorbar=False)
        ax.set_title(name.upper())
    fig.suptitle("Same signal, four auditory scales (all invertible)")
    d = os.path.join(os.path.dirname(__file__), "_output")
    os.makedirs(d, exist_ok=True)
    out = os.path.join(d, "06_auditory_gallery.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
