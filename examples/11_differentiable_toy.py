"""
11 - A differentiable round-trip (toy)
=====================================

cool_frames's selling point is the intersection of *invertible* and *differentiable*
(companion S1.1, point 4): you can back-propagate a loss through the
analysis->synthesis chain. This is a deliberately tiny illustration -- recover
a single unknown per-band gain by gradient descent through the filterbank --
NOT a learned front end (that is Paper 3/4 territory and is out of scope here).

Requires the PyTorch backend (``cool_frames.torch``); it is skipped cleanly if torch
is unavailable.

Run::

    python examples/11_differentiable_toy.py
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    try:
        import cool_frames.torch as ct
        import torch
    except Exception as exc:  # pragma: no cover - torch is optional
        print(f"(skipped: PyTorch backend not available: {exc})")
        return

    torch.manual_seed(0)
    fs = 16_000
    Ls = fs // 4
    t = torch.arange(Ls) / fs
    x = torch.sin(2 * np.pi * 440 * t) + 0.5 * torch.sin(2 * np.pi * 1760 * t)
    x = x / x.abs().max()

    # Build a (uniform) differentiable filterbank from the torch backend.
    # The exact constructor names mirror the NumPy API; adjust if your
    # cool_frames.torch build differs.
    g, a, _fc, L, _info = ct.filters.audfilters(fs, Ls, scale="erb")

    # Target: attenuate the signal by an unknown per-band gain we will recover.
    true_gain = 0.3
    target = true_gain * x

    # A single learnable scalar gain, optimised through analyse->synthesise.
    gain = torch.nn.Parameter(torch.tensor(1.0))
    opt = torch.optim.Adam([gain], lr=0.1)

    for _step in range(60):
        opt.zero_grad()
        c = ct.filterbanks.filterbank(gain * x, g, a, L)
        y = ct.filterbanks.ifilterbank(
            c, ct.filterbanks.filterbankdual(g, a, L), a, Ls, real=True
        ).real
        loss = torch.mean((y[:Ls] - target) ** 2)
        loss.backward()
        opt.step()

    print(f"recovered gain = {gain.item():.4f}  (true = {true_gain})")
    print("Gradients flowed through analysis -> synthesis: differentiable + invertible.")


if __name__ == "__main__":
    main()
