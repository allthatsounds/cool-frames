"""
10 - Frame-health report
========================

A one-call inspection of a filterbank's frame properties -- the questions that
only *exist* for an invertible frame: what are my bounds, is it tight, how
well-conditioned is it? Uses ``cool_frames.filterbanks.analyze_filterbank`` /
``print_report``.

Run::

    python examples/10_frame_health.py
"""

from __future__ import annotations

from cool_frames.numpy.filterbanks import analyze_filterbank, filterbanktight, print_report
from cool_frames.numpy.filters import audfilters, cqtfilters


def report(name, g, a, L):
    print(f"\n=== {name} ===")
    rep = analyze_filterbank(g, a, L)
    print_report(rep)
    return rep


def main() -> None:
    fs = 16_000
    Ls = fs

    g, a, _fc, L, _info = audfilters(fs, Ls, scale="erb")
    report("audfilters (raw)", g, a, L)

    gt = filterbanktight(g, a, L)
    report("audfilters (tightened)", gt, a, L)

    g, a, _fc, L, _info = cqtfilters(fs, Ls, fmin=50.0, fmax=fs / 2, bins=12)
    report("cqtfilters (raw)", g, a, L)


if __name__ == "__main__":
    main()
