#!/usr/bin/env python3
"""Print LTFAT's per-designer TFR convention from the exported reference.

Not a test — a report. Run it after ``export_sqtfr_reference`` in MATLAB:

    python tests/crosslang/report_sqtfr.py

For each designer it prints the gamma LTFAT actually used, per channel, and
checks it against each convention currently documented on
``filterbankconstphase``. Whichever line comes out at a flat ratio of 1.0 is
the right one; if none does, the printed gamma column *is* the answer and the
docstring needs replacing with whatever formula reproduces it.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

_REF = pathlib.Path(__file__).resolve().parents[1] / "reference_data"
DESIGNERS = ["audfilters", "cqtfilters", "waveletfilters", "warpedfilters"]


def main() -> int:
    try:
        import scipy.io
    except ImportError:
        print("scipy is required to read the .mat files")
        return 1

    found = 0
    for designer in DESIGNERS:
        path = _REF / f"sqtfr_{designer}.mat"
        if not path.exists():
            print(f"\n=== {designer} ===\n  no {path.name} — not exported")
            continue
        found += 1
        d = scipy.io.loadmat(str(path), squeeze_me=True)

        M = int(d["M"])
        L = int(d["L"])
        N = np.asarray(d["N"], int).ravel()
        fc_n = np.asarray(d["fc_n"], float).ravel()
        tfr = np.asarray(d["tfr"], float).ravel()
        src = str(d["tfr_source"])

        print(f"\n=== {designer} ===   M={M}  L={L}")
        print(f"  info fields : {list(np.atleast_1d(d['info_fields']))}")
        print(f"  tfr source  : {src}")
        if src == "absent" or not np.isfinite(tfr).any():
            print("  -> LTFAT exposes no info.tfr for this designer.")
            print("     The convention has to come from the gradients instead;")
            print("     see the gamma column below.")

        # gamma recovered from LTFAT's own two gradient exports.
        #
        # Use *fgrad*, not tgrad.  fgrad is scaled by exactly gamma/(2 pi) * N,
        # a pure multiply, so the ratio between the two exports is gamma to
        # machine precision.  tgrad looks like it should work too -- it carries
        # a 1/gamma -- but the tfr-difference correction it also carries is
        # zero at sqtfr = 1 and non-zero at sqtfr = sqrt(tfr), so that route
        # silently returns something that is not gamma.  (Measured: the fgrad
        # route reproduces info.tfr to 2e-16; the tgrad route disagrees on
        # every channel.)
        fg_one = np.split(np.asarray(d["fgrad_one"], float).ravel(),
                          np.cumsum(N)[:-1])
        fg_tfr = np.split(np.asarray(d["fgrad_tfr"], float).ravel(),
                          np.cumsum(N)[:-1])
        rec = np.full(M, np.nan)
        for m in range(M):
            z1 = fg_one[m]
            ok = np.abs(z1) > 1e-14 * max(np.abs(z1).max(), 1e-30)
            if ok.sum():
                rec[m] = float(np.median(fg_tfr[m][ok] / z1[ok]))

        if np.isfinite(rec).any():
            g = rec[np.isfinite(rec)]
            print(f"  gamma (LTFAT): median {np.median(g):.6g}  "
                  f"range [{g.min():.4g}, {g.max():.4g}]")
            print(f"    first 8     : {np.array2string(rec[:8], precision=5)}")
            flat = np.allclose(g, g[0], rtol=1e-6)
            print(f"    constant across channels: {flat}")
            for label, cand in (("ones(M)", np.ones(M)),):
                r = rec / cand
                r = r[np.isfinite(r)]
                if r.size:
                    print(f"    gamma / {label:<12s} median {np.median(r):.6g}"
                          f"  (1.0 would confirm this convention)")

        # does the exported tfr agree with what the gradients imply?
        if np.isfinite(tfr).any() and np.isfinite(rec).any():
            both = np.isfinite(tfr) & np.isfinite(rec)
            rel = np.abs(rec[both] - tfr[both]) / np.maximum(np.abs(tfr[both]), 1e-30)
            print(f"  self-consistency (recovered vs info.tfr): "
                  f"max rel {np.nanmax(rel):.2e}")

    if not found:
        print("\nNo reference files. In MATLAB, with LTFAT on the path:")
        print("    >> cd <repo>/tests")
        print("    >> export_sqtfr_reference")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
