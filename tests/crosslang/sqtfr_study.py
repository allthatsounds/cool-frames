#!/usr/bin/env python3
"""Everything the sqtfr investigation measures, in one runnable place.

    python tests/crosslang/sqtfr_study.py            # all sections
    python tests/crosslang/sqtfr_study.py --only bank
    python tests/crosslang/sqtfr_study.py --designer cqtfilters

Reads whatever ``tests/reference_data/sqtfr_*.mat`` exists.  Generate those
first, in MATLAB with LTFAT on the path::

    >> cd <repo>/tests
    >> export_sqtfr_reference

Four sections, in dependency order -- each is only meaningful if the one
before it came out clean:

  bank    Do cool-frames and LTFAT build the same filters?  Until this holds,
          designer drift masquerades as a convention error.  Expect the two to
          differ by a constant gain of sqrt(L) (an FFT normalisation that
          cancels in every log-magnitude difference) and to disagree at the DC
          and Nyquist complements, which cool-frames redesigned deliberately.
  kernel  Given LTFAT's own inputs, does cool-frames' gradient function return
          LTFAT's gradients?  With ``edge_mode='ltfat'`` this should be
          1.00000000 everywhere.  The default rescales the two one-sided edge
          channels by 2 -- deliberate, see _fbphasegradfrommag.
  gamma   What gamma did LTFAT use?  Recovered from its own two gradient
          exports via fgrad (a pure scaling, so this is exact).  Do NOT use
          tgrad for this: it also carries the tfr-difference correction, which
          is zero at sqtfr=1 and non-zero at sqtfr=sqrt(tfr), so that route
          silently returns something that is not gamma.
  scan    Which sqtfr actually reconstructs best on cool-frames' own bank,
          over several probe signals.  Single-probe optima disagree wildly, so
          the point of this section is the spread, not any one cell.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")

_REPO = pathlib.Path(__file__).resolve().parents[2]
_REF = pathlib.Path(__file__).resolve().parents[1] / "reference_data"

# Run from anywhere, with or without an installed cool-frames.  Python puts
# the *script's* directory on sys.path, not the working directory, so running
# this from the repo root does not make the in-tree package importable; add
# the repo root explicitly, but only as a fallback so that an installed
# cool-frames still wins.
try:
    import cool_frames  # noqa: F401
except ImportError:
    if (_REPO / "cool_frames" / "__init__.py").exists():
        sys.path.insert(0, str(_REPO))

# How to rebuild each designer's bank in cool-frames with the SAME parameters
# export_sqtfr_reference.m used.  Keep the two in step: if you change a call
# there, change it here.
BUILDERS = {
    "audfilters": lambda fs, Ls: __import__(
        "cool_frames.filters", fromlist=["audfilters"]).audfilters(fs, Ls),
    "cqtfilters": lambda fs, Ls: __import__(
        "cool_frames.filters", fromlist=["cqtfilters"]).cqtfilters(
            fs, Ls, fmin=50.0, fmax=fs / 2 - 100, bins=12),
    # LTFAT 2.6.0 takes the `scales` form, with no fs (the bank is a set of
    # dilations; fs only maps it to Hz).  highpass='none' is what reproduces
    # it -- the wavelet set already ends at Nyquist and cool-frames' default
    # 'auto' appends a second channel there.
    "waveletfilters": lambda fs, Ls: __import__(
        "cool_frames.filters", fromlist=["waveletfilters"]).waveletfilters(
            fs, Ls, scales=4.0 * 2.0 ** (-__import__("numpy").arange(64) / 12),
            highpass="none"),
    "warpedfilters": None,   # needs the warping pair; add one if you export it
}


def _load(designer):
    import scipy.io
    p = _REF / f"sqtfr_{designer}.mat"
    return scipy.io.loadmat(str(p), squeeze_me=True) if p.exists() else None


def _build(designer, fs, Ls):
    b = BUILDERS.get(designer)
    if b is None:
        return None
    try:
        return b(fs, Ls)
    except Exception as exc:
        print(f"    cool-frames {designer} would not build: "
              f"{type(exc).__name__}: {exc}")
        return None


def _responses(g, L):
    """Full-length |H| per channel as (L, M)."""
    M = len(g)
    out = np.zeros((L, M))
    for m in range(M):
        H = np.asarray(g[m]["H"](L)).ravel()
        fo = int(g[m]["foff"](L)) if callable(g[m]["foff"]) else int(g[m]["foff"])
        col = np.zeros(L, complex)
        col[(np.arange(fo, fo + len(H))) % L] = H
        out[:, m] = np.abs(col)
    return out


# ---------------------------------------------------------------------------

def section_bank(designer, d):
    from cool_frames.numpy.filterbanks._utils import normalise_a
    fs, Ls, L = float(d["fs"]), int(d["Ls"]), int(d["L"])
    built = _build(designer, fs, Ls)
    if built is None:
        return
    g, a, fc_hz, _L_cf, _ = built
    M = len(g)
    lt = np.abs(np.asarray(d["gf"]))
    if lt.ndim == 1:
        lt = lt.reshape(-1, 1)
    if lt.shape != (L, M):
        print(f"    shape mismatch: LTFAT {lt.shape} vs cool-frames {(L, M)} "
              "-> different channel counts, nothing further is comparable")
        return
    cf = _responses(g, L)

    fc_lt = np.asarray(d["fc_n"], float).ravel() * fs / 2.0
    an = normalise_a(a, M)
    a_cf = np.array([an[m, 0] / an[m, 1] for m in range(M)], float)
    a_lt = np.asarray(d["a_rat"], float).ravel()
    print(f"    fc  max |diff| : {np.max(np.abs(fc_lt - np.asarray(fc_hz, float))):.3g} Hz")
    print(f"    a   max |diff| : {np.max(np.abs(a_lt - a_cf)):.3g}")

    gains = np.array([cf[:, m].max() / lt[:, m].max()
                      for m in range(M) if lt[:, m].max() > 0 and cf[:, m].max() > 0])
    if gains.size:
        print(f"    gain CF/LT     : median {np.median(gains):.5g}  "
              f"cv {np.std(gains)/max(abs(np.mean(gains)), 1e-30):.2e}   "
              f"(sqrt(L) = {np.sqrt(L):.5g})")

    # NaN, not 0, where a channel is empty on either side: an all-zero filter
    # is *not comparable*, and scoring it as a perfect match hides exactly the
    # case worth seeing (LTFAT's audfilters Nyquist channel is empty).
    rel = np.full(M, np.nan)
    for m in range(M):
        n1, n2 = np.linalg.norm(lt[:, m]), np.linalg.norm(cf[:, m])
        if n1 > 0 and n2 > 0:
            rel[m] = np.linalg.norm(lt[:, m]/n1 - cf[:, m]/n2)
        else:
            print(f"    channel {m}: empty on "
                  f"{'LTFAT' if n1 == 0 else ''}{' and ' if n1 == n2 == 0 else ''}"
                  f"{'cool-frames' if n2 == 0 else ''} -- not comparable")

    def _fmt(x):
        return "n/a" if not np.isfinite(x) else f"{x:.3e}"

    # A flat tolerance cannot judge this.  A constant-Q bank's lowest channels
    # are only 5-6 bins wide, and a 5-sample Hann is a poor Hann: its own
    # distance to the ideal window is ~0.4, so two *correct* implementations
    # sit ~0.27 apart there purely from where the samples land.  The same
    # designs 200 bins up agree to 6e-3.  So the criterion is relative: a
    # channel only counts as differing if the two implementations are further
    # from each other than either is from the continuum window they are both
    # discretising.
    xs = np.linspace(-0.5, 0.5, 501)
    ideal = 0.5 * (1.0 + np.cos(2 * np.pi * xs))
    ideal = ideal / np.linalg.norm(ideal)

    def _shape(v):
        nz = np.nonzero(v > 1e-12 * v.max())[0]
        seg = v[nz.min():nz.max() + 1]
        u = np.linspace(-0.5, 0.5, seg.size)
        q = np.interp(xs, u, seg / seg.max())
        return q / np.linalg.norm(q)

    exceed = []
    for m in range(1, M - 1):
        if not np.isfinite(rel[m]):
            continue
        own = max(float(np.linalg.norm(_shape(lt[:, m]) - ideal)),
                  float(np.linalg.norm(_shape(cf[:, m]) - ideal)))
        if rel[m] > 1.5 * own:
            exceed.append(m)

    inner = rel[1:-1]
    fin = inner[np.isfinite(inner)]
    print(f"    |H| unit-norm distance: inner median {_fmt(np.median(fin) if fin.size else np.nan)} "
          f"max {_fmt(fin.max() if fin.size else np.nan)} | "
          f"edges {_fmt(rel[0])}, {_fmt(rel[-1])}")
    print(f"    vs own discretisation error: {len(exceed)}/{int(np.isfinite(inner).sum())} "
          f"inner channels exceed it"
          + (f" -> {exceed[:8]}" if exceed else ""))
    verdict = "agree (to within discretisation)" if not exceed else "DIFFER"
    print(f"    -> inner channels {verdict}; edges differ (expected: cool-frames "
          "redesigned the DC and Nyquist complements)")


def section_kernel(designer, d):
    from cool_frames.numpy.phase._fbphasegradfrommag import (
        comp_filterbankneighbors,
        comp_filterbankphasegradfrommag,
    )
    M = int(d["M"])
    N = np.asarray(d["N"], int).ravel()
    a = np.asarray(d["a_rat"], float).ravel()
    fc = np.asarray(d["fc_n"], float).ravel()
    NE, PI = comp_filterbankneighbors(a.astype(int), M, N, do_real=True)
    chan = np.repeat(np.arange(M), N)

    for mags, tkey in (("abss_scaled", "tgrad_one"), ("abss_raw", "tgrad_one_raw")):
        if mags not in d:
            continue
        ref = np.asarray(d[tkey], float).ravel()
        ok = np.abs(ref) > 1e-12 * np.abs(ref).max()
        line = f"    {mags.replace('abss_', '') :7s}"
        for em in ("ltfat", "rescaled"):
            tg, _f, _ = comp_filterbankphasegradfrommag(
                np.asarray(d[mags], float).ravel(), N, a, M, np.ones(M), fc,
                NE, PI, gderivweight=float(d["gderivweight"]), edge_mode=em)
            dev = tg - np.repeat(fc, N)
            r = dev[ok] / ref[ok]
            e = (chan[ok] == 0) | (chan[ok] == M - 1)
            line += (f"  edge_mode={em:8s} inner {np.median(r[~e]):.8f} "
                     f"edge {np.median(r[e]):.8f}")
        print(line)


def section_gamma(designer, d):
    M = int(d["M"])
    N = np.asarray(d["N"], int).ravel()
    tfr = np.asarray(d["tfr"], float).ravel()
    src = str(d["tfr_source"])
    print(f"    info fields : {list(np.atleast_1d(d['info_fields']))}")
    print(f"    tfr source  : {src}")
    fg1 = np.split(np.asarray(d["fgrad_one"], float).ravel(), np.cumsum(N)[:-1])
    fgT = np.split(np.asarray(d["fgrad_tfr"], float).ravel(), np.cumsum(N)[:-1])
    rec = np.full(M, np.nan)
    for m in range(M):
        z = fg1[m]
        ok = np.abs(z) > 1e-14 * max(float(np.abs(z).max()), 1e-30)
        if ok.sum():
            rec[m] = float(np.median(fgT[m][ok] / z[ok]))
    good = np.isfinite(rec)
    if not good.any():
        print("    gamma not recoverable (no fgrad_tfr -- designer had no info.tfr)")
        return
    gv = rec[good]
    print(f"    gamma       : median {np.median(gv):.6g}  "
          f"range [{gv.min():.4g}, {gv.max():.4g}]  "
          f"constant={np.allclose(gv, gv[0], rtol=1e-6)}")
    print(f"    first 6     : {np.array2string(rec[:6], precision=5)}")
    if np.isfinite(tfr).any():
        b = good & np.isfinite(tfr)
        rel = np.abs(rec[b] - tfr[b]) / np.maximum(np.abs(tfr[b]), 1e-30)
        print(f"    vs info.tfr : max rel {np.nanmax(rel):.2e}  "
              f"(>1e-9 means the export or this arithmetic is wrong)")


def section_scan(designer, d, ks=(1, 2, 4)):
    from cool_frames.filterbanks import filterbank, filterbankdual, ifilterbank
    from cool_frames.numpy.filterbanks._utils import normalise_a
    from cool_frames.phase import filterbankconstphase

    fs, Ls = float(d["fs"]), int(d["Ls"])
    tfr = np.abs(np.asarray(d["tfr"], float).ravel())
    built = _build(designer, fs, Ls)
    if built is None:
        return
    g, a, fc_hz, L, _ = built
    M = len(g)
    if len(tfr) != M:
        print(f"    LTFAT tfr has {len(tfr)} entries, cool-frames bank has {M} "
              "channels -> cannot transplant")
        return
    an = normalise_a(a, M)
    a_int = np.array([int(an[m, 0]) for m in range(M)])

    gd = filterbankdual(g, an, L, real=True)

    # Refuse to tabulate reconstruction quality unless the bank actually
    # reconstructs.  Measure it -- do not infer it.
    #
    # The obvious guard, min(filterbankresponse) > 0, is not enough and is
    # actively misleading: that estimator assumes the frame operator is
    # diagonal in frequency, which holds only for a painless bank.  For a
    # waveletfilters bank it reported A = 1.658, kappa = 2.28 -- a healthy
    # frame -- while filterbankbounds_svd, the exact eigenvalue oracle, gave
    # A = 0.0. Not a frame at all. Both filterbankdual and filterbanktight
    # then returned duals with 74-91% reconstruction error, and even
    # ifilterbankiter converged to a 12% residual, because the bank genuinely
    # annihilates a subspace. Every column of the scan -- the signal path and
    # the zero-phase floor included -- was measuring that, not phase
    # retrieval, and the table still looked like a table.
    #
    # One analysis plus one synthesis costs nothing and catches all of it.
    _x = np.random.default_rng(0).standard_normal(L)
    _y = np.real(ifilterbank(filterbank(_x, g, an, L=L), gd, an, Ls=L, real=True))
    _pr = float(np.linalg.norm(_x - _y) / np.linalg.norm(_x))
    if _pr > 1e-8:
        print(f"    this bank does not reconstruct: analysis -> synthesis "
              f"round-trip relative error = {_pr:.3e}")
        print("    (min(filterbankresponse) can still look healthy here -- it "
              "assumes a painless")
        print("     bank. Cross-check with filterbankbounds_svd.) Skipping the "
              "scan: with a")
        print("     broken dual, every column measures the dual, not phase "
              "retrieval.")
        return
    t = np.arange(Ls) / fs
    rng = np.random.default_rng(1)
    probes = {
        "sweep+tones": (0.5*np.sin(2*np.pi*(100*t + 2900/(2*(Ls/fs))*t**2))
                        + np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*1000*t)),
        "white noise": rng.standard_normal(Ls),
        "AM-FM": (1 + 0.5*np.sin(2*np.pi*5*t))*np.sin(
            2*np.pi*800*t + 300*np.sin(2*np.pi*3*t)),
    }

    hdr = f"    {'probe':>12s}"
    for k in ks:
        hdr += f" {f'{k}x LTFAT':>11s}"
    for k in ks:
        hdr += f" {f'{k}x ones':>10s}"
    hdr += f" {'signal':>9s} {'zero':>8s}"
    print(hdr)

    def one_probe(pname, sig):
        # A function, not a loop body: the closures below would otherwise
        # capture the loop variables and every row would score the last probe.
        sig = sig / np.max(np.abs(sig))
        xp = np.zeros(L)
        xp[:Ls] = sig
        c = filterbank(xp, g, an, L=L)
        s_list = [np.abs(np.asarray(ci).ravel()) for ci in c]
        target = np.concatenate([np.abs(np.asarray(x).ravel()) for x in c])

        def sc(cr):
            y = np.real(ifilterbank(cr, gd, an, Ls=L, real=True))
            e = np.concatenate([np.abs(np.asarray(x).ravel())
                                for x in filterbank(y, g, an, L=L)])
            return 20*np.log10(np.linalg.norm(target - e)
                               / (np.linalg.norm(target) + 1e-30))

        def trial(sq):
            cr, _ = filterbankconstphase(s_list, a_int, np.asarray(fc_hz, float),
                                         sqtfr=sq, fs=fs, tol=1e-6, rng=0)
            return sc(cr)

        row = f"    {pname:>12s}"
        for k in ks:
            row += f" {trial(np.sqrt(k*tfr)):11.2f}"
        for k in ks:
            row += f" {trial(np.sqrt(k*np.ones(M))):10.2f}"
        cr, _ = filterbankconstphase(xp, g, an, L, fc_hz, tol=1e-6)
        row += f" {sc(cr):9.2f}"
        row += f" {sc([x.astype(complex) for x in s_list]):8.2f}"
        print(row)

    for pname, sig in probes.items():
        one_probe(pname, sig)


SECTIONS = {"bank": section_bank, "kernel": section_kernel,
            "gamma": section_gamma, "scan": section_scan}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=list(SECTIONS), action="append",
                    help="run just this section (repeatable)")
    ap.add_argument("--designer", action="append",
                    help="restrict to this designer (repeatable)")
    args = ap.parse_args()

    try:
        import scipy.io  # noqa: F401
    except ImportError:
        print("scipy is required to read the .mat files")
        return 1

    found = sorted(p.stem.replace("sqtfr_", "") for p in _REF.glob("sqtfr_*.mat"))
    if args.designer:
        found = [x for x in found if x in args.designer]
    if not found:
        print(f"No reference files in {_REF}.")
        print("In MATLAB, with LTFAT on the path:")
        print("    >> cd <repo>/tests")
        print("    >> export_sqtfr_reference")
        return 1

    want = args.only or list(SECTIONS)
    for designer in found:
        d = _load(designer)
        print(f"\n{'='*74}\n{designer}   M={int(d['M'])}  L={int(d['L'])}  "
              f"fs={float(d['fs']):g}  Ls={int(d['Ls'])}")
        for name in SECTIONS:
            if name not in want:
                continue
            print(f"  -- {name} --")
            try:
                SECTIONS[name](designer, d)
            except Exception as exc:
                print(f"    section failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
