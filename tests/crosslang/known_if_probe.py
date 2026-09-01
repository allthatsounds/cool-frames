"""Which phase-gradient path is right?  Ask a signal whose answer is known.

The signal path (derivative filters) and the magnitude path
(comp_filterbankphasegradfrommag) disagree on interior channels -- in this
package and, measurably, in MATLAB LTFAT too (tests/compare_paths_ltfat.m).
Comparing them against each other cannot say which one is wrong.

A pure tone can.  Its instantaneous frequency is its frequency, everywhere,
so both estimators have a known target.  Run:

    python tests/crosslang/known_if_probe.py

Result, at the time of writing: the signal path recovers the tone frequency
exactly (0.0 % on every probe, both designers) and the magnitude estimator
misses by 0.4-10.7 %.  So the signal path is ground truth and the deviation
belongs to the magnitude estimator.

Read the conclusion carefully.  This shows that *this* estimator -- the one
LTFAT specifies and this package reproduces bit for bit -- is inexact.  It
does not show that magnitude-only estimation is intrinsically limited; that
would need a different algorithm to have been tried and to have failed too.
"""
import pathlib
import sys
import warnings

import numpy as np

warnings.simplefilter("ignore")
_REPO = str(pathlib.Path(__file__).resolve().parents[2])
if _REPO not in sys.path:                      # running from a checkout, not an install
    sys.path.insert(0, _REPO)
from cool_frames.filterbanks import filterbank  # noqa: E402
from cool_frames.filters import audfilters, cqtfilters  # noqa: E402
from cool_frames.numpy.phase import filterbankphasegrad  # noqa: E402
from cool_frames.numpy.phase._fbphasegradfrommag import (  # noqa: E402
    comp_filterbankneighbors,
    comp_filterbankphasegradfrommag,
)

fs, Ls = 8000, 4096
t = np.arange(Ls)/fs

def both_paths(f, g, a, L, fc, tfr):
    M=len(g)
    ar = np.asarray(a, float)
    ar = ar[:, 0] / ar[:, 1] if ar.ndim == 2 else ar
    N=np.ceil(L/ar).astype(int)
    tg_s, _fg_s, _s, _c = filterbankphasegrad(f, g, a, L)
    tg_s=np.concatenate([np.asarray(x).ravel() for x in tg_s])
    cc=filterbank(f,g,a,L=L)
    mag=np.concatenate([np.abs(np.asarray(x)).ravel() for x in cc])
    NEIGH,pos = comp_filterbankneighbors(ar.astype(int), M, N, do_real=True)
    fcn = 2.0*np.asarray(fc,float)/fs
    tg_m,_,_ = comp_filterbankphasegradfrommag(
        mag, N, ar, M, np.sqrt(np.nan_to_num(tfr,nan=1.0)), fcn, NEIGH, pos)
    return tg_s, tg_m, mag, N

for name, build, kw in (("audfilters",audfilters,{}),
                        ("cqtfilters",cqtfilters,dict(fmin=50.,fmax=3900.,bins=12))):
    g,a,fc,L,info = build(fs, Ls, **kw)
    tfr = np.asarray(info["tfr"],float)
    print(f"\n===== {name}  M={len(g)}  L={L}")
    print(f"{'probe':>22s} {'true IF':>9s} {'signal path':>26s} {'magnitude path':>26s}")
    for label, sig, true_if in (
        ("tone 440 Hz", np.sin(2*np.pi*440*t), 440.0),
        ("tone 1000 Hz", np.sin(2*np.pi*1000*t), 1000.0),
        ("tone 2500 Hz", np.sin(2*np.pi*2500*t), 2500.0),
    ):
        tg_s, tg_m, mag, N = both_paths(sig, g, a, L, fc, tfr)
        live = mag > 0.3*mag.max()          # only the cells the tone actually excites
        if live.sum() < 5:
            print(f"{label:>22s}  too few live cells")
            continue
        s_hz = np.median(tg_s[live])*fs/2
        m_hz = np.median(tg_m[live])*fs/2
        print(f"{label:>22s} {true_if:9.1f} "
              f"{s_hz:12.1f} Hz ({100*abs(s_hz-true_if)/true_if:5.1f}% err) "
              f"{m_hz:12.1f} Hz ({100*abs(m_hz-true_if)/true_if:5.1f}% err)")
