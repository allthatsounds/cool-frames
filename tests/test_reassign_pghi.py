"""
test_reassign_pghi.py
======================
Compare reassignment-enhanced PGHI (RA-PGHI) against existing baselines
on spectral convergence (SC) and reconstruction quality.

Baselines:
  1. rtpghifb_nonuniform   — heap PGHI with magnitude-derived gradients
  2. constphase_nonuniform  — fixed-order PGHI (batch_tgrad, two_phase)
  3. constphase_nonuniform  — fixed-order PGHI (causal)
  4. FGLA-50               — 50 iterations of fast Griffin-Lim
  5. RAAR-100              — 100 iterations of RAAR (β=0.9)

New methods:
  6. reassign_pghi          — oracle exact gradients + heap PGHI
  7. reassign_pghi_iter(1)  — 1 iteration of RA-PGHI (from zero init)
  8. reassign_pghi_iter(3)  — 3 iterations of RA-PGHI
  9. reassign_pghi_iter(5)  — 5 iterations of RA-PGHI

Metrics:
  - Round-trip spectral convergence (SC): synthesis → re-analysis → compare
  - Direct SC: compare reconstructed coefficients against original
"""
from __future__ import annotations

import pytest

# The reassignment-PGHI stack (reassign_pghi/_iter, rtpghifb_nonuniform,
# constphase_nonuniform, raar) was moved out of cool_frames.numpy.phase to the
# audioeffects package in the 2026-06 consolidation; this comparison test
# belongs there. Skipped here until it is relocated/repointed.
pytest.skip(
    "reassign-PGHI baselines moved to audioeffects; relocate this test there",
    allow_module_level=True,
)

import os
import sys
import time

import numpy as np

# Add filterbank to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cool_frames.numpy.filters import audfilters, cqtfilters
from cool_frames.numpy.filterbanks import filterbank, ifilterbank
from cool_frames.numpy.filterbanks._frame import filterbankdual
from cool_frames.numpy.phase import (
    constphase_nonuniform,
    filterbankconstphase,
    filterbankphasegrad,
    gla,
    raar,
    reassign_pghi,
    reassign_pghi_iter,
    rtpghifb_nonuniform,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sc_roundtrip(s_list, c_rec, g, a, L):
    """Round-trip spectral convergence (dB).

    Synthesise from c_rec using dual frame, re-analyse, compare magnitudes
    against original s_list.
    """
    gd = filterbankdual(g, a, L)
    x_rec = ifilterbank(c_rec, gd, a, Ls=L)
    c_re = filterbank(x_rec, g, a, L=L)

    num = sum(np.sum(np.abs(np.abs(c_re[m]) - s_list[m]) ** 2)
              for m in range(len(s_list)))
    den = sum(np.sum(s_list[m] ** 2) for m in range(len(s_list)))
    if den < 1e-30:
        return -np.inf
    return -10 * np.log10(num / den)


def sc_direct(c_orig, c_rec):
    """Direct spectral convergence (dB).

    Compare coefficient magnitudes directly (no round-trip).
    """
    num = sum(np.sum(np.abs(np.abs(c_rec[m]) - np.abs(c_orig[m])) ** 2)
              for m in range(len(c_orig)))
    den = sum(np.sum(np.abs(c_orig[m]) ** 2) for m in range(len(c_orig)))
    if den < 1e-30:
        return -np.inf
    return -10 * np.log10(num / den)


def phase_error(c_orig, c_rec):
    """Mean absolute phase error (radians) on above-threshold coefficients."""
    errs = []
    for m in range(len(c_orig)):
        mask = np.abs(c_orig[m]) > 1e-6 * np.max(np.abs(c_orig[m]))
        if mask.any():
            pe = np.abs(np.angle(c_rec[m][mask] * np.conj(c_orig[m][mask])))
            errs.append(np.mean(pe))
    return np.mean(errs) if errs else np.nan


# ---------------------------------------------------------------------------
# Test signals
# ---------------------------------------------------------------------------

def make_test_signals(fs=16000, dur=2.0):
    """Generate a set of test signals."""
    n = int(fs * dur)
    t = np.arange(n) / fs
    signals = {}

    # 1. Simple sinusoid
    signals["sine_440"] = 0.5 * np.sin(2 * np.pi * 440 * t)

    # 2. Two sinusoids (close frequencies — hard for PGHI)
    signals["two_sines"] = (0.3 * np.sin(2 * np.pi * 440 * t)
                            + 0.3 * np.sin(2 * np.pi * 480 * t))

    # 3. Chirp (linear sweep)
    f0, f1 = 200, 2000
    signals["chirp"] = 0.5 * np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * dur) * t ** 2))

    # 4. Click train (impulses — hard for magnitude-based gradient estimation)
    click = np.zeros(n)
    for k in range(0, n, int(fs * 0.1)):
        click[k] = 1.0
    signals["clicks"] = click

    # 5. White noise
    rng = np.random.default_rng(42)
    signals["noise"] = 0.3 * rng.standard_normal(n)

    # 6. AM signal (amplitude modulation)
    carrier = np.sin(2 * np.pi * 1000 * t)
    modulator = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    signals["am_signal"] = modulator * carrier

    # 7. Multi-component (realistic music-like)
    signals["multi"] = sum(
        (0.2 / (k + 1)) * np.sin(2 * np.pi * 220 * (k + 1) * t + rng.uniform(0, 2 * np.pi))
        for k in range(8)
    )

    return signals


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def run_comparison(fb_type="aud", fs=16000, dur=2.0):
    """Run all methods on all test signals, print results."""
    signals = make_test_signals(fs, dur)
    n = int(fs * dur)

    # --- Build filterbank ---
    if fb_type == "aud":
        g, a, fc, L, _info = audfilters(fs, n)
        fb_name = f"audfilters (ERBlet, {len(g)} channels)"
    elif fb_type == "cqt":
        g, a, fc, L, _info = cqtfilters(fs, n, fmin=50, fmax=fs // 2, bins=12)
        fb_name = f"cqtfilters ({len(g)} channels)"
    else:
        raise ValueError(f"Unknown fb_type: {fb_type}")

    a_arr = np.array([int(ai[0]) if hasattr(ai, '__len__') else int(ai)
                       for ai in np.atleast_2d(a)])
    if a_arr.ndim == 2:
        a_arr = a_arr[:, 0]
    a_arr = np.asarray(a, dtype=int).ravel() if np.ndim(a) <= 1 else np.asarray(a)[:, 0].astype(int)
    M = len(g)

    # Get tfr from filters
    tfr = np.ones(M)
    for m in range(M):
        if "tfr" in g[m]:
            tfr_val = g[m]["tfr"]
            tfr[m] = float(tfr_val(L) if callable(tfr_val) else tfr_val)

    # Normalised centre frequencies
    fc_arr = np.asarray(fc, dtype=float)
    fc_norm = fc_arr / fs * 2.0

    print(f"\n{'='*80}")
    print(f"Filterbank: {fb_name}")
    print(f"Signal: {fs} Hz, {dur}s ({n} samples), L={L}")
    print(f"{'='*80}")

    # Header
    methods = [
        "rtpghifb_nonuniform",
        "constphase(batch_tgrad)",
        "constphase(causal)",
        "FGLA-50",
        "RAAR-100",
        "RA-PGHI (oracle)",
        "RA-PGHI iter=1",
        "RA-PGHI iter=3",
        "RA-PGHI iter=5",
    ]

    # Print table header
    print(f"\n{'Signal':15s} | ", end="")
    for m in methods:
        print(f"{m:>22s}", end=" | ")
    print()
    print("-" * (15 + 3 + (22 + 3) * len(methods)))

    all_results = {}

    for sig_name, x in signals.items():
        # Analyse
        c_orig = filterbank(x, g, a, L=L)
        s_list = [np.abs(ci) for ci in c_orig]

        results = {}
        timings = {}

        # 1. rtpghifb_nonuniform (heap + mag-derived gradients)
        t0 = time.time()
        c_heap, _, _, _ = rtpghifb_nonuniform(s_list, a_arr, fc_norm, tfr, tol=1e-6)
        timings["rtpghifb_nonuniform"] = time.time() - t0
        results["rtpghifb_nonuniform"] = sc_roundtrip(s_list, c_heap, g, a, L)

        # 2. constphase_nonuniform (batch_tgrad, two_phase)
        t0 = time.time()
        c_hyb, _, _, _ = constphase_nonuniform(
            s_list, a_arr, fc_norm, tfr,
            integration="two_phase", gradients="batch_tgrad")
        timings["constphase(batch_tgrad)"] = time.time() - t0
        results["constphase(batch_tgrad)"] = sc_roundtrip(s_list, c_hyb, g, a, L)

        # 3. constphase_nonuniform (causal)
        t0 = time.time()
        c_cau, _, _, _ = constphase_nonuniform(
            s_list, a_arr, fc_norm, tfr,
            integration="two_phase", gradients="causal")
        timings["constphase(causal)"] = time.time() - t0
        results["constphase(causal)"] = sc_roundtrip(s_list, c_cau, g, a, L)

        # 4. FGLA-50
        t0 = time.time()
        c_fgla, _, _, _ = gla(s_list, g, a_arr, L=L, real=True,
                               maxit=50, tol=0.0, method='fgla', startphase='zero')
        timings["FGLA-50"] = time.time() - t0
        results["FGLA-50"] = sc_roundtrip(s_list, c_fgla, g, a, L)

        # 5. RAAR-100
        t0 = time.time()
        c_raar, _, _, _ = raar(s_list, g, a_arr, L=L, real=True,
                                maxit=100, tol=0.0, beta=0.9)
        timings["RAAR-100"] = time.time() - t0
        results["RAAR-100"] = sc_roundtrip(s_list, c_raar, g, a, L)

        # 6. RA-PGHI (oracle — uses original signal)
        t0 = time.time()
        c_ra = reassign_pghi(x, g, a, L, fc)
        timings["RA-PGHI (oracle)"] = time.time() - t0
        results["RA-PGHI (oracle)"] = sc_roundtrip(s_list, c_ra, g, a, L)

        # 7–9. RA-PGHI iterative (1, 3, 5 iterations)
        for n_it in [1, 3, 5]:
            key = f"RA-PGHI iter={n_it}"
            t0 = time.time()
            c_rai = reassign_pghi_iter(s_list, g, a, L, fc, n_iter=n_it)
            timings[key] = time.time() - t0
            results[key] = sc_roundtrip(s_list, c_rai, g, a, L)

        # Print row
        print(f"{sig_name:15s} | ", end="")
        for m in methods:
            sc_val = results[m]
            print(f"{sc_val:19.2f} dB", end=" | ")
        print()

        all_results[sig_name] = results

    # Summary: mean SC across signals
    print("-" * (15 + 3 + (22 + 3) * len(methods)))
    print(f"{'MEAN':15s} | ", end="")
    for m in methods:
        vals = [all_results[s][m] for s in all_results]
        mean_sc = np.mean(vals)
        print(f"{mean_sc:19.2f} dB", end=" | ")
    print()

    # Timing summary
    print(f"\n{'--- Timing (last signal) ---':40s}")
    for m in methods:
        print(f"  {m:30s}: {timings[m]:7.3f}s")

    return all_results


# ---------------------------------------------------------------------------
# Phase error analysis (oracle mode only)
# ---------------------------------------------------------------------------

def run_phase_error_analysis(fb_type="aud", fs=16000, dur=2.0):
    """Compare phase errors for oracle methods."""
    signals = make_test_signals(fs, dur)
    n = int(fs * dur)

    if fb_type == "aud":
        g, a, fc, L, _info = audfilters(fs, n)
    else:
        g, a, fc, L, _info = cqtfilters(fs, n, fmin=50, fmax=fs // 2, bins=12)

    a_arr = np.asarray(a, dtype=int).ravel() if np.ndim(a) <= 1 else np.asarray(a)[:, 0].astype(int)
    M = len(g)
    tfr = np.ones(M)
    for m in range(M):
        if "tfr" in g[m]:
            tfr_val = g[m]["tfr"]
            tfr[m] = float(tfr_val(L) if callable(tfr_val) else tfr_val)
    fc_norm = np.asarray(fc, dtype=float) / fs * 2.0

    print(f"\n{'='*60}")
    print(f"Phase Error Analysis ({fb_type}, {fs} Hz)")
    print(f"{'='*60}")

    methods = ["constphase(batch_tgrad)", "RA-PGHI (oracle)"]
    print(f"\n{'Signal':15s} | {'constphase':>15s} | {'RA-PGHI oracle':>15s}")
    print("-" * 55)

    for sig_name, x in signals.items():
        c_orig = filterbank(x, g, a, L=L)
        s_list = [np.abs(ci) for ci in c_orig]

        # constphase
        c_hyb, _, _, _ = constphase_nonuniform(
            s_list, a_arr, fc_norm, tfr,
            integration="two_phase", gradients="batch_tgrad")
        pe_hyb = phase_error(c_orig, c_hyb)

        # RA-PGHI oracle
        c_ra = reassign_pghi(x, g, a, L, fc)
        pe_ra = phase_error(c_orig, c_ra)

        print(f"{sig_name:15s} | {pe_hyb:12.4f} rad | {pe_ra:12.4f} rad")

    print()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("  Reassignment-Enhanced PGHI (RA-PGHI) — Baseline Comparison")
    print("=" * 80)

    # Run on ERBlet filterbank
    results_aud = run_comparison("aud", fs=16000, dur=1.0)

    # Phase error analysis
    run_phase_error_analysis("aud", fs=16000, dur=1.0)

    print("\nDone.")
