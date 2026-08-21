#!/usr/bin/env python3
"""
export_python.py
================
Generate cross-language reference .mat files from the Python filterbank port.

Computes the **same** quantities as export_matlab.m so that the orchestrator
can compare them field by field.

Usage
-----
    python export_python.py                        # → crosslang/results/python/
    python export_python.py --outdir /some/path    # custom output directory

Output files
------------
    params.mat             Signal, filterbank geometry
    filters.mat            Full-length transfer functions (L, M) complex
    filterbank_coeff.mat   c / ch / cd flat complex arrays + N vector
    layer0_ops.mat         Primitive operations (postpad, fftindex, etc.)
    layer1_filters.mat     Filter design outputs (audfilters, cqtfilters, firwin)
    layer2_frame.mat       Frame bounds, dual/tight frames, reconstruction
    engine_info.mat        Python version info
"""
from __future__ import annotations

import argparse
import datetime
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the filterbank package and ltfat_core are importable
_proj_root = Path(__file__).resolve().parent.parent
_fb_root = _proj_root / "filterbank"
_core_root = _proj_root / "ltfat_core"
for _p in [str(_fb_root), str(_core_root)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import scipy.io as sio

# --------------------------------------------------------------------------
# Import the Python filterbank port
# --------------------------------------------------------------------------
from cool_frames.numpy.core._core import (
    postpad,
    floor23,
    involute,
    modcent,
)
from cool_frames.numpy.filters._firwin import firwin
from cool_frames.numpy.filters._filters import filter_freqresp
from cool_frames.filters import audfilters, cqtfilters
from cool_frames.filterbanks import (
    filterbank,
    ifilterbank,
    filterbankdual,
    filterbankbounds,
)
from cool_frames.numpy.filterbanks._utils import normalise_a


def _save_mat(path: Path, data: dict, label: str = ""):
    """Save a dict as a v5 .mat file (scipy-compatible)."""
    sio.savemat(str(path), data, do_compression=True)
    size_kb = path.stat().st_size / 1024
    print(f"  {path.name:<36s}  {size_kb:5.0f} kB  {label}")


def export_python(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing Python reference data to:\n  {outdir}\n")

    # ── 1.  Fixed parameters and test signal ─────────────────────────────
    print("[1/7] params.mat ...")

    fs = 8000
    Ls = 1024
    t = np.arange(Ls, dtype=np.float64) / fs
    # Deterministic signal — no RNG, so MATLAB and Python produce identical inputs.
    # Three sinusoids at different frequencies + amplitudes.
    f = (np.sin(2 * np.pi * 440 * t)
         + 0.5 * np.sin(2 * np.pi * 1000 * t)
         + 0.3 * np.sin(2 * np.pi * 2500 * t))

    # ERB filterbank (same as export_reference_data.m)
    g, a_mat, fc_info, L, _info = audfilters(fs, Ls)
    M = len(g)

    # Normalise hop sizes
    a_norm = normalise_a(a_mat, M)
    a_num = a_norm[:, 0].astype(np.int32)
    a_den = a_norm[:, 1].astype(np.int32)
    a_rat = a_num.astype(np.float64) / a_den.astype(np.float64)

    # Center frequencies — normalise to match MATLAB convention
    # MATLAB info.fc is normalised: 0 = DC, 1 = Nyquist (fs/2)
    # Python audfilters returns fc in Hz
    fc_hz = np.asarray(fc_info, dtype=np.float64)
    fc_n = fc_hz / (fs / 2.0)  # normalise: Hz → fraction of Nyquist

    # Subband lengths
    N = np.ceil(L / a_rat).astype(np.int32)
    Nsum = int(np.sum(N))

    _save_mat(outdir / "params.mat", {
        "fs": np.int32(fs),
        "Ls": np.int32(Ls),
        "L": np.int32(L),
        "M": np.int32(M),
        "N": N,
        "Nsum": np.int32(Nsum),
        "a_num": a_num,
        "a_den": a_den,
        "a_rat": a_rat,
        "fc_n": fc_n,
        "f": f,
    })

    # ── 2.  filters.mat ──────────────────────────────────────────────────
    print("[2/7] filters.mat ...")
    G_cols = np.zeros((L, M), dtype=complex)
    for m in range(M):
        H_full, _ = filter_freqresp(g[m], L)
        G_cols[:, m] = H_full

    _save_mat(outdir / "filters.mat", {"G_cols": G_cols})

    # ── 3.  filterbank_coeff.mat ─────────────────────────────────────────
    print("[3/7] filterbank_coeff.mat ...")
    f_pad = postpad(f, L)
    c = filterbank(f_pad, g, a_mat)

    c_flat = np.concatenate([np.asarray(cm).ravel() for cm in c])

    _save_mat(outdir / "filterbank_coeff.mat", {
        "c_flat": c_flat,
    })

    # ── 4.  Layer 0: primitive operations ────────────────────────────────
    print("[4/7] layer0_ops.mat ...")
    L0 = {}

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    L0["postpad_shorter"] = postpad(x, 3)
    L0["postpad_longer"] = postpad(x, 8)
    L0["postpad_same"] = postpad(x, 5)


    L0["floor23_100"] = np.float64(floor23(100))

    v = np.array([1 + 2j, 3 + 4j, 5 + 6j, 7 + 8j])
    L0["involute_v"] = involute(v)

    L0["modcent_vals"] = modcent(np.arange(8), 8).astype(np.float64)

    _save_mat(outdir / "layer0_ops.mat", L0)

    # ── 5.  Layer 1: filter design ───────────────────────────────────────
    print("[5/7] layer1_filters.mat ...")
    L1 = {}

    # audfilters (fs=16000, Ls=4000)
    g_aud, a_aud, fc_aud, L_aud, _info = audfilters(16000, 4000)
    L1["aud_M"] = np.int32(len(g_aud))
    L1["aud_L"] = np.int32(L_aud)
    # Store fc in Hz (matches MATLAB layer1 which stores raw fc)
    L1["aud_fc"] = np.asarray(fc_aud, dtype=np.float64)
    a_aud_norm = normalise_a(a_aud, len(g_aud))
    L1["aud_a"] = a_aud_norm.astype(np.int32)

    G_aud = np.zeros((L_aud, len(g_aud)), dtype=complex)
    for m in range(len(g_aud)):
        H_full, _ = filter_freqresp(g_aud[m], L_aud)
        G_aud[:, m] = H_full
    L1["aud_G"] = G_aud

    # cqtfilters
    g_cqt, a_cqt, fc_cqt, L_cqt, _info = cqtfilters(16000, 4000, fmin=50, fmax=7000, bins=12)
    L1["cqt_M"] = np.int32(len(g_cqt))
    L1["cqt_L"] = np.int32(L_cqt)
    L1["cqt_fc"] = np.asarray(fc_cqt, dtype=np.float64)
    a_cqt_norm = normalise_a(a_cqt, len(g_cqt))
    L1["cqt_a"] = a_cqt_norm.astype(np.int32)

    G_cqt = np.zeros((L_cqt, len(g_cqt)), dtype=complex)
    for m in range(len(g_cqt)):
        H_full, _ = filter_freqresp(g_cqt[m], L_cqt)
        G_cqt[:, m] = H_full
    L1["cqt_G"] = G_cqt

    # firwin windows
    for wname in ["hann", "hamming", "blackman", "nuttall", "rect", "tria"]:
        L1[f"firwin_{wname}_64"] = firwin(wname, 64, norm="inf")
        L1[f"firwin_{wname}_128"] = firwin(wname, 128, norm="inf")

    _save_mat(outdir / "layer1_filters.mat", L1)

    # ── 6.  Layer 2: frame operations ────────────────────────────────────
    print("[6/7] layer2_frame.mat ...")
    L2 = {}

    # Frame bounds (audfilters)
    A_aud, B_aud = filterbankbounds(g_aud, a_aud, L_aud)
    L2["aud_frame_A"] = np.float64(A_aud)
    L2["aud_frame_B"] = np.float64(B_aud)

    # Frame bounds (cqtfilters)
    A_cqt, B_cqt = filterbankbounds(g_cqt, a_cqt, L_cqt)
    L2["cqt_frame_A"] = np.float64(A_cqt)
    L2["cqt_frame_B"] = np.float64(B_cqt)

    # Dual frame (audfilters) — evaluate transfer functions
    gd_aud = filterbankdual(g_aud, a_aud, L_aud)
    Gd_aud = np.zeros((L_aud, len(gd_aud)), dtype=complex)
    for m in range(len(gd_aud)):
        H_full, _ = filter_freqresp(gd_aud[m], L_aud)
        Gd_aud[:, m] = H_full
    L2["aud_Gd"] = Gd_aud

    # Perfect reconstruction (audfilters) — deterministic signal, no RNG
    t_aud = np.arange(L_aud, dtype=np.float64) / 16000.0
    f_test = (np.sin(2 * np.pi * 440 * t_aud)
              + 0.5 * np.sin(2 * np.pi * 1000 * t_aud)
              + 0.3 * np.sin(2 * np.pi * 2500 * t_aud))
    c_test = filterbank(f_test, g_aud, a_aud)
    r_test = ifilterbank(c_test, gd_aud, a_aud, real=True)
    L2["aud_recon_err"] = np.float64(
        np.linalg.norm(r_test - f_test) / np.linalg.norm(f_test)
    )
    L2["aud_f_test"] = f_test
    L2["aud_r_test"] = r_test

    c_flat_aud = np.concatenate([np.asarray(cm).ravel() for cm in c_test])
    L2["aud_c_flat"] = c_flat_aud
    L2["aud_N"] = np.array([len(cm) for cm in c_test], dtype=np.int32)

    # CQT reconstruction — deterministic signal
    gd_cqt = filterbankdual(g_cqt, a_cqt, L_cqt)
    t_cqt = np.arange(L_cqt, dtype=np.float64) / 16000.0
    f_cqt = (np.sin(2 * np.pi * 440 * t_cqt)
             + 0.5 * np.sin(2 * np.pi * 1000 * t_cqt)
             + 0.3 * np.sin(2 * np.pi * 2500 * t_cqt))
    c_cqt = filterbank(f_cqt, g_cqt, a_cqt)
    r_cqt = ifilterbank(c_cqt, gd_cqt, a_cqt, real=True)
    L2["cqt_recon_err"] = np.float64(
        np.linalg.norm(r_cqt - f_cqt) / np.linalg.norm(f_cqt)
    )

    _save_mat(outdir / "layer2_frame.mat", L2)

    # ── 7.  Engine info ──────────────────────────────────────────────────
    print("[7/7] engine_info.mat ...")
    _save_mat(outdir / "engine_info.mat", {
        "engine": f"Python {platform.python_version()} / NumPy {np.__version__}",
        "is_octave": np.int32(0),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

    # ── Summary ──────────────────────────────────────────────────────────
    mat_files = sorted(outdir.glob("*.mat"))
    print(f"\n=== Python export complete ===")
    print(f"Total files: {len(mat_files)}")
    for p in mat_files:
        print(f"  {p.name:<36s}  {p.stat().st_size / 1024:5.0f} kB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Python filterbank reference data")
    parser.add_argument("--outdir", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "python",
                        help="Output directory for .mat files")
    args = parser.parse_args()
    export_python(args.outdir)
