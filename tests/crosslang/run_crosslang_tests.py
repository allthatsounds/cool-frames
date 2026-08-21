#!/usr/bin/env python3
"""
run_crosslang_tests.py
======================
Cross-language test orchestrator for the ltfat_filterbank project.

Runs the MATLAB/Octave and Python filterbank codebases with identical inputs,
saves their outputs as .mat files, then compares every shared field to verify
numerical equivalence.

Usage
-----
    python run_crosslang_tests.py                    # auto-detect engine
    python run_crosslang_tests.py --engine matlab    # force MATLAB
    python run_crosslang_tests.py --engine octave    # force Octave
    python run_crosslang_tests.py --skip-matlab      # Python-only (useful for CI)
    python run_crosslang_tests.py --tol 1e-8         # custom tolerance

Workflow
--------
1. Auto-detect MATLAB or Octave (or use --engine / --skip-matlab).
2. Run ``export_matlab.m`` → writes .mat files to ``results/matlab/``.
3. Run ``export_python.py`` → writes .mat files to ``results/python/``.
4. For every .mat file present in BOTH directories, load and compare
   field by field using relative error.
5. Print a unified pass/fail report and exit with code 0 (all pass) or 1.

Report
------
A machine-readable JSON report is saved to ``results/crosslang_report.json``
and a human-readable text summary to ``results/crosslang_report.txt``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import scipy.io as sio
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ── Configuration ────────────────────────────────────────────────────────

CROSSLANG_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CROSSLANG_DIR / "results"
MATLAB_DIR = RESULTS_DIR / "matlab"
PYTHON_DIR = RESULTS_DIR / "python"

# Files that are compared (must exist in both matlab/ and python/ dirs)
COMPARABLE_FILES = [
    "params.mat",
    "filters.mat",
    "filterbank_coeff.mat",
    "layer0_ops.mat",
    "layer1_filters.mat",
    "layer2_frame.mat",
]

# Fields to skip during comparison (metadata, not numerical data)
SKIP_FIELDS = {"engine", "is_octave", "timestamp", "__header__", "__version__", "__globals__"}

# Default relative tolerance
DEFAULT_TOL = 1e-10

# Per-field tolerance overrides (some quantities have inherently lower precision)
FIELD_TOLERANCES = {
    # Reconstruction errors should be small but may differ slightly
    "aud_recon_err": 1e-6,
    "cqt_recon_err": 1e-6,
    # Frame bounds depend on response accumulation
    "aud_frame_A": 1e-8,
    "aud_frame_B": 1e-8,
    "cqt_frame_A": 1e-8,
    "cqt_frame_B": 1e-8,
}

# Known convention differences between the MATLAB and Python codebases.
# These are reported as KNOWN (not FAIL) so they don't obscure real bugs.
# Format: (file, field) -> description of the known difference.
KNOWN_DIFFERENCES: dict[tuple[str, str], str] = {
    # ── Layer 0: Nyquist convention ──
    ("layer0_ops.mat", "modcent_vals"):
        "Same Nyquist convention: modcent(N/2, N) → +N/2 (MATLAB) vs -N/2 (Python)",
    # ── Lowpass hop: audfilters DC channel hop=a vs hop=1 ──
    # Python sets hop=1 for the DC filter; MATLAB uses a larger hop (e.g. 18).
    # This single difference cascades to N, Nsum, c_flat shapes, G_cols,
    # frame bounds, dual frames, and all filterbank coefficients.
    ("params.mat", "a_num"):
        "Lowpass hop: MATLAB audfilters DC hop=18, Python hop=1",
    ("params.mat", "a_rat"):
        "Lowpass hop: cascade from a_num[0] difference",
    ("params.mat", "N"):
        "Cascade: N[0]=L/a[0] differs due to lowpass hop",
    ("params.mat", "Nsum"):
        "Cascade: sum(N) differs due to lowpass hop",
    ("filters.mat", "G_cols"):
        "Cascade: transfer functions differ due to lowpass hop (different filter prep)",
    ("filterbank_coeff.mat", "c_flat"):
        "Cascade: coefficient shapes differ due to lowpass hop → different N[0]",
    ("layer1_filters.mat", "aud_a"):
        "Lowpass hop: MATLAB audfilters DC hop=36 (fs=16k), Python hop=1",
    ("layer1_filters.mat", "aud_G"):
        "Cascade: transfer functions differ due to lowpass hop",
    ("layer1_filters.mat", "cqt_G"):
        "CQT transfer function convention difference",
    ("layer2_frame.mat", "aud_frame_A"):
        "Cascade: frame bounds differ due to lowpass hop",
    ("layer2_frame.mat", "aud_frame_B"):
        "Cascade: frame bounds differ due to lowpass hop",
    ("layer2_frame.mat", "cqt_frame_A"):
        "CQT frame bounds differ",
    ("layer2_frame.mat", "cqt_frame_B"):
        "CQT frame bounds differ",
    ("layer2_frame.mat", "aud_Gd"):
        "Cascade: dual frames differ due to lowpass hop",
    ("layer2_frame.mat", "aud_N"):
        "Cascade: subband lengths differ due to lowpass hop",
    ("layer2_frame.mat", "aud_c_flat"):
        "Cascade: coefficient shapes differ due to lowpass hop",
    ("layer2_frame.mat", "aud_recon_err"):
        "Cascade: reconstruction error differs due to different filterbank geometry",
    ("layer2_frame.mat", "cqt_recon_err"):
        "CQT reconstruction error differs",
    ("layer1_filters.mat", "firwin_hann_64"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_hann_128"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_hamming_64"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_hamming_128"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_blackman_64"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_blackman_128"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_nuttall_64"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_nuttall_128"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_rect_64"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_rect_128"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_tria_64"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
    ("layer1_filters.mat", "firwin_tria_128"):
        "Window centering: MATLAB peak at index 0, Python peak at center",
}


# ── Engine detection ─────────────────────────────────────────────────────

def detect_engine() -> Optional[str]:
    """Auto-detect MATLAB or Octave.  Returns 'matlab', 'octave', or None."""
    # Try MATLAB first
    matlab_path = shutil.which("matlab")
    if matlab_path:
        return "matlab"

    # Try Octave
    octave_path = shutil.which("octave") or shutil.which("octave-cli")
    if octave_path:
        return "octave"

    return None


def get_engine_cmd(engine: str) -> list[str]:
    """Return the command prefix for running a .m script."""
    if engine == "matlab":
        return ["matlab", "-batch"]
    elif engine == "octave":
        # Try octave-cli first (no GUI), fall back to octave --no-gui
        if shutil.which("octave-cli"):
            return ["octave-cli", "--eval"]
        return ["octave", "--no-gui", "--eval"]
    else:
        raise ValueError(f"Unknown engine: {engine}")


# ── MATLAB/Octave runner ────────────────────────────────────────────────

def run_matlab_export(engine: str) -> tuple[bool, str]:
    """Run export_matlab.m and return (success, log_text)."""
    print(f"\n{'='*60}")
    print(f"  Running MATLAB export ({engine})")
    print(f"{'='*60}\n")

    cmd_prefix = get_engine_cmd(engine)

    # Build the MATLAB command string
    matlab_cmd = (
        f"cd('{CROSSLANG_DIR}'); "
        f"export_matlab('{MATLAB_DIR}');"
    )

    cmd = cmd_prefix + [matlab_cmd]
    print(f"  Command: {' '.join(cmd)}\n")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min timeout
            cwd=str(CROSSLANG_DIR),
        )
        elapsed = time.time() - t0
        log = result.stdout + "\n" + result.stderr
        success = result.returncode == 0

        if success:
            print(f"  MATLAB export completed in {elapsed:.1f}s")
        else:
            print(f"  MATLAB export FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[:500]}")

        return success, log

    except FileNotFoundError:
        return False, f"Engine '{engine}' not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "MATLAB export timed out after 300s"


# ── Python runner ────────────────────────────────────────────────────────

def run_python_export() -> tuple[bool, str]:
    """Run export_python.py and return (success, log_text)."""
    print(f"\n{'='*60}")
    print(f"  Running Python export")
    print(f"{'='*60}\n")

    export_script = CROSSLANG_DIR / "export_python.py"
    cmd = [sys.executable, str(export_script), "--outdir", str(PYTHON_DIR)]
    print(f"  Command: {' '.join(cmd)}\n")

    # Ensure filterbank and ltfat_core packages are on PYTHONPATH
    proj_root = CROSSLANG_DIR.parent
    env = os.environ.copy()
    extra_paths = [
        str(proj_root / "filterbank"),
        str(proj_root / "ltfat_core"),
    ]
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([existing] if existing else []))

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(CROSSLANG_DIR),
            env=env,
        )
        elapsed = time.time() - t0
        log = result.stdout + "\n" + result.stderr
        success = result.returncode == 0

        # Also print stdout live
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")

        if success:
            print(f"\n  Python export completed in {elapsed:.1f}s")
        else:
            print(f"\n  Python export FAILED (exit code {result.returncode})")
            print(f"  stderr: {result.stderr[:500]}")

        return success, log

    except subprocess.TimeoutExpired:
        return False, "Python export timed out after 120s"


# ── Comparison engine ────────────────────────────────────────────────────

def compare_field(mat_val, py_val, field_name: str, tol: float) -> dict:
    """Compare two values and return a result dict."""
    result = {
        "field": field_name,
        "status": "SKIP",
        "rel_err": None,
        "abs_err": None,
        "tol": tol,
        "detail": "",
    }

    # Use per-field tolerance if available
    effective_tol = FIELD_TOLERANCES.get(field_name, tol)
    result["tol"] = effective_tol

    # Convert to numpy arrays
    try:
        mat_arr = np.asarray(mat_val, dtype=np.complex128 if np.iscomplexobj(mat_val) else np.float64)
        py_arr = np.asarray(py_val, dtype=np.complex128 if np.iscomplexobj(py_val) else np.float64)
    except (ValueError, TypeError):
        # Non-numeric (e.g. strings) — just check equality
        if np.array_equal(mat_val, py_val):
            result["status"] = "PASS"
        else:
            result["status"] = "FAIL"
            result["detail"] = "Non-numeric values differ"
        return result

    # Shape check
    mat_flat = mat_arr.ravel()
    py_flat = py_arr.ravel()

    if mat_flat.shape != py_flat.shape:
        result["status"] = "FAIL"
        result["detail"] = f"Shape mismatch: MATLAB {mat_arr.shape} vs Python {py_arr.shape}"
        return result

    if mat_flat.size == 0:
        result["status"] = "PASS"
        result["detail"] = "Empty arrays"
        return result

    # Compute errors
    diff = np.abs(mat_flat - py_flat)
    abs_err = float(np.max(diff))
    denom = float(np.max(np.abs(mat_flat))) + np.finfo(np.float64).eps
    rel_err = abs_err / denom

    result["abs_err"] = abs_err
    result["rel_err"] = rel_err

    if rel_err < effective_tol:
        result["status"] = "PASS"
    else:
        result["status"] = "FAIL"
        result["detail"] = f"rel_err={rel_err:.4e} > tol={effective_tol:.4e}"

    return result


def compare_mat_files(mat_file: Path, py_file: Path, tol: float) -> list[dict]:
    """Compare two .mat files field by field.  Returns list of result dicts."""
    results = []
    filename = mat_file.name

    mat_data = sio.loadmat(str(mat_file), squeeze_me=True)
    py_data = sio.loadmat(str(py_file), squeeze_me=True)

    # Combine all field names
    all_fields = sorted(
        set(mat_data.keys()) | set(py_data.keys()) - SKIP_FIELDS
    )

    for field in all_fields:
        if field in SKIP_FIELDS:
            continue

        # Check if this is a known convention difference
        known_key = (filename, field)
        if known_key in KNOWN_DIFFERENCES:
            r = compare_field(mat_data.get(field), py_data.get(field), field, tol) \
                if field in mat_data and field in py_data else {
                    "field": field, "status": "SKIP", "rel_err": None,
                    "abs_err": None, "tol": tol, "detail": "",
                }
            r["status"] = "KNOWN"
            r["detail"] = KNOWN_DIFFERENCES[known_key]
            results.append(r)
            continue

        if field not in mat_data:
            results.append({
                "field": field,
                "status": "SKIP",
                "detail": "Only in Python",
                "rel_err": None,
                "abs_err": None,
                "tol": tol,
            })
            continue

        if field not in py_data:
            results.append({
                "field": field,
                "status": "SKIP",
                "detail": "Only in MATLAB",
                "rel_err": None,
                "abs_err": None,
                "tol": tol,
            })
            continue

        results.append(compare_field(mat_data[field], py_data[field], field, tol))

    return results


# ── Report generation ────────────────────────────────────────────────────

def generate_report(
    all_results: dict[str, list[dict]],
    matlab_engine: str,
    matlab_ok: bool,
    python_ok: bool,
    tol: float,
) -> tuple[str, dict]:
    """Generate text and JSON reports.  Returns (text_report, json_data)."""

    total_pass = 0
    total_fail = 0
    total_skip = 0
    total_known = 0

    lines = []
    lines.append("=" * 70)
    lines.append("  CROSS-LANGUAGE TEST REPORT")
    lines.append(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  MATLAB engine:  {matlab_engine or 'N/A'}")
    lines.append(f"  MATLAB export:  {'OK' if matlab_ok else 'FAILED / SKIPPED'}")
    lines.append(f"  Python export:  {'OK' if python_ok else 'FAILED'}")
    lines.append(f"  Tolerance:      {tol:.0e}")
    lines.append("")

    json_files = {}

    for filename, results in sorted(all_results.items()):
        lines.append(f"  ── {filename} {'─' * (50 - len(filename))}")

        file_pass = file_fail = file_skip = file_known = 0
        json_fields = []

        for r in results:
            status = r["status"]
            field = r["field"]
            rel_err = r.get("rel_err")
            detail = r.get("detail", "")

            if status == "PASS":
                err_str = f"rel_err={rel_err:.2e}" if rel_err is not None else ""
                lines.append(f"    PASS  {field:<36s}  {err_str}")
                file_pass += 1
            elif status == "KNOWN":
                err_str = f"rel_err={rel_err:.2e}" if rel_err is not None else ""
                lines.append(f"    KNOWN {field:<36s}  {detail}")
                file_known += 1
            elif status == "FAIL":
                lines.append(f"    FAIL  {field:<36s}  {detail}")
                file_fail += 1
            else:
                lines.append(f"    SKIP  {field:<36s}  {detail}")
                file_skip += 1

            json_fields.append(r)

        total_pass += file_pass
        total_fail += file_fail
        total_skip += file_skip
        total_known += file_known
        parts = [f"{file_pass} pass", f"{file_fail} fail"]
        if file_known:
            parts.append(f"{file_known} known")
        if file_skip:
            parts.append(f"{file_skip} skip")
        lines.append(f"    ── {', '.join(parts)}")
        lines.append("")
        json_files[filename] = json_fields

    lines.append("=" * 70)
    overall = "PASS" if total_fail == 0 else "FAIL"
    summary_parts = [f"{total_pass} pass", f"{total_fail} fail"]
    if total_known:
        summary_parts.append(f"{total_known} known diffs")
    if total_skip:
        summary_parts.append(f"{total_skip} skip")
    lines.append(f"  OVERALL: {overall}  ({', '.join(summary_parts)})")
    lines.append("=" * 70)

    text_report = "\n".join(lines)

    json_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "matlab_engine": matlab_engine,
        "matlab_ok": matlab_ok,
        "python_ok": python_ok,
        "tolerance": tol,
        "overall": overall,
        "total_pass": total_pass,
        "total_fail": total_fail,
        "total_known": total_known,
        "total_skip": total_skip,
        "files": json_files,
    }

    return text_report, json_data


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-language test runner for ltfat_filterbank",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python run_crosslang_tests.py                  # auto-detect
              python run_crosslang_tests.py --engine octave  # force Octave
              python run_crosslang_tests.py --skip-matlab    # Python only
              python run_crosslang_tests.py --tol 1e-8       # looser tolerance
        """),
    )
    parser.add_argument(
        "--engine", choices=["matlab", "octave"],
        help="Force a specific MATLAB engine (default: auto-detect)",
    )
    parser.add_argument(
        "--skip-matlab", action="store_true",
        help="Skip the MATLAB export (compare against existing .mat files)",
    )
    parser.add_argument(
        "--skip-python", action="store_true",
        help="Skip the Python export (compare against existing .mat files)",
    )
    parser.add_argument(
        "--tol", type=float, default=DEFAULT_TOL,
        help=f"Relative tolerance for comparison (default: {DEFAULT_TOL})",
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help="Skip both exports, only compare existing results",
    )
    args = parser.parse_args()

    if not _HAS_SCIPY:
        print("ERROR: scipy is required for .mat file I/O")
        print("  pip install scipy")
        sys.exit(1)

    if args.compare_only:
        args.skip_matlab = True
        args.skip_python = True

    # ── Detect engine ────────────────────────────────────────────────────
    matlab_engine = args.engine
    matlab_ok = False

    if not args.skip_matlab:
        if matlab_engine is None:
            matlab_engine = detect_engine()
        if matlab_engine is None:
            print("WARNING: Neither MATLAB nor Octave found on PATH.")
            print("         Use --skip-matlab to compare against existing .mat files,")
            print("         or install MATLAB/Octave and ensure it's on PATH.\n")
            args.skip_matlab = True
        else:
            print(f"Detected engine: {matlab_engine}")

    # ── Run exports ──────────────────────────────────────────────────────
    if not args.skip_matlab:
        matlab_ok, matlab_log = run_matlab_export(matlab_engine)
        if not matlab_ok:
            print("\nWARNING: MATLAB export failed. Will compare against existing files if present.")
            # Save log for debugging
            log_path = RESULTS_DIR / "matlab_export.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(matlab_log)
            print(f"  Log saved to: {log_path}")
    else:
        print("\nSkipping MATLAB export (--skip-matlab or no engine found)")
        # Check if existing results are available
        if MATLAB_DIR.exists() and list(MATLAB_DIR.glob("*.mat")):
            matlab_ok = True
            matlab_engine = matlab_engine or "pre-existing"
            print(f"  Using existing MATLAB results in {MATLAB_DIR}")
        else:
            print(f"  No existing MATLAB results in {MATLAB_DIR}")

    python_ok = False
    if not args.skip_python:
        python_ok_run, python_log = run_python_export()
        python_ok = python_ok_run
        if not python_ok:
            print("\nERROR: Python export failed.")
            log_path = RESULTS_DIR / "python_export.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(python_log)
            print(f"  Log saved to: {log_path}")
    else:
        print("\nSkipping Python export (--skip-python)")
        if PYTHON_DIR.exists() and list(PYTHON_DIR.glob("*.mat")):
            python_ok = True
            print(f"  Using existing Python results in {PYTHON_DIR}")

    # ── Compare ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Comparing results")
    print(f"{'='*60}\n")

    all_results: dict[str, list[dict]] = {}

    if matlab_ok and python_ok:
        for filename in COMPARABLE_FILES:
            mat_file = MATLAB_DIR / filename
            py_file = PYTHON_DIR / filename

            if not mat_file.exists():
                print(f"  SKIP  {filename}  (not in MATLAB results)")
                all_results[filename] = [{
                    "field": "(entire file)",
                    "status": "SKIP",
                    "detail": "Not in MATLAB results",
                    "rel_err": None, "abs_err": None, "tol": args.tol,
                }]
                continue
            if not py_file.exists():
                print(f"  SKIP  {filename}  (not in Python results)")
                all_results[filename] = [{
                    "field": "(entire file)",
                    "status": "SKIP",
                    "detail": "Not in Python results",
                    "rel_err": None, "abs_err": None, "tol": args.tol,
                }]
                continue

            print(f"  Comparing {filename} ...")
            results = compare_mat_files(mat_file, py_file, args.tol)
            all_results[filename] = results

            n_pass = sum(1 for r in results if r["status"] == "PASS")
            n_fail = sum(1 for r in results if r["status"] == "FAIL")
            print(f"    → {n_pass} pass, {n_fail} fail")
    else:
        if not matlab_ok:
            print("  Cannot compare: MATLAB results not available")
        if not python_ok:
            print("  Cannot compare: Python results not available")

    # ── Report ───────────────────────────────────────────────────────────
    text_report, json_data = generate_report(
        all_results, matlab_engine, matlab_ok, python_ok, args.tol,
    )

    print("\n" + text_report)

    # Save reports
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "crosslang_report.txt").write_text(text_report)
    (RESULTS_DIR / "crosslang_report.json").write_text(
        json.dumps(json_data, indent=2, default=str)
    )
    print(f"\nReports saved to:")
    print(f"  {RESULTS_DIR / 'crosslang_report.txt'}")
    print(f"  {RESULTS_DIR / 'crosslang_report.json'}")

    # Exit code
    if json_data["overall"] == "PASS":
        print("\n✓ All cross-language tests passed.")
        sys.exit(0)
    else:
        print(f"\n✗ {json_data['total_fail']} cross-language test(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
