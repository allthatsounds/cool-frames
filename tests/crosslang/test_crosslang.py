"""Cross-language MATLAB <-> Python regression gate.

Runs the vendored cross-language harness in ``--skip-matlab`` mode: it
regenerates the Python reference data with the *current* cool_frames and diffs it,
field by field, against the committed MATLAB reference ``.mat`` files in
``results/matlab/``. A non-zero exit means a numerical divergence has crept
in --- i.e. one of the three reconciled MATLAB/Python differences (low-pass
hop, ``firwin`` centring, Nyquist convention) has silently reappeared, or a
new one has.

This test is **opt-in**: it only runs when ``COOL_FRAMES_RUN_CROSSLANG=1`` so that a
plain ``pytest tests/`` does not depend on scipy.io / the reference bundle.
The dedicated ``crosslang`` CI job sets that variable.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

pytestmark = pytest.mark.skipif(
    os.environ.get("COOL_FRAMES_RUN_CROSSLANG") != "1",
    reason="set COOL_FRAMES_RUN_CROSSLANG=1 to run the MATLAB<->Python cross-language diff",
)


def test_crosslang_no_divergence():
    pytest.importorskip("scipy")
    ref = HERE / "results" / "matlab"
    assert ref.is_dir() and any(ref.glob("*.mat")), (
        f"reference .mat files missing under {ref}"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(HERE / "run_crosslang_tests.py"),
            "--skip-matlab",
            "--tol",
            "1e-8",
        ],
        cwd=str(HERE),
        capture_output=True,
        text=True,
    )
    # Surface the harness's own report in the pytest output on failure.
    print(result.stdout[-6000:])
    if result.stderr:
        print(result.stderr[-2000:], file=sys.stderr)
    assert result.returncode == 0, (
        "Cross-language MATLAB<->Python divergence detected "
        "(see the harness report above)."
    )
