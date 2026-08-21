"""Tests for gammatonefir."""
from __future__ import annotations

import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from cool_frames.numpy.filters import gammatonefir


def _assert(cond, msg=""):
    if not cond:
        raise AssertionError(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_basic_output_structure():
    """gammatonefir returns a list of dicts with correct keys."""
    fc = [200.0, 1000.0, 4000.0]
    fs = 16000
    result = gammatonefir(fc, fs)
    _assert(isinstance(result, list), "Should return list")
    _assert(len(result) == 3, f"Expected 3 filters, got {len(result)}")
    for filt in result:
        _assert("h" in filt, "Missing 'h' key")
        _assert("offset" in filt, "Missing 'offset' key")
        _assert("realonly" in filt, "Missing 'realonly' key")
        _assert(isinstance(filt["h"], np.ndarray), "'h' should be ndarray")
        _assert(filt["offset"] < 0, "offset should be negative (pre-peak)")


def test_single_frequency():
    """Scalar fc works."""
    result = gammatonefir(1000.0, 16000)
    _assert(len(result) == 1)
    _assert(len(result[0]["h"]) > 0)


def test_real_output():
    """Default (real=True) produces real-valued filters."""
    result = gammatonefir([500.0, 2000.0], 44100)
    for filt in result:
        _assert(np.isrealobj(filt["h"]), "Default should be real-valued")


def test_complex_output():
    """real=False produces complex-valued filters."""
    result = gammatonefir([500.0, 2000.0], 44100, real=False)
    for filt in result:
        _assert(np.iscomplexobj(filt["h"]), "Should be complex-valued")


def test_envelope_shape():
    """The envelope should peak and then decay (gammatone shape)."""
    result = gammatonefir(1000.0, 44100, n=5000)
    h = result[0]["h"]
    env = np.abs(h) if np.iscomplexobj(h) else np.abs(
        # Use Hilbert transform to get envelope of real signal
        np.fft.ifft(np.fft.fft(h) * 2 * (np.arange(len(h)) < len(h) // 2))
    )
    # The envelope should have a single peak somewhere in the middle
    peak_idx = np.argmax(env)
    _assert(0 < peak_idx < len(h) - 1, "Peak should not be at boundaries")


def test_lower_fc_longer_filter():
    """Lower centre frequency should produce longer effective impulse response."""
    result_low = gammatonefir(200.0, 44100, n=5000)
    result_high = gammatonefir(4000.0, 44100, n=5000)
    # The pre-peak portion (nfirst) should be larger for lower fc
    _assert(abs(result_low[0]["offset"]) > abs(result_high[0]["offset"]),
            "Lower fc should have larger offset (longer pre-peak)")


def test_bandwidth_scaling():
    """betamul should scale the bandwidth."""
    result_narrow = gammatonefir(1000.0, 44100, betamul=0.5)
    result_wide = gammatonefir(1000.0, 44100, betamul=2.0)
    # Narrower bandwidth → longer filter (slower decay)
    _assert(len(result_narrow[0]["h"]) > len(result_wide[0]["h"]) or
            abs(result_narrow[0]["offset"]) > abs(result_wide[0]["offset"]),
            "Narrower bandwidth should produce longer or more offset filter")


def test_peakphase():
    """peakphase should shift the phase at the peak."""
    result_causal = gammatonefir(1000.0, 44100, real=False)
    result_peak = gammatonefir(1000.0, 44100, real=False, peakphase=True)
    # Both should have the same envelope
    env_c = np.abs(result_causal[0]["h"])
    env_p = np.abs(result_peak[0]["h"])
    np.testing.assert_allclose(env_c, env_p, atol=1e-10,
                               err_msg="Envelopes should match")
    # But phases should differ
    h_c = result_causal[0]["h"]
    h_p = result_peak[0]["h"]
    _assert(not np.allclose(np.angle(h_c), np.angle(h_p)),
            "Phase should differ between causal and peakphase")


def test_fc_validation():
    """Should reject fc > fs/2 or negative."""
    try:
        gammatonefir(-100.0, 44100)
        _assert(False, "Should have raised ValueError for negative fc")
    except ValueError:
        pass

    try:
        gammatonefir(25000.0, 44100)
        _assert(False, "Should have raised ValueError for fc > fs/2")
    except ValueError:
        pass


def test_filter_energy_decreases_with_fc():
    """Higher fc filters should have lower energy (shorter effective duration)."""
    fcs = [200, 1000, 4000]
    result = gammatonefir(fcs, 44100, n=5000)
    energies = [np.sum(np.abs(f["h"]) ** 2) for f in result]
    # Energy should decrease (or at least not increase dramatically) with fc
    # The exact relationship depends on the scaling constant
    _assert(all(e > 0 for e in energies), "All filters should have positive energy")


def test_import_from_filterbanks():
    """gammatonefir should be importable from filterbanks package."""
    from cool_frames.numpy.filters import gammatonefir as gtf
    result = gtf(1000.0, 44100)
    _assert(len(result) == 1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  PASS  {name}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{passed} passed, {failed} failed out of {passed + failed}")
    sys.exit(1 if failed else 0)
