"""
Filterbank analysis/synthesis benchmarks.

Run with:
    pytest benchmarks/ --benchmark-only
    pytest benchmarks/ --benchmark-autosave          # save baseline
    pytest benchmarks/ --benchmark-compare=baseline  # compare to baseline
"""

import pytest

import numpy as np
from cool_frames.numpy.filterbanks import filterbank, filterbankdual, ifilterbank
from cool_frames.numpy.filters import audfilters

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def erb_setup():
    fs, Ls = 16000, 16000
    g, a, _fc, L, _info = audfilters(fs, Ls)
    gd = filterbankdual(g, a, L, real=True)
    rng = np.random.default_rng(0)
    f = rng.standard_normal(Ls)
    return dict(g=g, a=a, gd=gd, L=L, Ls=Ls, f=f)


@pytest.fixture(scope="module")
def erb_setup_long():
    fs, Ls = 16000, 16000 * 10  # 10 seconds
    g, a, _fc, L, _info = audfilters(fs, Ls)
    gd = filterbankdual(g, a, L, real=True)
    rng = np.random.default_rng(1)
    f = rng.standard_normal(Ls)
    return dict(g=g, a=a, gd=gd, L=L, Ls=Ls, f=f)


# ---------------------------------------------------------------------------
# NumPy filterbank benchmarks
# ---------------------------------------------------------------------------


class BenchFilterbankAnalysis:
    """filterbank() — analysis (NumPy)."""

    def test_erb_1s(self, benchmark, erb_setup):
        s = erb_setup
        benchmark(filterbank, s["f"], s["g"], s["a"], L=s["L"])

    def test_erb_10s(self, benchmark, erb_setup_long):
        s = erb_setup_long
        benchmark(filterbank, s["f"], s["g"], s["a"], L=s["L"])


class BenchFilterbankSynthesis:
    """ifilterbank() — synthesis (NumPy)."""

    def test_erb_1s(self, benchmark, erb_setup):
        s = erb_setup
        c = filterbank(s["f"], s["g"], s["a"], L=s["L"])
        benchmark(ifilterbank, c, s["gd"], s["a"], Ls=s["Ls"], real=True)


class BenchFilterbankRoundTrip:
    """Full analysis + synthesis round-trip (NumPy)."""

    def test_erb_1s(self, benchmark, erb_setup):
        s = erb_setup

        def roundtrip():
            c = filterbank(s["f"], s["g"], s["a"], L=s["L"])
            return ifilterbank(c, s["gd"], s["a"], Ls=s["Ls"], real=True)

        benchmark(roundtrip)
