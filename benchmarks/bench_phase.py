"""
Phase retrieval benchmarks (NumPy backend).

Run with:
    pytest benchmarks/bench_phase.py --benchmark-only
"""

import pytest

import numpy as np
from cool_frames.numpy.filterbanks import filterbank, filterbankdual
from cool_frames.numpy.filters import audfilters
from cool_frames.numpy.phase import filterbankconstphase, filterbankphasegrad, gla


@pytest.fixture(scope="module")
def phase_setup():
    fs, Ls = 16000, 16000
    g, a, fc, L, _info = audfilters(fs, Ls)
    gd = filterbankdual(g, a, L, real=True)
    rng = np.random.default_rng(0)
    f = rng.standard_normal(Ls)
    c = filterbank(f, g, a, L=L)
    s = [np.abs(cm) for cm in c]
    return dict(g=g, a=a, fc=fc, gd=gd, L=L, Ls=Ls, f=f, c=c, s=s)


class BenchPhaseGrad:
    """filterbankphasegrad() — phase gradient computation."""

    def test_phasegrad(self, benchmark, phase_setup):
        s = phase_setup
        benchmark(filterbankphasegrad, s["f"], s["g"], s["a"], s["L"])


class BenchGLA:
    """Griffin-Lim algorithm (NumPy)."""

    def test_gla_10iter(self, benchmark, phase_setup):
        p = phase_setup
        benchmark(gla, p["s"], p["g"], p["a"], L=p["L"], Ls=p["Ls"], real=True, maxit=10)

    def test_gla_100iter(self, benchmark, phase_setup):
        p = phase_setup
        benchmark.pedantic(
            gla,
            args=(p["s"], p["g"], p["a"]),
            kwargs=dict(L=p["L"], Ls=p["Ls"], real=True, maxit=100),
            rounds=3,
            iterations=1,
        )


class BenchConstphase:
    """filterbankconstphase() — PGHI heap reconstruction."""

    def test_constphase(self, benchmark, phase_setup):
        p = phase_setup
        benchmark(filterbankconstphase, p["f"], p["g"], p["a"], L=p["L"], fc=p["fc"])
