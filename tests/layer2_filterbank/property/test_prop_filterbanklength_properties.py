"""
test_prop_filterbanklength_properties.py
=========================================
Python port of:
    layer2_filterbank/property/PropFilterbanklengthProperties.m

filterbanklength(Ls, a) returns the smallest L >= Ls divisible by lcm(a).

Properties tested:
(1) L >= Ls
(2) Idempotence: filterbanklength(L, a) == L
(3) Divisibility by a (uniform hop)
(4) Divisibility by lcm(a) (non-uniform hop)
(5) Monotone in Ls
(6) L == Ls when Ls is already divisible by lcm(a)

NOTE: Properties (1)-(6) only require numpy (lcm arithmetic), so the
reference-level class runs unconditionally; the impl class adds a smoke
test that calls the actual installed function.
"""

from __future__ import annotations

import math

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Pure-numpy reference: filterbanklength_ref
# ---------------------------------------------------------------------------

def filterbanklength_ref(Ls: int, a) -> int:
    """
    Smallest integer L >= Ls such that L is divisible by lcm(a).
    """
    a_arr = np.asarray(a, dtype=int).ravel()
    a_vals = a_arr[a_arr[:, None].shape[0] > 0] if a_arr.ndim > 1 else a_arr
    # For 2-column a arrays (LTFAT convention), take first column
    if np.asarray(a).ndim == 2:
        a_vals = np.asarray(a)[:, 0].astype(int)
    else:
        a_vals = np.asarray(a, dtype=int).ravel()
    lcm_a = int(a_vals[0])
    for v in a_vals[1:]:
        lcm_a = math.lcm(lcm_a, int(v))
    remainder = Ls % lcm_a
    if remainder == 0:
        return Ls
    return Ls + (lcm_a - remainder)


# ---------------------------------------------------------------------------
# Reference tests (unconditional numpy)
# ---------------------------------------------------------------------------

class TestFilterbanklengthReference:
    """PropFilterbanklengthProperties: reference arithmetic, no impl required."""

    @pytest.mark.parametrize("Ls,a", [
        (1, 1), (100, 1), (511, 2), (512, 4), (1000, 8), (1023, 16), (4095, 32),
    ])
    def test_geq_ls(self, Ls, a):
        L = filterbanklength_ref(Ls, [a])
        assert L >= Ls, f"filterbanklength_ref({Ls},{a})={L} < Ls"

    @pytest.mark.parametrize("Ls,a", [(100, 2), (512, 4), (1000, 8), (1023, 16)])
    def test_idempotent(self, Ls, a):
        L1 = filterbanklength_ref(Ls, [a])
        L2 = filterbanklength_ref(L1, [a])
        assert L1 == L2, f"Not idempotent: filterbanklength_ref({Ls},{a})={L1}, second={L2}"

    @pytest.mark.parametrize("Ls,a", [(100, 4), (512, 8), (1023, 16), (4095, 32)])
    def test_divisibility_uniform(self, Ls, a):
        L = filterbanklength_ref(Ls, [a])
        assert L % a == 0, f"filterbanklength_ref({Ls},{a})={L} not divisible by {a}"

    def test_divisibility_nonuniform(self):
        Ls = 1024
        for a_set in [[2, 4], [3, 6], [4, 8, 16], [2, 3, 4]]:
            L     = filterbanklength_ref(Ls, a_set)
            a_lcm = a_set[0]
            for v in a_set[1:]:
                a_lcm = math.lcm(a_lcm, v)
            assert L % a_lcm == 0, \
                f"a={a_set}: L={L} not divisible by lcm={a_lcm}"

    def test_monotone_in_ls(self):
        for a in [4, 8, 16]:
            prev = 0
            for Ls in range(1, 1000, 50):
                L = filterbanklength_ref(Ls, [a])
                assert L >= prev, \
                    f"a={a}: not monotone at Ls={Ls} (prev={prev}, now={L})"
                prev = L

    def test_exact_when_already_divisible(self):
        for a in [2, 4, 8, 16]:
            for k in range(1, 11):
                Ls = a * k * 16
                L  = filterbanklength_ref(Ls, [a])
                assert L == Ls, \
                    f"a={a}, Ls={Ls} already divisible but L={L} != Ls"


# ---------------------------------------------------------------------------
# Impl-level smoke test
# ---------------------------------------------------------------------------

@pytest.mark.requires_impl
class TestFilterbanklengthImpl:
    """PropFilterbanklengthProperties: installed filterbanklength."""

    @pytest.mark.parametrize("Ls,a", [
        (1, 1), (100, 2), (512, 4), (1023, 8), (4095, 16),
    ])
    def test_geq_ls(self, needs_impl, Ls, a):
        from cool_frames.filters import filterbanklength  # type: ignore
        L = filterbanklength(Ls, a)
        assert L >= Ls

    @pytest.mark.parametrize("Ls,a", [(100, 2), (512, 4), (1023, 8)])
    def test_idempotent(self, needs_impl, Ls, a):
        from cool_frames.filters import filterbanklength  # type: ignore
        L1 = filterbanklength(Ls, a)
        L2 = filterbanklength(L1, a)
        assert L1 == L2

    @pytest.mark.parametrize("Ls,a", [(100, 4), (512, 8), (1023, 16), (4095, 32)])
    def test_divisibility_uniform(self, needs_impl, Ls, a):
        from cool_frames.filters import filterbanklength  # type: ignore
        L = filterbanklength(Ls, a)
        assert L % a == 0, f"filterbanklength({Ls},{a})={L} not divisible by {a}"

    def test_divisibility_nonuniform(self, needs_impl):
        from cool_frames.filters import filterbanklength  # type: ignore
        Ls = 1024
        for a_set in [[2, 4], [3, 6], [4, 8, 16]]:
            L     = filterbanklength(Ls, a_set)
            a_lcm = a_set[0]
            for v in a_set[1:]:
                a_lcm = math.lcm(a_lcm, v)
            assert L % a_lcm == 0

    def test_monotone_in_ls(self, needs_impl):
        from cool_frames.filters import filterbanklength  # type: ignore
        for a in [4, 8]:
            prev = 0
            for Ls in range(1, 500, 50):
                L = filterbanklength(Ls, a)
                assert L >= prev
                prev = L

    def test_exact_when_already_divisible(self, needs_impl):
        from cool_frames.filters import filterbanklength  # type: ignore
        for a in [2, 4, 8, 16]:
            Ls = a * 64
            L  = filterbanklength(Ls, a)
            assert L == Ls, f"a={a}, Ls={Ls}: expected L={Ls}, got {L}"

    def test_matches_reference(self, needs_impl):
        from cool_frames.filters import filterbanklength  # type: ignore
        for Ls in [100, 512, 1000, 1023, 4095]:
            for a in [1, 2, 4, 8, 16]:
                L_ref  = filterbanklength_ref(Ls, [a])
                L_impl = filterbanklength(Ls, a)
                assert L_impl == L_ref, \
                    f"filterbanklength({Ls},{a}): impl={L_impl}, ref={L_ref}"
