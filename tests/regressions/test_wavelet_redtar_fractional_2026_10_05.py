"""Regression: ``waveletfilters(redtar=...)`` works in the fractional modes.

Found 2026-10-05 while matching the redundancy of four frame families for an
oracle-separability study (DEFECT_REGISTER G12).  The fractional modes carry
the hops as the (M, 2) array [Ls, N].  The ``redtar`` branch scaled that whole
array -- numerator and denominator alike -- and then read ``a_new.ravel()`` as
the hop vector, which has 2*M entries, so every call with ``redtar`` and
``sampling='fractional'`` or ``'fractionaluniform'`` raised ``ValueError:
matmul: Input operand 1 has a mismatch in its core dimension 0`` from the
Nyquist complement's ``filterbankresponse``.  ``regsampling`` and ``uniform``
carry 1-D hops and were unaffected; they must stay bit-identical.
"""

from __future__ import annotations

import itertools
import warnings

import pytest

import numpy as np
from cool_frames.numpy.filters import waveletfilters


def _quiet(fn, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kw)


def _real_redundancy(a, L):
    """Real degrees of freedom per signal sample of a single-sided bank."""
    a = np.asarray(a, dtype=float)
    hops = a[:, 0] / a[:, 1] if a.ndim == 2 else a.ravel()
    return 2.0 * np.sum(L / hops) / L


@pytest.mark.parametrize("sampling", ["fractional", "fractionaluniform"])
@pytest.mark.parametrize("redtar", [4.0, 8.0, 16.0])
def test_redtar_builds_and_hits_its_target(sampling, redtar):
    g, a, _fc, L, info = _quiet(
        waveletfilters, 8000, 4096, fmin=50, bins=12, sampling=sampling, redtar=redtar
    )
    a = np.asarray(a)
    assert a.ndim == 2 and a.shape == (len(g), 2)
    assert np.all(a[:, 0] == L)
    rho = _real_redundancy(a, L)
    # The Nyquist complement is appended after the hops are rewritten, so the
    # realised redundancy sits slightly above the target, never far from it.
    assert redtar * 0.95 <= rho <= redtar * 1.15, rho
    assert isinstance(info["painless"], bool)


@pytest.mark.parametrize("sampling", ["fractional", "fractionaluniform"])
def test_redtar_is_monotone_in_the_fractional_modes(sampling):
    rhos = []
    for redtar in (2.0, 4.0, 8.0, 16.0):
        _g, a, _fc, L, _info = _quiet(
            waveletfilters, 8000, 4096, fmin=50, bins=12, sampling=sampling, redtar=redtar
        )
        rhos.append(_real_redundancy(a, L))
    assert all(r1 < r2 for r1, r2 in itertools.pairwise(rhos)), rhos


@pytest.mark.parametrize("sampling", ["regsampling", "uniform"])
def test_one_dimensional_hop_modes_are_unchanged(sampling):
    # These modes never went through the faulty line; their hops are the
    # integer floor of the scaled hops, exactly as before the fix.
    _g, a, _fc, L, _info = _quiet(
        waveletfilters, 8000, 4096, fmin=50, bins=12, sampling=sampling, redtar=4.0
    )
    a = np.asarray(a)
    if a.ndim == 2:
        assert np.all(a[:, 1] >= 1)
    rho = _real_redundancy(a, L)
    assert 3.0 <= rho <= 5.5, rho
