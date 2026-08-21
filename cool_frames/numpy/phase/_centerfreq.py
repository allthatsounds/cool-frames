"""
numpy/phase/_centerfreq.py
==========================
Estimate normalised centre frequencies directly from filter transfer functions.

Several phase-retrieval methods need to know roughly where each channel sits in
frequency (``spsi`` to advance its phase accumulator, ``gsrtisila`` to seed an
unwrapping phase).  The filter dictionaries produced by the constructors in
``cool_frames.numpy.filters`` do not carry a ``fc`` entry, so the value has to be
recovered from the frequency response.

Doing that properly matters: the previous fallback was ``fc[m] = m / M``, a
linear ramp that reaches ``0.96`` cycles per sample — nearly twice Nyquist —
for the top channel of any bank, regardless of the actual filter layout.
"""

from __future__ import annotations

import numpy as np


def filter_center_frequencies(g: list[dict], L: int) -> np.ndarray:
    """Normalised centre frequency (cycles per sample) of each filter.

    Parameters
    ----------
    g : list of M filter dicts, each with ``H`` (transfer function samples,
        array or callable) and ``foff`` (DFT-bin offset of the first sample).
    L : int — DFT length the filters are evaluated on.

    Returns
    -------
    (M,) float array in ``[0, 0.5]``.

    Notes
    -----
    The estimate is the energy-weighted centroid of ``|H|^2`` over the filter's
    support, which is stabler than the bare argmax for flat-topped filters and
    identical to it for sharply peaked ones.  Frequencies above Nyquist are
    folded back, so a two-sided filter reports its positive-frequency centre.
    """
    M = len(g)
    fc = np.zeros(M, dtype=float)

    for m in range(M):
        gm = g[m]
        H = gm.get("H")
        if callable(H):
            H = H(L)
        if H is None:
            continue
        H = np.asarray(H).ravel()
        if H.size == 0:
            continue

        # `foff` is sometimes a plain int and sometimes a callable of L, in the
        # same way as `H`.
        foff_spec = gm.get("foff", 0)
        if callable(foff_spec):
            foff_spec = foff_spec(L)
        foff = 0 if foff_spec is None else int(np.asarray(foff_spec).ravel()[0])
        bins = (foff + np.arange(H.size)) % L
        w = np.abs(H) ** 2
        total = float(np.sum(w))
        if total <= 0:
            continue

        # Centroid on the unit circle, so that a support wrapping around bin 0
        # does not average to the middle of the spectrum.
        theta = 2.0 * np.pi * bins / L
        ang = np.angle(np.sum(w * np.exp(1j * theta)) / total)
        f = (ang / (2.0 * np.pi)) % 1.0
        fc[m] = f if f <= 0.5 else 1.0 - f

    return fc
