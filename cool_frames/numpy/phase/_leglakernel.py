"""
numpy/phase/_leglakernel.py
===========================
The truncated projection kernel that gives LEGLA its name.

Background
----------
Every Griffin-Lim-style iteration alternates between two projections: onto the
magnitude constraint set (trivial), and onto the range of the analysis
operator.  The second one, ``P = A A^+``, is what costs: a full synthesis
followed by a full analysis.

Le Roux's observation is that ``P`` is a *convolution*.  Writing ``g_m`` for the
analysis filters and ``gd_m`` for the dual (synthesis) filters, with per-channel
hop ``a_m``::

    (P c)_m[n] = sum_{m'} sum_{n'} k_{m,m'}[(n a_m - n' a_m') mod L] c_{m'}[n']

where the kernel is a cross-correlation, computed once in the frequency domain::

    k_{m,m'} = ifft( conj(G_m) * Gd_{m'} )

Most of that kernel is negligible — filters that sit in different parts of the
spectrum barely overlap — so discarding entries below ``relthr`` times the
kernel's peak leaves a sparse operator that can be applied directly, skipping
entire channel pairs.  On a 23-channel ERB bank at ``relthr=1e-3``, 107 of 529
channel pairs survive and 6.8 % of the lags.

Real mode
---------
With ``real=True`` the synthesis takes the real part, so ``P`` is only
*real*-linear and picks up a second, conjugate-linear term::

    (P c)_m[n] = sum k_{m,m'}[tau] c_{m'}[n'] + sum q_{m,m'}[tau] conj(c_{m'}[n'])

with ``q_{m,m'} = ifft( conj(G_m) * conj(Gd_{m'})(-.) )``.  Both kernels are
built and truncated the same way.

Correctness
-----------
With ``relthr=0`` (nothing discarded) this reproduces the full
synthesise-then-analyse projection to machine precision — that equivalence is
asserted in ``tests/layer3_repr/unit/test_legla_kernel.py`` and is the reason to
trust the truncated version.

Cost
----
Truncation makes each *step* cheaper than the full projection only when the
bank is large enough, or ``relthr`` loose enough, that whole channel pairs drop
out.  For a small bank the full analysis-synthesis (two FFTs per channel) is
hard to beat, and ``gla`` remains the better choice.  The point of the
truncated kernel is that it is a genuinely different, controllable operator —
not that it is unconditionally faster.
"""

from __future__ import annotations

import numpy as np

# Refuse to materialise a kernel bigger than this many stored entries.  At
# 16 bytes per complex value plus two 4-byte indices, 2e7 entries is ~0.5 GB.
_MAX_ENTRIES = 20_000_000


def _full_transfer_function(filt: dict, L: int) -> np.ndarray:
    """Materialise one filter's transfer function over all ``L`` DFT bins."""
    H = filt.get("H")
    if callable(H):
        H = H(L)
    if H is None:
        return np.zeros(L, dtype=complex)
    H = np.asarray(H).ravel()

    foff = filt.get("foff", 0)
    if callable(foff):
        foff = foff(L)
    foff = 0 if foff is None else int(np.asarray(foff).ravel()[0])

    out = np.zeros(L, dtype=complex)
    np.add.at(out, (foff + np.arange(H.size)) % L, H)
    return out


class LeglaKernel:
    """Precomputed truncated projection kernel for a given filterbank.

    Built once per ``legla`` call and applied every iteration.

    Parameters
    ----------
    g, gd : analysis and dual (synthesis) filter dicts
    hops : (M,) int — per-channel hop sizes
    N : (M,) int — per-channel frame counts
    L : int — DFT length
    real : bool — whether synthesis takes the real part
    relthr : float — discard kernel entries below ``relthr * max|k|``.
        ``0.0`` keeps everything and reproduces the exact projection.
    zero_self_term : bool — ``variant='modtrunc'``; drop ``k_{m,m}[0]``, the
        coefficient's own contribution to its update.
    """

    def __init__(self, g, gd, hops, N, L, *, real, relthr, zero_self_term=False):
        self.M = len(g)
        self.N = list(N)
        self.L = int(L)
        self.real = bool(real)
        self.relthr = float(relthr)

        Ga = np.stack([_full_transfer_function(g[m], L) for m in range(self.M)])
        Gd = np.stack([_full_transfer_function(gd[m], L) for m in range(self.M)])
        mirror_idx = (-np.arange(L)) % L

        # Pass 1: the global kernel peak, needed before anything can be
        # thresholded.  Pairs whose spectra do not overlap contribute nothing
        # and are skipped here and below.
        overlap = (np.abs(Ga) > 0) @ (np.abs(Gd) > 0).T
        peak = 0.0
        for m in range(self.M):
            for mp in range(self.M):
                if not overlap[m, mp]:
                    continue
                km = np.fft.ifft(np.conj(Ga[m]) * Gd[mp])
                peak = max(peak, float(np.max(np.abs(km))))
                if self.real:
                    qm = np.fft.ifft(np.conj(Ga[m]) * np.conj(Gd[mp])[mirror_idx])
                    peak = max(peak, float(np.max(np.abs(qm))))
        self.peak = peak
        cutoff = relthr * peak

        # Pass 2: build the sparse operator over the concatenated coefficient
        # vector.  `offs[m]` is where channel m starts in that vector.
        self.offs = np.concatenate([[0], np.cumsum(self.N)]).astype(int)
        ntot = int(self.offs[-1])
        self.ntot = ntot

        rows_k, cols_k, vals_k = [], [], []
        rows_q, cols_q, vals_q = [], [], []
        entries = 0

        for m in range(self.M):
            n_idx = np.arange(self.N[m])
            t_base = (n_idx * int(hops[m])) % L
            for mp in range(self.M):
                if not overlap[m, mp]:
                    continue
                ap = int(hops[mp])
                km = np.fft.ifft(np.conj(Ga[m]) * Gd[mp])
                qm = (
                    np.fft.ifft(np.conj(Ga[m]) * np.conj(Gd[mp])[mirror_idx])
                    if self.real
                    else None
                )
                if zero_self_term and m == mp:
                    km = km.copy()
                    km[0] = 0.0

                for kern, rows, cols, vals in (
                    (km, rows_k, cols_k, vals_k),
                    (qm, rows_q, cols_q, vals_q),
                ):
                    if kern is None:
                        continue
                    taus = np.flatnonzero(np.abs(kern) > cutoff)
                    if taus.size == 0:
                        continue
                    for tau in taus:
                        t = (t_base - int(tau)) % L
                        ok = (t % ap) == 0
                        if not np.any(ok):
                            continue
                        npi = t[ok] // ap
                        inside = npi < self.N[mp]
                        if not np.any(inside):
                            continue
                        r = n_idx[ok][inside] + self.offs[m]
                        c = npi[inside] + self.offs[mp]
                        rows.append(r)
                        cols.append(c)
                        vals.append(np.full(r.size, kern[tau]))
                        entries += r.size

                if entries > _MAX_ENTRIES:
                    raise MemoryError(
                        f"legla: the truncated projection kernel needs more than "
                        f"{_MAX_ENTRIES:,} stored entries at relthr={relthr:g}. "
                        f"Raise `relthr` to truncate harder, or use `gla`, whose "
                        f"full analysis-synthesis projection has no such cost."
                    )

        self._P = self._assemble(rows_k, cols_k, vals_k, ntot)
        self._Q = self._assemble(rows_q, cols_q, vals_q, ntot) if self.real else None
        self.nnz = int(self._P.nnz + (self._Q.nnz if self._Q is not None else 0))

    @staticmethod
    def _assemble(rows, cols, vals, ntot):
        from scipy.sparse import coo_matrix

        if not rows:
            return coo_matrix((ntot, ntot), dtype=complex).tocsr()
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals)
        return coo_matrix((v, (r, c)), shape=(ntot, ntot), dtype=complex).tocsr()

    def project(self, c: list[np.ndarray]) -> list[np.ndarray]:
        """Apply the truncated projection to a coefficient list."""
        x = np.concatenate([np.asarray(cm).ravel() for cm in c])
        y = self._P @ x
        if self._Q is not None:
            y = y + self._Q @ np.conj(x)
        return [y[self.offs[m] : self.offs[m + 1]] for m in range(self.M)]

    def __repr__(self):  # pragma: no cover - diagnostics only
        return (
            f"LeglaKernel(M={self.M}, L={self.L}, real={self.real}, "
            f"relthr={self.relthr:g}, nnz={self.nnz:,})"
        )
