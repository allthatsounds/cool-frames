"""
numpy/filterbanks/_analysis.py
=========================
Matrix-spectral analysis of time-frequency coefficient matrices.

Accepts arbitrary TF coefficient matrices — both uniformly and non-uniformly
sampled (as a list of per-channel arrays with varying lengths) — and returns a
comprehensive dictionary of matrix-spectral properties relevant to signal
processing and machine learning.

Public API
----------
``analyze_coefficients``
    Analyse a coefficient matrix (list-of-arrays or 2-D array).

``analyze_filterbank``
    Build a filterbank, analyse a test signal, and report properties of
    both the coefficients *and* the underlying linear operator.

``analyze_frame_operator``
    Materialise the frame operator S_g for small L and report its full
    matrix-spectral properties.

``print_report``
    Pretty-print the dict returned by any of the above.

Examples
--------
>>> import numpy as np
>>> from cool_frames.numpy.filterbanks import (
...     analyze_coefficients, analyze_filterbank, filterbank)
>>> from cool_frames.numpy.filters import audfilters, filterbanklength
>>> g, a, fc, _, _info = audfilters(8000, 512)
>>> L = filterbanklength(512, a)
>>> x = np.random.default_rng(0).standard_normal(L)
>>> c = filterbank(x, g, a)
>>> report = analyze_coefficients(c, a)
>>> sorted(report)[:2]
['coherence', 'dynamics']
>>> report = analyze_filterbank(g, a, L)   # or in one shot
>>> report['coefficients']['energy']['total'] > 0
True

``print_report(report)`` pretty-prints any of these to stdout.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

from ..core._core import setnorm
from ._firtools import transferfunction

# ====================================================================
# Internal helpers
# ====================================================================

def _as_hop_vector(a, M: int) -> np.ndarray:
    """Convert *a* (scalar, 1-D, or M×2 rational) into a 1-D float hop vector."""
    a = np.asarray(a)
    if a.ndim == 0:
        return np.full(M, float(a))  # type: ignore[no-any-return]
    if a.ndim == 2 and a.shape[1] == 2:
        return a[:, 0].astype(float) / a[:, 1].astype(float)  # type: ignore[no-any-return]
    return a.ravel().astype(float)  # type: ignore[no-any-return]


def _coeff_list(c) -> list[np.ndarray]:
    """Normalise coefficients to a list of 1-D complex arrays.

    A stacked array from ``filterbank(..., stack=True)`` is ``(N, M)`` —
    time down the rows, channels across the columns, the same convention
    ``plotfilterbank`` uses.  Until v0.1.1 this read it as ``(M, N)`` and took
    the *rows*, so every per-channel statistic was computed over the wrong
    axis: a 23-channel bank was reported as having M = 16, with the peak
    channel, Gini coefficient and entropy all computed across time slices
    instead of channels.  No error was raised.
    """
    if isinstance(c, np.ndarray) and c.ndim == 2:
        # Uniform (N, M) matrix -> list of M columns
        return [c[:, m] for m in range(c.shape[1])]
    return [np.asarray(cm).ravel() for cm in c]


def _full_freqresp(filt: dict, L: int) -> np.ndarray:
    """Expand a (lazy or materialised) filter dict into a length-L frequency
    response.

    ``H``/``foff`` are callables ``f(L)`` for lazily-defined filters (the
    designer output) and plain values for materialised filters (e.g. the
    output of ``filterbanktight`` / ``filterbankdual``).
    """
    H_raw = filt['H']
    H = np.asarray(H_raw(L) if callable(H_raw) else H_raw, dtype=complex)
    foff_raw = filt.get('foff', 0)
    foff = int(foff_raw(L) if callable(foff_raw) else foff_raw)
    out = np.zeros(L, dtype=complex)
    for i in range(len(H)):
        out[(foff + i) % L] = H[i]
    return out


# ====================================================================
# 1.  Coefficient-level analysis
# ====================================================================

def analyze_coefficients(
    c,
    a=None,
    *,
    signal_energy: float | None = None,
) -> dict:
    """Analyse matrix-spectral properties of a TF coefficient matrix.

    Parameters
    ----------
    c : list of M arrays (non-uniform) or (N, M) ndarray (uniform)
        Filterbank coefficients as returned by ``filterbank()`` — the stacked
        form is ``(N, M)``, i.e. what ``filterbank(..., stack=True)`` returns.
    a : hop sizes (scalar, (M,), or (M, 2)).  Optional — used for
        per-channel bandwidth weighting.  If *None*, unit hops assumed.
    signal_energy : ‖x‖² of the original signal.  If provided, enables
        frame-bound related diagnostics.

    Returns
    -------
    dict with sections:
        ``shape``     – dimensions (M, N_list, Nsum, uniform)
        ``energy``    – per-channel and total energy
        ``sparsity``  – coefficient sparsity measures
        ``dynamics``  – dynamic range and peak statistics
        ``coherence`` – inter-channel correlation
    """
    cl = _coeff_list(c)
    M = len(cl)
    N_list = [len(cm) for cm in cl]
    Nsum = sum(N_list)
    uniform = len(set(N_list)) == 1

    # ------ hop sizes ------
    if a is not None:
        hops = _as_hop_vector(a, M)
    else:
        hops = np.ones(M)

    # ------ per-channel energy ------
    channel_energies = np.array([np.sum(np.abs(cm) ** 2) for cm in cl])
    total_energy = float(channel_energies.sum())

    # ------ sparsity ------
    all_coefs = np.concatenate([np.abs(cm) for cm in cl])
    l1_norm = float(np.sum(all_coefs))
    l2_norm = float(np.sqrt(np.sum(all_coefs ** 2)))
    # Hoyer sparsity: (√N - L1/L2) / (√N - 1), in [0, 1]
    sqrt_n = np.sqrt(Nsum)
    if l2_norm > 1e-15 and sqrt_n > 1:
        hoyer = (sqrt_n - l1_norm / l2_norm) / (sqrt_n - 1)
    else:
        hoyer = 0.0

    # Fraction of coefficients above 1% of max
    max_abs = float(all_coefs.max()) if len(all_coefs) > 0 else 0.0
    if max_abs > 0:
        active_fraction = float(np.sum(all_coefs > 0.01 * max_abs)) / Nsum
    else:
        active_fraction = 0.0

    # ------ dynamics ------
    channel_peaks = np.array([float(np.max(np.abs(cm))) if len(cm) > 0 else 0.0
                              for cm in cl])
    dynamic_range_db = float('inf')
    if channel_peaks.min() > 0:
        dynamic_range_db = 20 * np.log10(channel_peaks.max() / channel_peaks.min())

    # ------ inter-channel correlation ------
    if uniform and M >= 2:
        N = N_list[0]
        # Build M × N magnitude matrix
        mag_matrix = np.array([np.abs(cm) for cm in cl])  # (M, N)
        # Centre
        mag_matrix -= mag_matrix.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(mag_matrix, axis=1, keepdims=True)
        norms = np.where(norms > 1e-15, norms, 1.0)
        mag_normed = mag_matrix / norms
        corr = mag_normed @ mag_normed.T  # (M, M) correlation
        # Off-diagonal statistics
        mask = ~np.eye(M, dtype=bool)
        offdiag = corr[mask]
        mean_corr = float(np.mean(np.abs(offdiag)))
        max_corr = float(np.max(np.abs(offdiag)))
    else:
        mean_corr = None
        max_corr = None

    # ------ energy distribution ------
    if total_energy > 1e-15:
        energy_dist = channel_energies / total_energy
        energy_entropy = float(-np.sum(energy_dist[energy_dist > 0] *
                                        np.log(energy_dist[energy_dist > 0])))
        energy_gini = _gini(channel_energies)
        peak_channel = int(np.argmax(channel_energies))
        peak_fraction = float(channel_energies[peak_channel] / total_energy)
    else:
        energy_entropy = 0.0
        energy_gini = 0.0
        peak_channel = 0
        peak_fraction = 0.0

    # ------ frame bound diagnostics ------
    frame_diag = {}
    if signal_energy is not None and signal_energy > 0:
        frame_energy_ratio = total_energy / signal_energy
        frame_diag['energy_ratio'] = frame_energy_ratio

    result = {
        'shape': {
            'M': M,
            'N_list': N_list,
            'Nsum': Nsum,
            'uniform': uniform,
            'N': N_list[0] if uniform else None,
        },
        'energy': {
            'total': total_energy,
            'per_channel': channel_energies,
            'peak_channel': peak_channel,
            'peak_fraction': peak_fraction,
            'entropy': energy_entropy,
            'gini': energy_gini,
        },
        'sparsity': {
            'hoyer': hoyer,
            'active_fraction': active_fraction,
            'l1_norm': l1_norm,
            'l2_norm': l2_norm,
            'l1_l2_ratio': l1_norm / (l2_norm + 1e-15),
        },
        'dynamics': {
            'dynamic_range_dB': dynamic_range_db,
            'channel_peaks': channel_peaks,
            'max_abs': max_abs,
        },
        'coherence': {
            'mean_abs_correlation': mean_corr,
            'max_abs_correlation': max_corr,
        },
    }
    if frame_diag:
        result['frame'] = frame_diag

    return result


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative array, in [0, 1]."""
    v = np.sort(np.abs(values.ravel()))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * v) / (n * np.sum(v))) - (n + 1) / n)


# ====================================================================
# 2.  Filterbank / frame-operator analysis
# ====================================================================

# NOTE (v0.1.1): every `filterbank`/`ifilterbank` call in this module now
# passes the caller's `L` explicitly.  Omitting it let `filterbanklength`
# silently round up to the lcm-multiple, so the coefficient counts were
# computed at one length while the frame section used another: redundancy came
# out inflated by L_actual/L_requested (3.63 reported against a true 2.15), and
# a *false* "the painless condition is not satisfied" note fired to explain the
# resulting mismatch on a bank that is painless.


def analyze_filterbank(
    g: list[dict],
    a,
    L: int,
    *,
    real: bool = True,
    materialise: bool | None = None,
    n_probe: int = 200,
) -> dict:
    """Full matrix-spectral analysis of a filterbank as a linear operator.

    Parameters
    ----------
    g : list of M filter dicts
    a : hop sizes
    L : signal length
    real : whether this is a real (single-sided) filterbank
    materialise : if True, build the full L×L frame operator matrix.
        Default: auto (True if L ≤ 512).
    n_probe : number of random probe vectors for statistical estimates

    Returns
    -------
    dict with sections:
        ``filterbank``  – basic filterbank parameters
        ``frame``       – frame bounds A, B, condition number κ
        ``operator``    – Lipschitz, gradient bounds, spectral properties
        ``coherence``   – mutual coherence between filters
        ``rank``        – effective rank, nuclear norm, spectral gap
        ``coefficients``– coefficient-level analysis of a test signal
    """
    from ._core import filterbank, ifilterbank
    from ._frame import filterbankbounds

    M = len(g)
    hops = _as_hop_vector(a, M)

    # ------ frame bounds ------
    if real:
        A, B = filterbankbounds(g, a, L)
    else:
        A, B = filterbankbounds(g, a, L, real=False)
    kappa = B / A if A > 0 else float('inf')

    # ------ N_list and redundancy ------
    x_dummy = np.zeros(L)
    c_dummy = filterbank(x_dummy, g, a, L=L)
    N_list = [len(np.asarray(cm).ravel()) for cm in c_dummy]
    Nsum = sum(N_list)
    redundancy = Nsum / L

    # ------ probe with random vectors ------
    rng = np.random.default_rng(0)
    frame_op_ratios = []      # ‖S_g x‖ / ‖x‖
    coeff_energy_ratios = []  # Σ|c_m|² / ‖x‖²

    for _ in range(n_probe):
        x = rng.standard_normal(L)
        x_norm_sq = np.dot(x, x)

        c = filterbank(x, g, a, L=L)
        coeff_energy = sum(np.sum(np.abs(np.asarray(cm).ravel()) ** 2) for cm in c)
        coeff_energy_ratios.append(coeff_energy / x_norm_sq)

        Sgx = np.real(np.asarray(ifilterbank(c, g, a, Ls=L, real=real)))[:L]
        frame_op_ratios.append(np.linalg.norm(Sgx) / np.sqrt(x_norm_sq))

    frame_op_ratios = np.array(frame_op_ratios)  # type: ignore[assignment]
    coeff_energy_ratios = np.array(coeff_energy_ratios)  # type: ignore[assignment]

    # ------ mutual coherence ------
    freq_resps = [_full_freqresp(g[m], L) for m in range(M)]
    norms_H = [np.linalg.norm(H) for H in freq_resps]
    max_coh = 0.0
    for m in range(M):
        if norms_H[m] < 1e-15:
            continue
        for n in range(m + 1, M):
            if norms_H[n] < 1e-15:
                continue
            coh = abs(np.vdot(freq_resps[m], freq_resps[n])) / (norms_H[m] * norms_H[n])
            max_coh = max(max_coh, coh)

    # ------ materialised operator analysis (eigenvalues, etc.) ------
    if materialise is None:
        materialise = L <= 512

    eig_info = {}
    if materialise:
        eig_info = analyze_frame_operator(g, a, L, real=real)

    # ------ coefficient analysis on a deterministic test signal ------
    #
    # NOTE: the 8000 here is *not* a sampling rate and this is not a bug, though
    # it reads like one and was once filed as one.  ``t`` is a sample index, so
    # ``440 * t / 8000`` is a digital frequency of 0.055 cycles/sample — the
    # tones sit at 0.110, 0.250 and 0.625 of Nyquist whatever the bank's real
    # sampling rate is, and none of them can alias.  The probe is deliberately
    # scale-invariant: it exercises the same relative part of the spectrum for a
    # 4 kHz bank as for a 48 kHz one.
    #
    # Rewriting these as Hz against the filters' own ``fs`` is numerically
    # identical (checked to 5e-14) and buys nothing, so don't.
    t = np.arange(L, dtype=float)
    x_test = (np.sin(2 * np.pi * 440 * t / 8000)
              + 0.5 * np.sin(2 * np.pi * 1000 * t / 8000)
              + 0.3 * np.sin(2 * np.pi * 2500 * t / 8000))
    c_test = filterbank(x_test, g, a, L=L)
    coeff_report = analyze_coefficients(
        c_test, a, signal_energy=float(np.dot(x_test, x_test)))

    result = {
        'filterbank': {
            'M': M,
            'L': L,
            'hops': hops,
            'N_list': N_list,
            'Nsum': Nsum,
            'redundancy': redundancy,
            'uniform': len(set(N_list)) == 1,
        },
        'frame': {
            'A': A,
            'B': B,
            'kappa': kappa,
            'optimal_learning_rate': 2.0 / (A + B) if (A + B) > 0 else float('inf'),
            'convergence_rate': (kappa - 1) / (kappa + 1) if kappa > 0 else 1.0,
        },
        'operator': {
            'lipschitz_analysis': float(np.sqrt(B / 2)) if B > 0 else 0.0,
            'lipschitz_frame_op': float(B),
            'frame_op_ratio_min': float(np.asarray(frame_op_ratios).min()),  # type: ignore[union-attr]
            'frame_op_ratio_max': float(np.asarray(frame_op_ratios).max()),  # type: ignore[union-attr]
            'frame_op_ratio_mean': float(np.asarray(frame_op_ratios).mean()),  # type: ignore[union-attr]
            'coeff_energy_ratio_min': float(np.asarray(coeff_energy_ratios).min()),  # type: ignore[union-attr]
            'coeff_energy_ratio_max': float(np.asarray(coeff_energy_ratios).max()),  # type: ignore[union-attr]
        },
        'coherence': {
            'mutual_coherence': max_coh,
        },
        'coefficients': coeff_report,
    }

    if eig_info:
        result['eigenvalues'] = eig_info
        # Flag if materialised eigenvalues exceed frequency-domain bounds
        # (happens when the painless condition is not satisfied)
        eig_min = eig_info['eigenvalues_min']
        eig_max = eig_info['eigenvalues_max']
        if eig_min < A - 1e-4 or eig_max > B + 1e-4:
            result['eigenvalues']['bounds_note'] = (  # type: ignore[index]
                f"Materialised eigenvalues [{eig_min:.2f}, {eig_max:.2f}] "
                f"exceed frequency-domain bounds [{A:.2f}, {B:.2f}]. "
                f"This is expected when the painless condition is not "
                f"satisfied (filter support > hop size in DFT domain)."
            )

    return result


# ====================================================================
# 3.  Frame operator materialisation
# ====================================================================

def analyze_frame_operator(
    g: list[dict],
    a,
    L: int,
    *,
    real: bool = True,
) -> dict:
    """Materialise the L×L frame operator and compute its spectral properties.

    Parameters
    ----------
    g, a, L : filterbank specification
    real : use ``2·real(...)`` convention

    Returns
    -------
    dict with:
        ``eigenvalues`` – sorted eigenvalues of S_g
        ``erank``       – effective rank (Shannon-entropy measure)
        ``nuclear_norm``– ‖S_g‖_* = tr(S_g)
        ``frobenius_norm`` – ‖S_g‖_F
        ``operator_norm``  – ‖S_g‖_2 (largest eigenvalue)
        ``spectral_gap``   – largest gap between consecutive eigenvalues
        ``spectral_gap_ratio`` – max gap / median gap
        ``diag_energy_ratio``  – fraction of ‖S_g‖_F² on the diagonal
        ``truncation_90``  – rank-k needed for 90% Frobenius energy
        ``symmetry_error`` – ‖S - Sᵀ‖ / ‖S‖ (should be ~0)
        ``positive_definite`` – whether all eigenvalues > 0
    """
    from ._core import filterbank, ifilterbank

    # This materialises the full L x L frame operator (L analysis/synthesis
    # passes) and eigendecomposes it -- O(L^2) memory and O(L^3) time. Warn for
    # large L so the cost is not a surprise; the cheap bounds come from
    # filterbankbounds / filterbankbounds_svd.
    if L > 2048:
        warnings.warn(
            f"analyze_frame_operator materialises and eigendecomposes the full "
            f"{L}x{L} frame operator (O(L^2) memory, O(L^3) time); for L={L} this "
            f"may take a while. For just the frame bounds use filterbankbounds "
            f"(cheap, painless) or filterbankbounds_svd (exact).",
            stacklevel=2)

    S = np.zeros((L, L))
    for j in range(L):
        ej = np.zeros(L)
        ej[j] = 1.0
        c = filterbank(ej, g, a, L=L)
        Sej = np.real(np.asarray(ifilterbank(c, g, a, Ls=L, real=real)))
        S[:, j] = Sej[:L]

    S_sym = 0.5 * (S + S.T)
    eigvals = np.sort(np.linalg.eigvalsh(S_sym))
    eigvals_pos = eigvals[eigvals > 1e-12]

    # Effective rank
    if len(eigvals_pos) > 0:
        p = eigvals_pos / eigvals_pos.sum()
        entropy = float(-np.sum(p * np.log(p)))
        erank = float(np.exp(entropy))
    else:
        erank = 0.0

    # Norms
    nuclear_norm = float(np.trace(S))
    frobenius_norm = float(np.linalg.norm(S, 'fro'))
    operator_norm = float(eigvals.max()) if len(eigvals) > 0 else 0.0

    # Spectral gap
    gaps = np.diff(eigvals)
    if len(gaps) > 0:
        max_gap = float(np.max(gaps))
        median_gap = float(np.median(gaps))
        gap_ratio = max_gap / median_gap if median_gap > 1e-15 else float('inf')
    else:
        max_gap = 0.0
        median_gap = 0.0
        gap_ratio = 0.0

    # Diagonal energy ratio
    diag_energy = float(np.sum(np.diag(S) ** 2))
    frob_sq = frobenius_norm ** 2
    diag_ratio = diag_energy / frob_sq if frob_sq > 0 else 0.0

    # Truncation rank for 90% energy
    eig_desc = eigvals_pos[::-1]
    cum_energy = np.cumsum(eig_desc ** 2) / (np.sum(eig_desc ** 2) + 1e-15)
    k_90 = int(np.searchsorted(cum_energy, 0.90)) + 1

    # Symmetry
    sym_err = float(np.linalg.norm(S - S.T) / (np.linalg.norm(S) + 1e-15))

    return {
        'eigenvalues': eigvals,
        'eigenvalues_min': float(eigvals.min()) if len(eigvals) > 0 else 0.0,
        'eigenvalues_max': float(eigvals.max()) if len(eigvals) > 0 else 0.0,
        'erank': erank,
        'nuclear_norm': nuclear_norm,
        'frobenius_norm': frobenius_norm,
        'operator_norm': operator_norm,
        'spectral_gap': max_gap,
        'spectral_gap_ratio': gap_ratio,
        'diag_energy_ratio': diag_ratio,
        'truncation_90': k_90,
        'symmetry_error': sym_err,
        'positive_definite': bool(eigvals.min() > 0) if len(eigvals) > 0 else False,
    }


# ====================================================================
# 4.  Pretty-print report
# ====================================================================

def print_report(report: dict, *, file=None) -> None:
    """Pretty-print an analysis report to *file* (default: stdout).

    Works with dicts returned by :func:`analyze_coefficients`,
    :func:`analyze_filterbank`, or :func:`analyze_frame_operator`.
    """
    out = file or sys.stdout

    def _p(text=''):
        print(text, file=out)

    def _section(title):
        _p()
        _p(f"{'─' * 60}")
        _p(f"  {title}")
        _p(f"{'─' * 60}")

    def _kv(key, value, unit='', indent=4):
        pad = ' ' * indent
        if isinstance(value, float):
            if abs(value) > 1e6 or (0 < abs(value) < 1e-3):
                _p(f"{pad}{key:.<36s} {value:>12.4e} {unit}")
            else:
                _p(f"{pad}{key:.<36s} {value:>12.4f} {unit}")
        elif isinstance(value, (int, np.integer)):
            _p(f"{pad}{key:.<36s} {value:>12d} {unit}")
        elif isinstance(value, bool):
            _p(f"{pad}{key:.<36s} {'yes' if value else 'no':>12s} {unit}")
        elif isinstance(value, np.ndarray):
            if value.size <= 8:
                _p(f"{pad}{key:.<36s} {np.array2string(value, precision=4, separator=', ')}")
            else:
                _p(f"{pad}{key:.<36s} [{value.min():.4f} .. {value.max():.4f}] ({value.size} values)")
        elif value is None:
            _p(f"{pad}{key:.<36s} {'N/A':>12s}")
        else:
            _p(f"{pad}{key:.<36s} {value!s:>12s}")

    _p()
    _p("  FILTERBANK MATRIX-SPECTRAL ANALYSIS")
    _p(f"  {'=' * 40}")

    # Detect report type and print accordingly
    if 'filterbank' in report:
        _print_filterbank_report(report, _section, _kv, _p)
    elif 'shape' in report:
        _print_coeff_report(report, _section, _kv, _p)
    elif 'eigenvalues' in report:
        _print_eigenvalue_report(report, _section, _kv, _p)
    else:
        _p("  (unrecognised report format)")

    _p()


def _print_filterbank_report(report, _section, _kv, _p):
    fb = report['filterbank']
    _section("Filterbank structure")
    _kv("Channels (M)", fb['M'])
    _kv("Signal length (L)", fb['L'])
    _kv("Total coefficients (Nsum)", fb['Nsum'])
    _kv("Redundancy (Nsum/L)", fb['redundancy'])
    _kv("Uniform hops", fb['uniform'])
    if not fb['uniform']:
        _kv("Hop sizes", fb['hops'])

    fr = report['frame']
    _section("Frame bounds")
    _kv("Lower bound (A)", fr['A'])
    _kv("Upper bound (B)", fr['B'])
    _kv("Condition number (kappa = B/A)", fr['kappa'])
    _kv("Optimal learning rate (2/(A+B))", fr['optimal_learning_rate'])
    _kv("GD convergence rate ((k-1)/(k+1))", fr['convergence_rate'])

    op = report['operator']
    _section("Operator properties")
    _kv("Lipschitz (analysis)", op['lipschitz_analysis'])
    _kv("Lipschitz (frame operator)", op['lipschitz_frame_op'])
    _kv("Frame op ratio min", op['frame_op_ratio_min'])
    _kv("Frame op ratio max", op['frame_op_ratio_max'])
    _kv("Frame op ratio mean", op['frame_op_ratio_mean'])
    _kv("Coeff energy ratio min", op['coeff_energy_ratio_min'])
    _kv("Coeff energy ratio max", op['coeff_energy_ratio_max'])

    coh = report['coherence']
    _section("Coherence")
    _kv("Mutual coherence (mu)", coh['mutual_coherence'])

    if 'eigenvalues' in report:
        _print_eigenvalue_report(report['eigenvalues'], _section, _kv, _p)

    _section("Coefficients (test signal)")
    _print_coeff_report(report['coefficients'], _section, _kv, _p,
                        sub=True)


def _print_coeff_report(report, _section, _kv, _p, sub=False):
    if not sub:
        sh = report['shape']
        _section("Shape")
        _kv("Channels (M)", sh['M'])
        _kv("Total coefficients", sh['Nsum'])
        _kv("Uniform", sh['uniform'])

    en = report['energy']
    if not sub:
        _section("Energy distribution")
    _kv("Total energy", en['total'])
    _kv("Peak channel", en['peak_channel'])
    _kv("Peak channel fraction", en['peak_fraction'])
    _kv("Energy entropy", en['entropy'])
    _kv("Energy Gini coefficient", en['gini'])

    sp = report['sparsity']
    _kv("Hoyer sparsity", sp['hoyer'])
    _kv("Active fraction (>1% of max)", sp['active_fraction'])
    _kv("L1/L2 ratio", sp['l1_l2_ratio'])

    dy = report['dynamics']
    _kv("Dynamic range", dy['dynamic_range_dB'], 'dB')

    co = report['coherence']
    _kv("Mean inter-channel |corr|", co['mean_abs_correlation'])
    _kv("Max inter-channel |corr|", co['max_abs_correlation'])

    if 'frame' in report:
        fr = report['frame']
        _kv("Energy / signal energy", fr.get('energy_ratio'))


def _print_eigenvalue_report(report, _section, _kv, _p):
    _section("Frame operator eigenvalues")
    _kv("Min eigenvalue", report['eigenvalues_min'])
    _kv("Max eigenvalue", report['eigenvalues_max'])
    _kv("Effective rank", report['erank'])
    _kv("Nuclear norm (trace)", report['nuclear_norm'])
    _kv("Frobenius norm", report['frobenius_norm'])
    _kv("Operator norm", report['operator_norm'])
    _kv("Spectral gap (max)", report['spectral_gap'])
    _kv("Spectral gap ratio", report['spectral_gap_ratio'])
    _kv("Diagonal energy ratio", report['diag_energy_ratio'])
    _kv("Rank for 90% energy", report['truncation_90'])
    _kv("Symmetry error", report['symmetry_error'])
    _kv("Positive definite", report['positive_definite'])


# ====================================================================
# Diagnostics moved here from the operator core (2026-06-12)
# ====================================================================

def pgrpdelay(g: dict, L: int) -> np.ndarray:
    """Group delay of a filter with periodic boundary conditions.

    Uses the second-order centred finite-difference approximation of
    the negative phase derivative, with unwrapping.  The result is in
    **samples**.

    Parameters
    ----------
    g : dict
        Filter dict.
    L : int
        DFT length.

    Returns
    -------
    ggd : (L,) real ndarray
        Group delay in samples at each of the *L* DFT frequency bins.
    """
    H = transferfunction(g, L)
    phase = np.angle(H)

    # Forward difference with phase-unwrap
    tgrad_1 = phase - np.roll(phase, -1)
    tgrad_1 -= 2.0 * np.pi * np.round(tgrad_1 / (2.0 * np.pi))

    # Backward difference with phase-unwrap
    tgrad_2 = np.roll(phase, 1) - phase
    tgrad_2 -= 2.0 * np.pi * np.round(tgrad_2 / (2.0 * np.pi))

    # Centred average, convert from radians/bin to samples
    ggd = (tgrad_1 + tgrad_2) / 2.0
    ggd = ggd / (2.0 * np.pi) * L

    return ggd  # type: ignore[no-any-return]


def _negative_half_is_redundant(H: np.ndarray) -> bool:
    """Is there anything on the negative-frequency half worth plotting?

    Two quite different filters answer "no", and ``magresp`` has to catch both:

    * a **real-valued** filter (a real FIR, or a real window's transfer
      function), whose negative half is the conjugate mirror of the positive
      one — redundant;
    * a **single-sided band-limited** filter, as the auditory and constant-Q
      designers produce, whose negative half is simply empty.

    A genuinely two-sided complex filter answers "yes" to neither and must be
    plotted over the full circle.

    ``np.isrealobj`` does not settle the first case: ``np.fft.fft`` of a real
    signal is complex-typed, so a real FIR filter is not a real *object*.  Test
    conjugate symmetry instead.
    """
    H = np.asarray(H).ravel()
    n = H.size
    if n < 2:
        return True
    if np.isrealobj(H):
        return True

    # Real-valued in time <=> conjugate-symmetric in frequency.
    mirrored = np.concatenate([[H[0]], H[:0:-1]])
    scale = float(np.max(np.abs(H))) or 1.0
    if np.max(np.abs(H - np.conj(mirrored))) <= 1e-8 * scale:
        return True

    # Otherwise: is the negative half empty?  Same 0.3 threshold, and the same
    # factor-of-five separation, as `filterbank_is_real` uses for a whole bank.
    half = n // 2
    pos_e = float(np.sum(np.abs(H[1:half]) ** 2))
    neg_e = float(np.sum(np.abs(H[half + 1:]) ** 2))
    return neg_e <= 0.3 * max(pos_e, 1e-30)


def magresp(g, L: int | None = None, *,
            fs: float | None = None,
            norm: str = "null",
            dynrange: float | None = None,
            posfreq: bool | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Magnitude response of a filter or window.

    Unlike the MATLAB ``magresp`` which plots directly, this function
    returns the frequency axis and dB values so the caller can plot
    however they like.

    Parameters
    ----------
    g : dict or array_like
        Filter dict or FIR coefficient array.
    L : int or None
        DFT length.  For FIR inputs defaults to ``13*len(g)+47``
        (smooth interpolation).  For filter dicts, required.
    fs : float or None
        Sampling rate.  ``None`` → normalised frequency axis [0, 1].
    norm : str
        Normalisation applied before computing the response.
    dynrange : float or None
        If given, clip the dB floor to ``max(dB) - dynrange``.
    posfreq : bool or None
        Show only positive frequencies?  ``None`` → auto (positive
        for real-valued signals).

    Returns
    -------
    freq : (K,) float ndarray
        Frequency axis (Hz if *fs* given, else normalised).
    mag_db : (K,) float ndarray
        Magnitude response in dB.
    """
    is_dict = isinstance(g, dict)

    if is_dict:
        if L is None:
            # Try to infer from filter
            if "h" in g:
                gl = len(g["h"])
                L = gl * 13 + 47
            else:
                raise ValueError("magresp: L is required for band-limited filters")
        H = transferfunction(g, L)
    else:
        g_arr = np.asarray(g)
        if L is None:
            L = len(g_arr) * 13 + 47
        # Magnitude response of a raw FIR array (magnitude is phase-invariant)
        H = np.fft.fft(g_arr, L)

    if norm.lower() not in ("null", "none", ""):
        H, _ = setnorm(H, norm)

    # Decide real-only.
    #
    # This used to key off the descriptor's ``realonly`` flag, which is the one
    # place in the package that flag is read — and the designers do not agree
    # about it.  ``cqtfilters`` sets ``realonly=1`` on 76 of its 78 channels
    # while ``audfilters`` sets 0 on all of its, although both produce
    # single-sided banks, so the same plotting call returned a one-sided axis
    # for a CQT filter and a two-sided one for an auditory filter.
    #
    # Ask the filter instead of the label: a single-sided filter has
    # negligible energy on the negative-frequency half.  This is the per-filter
    # form of the test ``filterbank_is_real`` applies to a whole bank, and it
    # agrees with it, so the plot and the transform now answer the same
    # question the same way.
    if posfreq is None:
        posfreq = _negative_half_is_redundant(H)

    if posfreq:
        mag = np.abs(H[:L // 2 + 1])
        if fs is None:
            freq = np.linspace(0, 1, len(mag))
        else:
            freq = np.linspace(0, fs / 2, len(mag))
    else:
        mag = np.abs(np.fft.fftshift(H))
        # The data is fftshift-ed, so the axis must be the fftshift-ed bin
        # frequencies — not a linspace.  ``linspace(-1, 1, L)`` steps by
        # 2/(L-1) where the bins step by 2/L, and it ends at +1, which is not
        # a bin: the true two-sided range is [-1, 1 - 2/L].  The result was an
        # axis stretched by one bin across its whole width, so every plotted
        # feature sat slightly off its real frequency and the error grew
        # towards the edges.
        if fs is None:
            freq = np.fft.fftshift(np.fft.fftfreq(L)) * 2.0
        else:
            freq = np.fft.fftshift(np.fft.fftfreq(L, d=1.0 / float(fs)))

    mag_db = 20.0 * np.log10(mag + np.finfo(float).tiny)

    if dynrange is not None:
        floor = np.max(mag_db) - dynrange
        mag_db = np.maximum(mag_db, floor)

    return freq, mag_db
