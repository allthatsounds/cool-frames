"""
numpy/layer3/_constphase.py
===========================
Phase reconstruction via heap-based PGHI (Weighted Phase Gradient Heap
Integration) for non-uniform filterbanks.

The traversal order follows a max-heap on coefficient magnitudes: the
highest-magnitude coefficient is the seed, and propagation ripples outward
through immediate time and frequency neighbours until every coefficient
is visited.  This matches the MATLAB ``comp_filterbankheapint`` /
``trapezheap_fb`` integration rules (trapezoidal rule in time and frequency).

MATLAB original: layer3/phase_processing/comp_filterbankconstphase.m
"""

from __future__ import annotations

import heapq
import math
import warnings

import numpy as np

from ..filterbanks._core import filterbank
from ..filterbanks._utils import normalise_a
from ..filters._design import filterbanklength
from ._phasegrad import filterbankphasegrad

# ---------------------------------------------------------------------------
# comp_filterbankneighbors – build the neighbour graph
# ---------------------------------------------------------------------------


def build_neighbor_map(N: list[int], a_int: list[int]) -> list[dict]:
    """Pre-compute time and frequency neighbour addresses for each subband.

    For subband m with N[m] samples and hop size a[m]:
      - time neighbours:   (m,  n±1)  → same subband, adjacent frame
      - freq neighbours:   (m±1, n')  → adjacent subband, nearest-time index

    Parameters
    ----------
    N : list of M ints
        Number of samples in each subband.
    a_int : list of M ints
        Integer hop sizes for each subband.

    Returns
    -------
    neighbors : list of M dicts, each containing:
        ``'time_prev'``, ``'time_next'`` : (N_m,) index arrays (same subband)
        ``'freq_prev'`` : tuple (m-1, indices) or None
        ``'freq_next'`` : tuple (m+1, indices) or None

    Examples
    --------
    >>> from cool_frames.numpy.phase._constphase import build_neighbor_map
    >>> N = [10, 12, 11]
    >>> a_int = [32, 32, 32]
    >>> nbrs = build_neighbor_map(N, a_int)
    >>> len(nbrs) == 3
    True
    """
    M = len(N)
    nbrs = []
    for m in range(M):
        Nm = N[m]
        am = a_int[m]

        t_prev = (np.arange(Nm) - 1) % Nm
        t_next = (np.arange(Nm) + 1) % Nm

        # Frequency-lower neighbour (m-1 → m)
        fp_m = None
        if m > 0:
            Nm_prev = N[m - 1]
            am_prev = a_int[m - 1]
            # Map time index in subband m → nearest time index in subband m-1
            fp_m = (np.arange(Nm) * am / am_prev).astype(int) % Nm_prev

        fn_m = None
        if m < M - 1:
            Nm_next = N[m + 1]
            am_next = a_int[m + 1]
            fn_m = (np.arange(Nm) * am / am_next).astype(int) % Nm_next

        nbrs.append(
            {
                "time_prev": t_prev,
                "time_next": t_next,
                "freq_prev": (m - 1, fp_m) if fp_m is not None else None,
                "freq_next": (m + 1, fn_m) if fn_m is not None else None,
            }
        )
    return nbrs


# ---------------------------------------------------------------------------
# fixed-order phase integration
# ---------------------------------------------------------------------------


def heap_pghi(
    abss: np.ndarray,
    tgrad: np.ndarray,
    fgrad: np.ndarray,
    N: list[int],
    a_int: list[int],
    fc_norm: np.ndarray,
    tol: float = 1e-6,
    phasetype: int = 0,
) -> np.ndarray:
    """Heap-based phase gradient integration (WPGHI) for filterbanks.

    Reconstructs phase using weighted phase gradient heap integration [constphase-balazs]_
    and non-iterative STFT phase reconstruction [constphase-prusa]_.

    Implements the MATLAB ``comp_filterbankheapint`` / ``trapezheap_fb``
    integration rules: trapezoidal rule in both time and frequency, with
    a max-heap on coefficient magnitudes driving the traversal order.

    The highest-magnitude coefficient is the seed (phase = 0).  Its
    immediate neighbours are pushed onto the heap with their integrated
    phases.  The heap always pops the highest-magnitude unvisited
    coefficient next, ensuring that phase propagates outward through
    reliable (high-energy) paths first.

    Parameters
    ----------
    abss    : (Nsum,) magnitude array (concatenated over channels)
    tgrad   : (Nsum,) normalised instantaneous frequency (from filterbankphasegrad)
    fgrad   : (Nsum,) group delay (from filterbankphasegrad)
    N       : list of M subband lengths
    a_int   : list of M integer hop sizes
    fc_norm : (M,) normalised centre frequencies in [0, 2]
              (fc_Hz / fs * 2, so 0 = DC, 1.0 = Nyquist)
    tol     : relative magnitude threshold for "reliable" phase
    phasetype : 0 = freq-invariant, 1 = time-invariant

    Returns
    -------
    phase : (Nsum,) phase array
        Reconstructed phase values.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.phase._constphase import heap_pghi
    >>> abss = np.array([1.0, 0.8, 0.6, 0.5])
    >>> tgrad = np.zeros(4)
    >>> fgrad = np.zeros(4)
    >>> N = [4]
    >>> a_int = [32]
    >>> fc_norm = np.array([0.0])
    >>> phase = heap_pghi(abss, tgrad, fgrad, N, a_int, fc_norm)
    >>> phase.shape
    (4,)

    References
    ----------
    .. [constphase-balazs] P. Balazs, M. Dörfler, N. Holighaus, F. Jaillet, G. Velasco, "Theory, implementation
           and applications of nonstationary Gabor frames," J. Comput. Appl. Math., vol. 236,
           no. 6, pp. 1481–1496, 2011.
    .. [constphase-prusa] Z. Průša, P. Balazs, P. L. Søndergaard, "A non-iterative method for STFT phase
           (re)construction," IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 25, no. 5,
           pp. 1091–1101, 2017. doi:10.1109/TASLP.2017.2678166
    """
    M = len(N)
    Nsum = int(sum(N))
    phase = np.zeros(Nsum)
    visited = np.zeros(Nsum, dtype=bool)

    # Channel offset table
    offsets = np.zeros(M + 1, dtype=int)
    for m in range(M):
        offsets[m + 1] = offsets[m] + N[m]

    def flat_idx(m: int, n: int) -> int:
        return offsets[m] + n % N[m]  # type: ignore[no-any-return]

    # --- Convert gradients to radians ---
    tgradw = tgrad * math.pi
    fgradw = -fgrad * math.pi

    # --- Helper: push all unvisited neighbours of (m, n) onto the heap ---
    def _push_neighbours(m: int, n: int, fi: int, heap: list):
        """Compute phase for each unvisited neighbour and push onto heap."""
        src_phase = phase[fi]

        # -- Time neighbours (same channel) --
        n_next = (n + 1) % N[m]
        if n_next != 0:  # avoid circular wrap
            fi_next = flat_idx(m, n_next)
            if not visited[fi_next]:
                p = src_phase + a_int[m] * (tgradw[fi] + tgradw[fi_next]) / 2
                heapq.heappush(heap, (-abss[fi_next], fi_next, p))

        n_prev = (n - 1) % N[m]
        if n != 0:  # avoid circular wrap
            fi_prev = flat_idx(m, n_prev)
            if not visited[fi_prev]:
                p = src_phase - a_int[m] * (tgradw[fi] + tgradw[fi_prev]) / 2
                heapq.heappush(heap, (-abss[fi_prev], fi_prev, p))

        # -- Frequency neighbours (across channels) --
        t_w = n * a_int[m]

        if m + 1 < M:
            n_fn = int(round(n * a_int[m] / a_int[m + 1])) % N[m + 1]
            fi_fn = flat_idx(m + 1, n_fn)
            if not visited[fi_fn]:
                t_fn = n_fn * a_int[m + 1]
                dt = t_fn - t_w
                df = fc_norm[m + 1] - fc_norm[m]
                if df < 0:
                    df += 2.0
                p = (
                    src_phase
                    + dt * (tgradw[fi] + tgradw[fi_fn]) / 2
                    + df * (fgradw[fi] + fgradw[fi_fn]) / 2
                )
                heapq.heappush(heap, (-abss[fi_fn], fi_fn, p))

        if m > 0:
            n_fp = int(round(n * a_int[m] / a_int[m - 1])) % N[m - 1]
            fi_fp = flat_idx(m - 1, n_fp)
            if not visited[fi_fp]:
                t_fp = n_fp * a_int[m - 1]
                dt = t_fp - t_w
                df = fc_norm[m - 1] - fc_norm[m]
                if df > 0:
                    df -= 2.0
                p = (
                    src_phase
                    + dt * (tgradw[fi] + tgradw[fi_fp]) / 2
                    + df * (fgradw[fi] + fgradw[fi_fp]) / 2
                )
                heapq.heappush(heap, (-abss[fi_fp], fi_fp, p))

    # --- Seed: highest-magnitude coefficient ---
    seed = int(np.argmax(abss))
    phase[seed] = 0.0
    visited[seed] = True
    m_seed = int(np.searchsorted(offsets, seed + 1) - 1)
    n_seed = seed - offsets[m_seed]

    # Max-heap (negate magnitude for Python's min-heap)
    heap: list = []
    _push_neighbours(m_seed, n_seed, seed, heap)

    # --- Main loop: pop highest-magnitude unvisited, propagate ---
    while heap:
        _neg_mag, fi, p = heapq.heappop(heap)
        if visited[fi]:
            continue
        phase[fi] = p
        visited[fi] = True

        m = int(np.searchsorted(offsets, fi + 1) - 1)
        n = fi - offsets[m]
        _push_neighbours(m, n, fi, heap)

    return phase


# Keep the old name as an alias for backward compatibility
fixed_order_pghi = heap_pghi


# ---------------------------------------------------------------------------
# filterbankconstphase – public API
# ---------------------------------------------------------------------------


def filterbankconstphase(
    f,
    g,
    a=None,
    L: int | None = None,
    fc: np.ndarray | None = None,
    tol: float = 1e-6,
    tgrad: list[np.ndarray] | None = None,
    fgrad: list[np.ndarray] | None = None,
    sqtfr: np.ndarray | None = None,
    fs: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> tuple:
    """Reconstruct phase for a filterbank using fixed-order PGHI.

    Phase reconstruction via heap-based phase gradient integration [constphase-balazs]_
    and non-iterative methods [constphase-prusa]_.

    Handles both uniform and non-uniform filter banks through the same
    flattened heap integrator (a uniform bank is simply all-equal hops; the
    former ``comp_ufilterbankconstphase`` 2-D specialisation is subsumed here).

    Two calling conventions:
    1. Signal path: filterbankconstphase(signal, filters, hops, L, fc)
    2. Magnitude path: filterbankconstphase(magnitudes_list, a_hops, fc_frequencies)

    Parameters
    ----------
    f     : signal (Ls,) or list of magnitude arrays
            Signal for computing c, tgrad, fgrad if not provided.
            OR pre-computed list of magnitude arrays.
    g     : list of M filter dicts (signal path) or hop sizes (magnitude path)
    a     : array-like or None
            Hop sizes (signal path) or centre frequencies (magnitude path).
    L     : DFT length (signal path only)
    fc    : (M,) centre frequencies in Hz (optional when f is signal).
    tol   : relative magnitude threshold
    tgrad : precomputed instantaneous frequencies (optional).  Same convention
            as ``filterbankphasegrad``: absolute normalised instantaneous
            frequency in [0, 2] with 2 == fs, *not* a deviation from ``fc``.
    fgrad : precomputed group delays (optional)
    sqtfr : (M,) sqrt of the per-channel time-frequency ratios.  Required on
            the magnitude path for the phase gradients to be estimated from
            the magnitudes; without it the gradients fall back to zero and
            PGHI degenerates to zero-phase reconstruction.

            **The convention is designer-specific**, and getting it wrong
            costs accuracy rather than raising.

            *What MATLAB LTFAT does.*  Its designers return ``info.tfr``, a
            function handle, and ``filterbankconstphase`` uses
            ``sqrt(info.tfr(L))``.  Recovered from an LTFAT export to 2e-16
            (``tests/crosslang/report_sqtfr.py``), it is strongly
            channel-dependent -- for ``audfilters`` at fs = 8 kHz, Ls = 4096 it
            spans 0.0298 to 719 across 29 channels.  cool-frames' designers do
            not currently expose an equivalent, which is the porting gap behind
            everything below.

            *Use LTFAT's convention.*  Round-trip spectral convergence (dB,
            lower better) on cool-frames' own banks, fs = 8 kHz, Ls = 4096,
            over three probes -- ``k`` scales gamma globally:

            ====================  =======  =======  =======  ======  ======
            audfilters             LTFAT    2xLTFAT  4xLTFAT  ones    signal
            ====================  =======  =======  =======  ======  ======
            sweep + tones           -9.8    -13.0    -15.7   -16.2   -27.3
            white noise             -9.9     -9.7     -8.5    -7.5   -11.2
            AM-FM                  -15.4    -15.2    -10.9   -12.6   -18.9
            ====================  =======  =======  =======  ======  ======

            ====================  =======  =======  =======  ======  ======
            cqtfilters             LTFAT    2xLTFAT  4xLTFAT  ones    signal
            ====================  =======  =======  =======  ======  ======
            sweep + tones           -5.8    -17.7    -13.6    -3.3   -25.0
            white noise            -11.3    -11.6    -10.5    -9.8   -11.4
            AM-FM                  -16.0    -10.5     -9.3    -8.4   -14.4
            ====================  =======  =======  =======  ======  ======

            ``sqrt(info.tfr(L))`` at ``k = 1`` is best or near-best on two of
            three probes for both designers, and it is what the reference
            implementation does.  Single-probe optima disagree wildly --
            ``ones`` looks 6 dB better than LTFAT on the audfilters sweep and
            2 dB *worse* on noise; ``2x`` looks 12 dB better on the cqtfilters
            sweep and 6 dB worse on AM-FM -- so tune this against your own
            material rather than trusting any one row.

            Not yet checked against LTFAT (those designers did not export):
            ``waveletfilters``, where the designer populates it and you
            evaluate ``g[m]['tfr'](L)`` (for Cauchy wavelets
            ``(alpha - 1) / (pi * fc**2 * L)``), and ``gabfilters``, where
            ``gamma = Cg * gl**2`` in the window length (``Cg_hann = 0.25645``,
            tabulated in ``_findgamma.py``).

            Do *not* use ``compute_tfr_from_filters``: it returns L/gamma, the
            support length, a different quantity.

            The remaining gap to the signal path is not a gamma problem.  The
            derivative-filter path needs no gamma at all and would otherwise
            settle this, but it does not reproduce the magnitude path on these
            banks under *any* gamma, on interior channels either -- so no
            choice of ``sqtfr`` closes it.  That, rather than the convention,
            is what limits magnitude-only phase retrieval here.

    fs    : sampling rate in Hz.  On the signal path it is read from the
            filters.  On the magnitude path, supplying it is what makes ``fc``
            unambiguous: without it, an ``fc`` that looks like Hz is normalised
            by assuming the top channel sits at Nyquist, which is exact for
            every single-sided designer here but wrong by about a factor of two
            for a two-sided bank.  That inference warns.
    rng   : seed or ``numpy.random.Generator`` for the random phase assigned to
            below-threshold coefficients.  Pass one for reproducible output;
            the default draws from a fresh local generator, which varies
            between calls but — unlike the global ``np.random`` this used to
            use — leaves the caller's random stream alone.

    Returns
    -------
    c_new    : list of M complex arrays with reconstructed phase
    usedmask : list of M boolean arrays, True where the phase was integrated
               rather than drawn at random because the coefficient sat below
               ``tol`` of the peak magnitude

    .. versionchanged:: 0.1.1
       Returns a 2-tuple.  It previously returned the coefficient list alone,
       while the torch backend already returned ``(c, usedmask)`` and this
       function's own annotation already said ``-> tuple`` (silenced with a
       ``# type: ignore``).  Unpack it::

           c, usedmask = filterbankconstphase(...)

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.phase import filterbankconstphase
    >>> x = np.random.randn(8000)
    >>> g, a, fc, L, _info = audfilters(8000, len(x))
    >>> c_recon, usedmask = filterbankconstphase(x, g, a, L, fc)
    >>> len(c_recon) == len(g)
    True
    >>> usedmask[0].dtype
    dtype('bool')

    References
    ----------
    .. [constphase-balazs] P. Balazs, M. Dörfler, N. Holighaus, F. Jaillet, G. Velasco, "Theory, implementation
           and applications of nonstationary Gabor frames," J. Comput. Appl. Math., vol. 236,
           no. 6, pp. 1481–1496, 2011.
    .. [constphase-prusa] Z. Průša, P. Balazs, P. L. Søndergaard, "A non-iterative method for STFT phase
           (re)construction," IEEE/ACM Trans. Audio, Speech, Lang. Process., vol. 25, no. 5,
           pp. 1091–1101, 2017. doi:10.1109/TASLP.2017.2678166
    """
    # Detect if f is a list (pre-computed magnitudes) vs signal array
    f_is_list = isinstance(f, (list, tuple))

    # Detect if g is a list of dicts (signal path) or numeric (magnitude path)
    g_is_filter_list = isinstance(g, (list, tuple)) and len(g) > 0 and isinstance(g[0], dict)

    if f_is_list and not g_is_filter_list:
        # Magnitude path: f is magnitudes list, g is hop sizes, a is frequencies
        abss_list = [np.asarray(fi) for fi in f]
        a_int_param = np.atleast_1d(g)
        fc_param = a  # In this calling convention, 3rd arg is fc
        M = len(abss_list)
        N = [len(np.asarray(ci).ravel()) for ci in abss_list]

        # Convert a_int_param to proper format
        if a_int_param.ndim == 1:
            a_int = [int(ai) for ai in a_int_param]
        else:
            a_int = [int(a_int_param[m, 0]) for m in range(M)]

        # Build fc_norm from fc_param (needed before gradient computation).
        #
        # The integrator and the gradient estimator both want centre
        # frequencies normalised to [0, 2] with 2 == fs.  When ``fs`` is given
        # that is a plain division and there is nothing to guess.
        #
        # When it is not, the sampling rate is genuinely underdetermined: `fc`
        # alone cannot distinguish a single-sided bank spanning [0, fs/2] from
        # a two-sided one spanning [0, fs).  The old code resolved this by
        # assuming the top channel sits at Nyquist, silently.  That assumption
        # turns out to be *exact* for every single-sided designer the package
        # ships — `audfilters`, `cqtfilters` (which appends a Nyquist channel
        # whatever `fmax` is set to) and `gabfilters(real=True)` all put their
        # last channel at fs/2 — and wrong by a factor of about two for a
        # two-sided bank, whose channels run past Nyquist.
        #
        # So the inference is kept, because it is right for the overwhelming
        # majority of callers and removing it would break them for no gain.
        # What is not kept is the silence: the assumption and the case it fails
        # in are now stated at the point of use, so a two-sided caller finds out
        # from a warning rather than from a bad reconstruction.
        if fc_param is None:
            fc_norm = np.zeros(M)
        else:
            fc_norm = np.asarray(fc_param, dtype=float).ravel()
            if fc_norm.size != M:
                raise ValueError(
                    f"filterbankconstphase: fc has {fc_norm.size} entries but "
                    f"there are {M} channels."
                )
            if fs is not None:
                fc_norm = fc_norm / float(fs) * 2.0
            elif np.max(np.abs(fc_norm)) > 2.0:
                fc_max = float(np.max(np.abs(fc_norm)))
                warnings.warn(
                    "filterbankconstphase: centre frequencies look like Hz "
                    f"(max {fc_max:g} > 2) but no 'fs' was given, so the "
                    f"sampling rate is being inferred as {2 * fc_max:g} Hz by "
                    "assuming the top channel is centred at Nyquist. That is "
                    "exact for audfilters, cqtfilters and gabfilters(real=True), "
                    "and wrong by about a factor of two for a two-sided bank "
                    "whose channels run past Nyquist. Pass fs=<sampling rate> "
                    "to remove the ambiguity, or normalise fc yourself to "
                    "[0, 2] where 2 corresponds to fs.",
                    stacklevel=2,
                )
                fc_norm = fc_norm / fc_max

        # When tgrad/fgrad not provided for magnitude-only input, compute them
        # from the magnitudes via comp_filterbankphasegradfrommag. Falling
        # back to zero gradients (the previous behaviour) made PGHI degenerate
        # to random-phase reconstruction — the "phase gradient" in PGHI
        # literally is the thing being integrated.
        if (tgrad is None or fgrad is None) and sqtfr is not None:
            from ._fbphasegradfrommag import (
                comp_filterbankneighbors,
                comp_filterbankphasegradfrommag,
            )

            N_arr = np.array(N, dtype=int)
            a_arr_int = np.array(a_int, dtype=int)
            NEIGH, posInfo = comp_filterbankneighbors(a_arr_int, M, N_arr, do_real=True)
            abss_flat_tmp = np.concatenate([np.asarray(s, dtype=float).ravel() for s in abss_list])
            sqtfr_arr = np.asarray(sqtfr, dtype=float).ravel()
            if sqtfr_arr.size == 1:
                sqtfr_arr = np.full(M, float(sqtfr_arr[0]))
            tgrad_flat, fgrad_flat, _logs = comp_filterbankphasegradfrommag(
                abss_flat_tmp,
                N_arr,
                a_arr_int,
                M,
                sqtfr_arr,
                fc_norm,
                NEIGH,
                posInfo,
            )
            # Re-split into per-channel arrays matching abss_list shapes
            tgrad = []
            fgrad = []
            offset = 0
            for m in range(M):
                nm = int(N_arr[m])
                tgrad.append(tgrad_flat[offset : offset + nm].reshape(abss_list[m].shape))
                fgrad.append(fgrad_flat[offset : offset + nm].reshape(abss_list[m].shape))
                offset += nm
        elif tgrad is None or fgrad is None:
            # No sqtfr given — fall back to zero gradients (degenerate but
            # preserves backward compatibility). PGHI quality will be poor.
            if tgrad is None:
                tgrad = [np.zeros_like(abss_list[m]) for m in range(M)]
            if fgrad is None:
                fgrad = [np.zeros_like(abss_list[m]) for m in range(M)]

        c = abss_list  # For shape preservation later
    else:
        # Signal path: f is signal, g is filters, a is hop sizes
        f = np.asarray(f)
        M = len(g)
        a_norm = normalise_a(a, M)
        a_int = [int(a_norm[m, 0]) for m in range(M)]

        if L is None:
            L = filterbanklength(len(f), a_norm)

        if tgrad is None or fgrad is None:
            tgrad_c, fgrad_c, _s_c, c = filterbankphasegrad(f, g, a_norm, L)
            if tgrad is None:
                tgrad = tgrad_c
            if fgrad is None:
                fgrad = fgrad_c
            abss_list = [np.abs(np.asarray(ci)) for ci in c]
        else:
            c = filterbank(f, g, a_norm, L=L)
            abss_list = [np.abs(np.asarray(ci)) for ci in c]

        # Normalised centre frequencies: fc_norm in [0, 2] where 2 = fs
        if fc is not None:
            # fc in Hz — extract fs from filter dict
            fs = float(g[0].get("fs", L))
            fc_norm = np.asarray(fc, dtype=float) / fs * 2.0
        else:
            # Estimate from filter foff: centre bin / L * 2
            fc_norm = np.zeros(M)
            for m in range(M):
                gm = g[m]
                if "H" in gm:
                    # ``H`` may be a callable(L) or an already-materialised
                    # array — both are valid filter descriptors, and the torch
                    # wrappers and ``prepare_filters`` produce the latter.
                    # Assuming callable made this branch raise
                    # ``TypeError: 'numpy.ndarray' object is not callable`` for
                    # every such bank, i.e. for the whole torch backend
                    # whenever ``fc`` was not passed explicitly.
                    H_raw = gm["H"]
                    H_vals = np.asarray(H_raw(L) if callable(H_raw) else H_raw)
                    fo = int(gm["foff"](L)) if callable(gm["foff"]) else int(gm["foff"])
                    n_h = len(H_vals)
                    # Weighted centre frequency
                    k_abs = (np.arange(fo, fo + n_h) % L).astype(float)
                    weights = np.abs(H_vals) ** 2
                    ws = weights.sum()
                    if ws > 0:
                        k_cent = k_abs.copy()
                        k_cent[k_cent > L / 2] -= L
                        fc_norm[m] = np.sum(k_cent * weights) / ws / L * 2
                    else:
                        fc_norm[m] = 0.0

        # Flatten into a single vector for the phase integrator
        N = [len(np.asarray(ci).ravel()) for ci in c]

    # Common phase integration (both paths)
    abss_flat = np.concatenate([np.asarray(s).ravel() for s in abss_list])
    tgrad_flat = np.concatenate([np.asarray(tg).ravel() for tg in tgrad])
    fgrad_flat = np.concatenate([np.asarray(fg).ravel() for fg in fgrad])

    phase_flat = fixed_order_pghi(abss_flat, tgrad_flat, fgrad_flat, N, a_int, fc_norm, tol=tol)

    # Assign random phases to below-threshold coefficients.
    #
    # Randomising here is correct and deliberate: the phase of a coefficient at
    # the noise floor carries no information, and integrating through it would
    # propagate that noise into its neighbours.  What was wrong is *which*
    # generator supplied it.  ``np.random.uniform`` draws from NumPy's global
    # state, which made this function (a) irreproducible — four identical calls
    # returned four different answers — and (b) a source of action at a
    # distance, silently advancing the caller's global random stream as a side
    # effect of doing a transform.
    #
    # A local Generator fixes both.  The default still varies between calls, so
    # nobody starts depending on a particular arbitrary phase, but it no longer
    # touches global state; pass ``rng=<int or Generator>`` for reproducibility.
    sMax = float(np.max(abss_flat)) if abss_flat.size > 0 else 1.0
    absthr = sMax * tol
    low_idx = np.where(abss_flat <= absthr)[0]
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    phase_flat[low_idx] = generator.uniform(0, 2 * np.pi, size=len(low_idx))

    # Re-split into per-channel arrays.
    #
    # ``usedmask`` marks the coefficients whose phase was *integrated* rather
    # than filled in at random because they sat below the threshold.  It used to
    # be accumulated here and then dropped on the floor; it is genuinely useful
    # (it tells you which part of the answer means anything) and LTFAT returns
    # it too, so it is returned rather than discarded.
    c_new = []
    usedmask = []
    offset = 0
    for m in range(M):
        nm = N[m]
        phi = phase_flat[offset : offset + nm]
        a_m = abss_list[m].ravel()
        shape = np.asarray(c[m]).shape
        c_new.append((a_m * np.exp(1j * phi)).reshape(shape))
        usedmask.append((a_m > absthr).reshape(shape))
        offset += nm

    return c_new, usedmask
