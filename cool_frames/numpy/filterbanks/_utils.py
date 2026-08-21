"""
filterbanks/_utils.py
=====================
Internal helpers and conversion utilities for filterbanks.

Contains:
- Hop-size normalisation (``normalise_a``, ``hop_float``)
- Filter preparation and compute dispatch (``prepare_filters``,
  ``comp_filterbank``, ``comp_ifilterbank``)
- Filter-bank window resolution (``filterbankwin``)
- Format conversion (``center_freqs``, ``nonu2ucfmt``, ``u2nonucfmt``,
  ``nonu2ufilterbank``)

MATLAB originals
----------------
  layer2/dispatch/filterbankwin.m
  layer2/dispatch/comp_filterbank_a.m
  layer2/dispatch/comp_filterbank.m
  layer2/dispatch/comp_ifilterbank.m
  layer2/utilities/center_freqs.m
  layer2/utilities/nonu2ucfmt.m
  layer2/utilities/u2nonucfmt.m
  layer2/utilities/nonu2ufilterbank.m
"""
from __future__ import annotations

import numpy as np

from ..core._core import (
    comp_filterbank_fft,
    comp_filterbank_fftbl,
    comp_filterbank_td,
    comp_ifilterbank_fft,
    comp_ifilterbank_fftbl,
    postpad,
)

# ---------------------------------------------------------------------------
# a-matrix normaliser
# ---------------------------------------------------------------------------

def normalise_a(a, M: int) -> np.ndarray:
    """Return a (M, 2) integer array of hop sizes.

    Single integer or (M,) vector → [a_m, 1] for each m.
    (M, 2) → fractional [numerator, denominator].

    Parameters
    ----------
    a : int, array_like, or (M, 2) array
        Hop size(s).
    M : int
        Number of channels (for broadcasting scalars).

    Returns
    -------
    np.ndarray
        Shape (M, 2) with columns [numerator, denominator].

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filterbanks._utils import normalise_a
    >>> a_norm = normalise_a(32, 8)
    >>> a_norm.shape
    (8, 2)
    >>> bool(np.all(a_norm[:, 0] == 32))
    True
    """
    a = np.asarray(a)
    if a.ndim == 0:
        a = np.full(M, int(a), dtype=int)

    if a.ndim == 1:
        a = a.ravel()
        if len(a) == 1:
            a = np.repeat(a, M)
        a = np.column_stack([a, np.ones(len(a), dtype=int)])

    elif a.ndim == 2 and a.shape[1] == 1:
        a = np.column_stack([a.ravel(), np.ones(len(a), dtype=int)])

    return a.astype(int)  # type: ignore[no-any-return]


def hop_float(a_norm: np.ndarray) -> np.ndarray:
    """Return the floating-point hop size array from a (M,2) array.

    Parameters
    ----------
    a_norm : (M, 2) ndarray
        Hop sizes in rational form [numerator, denominator].

    Returns
    -------
    np.ndarray
        Shape (M,) with floating-point hop values.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filterbanks._utils import hop_float
    >>> a_norm = np.array([[32, 1], [16, 2]])
    >>> a_f = hop_float(a_norm)
    >>> a_f
    array([32.,  8.])
    """
    return a_norm[:, 0] / a_norm[:, 1]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# filterbankwin – evaluate and validate filter bank windows
# ---------------------------------------------------------------------------

def filterbankwin(g: list, a, L: int | None = None):
    """Evaluate filter bank windows and return ready-to-use filter dicts.

    Resolves callable ``H`` / ``foff`` fields, handles string-based
    window specifications (``'dual'``, ``'realdual'``, ``'tight'``,
    ``'realtight'``), and validates filter dimensions.

    Parameters
    ----------
    g : list of filter dicts, or ``[type_str, g_inner]`` where
        *type_str* is one of ``'dual'``, ``'realdual'``, ``'tight'``,
        ``'realtight'``.
    a : hop sizes – scalar, (M,) vector, or (M, 2) fractional.
    L : int or None – transform length.  Required for band-limited
        filters with callable ``H``; optional for FIR filters.

    Returns
    -------
    g_out : list of filter dicts with evaluated (ndarray) ``H``/``foff``
    a_out : (M, 2) normalised hop-size array
    info  : dict with keys ``M``, ``ispainless``, ``isfir``,
            ``longestfilter``, ``gl``.

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks._utils import filterbankwin
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> g_ready, a_norm, info = filterbankwin(g, a, L)
    >>> info['M'] == len(g)
    True

    MATLAB original
    ---------------
    ``layer2/dispatch/filterbankwin.m``
    """
    # Handle string-based specs: ['dual', g_inner], etc.
    if (isinstance(g, (list, tuple)) and len(g) >= 2
            and isinstance(g[0], str)):
        from ._frame import filterbankdual, filterbanktight
        kind = g[0].lower()
        g_inner = g[1]
        # Recursively resolve the inner filter bank first
        g_inner, a_out, _info = filterbankwin(g_inner, a, L)
        L_use = L if L is not None else _info.get("L")
        if L_use is None:
            raise ValueError(
                f"filterbankwin: L is required for '{kind}' window spec")
        # `real=` distinguishes the complex (two-sided) construction from the
        # folded real-audio one.  Until v0.1.1 all four branches called the
        # same function with the default real=True, so 'dual' was a silent
        # synonym for 'realdual' and 'tight' for 'realtight' — the complex
        # variants were unreachable.
        if kind == "dual":
            g_out = filterbankdual(g_inner, a, L_use, real=False)
        elif kind == "realdual":
            g_out = filterbankdual(g_inner, a, L_use, real=True)
        elif kind == "tight":
            g_out = filterbanktight(g_inner, a, L_use, real=False)
        elif kind == "realtight":
            g_out = filterbanktight(g_inner, a, L_use, real=True)
        else:
            raise ValueError(
                f"filterbankwin: unsupported window type '{kind}'")
        return filterbankwin(g_out, a, L_use)

    M = len(g)
    a_norm = normalise_a(a, M)

    if L is None:
        # Try to use filters as-is; L will be resolved later by filterbank()
        g_ready = list(g)
        info = {
            "M": M,
            "ispainless": True,
            "isfir": all("h" in gm for gm in g),
            "longestfilter": 0,
            "gl": np.zeros(M, dtype=int),
        }
        if info["isfir"]:
            gls = np.array([len(np.asarray(gm["h"])) for gm in g])
            info["gl"] = gls
            info["longestfilter"] = int(gls.max()) if M > 0 else 0
        return g_ready, a_norm, info

    g_ready, m_td, m_fft, m_fftbl = prepare_filters(g, a_norm, L)

    # Build info dict
    gl = np.zeros(M, dtype=int)
    ispainless = True
    isfir = True

    for m in range(M):
        gm = g_ready[m]
        if "H" in gm and len(gm["H"]) > 0:
            afrac_m = a_norm[m, 0] / a_norm[m, 1]
            Nm = L / afrac_m
            if len(gm["H"]) > Nm:
                ispainless = False
            isfir = False
        elif "h" in gm:
            gl[m] = len(gm["h"])
        else:
            isfir = False

    info = {
        "M": M,
        "L": L,
        "ispainless": ispainless,
        "isfir": isfir,
        "longestfilter": int(gl.max()) if M > 0 and gl.max() > 0 else 0,
        "gl": gl,
    }

    return g_ready, a_norm, info


# ---------------------------------------------------------------------------
# Filter preparation: evaluate callables and classify
# ---------------------------------------------------------------------------

def prepare_filters(g: list[dict], a_norm: np.ndarray, L: int):
    """Evaluate any callable fields in the filter list and split into groups.

    Parameters
    ----------
    g : list of dict
        Filter descriptors (may have callable H/foff).
    a_norm : (M, 2) ndarray
        Normalized hop sizes.
    L : int
        DFT length.

    Returns
    -------
    g_ready  : list of dicts with ``'H'`` (ndarray) and ``'foff'`` (int), or
               ``'h'`` (ndarray) and ``'offset'`` (int)
    m_td     : indices of time-domain FIR filters
    m_fft    : indices of full-length frequency-domain filters
    m_fftbl  : indices of band-limited frequency-domain filters

    Examples
    --------
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks._utils import (
    ...     prepare_filters, normalise_a
    ... )
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> a_norm = normalise_a(a, len(g))
    >>> g_ready, m_td, m_fft, m_fftbl = prepare_filters(g, a_norm, L)
    >>> len(g_ready) == len(g)
    True
    """
    M        = len(g)
    g_ready  = []
    m_td     = []
    m_fft    = []
    m_fftbl  = []

    for m, gm in enumerate(g):
        gm_out = dict(gm)  # shallow copy

        # Evaluate callable H and foff
        if "H" in gm:
            H_val   = gm["H"](L) if callable(gm["H"])   else np.asarray(gm["H"])
            foff_v  = int(gm["foff"](L)) if callable(gm["foff"]) else int(gm["foff"])
            gm_out["H"]    = np.asarray(H_val, dtype=complex)
            gm_out["foff"] = foff_v

            # Scalar / zero sentinel
            if gm_out["H"].ndim == 0 or gm_out["H"].size == 0 or \
               (gm_out["H"].size == 1 and gm_out["H"].flat[0] == 0):
                gm_out["H"]    = np.zeros(0, dtype=complex)
                gm_out["foff"] = 0
                m_fftbl.append(m)

            # Full-length vs band-limited?
            elif len(gm_out["H"]) == L and a_norm[m, 1] == 1:
                m_fft.append(m)
            else:
                m_fftbl.append(m)

        elif "h" in gm:
            gm_out["h"]      = np.asarray(gm["h"])
            gm_out["offset"] = int(gm.get("offset", gm.get("delay", 0)))
            m_td.append(m)
        else:
            # Fallback: assume zero filter
            gm_out["H"]    = np.zeros(0, dtype=complex)
            gm_out["foff"] = 0
            m_fftbl.append(m)

        g_ready.append(gm_out)

    return g_ready, m_td, m_fft, m_fftbl


# ---------------------------------------------------------------------------
# Low-level compute dispatcher
# ---------------------------------------------------------------------------

def comp_filterbank(f: np.ndarray,
                    g: list[dict],
                    a_norm: np.ndarray) -> list[np.ndarray]:
    """Dispatch filterbank analysis to the correct low-level kernel.

    Parameters
    ----------
    f      : (L, W) complex signal (post-padded, post-FFT not required)
    g      : list of prepared filter dicts
    a_norm : (M, 2) hop-size matrix

    Returns
    -------
    c : list of M arrays (N_m, W)
        Filterbank coefficients (one per channel).

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks._utils import (
    ...     comp_filterbank, normalise_a, filterbankwin
    ... )
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> a_norm = normalise_a(a, len(g))
    >>> g_ready, a_norm, _ = filterbankwin(g, a, L)
    >>> f = np.random.randn(L, 1) + 1j * np.random.randn(L, 1)
    >>> c = comp_filterbank(f, g_ready, a_norm)
    >>> len(c) == len(g)
    True
    """
    L, W = f.shape
    M    = len(g)
    c    = [None] * M

    g_ready, m_td, m_fft, m_fftbl = prepare_filters(g, a_norm, L)

    # ---- time-domain FIR ----
    if m_td:
        g_td  = [g_ready[m]["h"] for m in m_td]
        skip  = np.array([g_ready[m]["offset"] for m in m_td])
        a_td  = a_norm[m_td, :]
        c_td  = comp_filterbank_td(f, g_td, a_td[:, 0], skip)
        for k, m in enumerate(m_td):
            c[m] = c_td[k]

    # ---- full-length FFT ----
    if m_fft:
        F      = np.fft.fft(f, axis=0)
        G_fft  = [g_ready[m]["H"] for m in m_fft]
        a_fft  = a_norm[m_fft, :]
        c_fft  = comp_filterbank_fft(F, G_fft, a_fft)
        for k, m in enumerate(m_fft):
            c[m] = c_fft[k]

    # ---- band-limited FFT ----
    if m_fftbl:
        if m_fft:
            F = F  # already computed
        else:
            F = np.fft.fft(f, axis=0)

        G_bl  = [g_ready[m]["H"]    for m in m_fftbl]
        fo_bl = np.array([g_ready[m]["foff"] for m in m_fftbl], dtype=int)
        a_bl  = a_norm[m_fftbl, :]
        ro_bl = np.array([g_ready[m].get("realonly", 0) for m in m_fftbl])

        # Skip zero filters (H is empty array → N_m bins of zeros)
        valid = [k for k, m in enumerate(m_fftbl) if len(G_bl[k]) > 0]
        zero_k = [k for k, m in enumerate(m_fftbl) if len(G_bl[k]) == 0]

        if valid:
            G_v   = [G_bl[k] for k in valid]
            fo_v  = fo_bl[valid]
            a_v   = a_bl[valid, :]
            ro_v  = ro_bl[valid]
            c_v   = comp_filterbank_fftbl(F, G_v, fo_v, a_v, ro_v)
            for i, k in enumerate(valid):
                m = m_fftbl[k]
                c[m] = c_v[i]

        for k in zero_k:
            m    = m_fftbl[k]
            Nm   = max(1, round(L / (a_norm[m, 0] / a_norm[m, 1])))
            c[m] = np.zeros((Nm, W), dtype=complex)

    return c  # type: ignore[return-value]


def comp_ifilterbank(c: list[np.ndarray],
                     g: list[dict],
                     a_norm: np.ndarray,
                     L: int) -> np.ndarray:
    """Dispatch filterbank synthesis.

    Parameters
    ----------
    c : list of (N_m, W) arrays
        Filterbank coefficients from comp_filterbank.
    g : list of prepared filter dicts
    a_norm : (M, 2) hop-size matrix
    L : int
        Output length.

    Returns
    -------
    F : (L, W) complex output in frequency domain
        Synthesized frequency-domain representation.

    Examples
    --------
    >>> import numpy as np
    >>> from cool_frames.numpy.filters import audfilters
    >>> from cool_frames.numpy.filterbanks._utils import (
    ...     comp_filterbank, comp_ifilterbank, normalise_a, filterbankwin
    ... )
    >>> g, a, fc, L, _info = audfilters(8000, 8000)
    >>> a_norm = normalise_a(a, len(g))
    >>> g_ready, a_norm, _ = filterbankwin(g, a, L)
    >>> f = np.random.randn(L, 1) + 1j * np.random.randn(L, 1)
    >>> c = comp_filterbank(f, g_ready, a_norm)
    >>> F_recon = comp_ifilterbank(c, g_ready, a_norm, L)
    >>> F_recon.shape == (L, 1)
    True
    """
    M = len(c)
    W = c[0].shape[1] if c[0].ndim > 1 else 1
    g_ready, m_td, m_fft, m_fftbl = prepare_filters(g, a_norm, L)

    F = np.zeros((L, W), dtype=complex)

    # ---- full-length FFT synthesis ----
    if m_fft:
        G_fft = [g_ready[m]["H"] for m in m_fft]
        a_fft = a_norm[m_fft, :]
        c_sub = [c[m].reshape(-1, 1) if c[m].ndim == 1 else c[m] for m in m_fft]
        F_fft = comp_ifilterbank_fft(c_sub, G_fft, a_fft, L)
        F += F_fft

    # ---- band-limited FFT synthesis ----
    if m_fftbl:
        G_bl  = [g_ready[m]["H"]    for m in m_fftbl]
        fo_bl = np.array([g_ready[m]["foff"] for m in m_fftbl], dtype=int)
        a_bl  = a_norm[m_fftbl, :]
        ro_bl = np.array([g_ready[m].get("realonly", 0) for m in m_fftbl])

        valid = [k for k, m in enumerate(m_fftbl) if len(G_bl[k]) > 0]
        if valid:
            c_sub = [c[m_fftbl[k]].reshape(-1, 1) if c[m_fftbl[k]].ndim == 1
                     else c[m_fftbl[k]] for k in valid]
            G_v   = [G_bl[k] for k in valid]
            fo_v  = fo_bl[valid]
            a_v   = a_bl[valid, :]
            ro_v  = ro_bl[valid]
            F_bl  = comp_ifilterbank_fftbl(c_sub, G_v, fo_v, a_v, ro_v, L)
            F += F_bl

    # Time-domain synthesis (convert back via IFFT, add).
    #
    # Two bugs lived here before v0.1.1:
    #
    #  * `skip` (the filter's `offset`) was read and then never used.  The
    #    analysis leg `comp_filterbank_td` *does* apply it, so synthesis was
    #    not the adjoint of analysis for any filter with a non-zero offset —
    #    which is every filter `gammatonefir` produces.  The resulting "frame
    #    operator" was non-symmetric with negative eigenvalues, and CG on it
    #    diverged.  The delay is a pure phase factor in the DFT domain.
    #
    #  * `np.atleast_2d` turns a 1-D (N,) array into a (1, N) *row*, so
    #    `cm[:, w]` was a length-1 slice that then broadcast one scalar across
    #    every bin.  The two frequency-domain branches above reshape to a
    #    column; this one has to as well.
    if m_td:
        k_bins = np.arange(L)
        for m in m_td:
            h = g_ready[m]["h"]
            skip = int(g_ready[m].get("offset", 0) or 0)
            am = int(a_norm[m, 0])
            H_m = np.fft.fft(postpad(h, L))
            # exp(-2j*pi*k*skip/L) is the DFT of a shift by `skip` samples;
            # conjugated below along with H_m, it undoes the analysis delay.
            delay_phase = np.exp(-2j * np.pi * k_bins * skip / L)
            H_m = H_m * delay_phase

            cm = c[m].reshape(-1, 1) if c[m].ndim == 1 else c[m]
            Nm = cm.shape[0]
            for w in range(W):
                Cm = np.fft.fft(cm[:, w])
                Cm_up = np.tile(Cm, am)[:L]
                F[:, w] += Cm_up * H_m.conj()

    return F


# ---------------------------------------------------------------------------
# pack_coefficients / unpack_coefficients – dense+mask batching for ML
# ---------------------------------------------------------------------------

def pack_coefficients(c: list) -> tuple:
    """Pack a ragged list of per-channel coefficient vectors into a dense
    ``(M, Nmax)`` array plus a boolean mask, for batched ML processing.

    Non-uniform filterbanks produce per-channel arrays of different lengths
    (``N_m``), which resist tensor batching. This stacks them into a padded
    ``(M, Nmax)`` array and returns a mask marking the valid (non-padding)
    entries. Inverse: :func:`unpack_coefficients`.

    Parameters
    ----------
    c : list of M 1-D arrays (per-channel coefficients)

    Returns
    -------
    dense : (M, Nmax) ndarray  (zero-padded; dtype follows the inputs)
    mask  : (M, Nmax) bool ndarray  (True where ``dense`` holds a real coefficient)
    """
    arrs = [np.asarray(cm).ravel() for cm in c]
    M = len(arrs)
    Nmax = max((a.size for a in arrs), default=0)
    dtype = np.result_type(*[a.dtype for a in arrs]) if arrs else complex
    dense = np.zeros((M, Nmax), dtype=dtype)
    mask = np.zeros((M, Nmax), dtype=bool)
    for m, a in enumerate(arrs):
        dense[m, : a.size] = a
        mask[m, : a.size] = True
    return dense, mask


def unpack_coefficients(dense: np.ndarray, mask: np.ndarray) -> list:
    """Inverse of :func:`pack_coefficients`: a ``(M, Nmax)`` dense array + mask
    back into the ragged list of M per-channel coefficient vectors."""
    dense = np.asarray(dense)
    mask = np.asarray(mask, dtype=bool)
    return [dense[m][mask[m]] for m in range(dense.shape[0])]
