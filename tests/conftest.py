"""
conftest.py – shared pytest fixtures for the LTFAT filterbank test suite.

Fixture hierarchy
-----------------
All reference fixtures are *session*-scoped: the .mat files are loaded once
per test session and shared across every test that requests them.

Reference data is produced by running in MATLAB:

    cd filterbank/
    export_reference_data()

which writes  tests/reference_data/*.mat  (v7 format, scipy.io.loadmat-ready).

If the reference data directory is absent, or scipy is not installed, every
fixture that returns reference data calls ``pytest.skip()`` so that the test is
reported as SKIPPED (not ERROR or FAILED).  Tests that do *not* need reference
data (pure structural / hypothesis tests) run unconditionally.

Implementation guard
--------------------
Tests that call into the not-yet-ported  cool_frames  package are marked
with ``@pytest.mark.requires_impl`` and will be skipped automatically until the
package is importable.  The skip is applied via the ``needs_impl`` fixture so
callers do not have to repeat the import guard.

Usage in test files
-------------------
    # structural test – always runs
    def test_shape(params):
        assert int(params["M"]) > 0

    # reference accuracy test – skipped until ref data exists
    @pytest.mark.requires_ref
    def test_tgrad_matches_reference(phasegrad_ref, params):
        N = params["N"]
        tgrad = split_channels(phasegrad_ref["tgrad_flat"], N)
        ...

    # implementation test – skipped until cool_frames is installed
    @pytest.mark.requires_impl
    def test_filterbankphasegrad_runs(needs_impl, params):
        from cool_frames.phase import filterbankphasegrad
        ...
"""

from __future__ import annotations

import pathlib

import pytest

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_TESTS_DIR = pathlib.Path(__file__).parent
_REF_DIR   = _TESTS_DIR / "reference_data"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_mat(name: str) -> dict | None:
    """
    Load  reference_data/<name>.mat  and return its contents as a dict.
    Returns None (without raising) if scipy is unavailable or the file is
    missing – callers are responsible for calling pytest.skip() if needed.
    """
    try:
        import scipy.io
    except ImportError:
        return None
    path = _REF_DIR / name
    if not path.exists():
        return None
    return scipy.io.loadmat(str(path), squeeze_me=True)


def split_channels(flat: np.ndarray, N: np.ndarray) -> list[np.ndarray]:
    """
    Split a flat (Nsum,) array into a list of M per-channel arrays.

    Parameters
    ----------
    flat : ndarray, shape (Nsum,)
    N    : ndarray, shape (M,), dtype int – subband lengths

    Returns
    -------
    list of M arrays with lengths N[0], N[1], …, N[M-1]
    """
    N = np.asarray(N, dtype=int)
    return np.split(flat, np.cumsum(N)[:-1])


def _check_ref(data: dict | None, filename: str) -> dict:
    """
    Return *data* if it is not None, otherwise call pytest.skip() with a
    descriptive message telling the user how to generate the file.
    """
    if data is None:
        pytest.skip(
            f"reference_data/{filename} not found or scipy unavailable. "
            "Run  export_reference_data()  in MATLAB to generate reference files."
        )
    return data


# ---------------------------------------------------------------------------
# Implementation availability
# ---------------------------------------------------------------------------

def _has_impl() -> bool:
    try:
        import cool_frames  # noqa: F401
        return True
    except ImportError:
        return False


HAS_IMPL = _has_impl()


@pytest.fixture(scope="session")
def needs_impl():
    """
    Request this fixture in any test that calls into cool_frames.
    The test is skipped automatically if the package is not yet installed.

    Example
    -------
    def test_something(needs_impl):
        from cool_frames.phase import filterbankphasegrad
        ...
    """
    if not HAS_IMPL:
        pytest.skip("cool_frames not yet implemented / installed.")


# ---------------------------------------------------------------------------
# Test-parameter fixtures  (no reference data required)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_params() -> dict:
    """
    Scalar test parameters matching make_test_params.m:
        fs=8000, Ls=1024, tol=1e-8, abs_tol=1e-12
    """
    return {
        "fs":      8000,
        "Ls":      1024,
        "tol":     1e-8,
        "abs_tol": 1e-12,
    }


@pytest.fixture(scope="session")
def test_signal(test_params) -> np.ndarray:
    """
    Deterministic test signal matching export_reference_data.m (rng(42)):
        f = sin(2π·440·t) + 0.5·sin(2π·1000·t) + 0.1·noise
    Shape: (Ls,)
    """
    fs = test_params["fs"]
    Ls = test_params["Ls"]
    t  = np.arange(Ls) / fs
    rng = np.random.default_rng(42)
    return (
        np.sin(2 * np.pi * 440  * t)
        + 0.5 * np.sin(2 * np.pi * 1000 * t)
        + 0.1 * rng.standard_normal(Ls)
    )


# ---------------------------------------------------------------------------
# Reference-data fixtures  (skip if absent)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def params(test_params) -> dict:
    """
    Filterbank geometry loaded from  reference_data/params.mat.

    Fields (all squeezed to scalars / 1-D arrays by scipy):
        fs, Ls, L, M, Nsum   – scalars (int)
        N                     – (M,) int32  subband lengths
        a_num, a_den          – (M,) int32  rational hop numerators/denominators
        a_rat                 – (M,) float64  a_num / a_den
        fc_n                  – (M,) float64  normalised center frequencies
        tfr                   – (M,) float64  time-frequency ratios at L
        f                     – (Ls,) float64  test signal
    """
    return _check_ref(_load_mat("params.mat"), "params.mat")


@pytest.fixture(scope="session")
def filters_ref(params) -> dict:
    """
    Full-length filter transfer functions from  reference_data/filters.mat.

    Fields:
        G_cols – (L, M) complex128  column m = DFT of filter m at length L
        foff   – (M,) int32  frequency offset of each bandlimited filter
    """
    return _check_ref(_load_mat("filters.mat"), "filters.mat")


@pytest.fixture(scope="session")
def coeff_ref(params) -> dict:
    """
    Filterbank coefficients from  reference_data/filterbank_coeff.mat.

    Fields (all flat, shape (Nsum,)):
        c_flat   – complex128  analysis coefficients
        ch_flat  – complex128  frequency-weighted coefficients
        cd_flat  – complex128  time-weighted coefficients
    """
    return _check_ref(_load_mat("filterbank_coeff.mat"), "filterbank_coeff.mat")


@pytest.fixture(scope="session")
def phasegrad_ref(params) -> dict:
    """
    Phase gradient from complex coefficients  (reference_data/phasegrad.mat).

    Fields (flat, shape (Nsum,)):
        tgrad_flat – float64  normalised instantaneous frequency
        fgrad_flat – float64  negative group delay (samples)
        cs_flat    – float64  spectrogram  |c|^2
    """
    return _check_ref(_load_mat("phasegrad.mat"), "phasegrad.mat")


@pytest.fixture(scope="session")
def neighbors_ref(params) -> dict:
    """
    Neighbour graph from  reference_data/neighbors.mat.

    Fields:
        NEIGH   – (Nsum, 6) int32  0-based flat indices, -1 = absent
        posInfo – (Nsum, 2) float64  col0 = channel (0-based), col1 = time (samples)
    """
    return _check_ref(_load_mat("neighbors.mat"), "neighbors.mat")


@pytest.fixture(scope="session")
def phasegrad_frommag_ref(params) -> dict:
    """
    Phase gradient from magnitude  (reference_data/phasegrad_frommag.mat).

    Fields (flat, shape (Nsum,)):
        tgrad_mag_flat – float64
        fgrad_mag_flat – float64
        logs_flat      – float64  log of scaled magnitude
        abss_flat      – float64  scaled magnitude input
        sqtfr          – (M,) float64  sqrt of TFR vector
        scal           – (M,) float64  natural-scaling factors
    """
    return _check_ref(_load_mat("phasegrad_frommag.mat"), "phasegrad_frommag.mat")


@pytest.fixture(scope="session")
def heapint_ref(params) -> dict:
    """
    Heap integration reference  (reference_data/heapint.mat).

    Fields (flat, shape (Nsum,)):
        s_flat             – float64  spectrogram input
        tgrad_flat         – float64  gradient input
        fgrad_flat         – float64  gradient input
        phase_timeinv_flat – float64  PHASETYPE_TIMEINV output
        phase_relgrad_flat – float64  PHASETYPE_RELGRAD output
        tol                – float64  scalar tolerance used
    """
    return _check_ref(_load_mat("heapint.mat"), "heapint.mat")


@pytest.fixture(scope="session")
def constphase_ref(params) -> dict:
    """
    Full filterbankconstphase output  (reference_data/constphase.mat).

    Fields (flat, shape (Nsum,)):
        c_cp_flat     – complex128  reconstructed complex coefficients
        newphase_flat – float64     unwrapped phase (radians)
        usedmask_flat – float64     0/1 mask of bins that were processed
    """
    return _check_ref(_load_mat("constphase.mat"), "constphase.mat")


@pytest.fixture(scope="session")
def reassign_ref(params) -> dict:
    """
    Reassignment reference  (reference_data/reassign.mat).

    Fields:
        sr_flat        – (Nsum,) float64  reassigned spectrogram values
        repos_lengths  – (Nsum,) int32    number of contributing bins per output bin
        repos_concat   – (K,)   int32    all contributing bin indices concatenated
    """
    return _check_ref(_load_mat("reassign.mat"), "reassign.mat")


@pytest.fixture(scope="session")
def unif_heapint_ref() -> dict:
    """
    Uniform filterbank heap integration  (reference_data/unif_heapint.mat).

    Fields:
        c_mat               – (L, M) complex128  analysis coefficients
        s_mat               – (L, M) float64     spectrogram
        tgrad_mat           – (L, M) float64
        fgrad_mat           – (L, M) float64
        phase_timeinv_mat   – (L, M) float64
        phase_relgrad_mat   – (L, M) float64
        N_unif, M, a_unif   – scalars (int)
        fc_n                – (M,) float64
    """
    return _check_ref(_load_mat("unif_heapint.mat"), "unif_heapint.mat")


# ---------------------------------------------------------------------------
# Convenience fixtures  (derived from reference data)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def M(params) -> int:
    """Number of filterbank channels."""
    return int(params["M"])


@pytest.fixture(scope="session")
def L(params) -> int:
    """System length."""
    return int(params["L"])


@pytest.fixture(scope="session")
def N(params) -> np.ndarray:
    """Subband lengths, shape (M,), dtype int."""
    return np.asarray(params["N"], dtype=int)


@pytest.fixture(scope="session")
def a_rat(params) -> np.ndarray:
    """Hop ratios a_num/a_den, shape (M,), dtype float64."""
    return np.asarray(params["a_rat"], dtype=float)


@pytest.fixture(scope="session")
def fc_n(params) -> np.ndarray:
    """Normalised center frequencies, shape (M,), dtype float64."""
    return np.asarray(params["fc_n"], dtype=float)


@pytest.fixture(scope="session")
def tfr(params) -> np.ndarray:
    """Time-frequency ratios at length L, shape (M,), dtype float64."""
    return np.asarray(params["tfr"], dtype=float)


@pytest.fixture(scope="session")
def c_channels(coeff_ref, N) -> list[np.ndarray]:
    """Analysis coefficients as list of M complex arrays."""
    return split_channels(coeff_ref["c_flat"], N)


@pytest.fixture(scope="session")
def tgrad_channels(phasegrad_ref, N) -> list[np.ndarray]:
    """Instantaneous frequency as list of M real arrays."""
    return split_channels(phasegrad_ref["tgrad_flat"], N)


@pytest.fixture(scope="session")
def fgrad_channels(phasegrad_ref, N) -> list[np.ndarray]:
    """Group delay as list of M real arrays."""
    return split_channels(phasegrad_ref["fgrad_flat"], N)


@pytest.fixture(scope="session")
def cs_channels(phasegrad_ref, N) -> list[np.ndarray]:
    """Spectrogram |c|^2 as list of M real arrays."""
    return split_channels(phasegrad_ref["cs_flat"], N)


# ---------------------------------------------------------------------------
# Synthetic filterbank  (no reference data required)
# Used by hypothesis property tests that need *some* valid filterbank.
# ---------------------------------------------------------------------------

# ===========================================================================
# Layer 0 – numpy reference implementations (module-level, not fixtures)
#
# These mirror the LTFAT MATLAB primitives using only NumPy.  Tests that use
# them run *unconditionally* (no cool_frames package required).  When
# the package is available, @pytest.mark.requires_impl tests compare the
# package output to these references.
# ===========================================================================

def pderiv_ref(f: np.ndarray, difforder: float = 4) -> np.ndarray:
    """Numpy reference for LTFAT ``pderiv`` (periodic derivative on [0,1))."""
    f = np.asarray(f)
    L = len(f)
    if difforder == 2:
        return L * (np.roll(f, -1) - np.roll(f, 1)) / 2
    elif difforder == 4:
        return L * (
            -np.roll(f, -2) + 8 * np.roll(f, -1)
            - 8 * np.roll(f, 1) + np.roll(f, 2)
        ) / 12
    else:  # Inf – spectral derivative
        n  = fftindex_ref(L, nyquistzero=True).astype(float)
        fd = 2 * np.pi * np.fft.ifft(1j * n * np.fft.fft(f))
        return np.real(fd) if np.isrealobj(f) else fd


def psech_ref(L: int, tfr: float = 1.0) -> np.ndarray:
    """Numpy reference for LTFAT ``psech`` (periodised hyperbolic-secant, unit L2-norm)."""
    safe  = 12
    sqrtl = np.sqrt(L)
    nk    = int(np.ceil(safe / np.sqrt(L / np.sqrt(tfr))))
    lr    = np.arange(L, dtype=float)
    g     = np.zeros(L)
    for k in range(-nk, nk + 1):
        g += 1.0 / np.cosh(np.pi * (lr / sqrtl - k * sqrtl) / np.sqrt(tfr))
    g *= np.sqrt(np.pi / (2 * np.sqrt(L * tfr)))
    return g


def involute_ref(x: np.ndarray) -> np.ndarray:
    """Numpy reference for LTFAT ``involute``: involute(f)(n)=conj(f(mod(-n,L)))."""
    x = np.asarray(x)
    return np.concatenate([[np.conj(x[0])], np.conj(x[1:][::-1])])


def modcent_ref(x: np.ndarray, r: float) -> np.ndarray:
    """Numpy reference for LTFAT ``modcent``: centred modulo in [-r/2, r/2)."""
    out = np.mod(np.asarray(x, dtype=float), float(r))
    out[out > r / 2] -= r
    return out


def fftindex_ref(N: int, nyquistzero: bool = False) -> np.ndarray:
    """
    Numpy reference for LTFAT ``fftindex``.

    Returns frequency indices in [-ceil(N/2)+1, floor(N/2)].
    For even N, the Nyquist bin (+N/2) sits at position N/2 in the output;
    set nyquistzero=True to zero it out (matching ``fftindex(N,0)``).
    """
    if N % 2 == 0:
        n = np.concatenate([np.arange(N // 2 + 1), np.arange(-N // 2 + 1, 0)])
        if nyquistzero:
            n = n.copy()
            n[N // 2] = 0
    else:
        n = np.concatenate([np.arange((N + 1) // 2), np.arange(-(N - 1) // 2, 0)])
    return n


def floor23_ref(n: int) -> int:
    """Numpy reference for LTFAT ``floor23``: largest 2-3 smooth number ≤ n."""
    if n <= 0:
        return 1
    best, p2 = 1, 1
    while p2 <= n:
        p3 = p2
        while p3 <= n:
            best = max(best, p3)
            p3 *= 3
        p2 *= 2
    return best


def comp_downs_ref(x: np.ndarray, a: int, skip: int = 0) -> np.ndarray:
    """Numpy reference for LTFAT ``comp_downs``: stride-based downsampling."""
    return x[skip::a]


def comp_ups_ref(x: np.ndarray, a: int) -> np.ndarray:
    """Numpy reference for LTFAT ``comp_ups``: zero-insertion upsampling."""
    out = np.zeros(len(x) * a, dtype=x.dtype)
    out[::a] = x
    return out


def postpad_ref(x: np.ndarray, L: int, val: float = 0.0) -> np.ndarray:
    """Numpy reference for LTFAT ``postpad``: pads or truncates x to length L."""
    x  = np.asarray(x)
    Lx = len(x)
    if Lx >= L:
        return x[:L].copy()
    return np.concatenate([x, np.full(L - Lx, val, dtype=x.dtype)])


def comp_extBoundary_ref(f: np.ndarray, extLen: int, mode: str) -> np.ndarray:
    """
    Numpy reference for LTFAT ``comp_extBoundary``.

    Assumes extLen ≤ L (sufficient for all unit / property tests).
    Modes: 'per', 'zpd', 'sym', 'symw', 'asym', 'asymw', 'sp0'.
    """
    f     = np.asarray(f)
    L     = len(f)
    fout  = np.zeros(L + 2 * extLen, dtype=f.dtype)
    fout[extLen:extLen + L] = f
    if extLen == 0:
        return fout
    legal = min(L, extLen)
    if mode in ('per', 'ppd'):
        fout[:extLen]     = f[L - extLen:]
        fout[L + extLen:] = f[:extLen]
    elif mode in ('zpd', 'zero'):
        pass
    elif mode in ('sym', 'even'):
        fout[:legal]                         = f[:legal][::-1]
        fout[L + extLen:L + extLen + legal]  = f[-legal:][::-1]
    elif mode == 'symw':
        lw = min(L - 1, extLen)
        fout[extLen - lw:extLen]           = f[1:lw + 1][::-1]
        fout[L + extLen:L + extLen + lw]   = f[L - lw - 1:L - 1][::-1]
    elif mode in ('asym', 'odd'):
        fout[:legal]                         = -f[:legal][::-1]
        fout[L + extLen:L + extLen + legal]  = -f[-legal:][::-1]
    elif mode == 'asymw':
        lw = min(L - 1, extLen)
        fout[extLen - lw:extLen]           = -f[1:lw + 1][::-1]
        fout[L + extLen:L + extLen + lw]   = -f[L - lw - 1:L - 1][::-1]
    elif mode == 'sp0':
        fout[:extLen]     = f[0]
        fout[L + extLen:] = f[-1]
    else:
        raise ValueError(f"comp_extBoundary_ref: unsupported mode '{mode}'")
    return fout


def _hann_fir(Lh: int) -> np.ndarray:
    """
    Hann FIR window of length Lh, unit L2-norm.

    Matches LTFAT  firwin('hann', Lh): period-L Hann window.
    """
    n = np.arange(Lh, dtype=float)
    w = 0.5 * (1.0 - np.cos(2 * np.pi * n / Lh))
    return w / np.linalg.norm(w)


# ===========================================================================
# Layer 0 – filterbank setup fixtures (no impl required)
# ===========================================================================

@pytest.fixture(scope="session")
def fft_fb() -> dict:
    """
    Rectangular full-length FFT filterbank matching  TestFFTFull.m.

    Keys: Ls, M, a, G, noise_real, noise_stereo, zeros_sig.
    G is a list of M (Ls,) complex DFT responses.
    """
    Ls  = 1024
    M   = 4
    a   = np.array([4, 8, 8, 16], dtype=int)
    bw  = Ls // (2 * M + 2)
    G   = []
    for m in range(M):
        g_full = np.zeros(Ls, dtype=complex)
        fc_bin = round((m + 1) * Ls / (2 * (M + 1)))
        lo, hi = max(0, fc_bin - bw // 2), min(Ls, fc_bin + bw // 2)
        g_full[lo:hi] = 1.0
        G.append(g_full)
    rng = np.random.default_rng(0)
    return {
        "Ls": Ls, "M": M, "a": a, "G": G,
        "noise_real":   rng.standard_normal(Ls),
        "noise_stereo": rng.standard_normal((Ls, 2)),
        "zeros_sig":    np.zeros(Ls),
    }


@pytest.fixture(scope="session")
def fftbl_fb() -> dict:
    """
    Bandlimited rectangular FFT filterbank matching  TestFFTBandlimited.m.

    Keys: Ls, M, bw, G_bl, foff, realonly, a, noise_real, noise_complex, zeros_sig.
    """
    Ls      = 1024
    M       = 4
    bw      = 8
    step    = Ls // (2 * M)
    G_bl    = [np.ones(bw, dtype=complex) for _ in range(M)]
    foff    = np.array([m * step for m in range(M)], dtype=int)
    realonly = np.zeros(M, dtype=int)
    a        = np.ones(M, dtype=int)
    rng = np.random.default_rng(1)
    return {
        "Ls": Ls, "M": M, "bw": bw,
        "G_bl": G_bl, "foff": foff, "realonly": realonly, "a": a,
        "noise_real":    rng.standard_normal(Ls),
        "noise_complex": (rng.standard_normal(Ls)
                          + 1j * rng.standard_normal(Ls)),
        "zeros_sig":     np.zeros(Ls),
    }


@pytest.fixture(scope="session")
def td_fb() -> dict:
    """
    FIR time-domain filterbank matching  TestTDPath.m.

    Keys: Ls, M, Lh, a, G_td, offset, noise_real, zeros_sig, impulse.
    G_td is a list of M unit-norm Hann FIR windows of length Lh.
    """
    Ls, M, Lh = 256, 3, 16
    a      = np.array([2, 4, 4], dtype=int)
    w      = _hann_fir(Lh)
    G_td   = [w.copy() for _ in range(M)]
    offset = np.zeros(M, dtype=int)
    rng    = np.random.default_rng(2)
    imp    = np.zeros(Ls); imp[0] = 1.0
    return {
        "Ls": Ls, "M": M, "Lh": Lh, "a": a,
        "G_td": G_td, "offset": offset,
        "noise_real": rng.standard_normal(Ls),
        "zeros_sig":  np.zeros(Ls),
        "impulse":    imp,
    }


@pytest.fixture(scope="session")
def polyphase_fb() -> dict:
    """
    FIR + DFT pair for  PropPolyphaseEquivalence.m.

    G_td and G_fft represent the SAME Hann FIR filters in time- and
    frequency-domain form.  Keys: Ls, M, Lh, a, G_td, G_fft, offset, noise_real.
    """
    Ls, M, Lh = 512, 3, 16
    a      = np.array([4, 8, 8], dtype=int)
    w      = _hann_fir(Lh)
    G_td   = [w.copy() for _ in range(M)]
    G_fft  = [np.fft.fft(postpad_ref(w, Ls)) for _ in range(M)]
    offset = np.zeros(M, dtype=int)
    rng    = np.random.default_rng(3)
    return {
        "Ls": Ls, "M": M, "Lh": Lh, "a": a,
        "G_td": G_td, "G_fft": G_fft, "offset": offset,
        "noise_real": rng.standard_normal(Ls),
    }


# ===========================================================================
# Layer 1 – numpy reference implementations
#
# firwin_ref: implements the WPE (whole-point even) LTFAT window convention
#   w[n] = window_fn(n/M)  for n = 0, 1, ..., M-1
# Peak is at index 0.  The LTFAT fftshift property w + fftshift(w) = 1 holds
# for hann and tria (PU); sine satisfies the tight-frame condition w^2 + fftshift(w^2) = 1.
# ===========================================================================

def firwin_ref(name: str, M: int) -> np.ndarray:
    """
    Numpy reference for LTFAT ``firwin``.

    Parameters
    ----------
    name : {'hann', 'sine', 'rect', 'tria'}
    M    : window length (positive integer)

    Returns
    -------
    w : np.ndarray, shape (M,)
        Zero-phase WPE window, peak at index 0.
    """
    n = np.arange(M, dtype=float)
    if name == "hann":
        return 0.5 * (1.0 + np.cos(2.0 * np.pi * n / M))
    elif name == "sine":
        return np.sqrt(np.maximum(0.0, 0.5 * (1.0 + np.cos(2.0 * np.pi * n / M))))
    elif name == "rect":
        w = np.ones(M)
        if M % 2 == 0:
            w[M // 2] = 0.0  # Nyquist bin x = -0.5 → 0
        return w
    elif name == "tria":
        return 1.0 - 2.0 * np.minimum(n, M - n) / M
    else:
        raise ValueError(f"firwin_ref: unknown window name '{name}'")


def synthetic_filterbank() -> dict:
    """
    A minimal valid filterbank for structural property tests.

    Uses M non-overlapping band-pass box filters at hop a=1, system length L.
    No reference data needed – can be used unconditionally.

    Returns a dict with keys:
        g    : list of M dicts, each with key 'H' (L-point DFT, complex128)
        a    : (M, 2) int array  [[1,1], [1,1], ...]
        L    : int
        M    : int
        fc_n : (M,) float64  centre frequencies normalised to [0, 1)
        tfr  : (M,) float64  time-frequency ratios (all equal to bin_width)
    """
    L = 256
    M = 8
    bin_width = L // M
    g = []
    fc_n = np.zeros(M)
    for m in range(M):
        H = np.zeros(L, dtype=complex)
        lo = m * bin_width
        hi = (m + 1) * bin_width
        H[lo:hi] = 1.0
        g.append({"H": H, "foff": 0})
        fc_n[m] = (lo + hi) / (2.0 * L)  # normalised to [0, 1)
    a = np.ones((M, 2), dtype=int)
    tfr = np.full(M, float(bin_width) / L)
    return {"g": g, "a": a, "L": L, "M": M, "fc_n": fc_n, "tfr": tfr}
