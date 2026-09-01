"""The closed-form admissibility predictor must agree with the measured frame
response for every designer, every scale.

The predictor is exact by construction: ``A = 0`` iff the filters leave a DFT
bin uncovered, and coverage is integer arithmetic on the design parameters.
These tests pin that agreement down so a change to any designer's geometry
that breaks it fails the build.
"""
import itertools
import math
import warnings

import pytest

import numpy as np
from cool_frames.diagnostics.admissibility import (
    max_overlap_for_kappa,
    predict_admissible,
    ripple_curve,
)
from cool_frames.filterbanks import filterbankresponse
from cool_frames.filters import (
    audfilters,
    cqtfilters,
    gabfilters,
    greenwoodfilters,
    warpedfilters,
    waveletfilters,
)
from cool_frames.numpy.filters._audscale import audfiltbw, audspace, audtofreq, freqtoaud
from cool_frames.numpy.filters._design import _scale_default
from cool_frames.numpy.filters._firwin import hann_winbw

SCALES = ["erb", "erb83", "bark", "mel", "mel1000", "greenwood",
          "linear", "log", "semitone", "third-octave"]


def measured_is_frame(g, a, L):
    R = np.real(np.asarray(filterbankresponse(g, a, L, real=True)))
    return bool(R.min() > 1e-12 * max(float(R.max()), 1e-300))


def aud_geometry(fs, Ls, scale, M):
    """Centres and supports of an ``audfilters`` bank, from parameters only."""
    spacing = _scale_default(scale, "spacing", None)
    bwmul = _scale_default(scale, "bwmul", None)
    fmin = float(audtofreq(spacing, scale))
    fmax = fs / 2.0
    u_min, u_max = float(freqtoaud(fmin, scale)), float(freqtoaud(fmax, scale))
    spacing = (u_max - u_min) / max(M - 1, 1)
    n = int(math.floor((u_max - u_min) / spacing)) + 1
    _fmax = float(audtofreq(u_min + (n - 1) * spacing, scale))
    while _fmax >= fs / 2.0 and n > 1:
        n -= 1
        _fmax = float(audtofreq(u_min + (n - 1) * spacing, scale))
    fc = np.asarray(audspace(fmin, _fmax, n, scale), dtype=float)
    fsupp = np.maximum(np.asarray(audfiltbw(fc, scale), float) / hann_winbw()
                       * bwmul, 4 / Ls * fs)
    return fc, fsupp


@pytest.mark.parametrize("scale", SCALES)
def test_predictor_matches_response_audfilters(scale):
    warnings.simplefilter("ignore")
    for fs in (8000, 16000, 44100):
        for Ls in (512, 1024):
            for M in range(4, 30):
                try:
                    g, a, _fc, L, _ = audfilters(fs, Ls, scale=scale, M=M)
                    fcp, fsp = aud_geometry(fs, Ls, scale, M)
                except Exception:
                    continue
                p = predict_admissible(
                    fcp, fsp, fs=fs, L=L,
                    fsupp_dc=2.0 * float(fcp[0]),
                    fsupp_nyq=2.0 * (fs / 2.0 - float(fcp[-1])))
                assert p["is_frame"] == measured_is_frame(g, a, L), (
                    f"{scale} fs={fs} Ls={Ls} M={M}: predicted "
                    f"{p['is_frame']}, measured {not p['is_frame']}")


def test_predictor_matches_response_greenwoodfilters():
    warnings.simplefilter("ignore")
    for fs in (8000, 16000):
        for Ls in (512, 1024):
            for M in range(4, 30):
                try:
                    g, a, _fc, L, _ = greenwoodfilters(fs, Ls, M=M)
                    fcp, fsp = aud_geometry(fs, Ls, "greenwood", M)
                except Exception:
                    continue
                p = predict_admissible(
                    fcp, fsp, fs=fs, L=L, fsupp_dc=2.0 * float(fcp[0]),
                    fsupp_nyq=2.0 * (fs / 2.0 - float(fcp[-1])))
                assert p["is_frame"] == measured_is_frame(g, a, L)


def test_predictor_matches_response_cqtfilters():
    warnings.simplefilter("ignore")
    for fs in (8000, 16000):
        for Ls in (512, 1024, 2048):
            for bins in (1, 2, 3, 4, 6, 12, 24):
                for fmin in (20.0, 50.0, 200.0):
                    for Qvar in (0.5, 1.0, 1.5):
                        try:
                            g, a, _fc, L, _ = cqtfilters(fs, Ls, fmin=fmin,
                                                        bins=bins, Qvar=Qvar)
                        except Exception:
                            continue
                        t = 2.0 ** (1.0 / bins)
                        n = int(math.floor(math.log(fs / 2.0 / fmin, t))) + 1
                        fcp = fmin * t ** np.arange(n, dtype=float)
                        fcp = fcp[fcp < fs / 2.0]
                        if fcp.size < 2:
                            continue
                        fsp = np.empty_like(fcp)
                        fsp[0] = fcp[0] * (t - 1 / t)
                        fsp[-1] = fcp[-1] * (t - 1 / t)
                        if fcp.size > 2:
                            fsp[1:-1] = fcp[2:] - fcp[:-2]
                        fsp = np.maximum(fsp * Qvar, 4 / Ls * fs)
                        p = predict_admissible(
                            fcp, fsp, fs=fs, L=L, fsupp_dc=2.0 * fmin,
                            fsupp_nyq=2.0 * (fs / 2.0 - float(fcp[-1])))
                        assert p["is_frame"] == measured_is_frame(g, a, L)


def test_predictor_matches_response_warpedfilters():
    warnings.simplefilter("ignore")

    def f2s(f):
        return np.log2(np.maximum(np.asarray(f, float), 1.0))

    def s2f(u):
        return 2.0 ** np.asarray(u, float)

    for fs in (8000, 16000):
        for Ls in (1024, 2048):
            for bins in (1, 2, 3, 4, 6, 12):
                for bwmul in (0.25, 0.5, 0.75, 1.0):
                    try:
                        out = warpedfilters(f2s, s2f, fs, 100.0, fs / 2.5,
                                            bins, Ls, bwmul=bwmul)
                        g, a, L = out[0], out[1], out[3]
                    except Exception:
                        continue
                    chan_min = math.floor(bins * f2s(100.0)) / bins
                    cmax = chan_min
                    while s2f(cmax) <= fs / 2.5:
                        cmax += 1.0 / bins
                    while s2f(cmax + bwmul) >= fs / 2.0:
                        cmax -= 1.0 / bins
                    u = np.arange(chan_min, cmax + 0.5 / bins, 1.0 / bins)
                    if u.size < 2:
                        continue
                    dc = math.ceil(2 * float(s2f(u[0] - 1.0 / bins + bwmul))) + 2
                    nyq = math.ceil(2 * (fs / 2.0
                                         - float(s2f(u[-1] + 1.0 / bins - bwmul)))) + 2
                    p = predict_admissible(
                        None, None, fs=fs, L=L, fsupp_dc=dc, fsupp_nyq=nyq,
                        min_win=1, warped=(u, s2f, bwmul))
                    assert p["is_frame"] == measured_is_frame(g, a, L)


def test_ripple_curve_partition_of_unity_points():
    # Hann-squared tiles exactly at overlap ratios 1/4 and 1/3, and gives
    # kappa = 2 at 1/2.
    assert ripple_curve(0.25) == pytest.approx(1.0, abs=1e-6)
    assert ripple_curve(1.0 / 3.0) == pytest.approx(1.0, abs=1e-6)
    assert ripple_curve(0.5) == pytest.approx(2.0, rel=1e-3)
    assert math.isinf(ripple_curve(1.0))


def test_ripple_curve_is_monotone_above_a_third():
    rs = np.linspace(0.35, 0.95, 25)
    ks = [ripple_curve(r) for r in rs]
    assert all(b >= a for a, b in itertools.pairwise(ks))


def test_max_overlap_inverts_the_ripple_curve():
    for target in (2.0, 5.0, 100.0):
        rho = max_overlap_for_kappa(target)
        assert ripple_curve(rho) <= target * (1 + 1e-6)
        assert ripple_curve(min(rho + 1e-3, 0.999)) >= target * (1 - 1e-6)


# ---------------------------------------------------------------------------
# construction-time check: every designer reports its own admissibility
# ---------------------------------------------------------------------------

DESIGNER_CASES = [
    ("audfilters", lambda: audfilters(16000, 1024, M=8), False),
    ("audfilters", lambda: audfilters(16000, 1024, M=25), True),
    ("cqtfilters", lambda: cqtfilters(16000, 1024, fmin=50.0, bins=12, Qvar=1.0), True),
    ("greenwoodfilters", lambda: greenwoodfilters(16000, 1024, M=20), True),
]


@pytest.mark.parametrize("designer,call,expected", DESIGNER_CASES)
def test_designer_reports_admissibility(designer, call, expected):
    """``info['admissible']`` must agree with the measured frame response."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, _fc, L, info = call()
    assert info["designer"] == designer
    assert info["admissible"]["is_frame"] is expected
    assert info["admissible"]["is_frame"] == measured_is_frame(g, a, L)


@pytest.mark.parametrize("designer,call,expected", DESIGNER_CASES)
def test_designer_reports_geometry(designer, call, expected):
    """The geometry the predictor needs is published in ``info``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _g, _a, fc, _L, info = call()
    assert "fsupp_inner" in info and "fsupp_dc" in info and "fsupp_nyq" in info
    assert len(info["fsupp_inner"]) == len(fc) - 2
    assert info["fsupp_dc"] > 0 and info["fsupp_nyq"] > 0


def test_non_admissible_geometry_warns():
    """A geometry below the floor must say so at construction time."""
    from cool_frames.diagnostics.admissibility import NotAFrameWarning

    with pytest.warns(NotAFrameWarning, match="not a frame"):
        audfilters(16000, 1024, M=8)


def test_admissible_geometry_does_not_warn():
    from cool_frames.diagnostics.admissibility import NotAFrameWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", NotAFrameWarning)
        audfilters(16000, 1024, M=25)


def test_warpedfilters_reports_admissibility():
    def f2s(f):
        return np.log2(np.maximum(np.asarray(f, float), 1.0))

    def s2f(u):
        return 2.0 ** np.asarray(u, float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = warpedfilters(f2s, s2f, 16000, 100.0, 6400.0, 4, 2048, bwmul=0.5)
    g, a, _fc, L, info = out[0], out[1], out[2], out[3], out[-1]
    assert info["designer"] == "warpedfilters"
    assert info["admissible"]["is_frame"] == measured_is_frame(g, a, L)


def test_warpedfilters_below_the_floor_warns():
    """2*bwmul > 1/bins is the warped floor; violate it and it must warn."""
    from cool_frames.diagnostics.admissibility import NotAFrameWarning

    def f2s(f):
        return np.log2(np.maximum(np.asarray(f, float), 1.0))

    def s2f(u):
        return 2.0 ** np.asarray(u, float)

    # bins=1, bwmul=0.25  ->  2*bwmul = 0.5 < 1/bins = 1
    with pytest.warns(NotAFrameWarning):
        warpedfilters(f2s, s2f, 16000, 100.0, 6400.0, 1, 2048, bwmul=0.25)


# ---------------------------------------------------------------------------
# gabfilters: the warping coordinate is frequency itself
# ---------------------------------------------------------------------------

def gab_geometry(fs, Ls, M, a, real=True):
    """Bin geometry of a ``gabfilters`` bank, from the parameters only.

    The prototype is built in the TIME domain: a length-M window zero-padded
    to L, transformed, then truncated to the M bins around its peak.  So every
    channel realises exactly M DFT bins -- whatever the window shape -- at

        A_k = k*(L/M) - M//2,    B_k = A_k + M - 1

    with ``L = dgtlength(Ls, a, M)`` a multiple of M.  There are no dead
    endpoint bins: the transformed window vanishes only at L-bin offsets that
    are multiples of L/M, and those bins carry another channel's peak.  Since
    the channels are uniform *in frequency*, the exact interval is expressed
    through the warped hook with ``scaletofreq`` the identity; the
    quarter-bin inset makes floor()/ceil() land on A and B for both parities
    of M.
    """
    b = math.lcm(int(a), int(M))
    L = int(math.ceil(Ls / b) * b)
    q = L // M
    M2 = M // 2 + 1 if real else M
    nyq_bin = L // 2
    A_top = (M2 - 1) * q - M // 2
    B_top = A_top + M - 1
    if real and B_top >= nyq_bin:
        # the top channel reaches Nyquist and folds to [min(A,L-B), L/2]
        last = M2 - 1
        W_nyq = 2 * (nyq_bin - min(A_top, L - B_top)) + 1
    else:
        # no channel plays the Nyquist complement; a one-bin stub is exact
        last = M2
        W_nyq = 1
    A = np.arange(1, last, dtype=float) * q - M // 2
    u = (A + M / 2.0) * fs / L
    bwmul = (M / 2.0 - 0.25) * fs / L
    return u, bwmul, M * fs / L, W_nyq * fs / L, L


def test_predictor_matches_response_gabfilters():
    warnings.simplefilter("ignore")
    for fs in (8000, 16000):
        for Ls in (512, 1024):
            for M in (4, 8, 9, 16, 17, 24, 25, 32, 33, 64):
                for a in (2, max(1, M // 4), M + 1):
                    if math.lcm(a, M) > 8 * Ls:
                        continue
                    g, aout, _fc, L, _ = gabfilters(fs, Ls, M=M, a=a)
                    u, bwmul, dc, nyq, Lg = gab_geometry(fs, Ls, M, a)
                    assert Lg == L
                    p = predict_admissible(
                        None, None, fs=fs, L=L, fsupp_dc=dc, fsupp_nyq=nyq,
                        min_win=1, window="rect",
                        warped=(u, lambda x: x, bwmul))
                    assert p["is_frame"] == measured_is_frame(g, aout, L), (
                        f"fs={fs} Ls={Ls} M={M} a={a}: predicted "
                        f"{p['is_frame']}, measured {not p['is_frame']}")


def test_predictor_matches_response_gabfilters_two_sided():
    warnings.simplefilter("ignore")
    for M in (8, 9, 16, 17, 32):
        for a in (2, max(1, M // 4), M + 1):
            g, aout, _fc, L, _ = gabfilters(16000, 1024, M=M, a=a, real=False)
            u, bwmul, dc, nyq, _L = gab_geometry(16000, 1024, M, a, real=False)
            p = predict_admissible(None, None, fs=16000, L=L, fsupp_dc=dc,
                                   fsupp_nyq=nyq, min_win=1, window="rect",
                                   warped=(u, lambda x: x, bwmul))
            assert p["is_frame"] == measured_is_frame(g, aout, L)


def test_predictor_matches_response_gabfilters_at_the_covering_boundary():
    """``L/M`` within a couple of bins of ``M`` is where the verdict turns:
    the channels abut at ``L/M == M`` and the parity of M decides whether the
    top one still reaches Nyquist."""
    warnings.simplefilter("ignore")
    for M in range(3, 40):
        for q in range(max(1, M - 2), M + 3):
            Ls = M * (q - 1) + 1            # a = 1  ->  L = dgtlength = M*q
            if Ls < 1:
                continue
            for real in (True, False):
                g, aout, _fc, L, _ = gabfilters(16000, Ls, M=M, a=1, real=real)
                assert L == M * q
                u, bwmul, dc, nyq, _L = gab_geometry(16000, Ls, M, 1, real=real)
                p = predict_admissible(None, None, fs=16000, L=L, fsupp_dc=dc,
                                       fsupp_nyq=nyq, min_win=1, window="rect",
                                       warped=(u, lambda x: x, bwmul))
                assert p["is_frame"] == measured_is_frame(g, aout, L), (
                    f"M={M} q={q} real={real}")


def test_gabfilters_foreign_window_length_reports_no_verdict():
    """A window array whose length is not M puts the transformed window's
    zeros out of step with the channel lattice, so the live support is no
    longer 'all of it': no verdict rather than a wrong one."""
    from cool_frames.numpy.filters._firwin import firwin

    warnings.simplefilter("ignore")
    w = firwin("hann", 12, norm="energy")
    _g, _a, _fc, _L, info = gabfilters(16000, 300, M=25, a=1, window=w)
    assert info["admissible"] is None


def test_gabfilters_reports_admissibility():
    """``info['admissible']`` must agree with the measured frame response."""
    warnings.simplefilter("ignore")
    for M, a, expected in ((256, 64, True), (16, 4, False), (24, 25, False)):
        g, aout, _fc, L, info = gabfilters(16000, 4096, M=M, a=a)
        assert info["designer"] == "gabfilters"
        assert info["admissible"]["is_frame"] is expected
        assert info["admissible"]["is_frame"] == measured_is_frame(g, aout, L)


def test_gabfilters_reports_geometry():
    warnings.simplefilter("ignore")
    _g, _a, fc, L, info = gabfilters(16000, 4096, M=256, a=64)
    assert "fsupp_inner" in info and "fsupp_dc" in info and "fsupp_nyq" in info
    assert len(info["fsupp_inner"]) == len(fc) - 2
    assert info["fsupp_dc"] > 0 and info["fsupp_nyq"] > 0
    # every channel realises the same M-bin support
    assert info["fsupp_dc"] == pytest.approx(256 * 16000 / L)
    assert np.allclose(info["fsupp_inner"], 256 * 16000 / L)


def test_gabfilters_below_the_floor_warns():
    """L/M > M leaves holes between the channels; it must say so."""
    from cool_frames.diagnostics.admissibility import NotAFrameWarning

    # M = 16, Ls = 4096  ->  L = 4096, L/M = 256 > 16
    with pytest.warns(NotAFrameWarning, match="not a frame"):
        gabfilters(16000, 4096, M=16, a=4)


def test_gabfilters_admissible_geometry_does_not_warn():
    from cool_frames.diagnostics.admissibility import NotAFrameWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", NotAFrameWarning)
        gabfilters(16000, 4096, M=256, a=64)      # L/M = 16 <= 256


def test_gabfilters_freq_axis_reports_no_verdict():
    """windowaxis='freq' has window-dependent dead bins: no verdict, not a
    guess."""
    warnings.simplefilter("ignore")
    _g, _a, _fc, _L, info = gabfilters(16000, 1024, M=64, a=16,
                                       windowaxis="freq")
    assert info["admissible"] is None


# ---------------------------------------------------------------------------
# waveletfilters: constant-Q supports, complements at both edges
# ---------------------------------------------------------------------------

def wavelet_geometry(fs, L, scales, wavelet, trunc_at=1e-5, min_win=4):
    """Bin geometry of a ``waveletfilters`` bank, from the parameters only.

    A wavelet of scale s is a dilation of the mother wavelet, so its support
    edges are fixed multiples of its centre frequency (the dilation cancels):
    ``f_lo = r0 * fc``, ``f_hi = r4 * fc`` with ``r0 = fsupp_[0]/peakpos`` and
    ``r4 = fsupp_[4]/peakpos``.  freqwavelet then occupies bins
    ``ceil(L/fs*f_lo) .. floor(L/fs*f_hi) - 1``.  Endpoint bins are alive, so
    the predictor is called with ``window='rect'``; an even-width channel is
    split into two odd intervals because ``_interval_linear`` can only build
    an odd one.
    """
    from cool_frames.numpy.filters._edge_filters import edge_params_from_geometry
    from cool_frames.numpy.filters._wavelet import wavelet_generator_func

    _fun, fsupp_, peakpos, _ca = wavelet_generator_func(
        wavelet, negative=False, efsuppthr=trunc_at, bwthr=10 ** (-3 / 10))
    basefc = 0.1
    basedil = peakpos / basefc
    scales = np.atleast_1d(np.asarray(scales, dtype=float)).ravel()
    scales_sorted = np.sort(scales)[::-1]
    flo = np.maximum(0.0, (fsupp_[0] / basedil) / scales)
    fhi = np.minimum(2.0, (fsupp_[4] / basedil) / scales)   # fs = 2 convention
    A = np.ceil(flo / 2.0 * L).astype(int)
    B = np.floor(fhi / 2.0 * L).astype(int) - 1

    fc_pred, fsupp_pred = [], []
    for a_m, b_m in zip(A.tolist(), B.tolist()):
        W = b_m - a_m + 1
        if W <= 0:
            continue
        if W % 2 == 1:
            pairs = [((a_m + W // 2) * fs / L, W * fs / L)]
        elif W == 2:
            pairs = [(a_m * fs / L, fs / L), ((a_m + 1) * fs / L, fs / L)]
        else:
            Wp = W - 1
            pairs = [((a_m + Wp // 2) * fs / L, Wp * fs / L),
                     ((a_m + 1 + Wp // 2) * fs / L, Wp * fs / L)]
        for f, s in pairs:
            fc_pred.append(f)
            fsupp_pred.append(s)

    # DC complement: lp_bw = 0.2 / scales_sorted[3] (fs = 2 convention)
    lp_bw = 0.2 / scales_sorted[3]
    taper_dc = 1 - scales_sorted[3] / scales_sorted[1]
    W_dc = max(min_win, round(L * lp_bw))
    W_dc = W_dc if W_dc % 2 else W_dc + 1
    if taper_dc <= 0:
        W_dc -= 2
    # Nyquist complement: 2*(1 - fc_last) wide in the fs = 2 convention
    fc_norm = (fsupp_[2] / basedil) / scales
    jmax = int(np.argmax(fc_norm))
    fsupp_hp, taper_hp = edge_params_from_geometry(
        float(fc_norm[jmax]), float(B[jmax] - A[jmax] + 2), 2.0,
        target="nyquist")
    W_hp = max(min_win, round(L * fsupp_hp / 2.0))
    W_hp = W_hp if W_hp % 2 else W_hp + 1
    if taper_hp <= 0:
        W_hp -= 2
    return (np.asarray(fc_pred, float), np.asarray(fsupp_pred, float),
            W_dc * fs / L, W_hp * fs / L)


def wavelet_default_scales(fs, fmin, fmax, bins):
    """The ``scales=None`` grid of ``waveletfilters``."""
    if fmax is None:
        fmax = fs / 2.0
    n = max(1, int(round(bins * np.log2(fmax / fmin))))
    fc_grid = fmin * (fmax / fmin) ** (np.arange(n + 1) / n)
    return np.maximum(0.05 * fs / fc_grid, 0.1)


def test_predictor_matches_response_waveletfilters():
    warnings.simplefilter("ignore")
    for fs in (8000, 16000):
        for Ls in (512, 1024):
            for bins in (1, 2, 4, 8):
                for fmin in (50.0, 200.0):
                    for wav in (["cauchy", 300], ["morlet", 4]):
                        for trunc_at in (1e-5, 1e-2):
                            try:
                                g, a, _fc, L, _ = waveletfilters(
                                    fs, Ls, fmin=fmin, bins=bins, wavelet=wav,
                                    trunc_at=trunc_at)
                            except Exception:
                                continue
                            sc = wavelet_default_scales(fs, fmin, None, bins)
                            fcp, fsp, dc, nyq = wavelet_geometry(
                                fs, L, sc, wav, trunc_at=trunc_at)
                            p = predict_admissible(
                                fcp, fsp, fs=fs, L=L, fsupp_dc=dc,
                                fsupp_nyq=nyq, min_win=1, window="rect")
                            assert p["is_frame"] == measured_is_frame(g, a, L), (
                                f"fs={fs} Ls={Ls} bins={bins} fmin={fmin} "
                                f"{wav} trunc_at={trunc_at}: predicted "
                                f"{p['is_frame']}, measured {not p['is_frame']}")


def test_waveletfilters_reports_admissibility():
    warnings.simplefilter("ignore")
    cases = [(dict(fmin=50.0, bins=12), True),
             (dict(fmin=50.0, bins=1), True),
             (dict(fmin=50.0, bins=2, wavelet=["cauchy", 600],
                   trunc_at=1e-2), False)]
    for kwargs, expected in cases:
        g, a, _fc, L, info = waveletfilters(16000, 1024, **kwargs)
        assert info["designer"] == "waveletfilters"
        assert info["admissible"]["is_frame"] is expected
        assert info["admissible"]["is_frame"] == measured_is_frame(g, a, L)


def test_waveletfilters_reports_geometry():
    warnings.simplefilter("ignore")
    _g, _a, fc, _L, info = waveletfilters(16000, 1024, fmin=50.0, bins=12)
    assert "fsupp_inner" in info and "fsupp_dc" in info and "fsupp_nyq" in info
    assert len(info["fsupp_inner"]) == len(fc) - 2
    assert info["fsupp_dc"] > 0 and info["fsupp_nyq"] > 0


def test_waveletfilters_below_the_floor_warns():
    """Sparse voices plus a hard truncation leave gaps between the wavelets."""
    from cool_frames.diagnostics.admissibility import NotAFrameWarning

    with pytest.warns(NotAFrameWarning, match="not a frame"):
        waveletfilters(16000, 1024, fmin=50.0, bins=2,
                       wavelet=["cauchy", 600], trunc_at=1e-2)


def test_waveletfilters_admissible_geometry_does_not_warn():
    from cool_frames.diagnostics.admissibility import NotAFrameWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", NotAFrameWarning)
        waveletfilters(16000, 1024, fmin=50.0, bins=12)


def test_waveletfilters_unrepresentable_layout_falls_back_to_measurement():
    """Layouts without one complement at each edge fall back to measuring the
    realised response.

    The closed-form predictor always assumes a DC and a Nyquist filter, so it
    cannot express these layouts.  It used to hand back ``None``, which reads
    as "fine" -- ``lowpass='none'`` leaves 79 DFT bins uncovered at the
    defaults and reported ``None``.  The fallback answers the same question
    (is any bin covered by nothing) exactly, from the realised bank; it just
    cannot say which parameter to change, hence ``source='measured'``.
    """
    warnings.simplefilter("ignore")
    for kwargs in (dict(lowpass="none"), dict(lowpass="repeat"),
                   dict(highpass="none"), dict(freqrange="complex"),
                   dict(freqrange="analytic")):
        _g, _a, _fc, _L, info = waveletfilters(16000, 1024, fmin=50.0, bins=6,
                                               **kwargs)
        adm = info["admissible"]
        assert adm is not None, kwargs
        assert adm["source"] == "measured", kwargs
        assert isinstance(adm["is_frame"], bool)
        assert (adm["n_hole_bins"] == 0) == adm["is_frame"]


def test_waveletfilters_lowpass_none_is_reported_as_not_a_frame():
    """The concrete case the fallback exists for: no DC filter, so bin 0 is
    covered by nothing and the lower frame bound is 0."""
    warnings.simplefilter("ignore")
    _g, _a, _fc, _L, info = waveletfilters(
        8000, 4096, scales=4 * 2.0 ** (-np.arange(64) / 12), lowpass="none")
    adm = info["admissible"]
    assert adm["is_frame"] is False
    assert adm["first_hole_bin"] == 0
    assert adm["n_hole_bins"] > 0


# ---------------------------------------------------------------------------
# The diagonal estimator vs the exact oracle
# ---------------------------------------------------------------------------
#
# `filterbankbounds` is closed-form and O(L*M), but it is the diagonal of the
# frame operator, which equals the operator only under the painless condition.
# Off the painless case it does not fail loudly -- it reports a plausible,
# healthy answer.  On the pre-v0.1.1 default `waveletfilters` bank it gave
# A = 1.658, kappa = 2.28 while `filterbankbounds_svd`, the exact eigenvalue
# oracle, gave A = 0.
#
# No test compared the two.  These do, at an L small enough for the O(L^2)
# oracle to be cheap.

@pytest.mark.parametrize("designer,kwargs", [
    ("audfilters", {}),
    ("cqtfilters", dict(fmin=100.0, fmax=3500.0, bins=6)),
    ("greenwoodfilters", {}),
    ("waveletfilters", dict(scales=4 * 2.0 ** (-np.arange(24) / 6))),
])
def test_diagonal_estimator_agrees_with_exact_oracle(designer, kwargs):
    """Every designer's default bank must be painless, so that the cheap
    estimator and the exact oracle report the same condition number."""
    from cool_frames.filterbanks import filterbankbounds, filterbankbounds_svd

    build = {"audfilters": audfilters, "cqtfilters": cqtfilters,
             "greenwoodfilters": greenwoodfilters,
             "waveletfilters": waveletfilters}[designer]
    warnings.simplefilter("ignore")
    g, a, _fc, L, _info = build(8000, 1024, **kwargs)

    A_est, B_est = filterbankbounds(g, a, L, real=True)
    A_ex, B_ex = filterbankbounds_svd(g, a, L, real=True)

    assert A_ex > 0, f"{designer}: exact lower frame bound is 0"
    # The two differ by a known factor of 2 on a folded real bank; the
    # condition number is what has to agree, and it agrees exactly.
    assert abs((B_est / A_est) - (B_ex / A_ex)) < 1e-6 * (B_ex / A_ex), (
        f"{designer}: estimator kappa {B_est / A_est:.4f} vs "
        f"oracle kappa {B_ex / A_ex:.4f}")


def test_non_painless_bank_is_where_the_estimator_lies():
    """The negative control for the test above.

    Kept as a test rather than a comment because it is the whole reason the
    comparison exists: on a non-painless bank the estimator is not merely
    imprecise, it is confidently wrong in the safe direction, and nothing in
    its output says so.
    """
    from cool_frames.filterbanks import filterbankbounds, filterbankbounds_svd

    warnings.simplefilter("ignore")
    g, a, _fc, L, info = waveletfilters(
        8000, 1024, scales=4 * 2.0 ** (-np.arange(24) / 6), painless=False)
    assert info["painless"] is False

    A_est, B_est = filterbankbounds(g, a, L, real=True)
    A_ex, _B_ex = filterbankbounds_svd(g, a, L, real=True)
    assert A_est > 0, "estimator should report a healthy-looking bank"
    assert A_ex == 0.0, "oracle should report it is not a frame"
