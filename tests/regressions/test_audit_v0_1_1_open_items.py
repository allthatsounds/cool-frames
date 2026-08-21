"""
test_audit_v0_1_1_open_items.py
===============================
Regression tests for the items the v0.1.1 audit recorded as **open** — verified
and measured, but not fixed in the first pass — and which have since been
closed.

The register lists them as carried-over items 1-12.  They divide into four
kinds, and the kind determines what a useful test looks like:

* **Parameters that do nothing.**  ``warpedfilters(min_win=...)`` was accepted,
  documented, and then hardcoded away at the point of use.  A test that merely
  calls the function passes whether or not the parameter is wired up, so these
  are tested by *varying* the parameter and asserting the output changes.

* **Silent wrong answers.**  The negative-frequency channels of a
  ``freqrange='complex'`` bank were built by evaluating a warp outside its
  domain.  Nothing raised; the filters were simply in the wrong place.  Tested
  against the invariant that makes them checkable at all — for a warp that is
  symmetric about DC, channel -f must be the mirror of channel +f.

* **Silent truncation.**  ``ifilterbank`` returning fewer samples than asked
  for.  Tested by asking for more and checking what comes back.

* **Interfaces that lie.**  An exported function that always raises; a wrapper
  missing the argument every one of its siblings takes; hardcoded output
  dtypes.  Tested by using them the way the rest of the family is used.
"""

from __future__ import annotations

import warnings

import pytest

import numpy as np

FS = 8000
LS = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_warp():
    """A log warp, which diverges at DC and so is symmetric about it.

    This is the case ``warpedfilters``' ``symmetry`` flag exists to detect, and
    the only one under which the negative-frequency channels are well defined.
    Deliberately *not* clamped (``np.log(max(f, tiny))``): a clamped log gives
    ``freqtoscale(0) = -27.6``, which is finite, so the flag comes out False and
    the branch under test is never entered.

    ``log(0) = -inf`` is therefore the point rather than an accident, so the
    divide-by-zero it reports is suppressed here rather than avoided.  The
    ``-inf`` then propagates into the window evaluation and becomes ``nan``,
    which ``comp_warpedfreqresponse`` zeroes out by design — hence
    ``invalid="ignore"`` in :func:`_build_warped` as well.  ``np.errstate`` is
    the right mechanism for these: they are NumPy floating-point conditions,
    and ``warnings.catch_warnings`` does not reliably suppress them under
    pytest's own warning capture.
    """

    def freqtoscale(f):
        with np.errstate(divide="ignore"):
            return np.log(np.asarray(f, dtype=float))

    def scaletofreq(s):
        return np.exp(np.asarray(s, dtype=float))

    return freqtoscale, scaletofreq


def _dense(gm, L):
    """Expand a filter descriptor to its full-length transfer function.

    ``H`` is a lazy closure, so evaluating it here re-enters the warp — outside
    whatever context :func:`_build_warped` was called in.  The ``-inf`` from
    ``log(0)`` reaches the window evaluation and becomes ``nan``, which
    ``comp_warpedfreqresponse`` zeroes out by design, so the ``invalid value``
    it reports is expected and belongs suppressed at the call, not globally.
    """
    H = gm.get("H")
    if callable(H):
        with np.errstate(divide="ignore", invalid="ignore"):
            H = H(L)
    H = np.asarray(H).ravel()
    foff = gm.get("foff", 0)
    if callable(foff):
        foff = foff(L)
    out = np.zeros(L, dtype=complex)
    if H.size:
        out[(int(foff or 0) + np.arange(H.size)) % L] = H
    return out


def _build_warped(**kw):
    from cool_frames.numpy.filters import warpedfilters

    f2s, s2f = _log_warp()
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")
        return warpedfilters(f2s, s2f, FS, 50.0, FS / 2, 4, LS, **kw)


# ---------------------------------------------------------------------------
# Open item 1: warpedfilters(min_win=...) was inert
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_warpedfilters_min_win_reaches_the_edge_filters():
    """``min_win`` was accepted and documented, then hardcoded to 1.

    Both edge builders were called with a literal ``min_win=1`` rather than the
    caller's value, so ``min_win=1`` and ``min_win=4096`` produced bit-identical
    banks — while the same parameter demonstrably changes ``audfilters``,
    ``cqtfilters`` and ``greenwoodfilters``.

    ``min_win`` sets a floor on the edge filters' support, so the test varies it
    and asserts the support actually grows.  Asserting "the output differs"
    would also pass if the parameter were wired to something unrelated.
    """

    def _support(min_win):
        g, _a, _fc, L, _info = _build_warped(min_win=min_win)
        sizes = []
        for gm in g:
            H = gm.get("H")
            if callable(H):
                # Same lazy-closure re-entry as in _dense; see the note there.
                with np.errstate(divide="ignore", invalid="ignore"):
                    H = H(L)
            sizes.append(np.asarray(H).ravel().size)
        return sizes

    small = _support(1)
    large = _support(LS)

    assert small != large, (
        "min_win=1 and min_win=Ls give identical banks — the parameter is not "
        "reaching the edge filters"
    )
    # It is a *floor* on the support, so nothing may shrink and the edges grow.
    assert all(b >= a for a, b in zip(small, large)), (
        "raising min_win shrank a filter's support; it is a lower bound"
    )
    assert large[0] > small[0], "the DC edge filter did not respond to min_win"


# ---------------------------------------------------------------------------
# Open item 2: freqrange='complex' negative-frequency channels
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_warped_complex_negative_channels_mirror_their_positive_twins():
    """The defining invariant for a bank built on a warp symmetric about DC.

    Three separate defects conspired here, all of them silent:

    1. ``warpedfilters`` computed a ``symmetry`` flag and dropped it —
       ``warpedblfilter`` took no such argument, so it could not be forwarded to
       ``comp_warpedfreqresponse``/``comp_warpedfoff``, which have always
       accepted it.  Every negative-fc channel was therefore built by evaluating
       the warp below zero, outside its domain.
    2. The mirrored branch of ``comp_warpedfoff`` had MATLAB's ``+1`` stripped
       from it as a 1-based indexing artifact.  It is not one — it compensates
       for the ``n-1`` in the ``H[::-1]`` reversal — so every mirrored channel
       landed exactly one bin low.
    3. The mirrored branch takes a deliberately wide window (~2B rather than the
       filter's own B-A) because the roll-and-reverse arithmetic needs it, and
       then never trimmed back.  The surplus is the aliased ``win_hi`` term that
       the positive twin discards: channel -2321.6 Hz came out with 3334 nonzero
       bins and 4.5x the energy of its +2321.6 Hz twin.

    Each of the three leaves the peak amplitudes intact, which is why "does it
    look like a filter" checks never caught any of them.  Mirror equality
    catches all three, and does so exactly — these are the same numbers in the
    reverse order, not an approximation, so the tolerance is 0.
    """
    g, _a, fc, L, _info = _build_warped(freqrange="complex")
    fca = np.asarray(fc, dtype=float)
    M = len(g)

    pairs = 0
    for m in range(1, M // 2 - 1):
        twin = np.flatnonzero(np.isclose(fca, -fca[m]))
        if twin.size == 0:
            continue
        pairs += 1
        pos = np.abs(_dense(g[m], L))
        neg = np.abs(_dense(g[int(twin[0])], L))
        # Mirror about DC: bin k of the negative channel answers to bin L-k.
        mirrored = np.concatenate([[neg[0]], neg[:0:-1]])

        assert np.array_equal(pos, mirrored), (
            f"channel {m} (fc={fca[m]:.1f} Hz) and its negative twin are not "
            f"mirror images: relative L2 difference "
            f"{np.linalg.norm(pos - mirrored) / max(np.linalg.norm(pos), 1e-30):.3e}, "
            f"support sizes {int(np.count_nonzero(pos))} vs "
            f"{int(np.count_nonzero(mirrored))}"
        )

    assert pairs >= 8, f"only {pairs} mirror pairs found; the fixture is too small to be a test"


@pytest.mark.requires_impl
def test_warped_real_bank_is_unchanged_by_the_symmetry_fix():
    """The symmetry work must not touch the real-valued path.

    ``do_symmetric`` only takes effect for a negative centre frequency, and a
    ``freqrange='real'`` bank has none.  Pinned because the fix threads a new
    argument through three functions that the real path also calls, and a
    default flipped the wrong way would silently change every warped bank in
    the package rather than only the complex ones.
    """
    g, _a, fc, L, _info = _build_warped(freqrange="real")

    assert np.all(np.asarray(fc, dtype=float) >= 0.0), "a real bank has no negative fc"
    for m, gm in enumerate(g):
        H = _dense(gm, L)
        assert np.any(np.abs(H) > 0), f"channel {m} of the real bank is empty"
        assert np.all(np.isfinite(H)), f"channel {m} of the real bank has non-finite entries"


# ---------------------------------------------------------------------------
# Open item 3: analyze_filterbank's probe signal
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_analyze_filterbank_probe_is_scale_invariant_and_never_aliases():
    """Open item 3 turned out to be a **misdiagnosis**, pinned so it stays fixed.

    The register recorded that ``analyze_filterbank`` "hard-codes fs = 8000 for
    its probe signal although the filters carry ``fs``, so on a 4 kHz bank its
    2500 Hz tone is above Nyquist".  Measuring it says otherwise: ``t`` is a
    *sample index*, so ``440 * t / 8000`` is a digital frequency of 0.055
    cycles/sample.  The three tones sit at 0.110, 0.250 and 0.625 of Nyquist
    regardless of the bank's real sampling rate, and the highest is at 0.3125
    cycles/sample — comfortably below the 0.5 that would alias.

    Rewriting the expression in Hz against the filters' own ``fs`` was tried and
    is numerically identical to 5e-14, so the code is unchanged and only the
    comment is.  This test exists because the expression *reads* like a bug: it
    pins the property that makes it correct, so the next person to reach for the
    obvious "fix" gets told why not.
    """
    probe = (440 / 8000, 1000 / 8000, 2500 / 8000)

    for f_digital in probe:
        assert f_digital < 0.5, (
            f"probe tone at {f_digital:.4f} cycles/sample is above Nyquist and would alias"
        )

    # And the analysis runs identically whatever the bank's sampling rate,
    # because the probe never referred to that rate in the first place.
    from cool_frames.numpy.filterbanks import analyze_filterbank
    from cool_frames.numpy.filters import audfilters

    L = 512
    t_idx = np.arange(L, dtype=float)
    expected = (
        np.sin(2 * np.pi * 440 * t_idx / 8000)
        + 0.5 * np.sin(2 * np.pi * 1000 * t_idx / 8000)
        + 0.3 * np.sin(2 * np.pi * 2500 * t_idx / 8000)
    )
    for fs in (4000, 16000):
        rescaled = (
            np.sin(2 * np.pi * 0.110 * (fs / 2) * t_idx / fs)
            + 0.5 * np.sin(2 * np.pi * 0.250 * (fs / 2) * t_idx / fs)
            + 0.3 * np.sin(2 * np.pi * 0.625 * (fs / 2) * t_idx / fs)
        )
        assert np.allclose(rescaled, expected, atol=1e-10), (
            f"the probe expressed as fractions of Nyquist at fs={fs} differs "
            f"from the literal expression — one of them is wrong"
        )

    for fs in (4000, 16000):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, a, _fc, Lb, _info = audfilters(fs, 512)
            rep = analyze_filterbank(g, a, Lb)
        assert rep["coefficients"]["energy"]["total"] > 0


# ---------------------------------------------------------------------------
# Open item 4: ifilterbank silently ignoring Ls > L
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_ifilterbank_warns_when_ls_exceeds_the_transform_length():
    """Asking for more samples than exist used to return fewer, silently.

    ``Ls > L`` fell through the trim branch and returned ``L`` samples with no
    indication the request had been ignored, so the caller's next operation saw
    an array of unexpected length — a shape error several frames away from the
    cause, or a silent broadcast.

    The coefficients genuinely determine only ``L`` samples, so returning ``L``
    is right; not saying so was not.
    """
    from cool_frames.numpy.filterbanks import filterbank, ifilterbank
    from cool_frames.numpy.filters import audfilters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, _fc, L, _info = audfilters(FS, LS)
        c = filterbank(np.random.default_rng(0).standard_normal(LS), g, a, L=L)

    with pytest.warns(UserWarning, match=r"exceeds the transform length"):
        out = ifilterbank(c, g, a, Ls=L + 500)
    assert np.asarray(out).shape[0] == L, "the returned length should still be L"

    # The ordinary case must stay quiet and exact.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        short = ifilterbank(c, g, a, Ls=LS)
    assert np.asarray(short).shape[0] == LS


# ---------------------------------------------------------------------------
# Open item 5: filterbankiter's real default
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_filterbankiter_derives_real_and_converges_on_the_flagship_bank():
    """The documented default diverged on the package's own flagship bank.

    ``filterbankiter`` defaulted to ``real=False`` while ``ifilterbank``,
    ``filterbankdual`` and ``filterbanktight`` all default to ``real=True`` — an
    asymmetry that was not merely stylistic: on ``audfilters(4000, 512)`` the
    default ran the full 100 iterations to a relative residual of 58, against
    9 iterations and 7.9e-07 in the correct mode.

    Flipping the default to ``True`` would only have moved the breakage onto
    two-sided banks, so it is derived from the filters instead, and both cases
    are pinned here.  The explicit override is pinned too, since that is what
    makes the derivation safe to rely on.
    """
    from cool_frames.numpy.filterbanks import filterbankiter
    from cool_frames.numpy.filters import audfilters, gabfilters

    rng = np.random.default_rng(0)
    x = rng.standard_normal(512)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, _fc, _L, _info = audfilters(4000, 512)
        _c, relres, niter = filterbankiter(x, g, a)
    final = float(np.atleast_1d(relres)[-1])
    assert final < 1e-5 and niter < 50, (
        f"single-sided bank did not converge: relres={final:.4g} after {niter} iterations"
    )

    # A genuinely two-sided bank must also converge, which is the case a
    # flipped default would have broken.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gc, ac = gabfilters(4000, 512, M=32, a=8, real=False)[:2]
        _c2, relres2, niter2 = filterbankiter(x, gc, ac)
    final2 = float(np.atleast_1d(relres2)[-1])
    assert final2 < 1e-5 and niter2 < 50, (
        f"two-sided bank did not converge: relres={final2:.4g} after {niter2} iterations"
    )

    # The override still reaches the algorithm — this is the old broken mode.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _c3, relres3, _n3 = filterbankiter(x, g, a, real=False)
    assert float(np.atleast_1d(relres3)[-1]) > 1.0, (
        "real=False no longer reproduces the divergence, so the override is not being honoured"
    )


@pytest.mark.requires_torch_impl
def test_torch_filterbankiter_matches_the_numpy_default():
    """The two backends must not disagree about their own default.

    The torch wrapper carried the same ``real=False`` default.  A default that
    differs between backends is its own defect — code ported from one to the
    other changes behaviour without changing a line — so it is fixed in parity
    and the parity is what gets asserted.
    """
    torch = pytest.importorskip("torch")
    from cool_frames.numpy.filterbanks import filterbankiter as np_iter
    from cool_frames.numpy.filters import audfilters
    from cool_frames.torch.filterbanks import filterbankiter as t_iter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, _fc, _L, _info = audfilters(4000, 512)
        x = np.random.default_rng(0).standard_normal(512)
        _cn, rn, itn = np_iter(x, g, a)
        _ct, rt, itt = t_iter(torch.as_tensor(x, dtype=torch.float64), g, a)

    assert itn == itt, f"iteration counts differ: numpy {itn}, torch {itt}"
    assert float(np.atleast_1d(rn)[-1]) == pytest.approx(float(rt), rel=1e-9)


# ---------------------------------------------------------------------------
# Open items 6-8: the torch wrappers and fixed output dtypes
# ---------------------------------------------------------------------------


@pytest.mark.requires_torch_impl
def test_hopfilters_is_no_longer_exported():
    """An exported name that can only ever raise is worse than no name.

    ``cool_frames.torch.filters.hopfilters`` was in ``__all__`` and raised
    ``NotImplementedError`` unconditionally, because there is no NumPy
    ``hopfilters`` for it to wrap — the name appears only in other modules'
    docstrings.  It advertised a capability the package does not have, and
    discovery (``dir()``, tab-completion, the API docs) listed it alongside
    designers that work.

    Removing it is safe precisely because it never returned: no caller can have
    been depending on its behaviour.
    """
    pytest.importorskip("torch")
    import cool_frames.torch.filters as tf

    assert "hopfilters" not in tf.__all__, "hopfilters is still advertised in __all__"
    assert not hasattr(tf, "hopfilters"), "hopfilters is still importable"


@pytest.mark.requires_torch_impl
def test_torch_firwin_takes_device_and_dtype_like_its_siblings():
    """``firwin`` was the one wrapper in the module without them.

    Every other wrapper in ``cool_frames.torch.filters`` accepts ``device`` and
    ``dtype``; ``firwin`` hardcoded ``torch.tensor(..., dtype=torch.float64)``
    on the CPU.  So the single window-building entry point was the one that
    forced a caller assembling a filterbank on the GPU to notice and move it by
    hand, and silently widened a float32 pipeline.
    """
    torch = pytest.importorskip("torch")
    from cool_frames.torch.filters import firwin

    w = firwin("hann", 64)
    assert w.dtype == torch.float64, "the default must stay float64"

    w32 = firwin("hann", 64, dtype=torch.float32)
    assert w32.dtype == torch.float32, "dtype is not reaching the result"
    assert torch.allclose(w32.double(), w, atol=1e-6), "the window itself changed"

    # device is accepted and honoured (CPU is the only device we can rely on
    # in CI, but the argument must at least be wired through rather than
    # rejected as unexpected).
    assert firwin("hann", 8, device="cpu").device.type == "cpu"


@pytest.mark.requires_impl
def test_output_dtypes_are_requestable_without_changing_the_default():
    """Three functions hardcoded their output dtype.

    ``filterbankresponse`` always returned float64, ``filterbankfreqz``
    complex128 and ``ifilterbankiter`` float64, with no way to ask for anything
    else.

    The default deliberately does not change.  The NumPy backend is a float64
    reference implementation throughout — ``filterbank`` itself widens a float32
    input to complex128 — so narrowing these three by default would make them
    inconsistent with the rest of the backend rather than more consistent, and
    genuine dtype polymorphism lives in the torch side.  What was missing was
    the ability to ask, and that is what this pins.
    """
    from cool_frames.numpy.filterbanks import (
        filterbank,
        filterbankfreqz,
        filterbankresponse,
        ifilterbankiter,
    )
    from cool_frames.numpy.filters import audfilters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, _fc, L, _info = audfilters(FS, 512)
        x = np.random.default_rng(0).standard_normal(512)
        c = filterbank(x, g, a)

        resp64 = np.asarray(filterbankresponse(g, a, L))
        resp32 = np.asarray(filterbankresponse(g, a, L, dtype=np.float32))
        fz128 = np.asarray(filterbankfreqz(g, a=a, L=L))
        fz64 = np.asarray(filterbankfreqz(g, a=a, L=L, dtype=np.complex64))
        rec64 = np.asarray(ifilterbankiter(c, g, a, Ls=512)[0])
        rec32 = np.asarray(ifilterbankiter(c, g, a, Ls=512, dtype=np.float32)[0])

    # Defaults unchanged.
    assert resp64.dtype == np.float64
    assert fz128.dtype == np.complex128
    assert rec64.dtype == np.float64

    # ...and each is now requestable.
    assert resp32.dtype == np.float32
    assert fz64.dtype == np.complex64
    assert rec32.dtype == np.float32

    # Narrowing must change precision, not values.
    assert np.allclose(resp32, resp64, rtol=1e-6)
    assert np.allclose(fz64, fz128, rtol=1e-6, atol=1e-6)
    assert np.allclose(rec32, rec64, rtol=1e-5, atol=1e-6)


@pytest.mark.requires_impl
def test_ifilterbankiter_derives_real_too():
    """The synthesis twin of the ``filterbankiter`` default defect.

    The register recorded the analysis side only.  ``ifilterbankiter`` carried
    the identical ``real=False`` default and reconstructed the flagship
    ``audfilters`` bank with **23 % error**, where the correct mode reaches
    4.5e-16 — machine precision.  Found by reading the two signatures next to
    each other while fixing the one that was filed.
    """
    from cool_frames.numpy.filterbanks import filterbank, ifilterbankiter
    from cool_frames.numpy.filters import audfilters, gabfilters

    x = np.random.default_rng(0).standard_normal(512)

    for name, build in (
        ("audfilters", lambda: audfilters(4000, 512)[:2]),
        ("gabfilters complex", lambda: gabfilters(4000, 512, M=32, a=8, real=False)[:2]),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, a = build()
            c = filterbank(x, g, a)
            out = np.real(np.asarray(ifilterbankiter(c, g, a, Ls=512)[0]))[:512]
        err = np.linalg.norm(out - x) / np.linalg.norm(x)
        assert err < 1e-10, f"{name}: round-trip error {err:.4g} with the derived real mode"


# ---------------------------------------------------------------------------
# Open item 9: two-sided frequency axes
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_magresp_two_sided_axis_matches_the_dft_bins():
    """The axis was a ``linspace``, the data was ``fftshift``-ed.

    ``linspace(-1, 1, L)`` steps by ``2/(L-1)`` where the DFT bins step by
    ``2/L``, and it ends at ``+1``, which is not a bin — the two-sided range is
    ``[-1, 1 - 2/L]``.  So the axis was stretched by one bin across its whole
    width and every plotted feature sat off its true frequency, by more towards
    the edges.  Nothing about the plot looked wrong, which is the point.
    """
    from cool_frames.numpy.filterbanks import magresp
    from cool_frames.numpy.filters import audfilters

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, _a, fc, L, _info = audfilters(FS, 512)
    freq, mag_db = magresp(g[5], L=L, fs=FS, posfreq=False)

    assert len(freq) == L
    assert np.all(np.diff(freq) > 0), "the two-sided axis must be ascending"
    # Exactly the fftshift-ed bin frequencies.
    assert np.allclose(freq, np.fft.fftshift(np.fft.fftfreq(L, d=1.0 / FS)))
    assert freq[0] == pytest.approx(-FS / 2)
    assert freq[-1] == pytest.approx(FS / 2 - FS / L)

    # And the response now peaks at the channel's actual centre frequency,
    # within a bin.
    assert abs(freq[int(np.argmax(mag_db))] - float(fc[5])) <= FS / L

    # Normalised axis keeps the same [-1, 1) convention it always claimed.
    fn, _ = magresp(g[5], L=L, posfreq=False)
    assert fn[0] == pytest.approx(-1.0)
    assert fn[-1] == pytest.approx(1.0 - 2.0 / L)


@pytest.mark.requires_impl
def test_plotfft_two_sided_does_not_wrap_mid_plot():
    """``fftfreq`` order runs 0..+Nyquist then -Nyquist..0.

    Plotting the coefficients against it in DFT order drew the line rightwards
    to Nyquist and then jumped back to -Nyquist, leaving a horizontal streak
    across the middle of every two-sided plot.  Both the axis *and* the data
    have to be shifted; shifting only one of them swaps the halves instead.
    """
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from cool_frames.numpy.filterbanks import plotfft

    x = np.random.default_rng(0).standard_normal(256)
    spec = np.fft.fft(x)
    ax = plotfft(spec, fs=FS)
    xd = np.asarray(ax.get_lines()[0].get_xdata())
    yd = np.asarray(ax.get_lines()[0].get_ydata())

    assert np.all(np.diff(xd) > 0), "the x-axis wraps, so the plotted line doubles back"
    assert xd[0] == pytest.approx(-FS / 2)

    # The data must have travelled with the axis: the value at each plotted
    # frequency is still that frequency's coefficient.
    assert np.allclose(yd, np.fft.fftshift(np.abs(spec)))


# ---------------------------------------------------------------------------
# Open items 11-12: dead code and the realonly inconsistency
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_dead_builders_are_gone():
    """Six pieces of code with no reachable caller.

    ``_design._build_lowpass``/``_build_highpass``,
    ``_cqtfilters._apply_taper_to_edge_filters``,
    ``_warpedfilters_design._comp_nyquistfilt``/``comp_zerofilt`` had zero call
    sites; ``core/_core.py`` computed ``a[:, 0] / a[:, 1]`` and discarded it.

    Dead code in a numerics library is not merely untidy: each of these looks
    like the function you want when you go looking for the edge-filter
    construction, and two of them are *near-misses* for the live builders that
    replaced them.  Pinned by name so they do not drift back in.
    """
    import cool_frames.numpy.core._core as core
    import cool_frames.numpy.filters._cqtfilters as cqt
    import cool_frames.numpy.filters._design as design
    import cool_frames.numpy.filters._warpedfilters_design as wfd

    for mod, name in (
        (design, "_build_lowpass"),
        (design, "_build_highpass"),
        (cqt, "_apply_taper_to_edge_filters"),
        (wfd, "_comp_nyquistfilt"),
        (wfd, "comp_zerofilt"),
    ):
        assert not hasattr(mod, name), f"{mod.__name__}.{name} is back"

    # The discarded statement had no effect, so the only way to pin it is to
    # confirm the function it lived in still behaves.
    assert hasattr(core, "comp_filterbank_fftbl")


@pytest.mark.requires_impl
def test_gabwin_no_longer_takes_arguments_it_never_read():
    """``_gabwin(g, a, M, L, norm)`` read only ``g``, ``M`` and ``norm``.

    ``a`` and ``L`` were threaded in from the caller and never touched, which
    invites exactly the wrong conclusion at the call site — that the window
    depends on the lattice.  It does not.
    """
    import inspect

    from cool_frames.numpy.filters._gabfilters import _gabwin

    params = list(inspect.signature(_gabwin).parameters)
    assert params == ["g", "M", "norm"], f"unexpected signature: {params}"

    w = _gabwin("hann", 32)
    assert np.asarray(w).size == 32


@pytest.mark.requires_impl
def test_magresp_default_axis_is_consistent_across_designers():
    """``posfreq`` keyed off ``realonly``, which the designers disagree about.

    ``cqtfilters`` sets ``realonly=1`` on 76 of 78 channels and ``audfilters``
    sets 0 on all of its, though both build single-sided banks.  So the same
    ``magresp`` call returned a one-sided axis for a CQT filter and a two-sided
    one for an auditory filter — a difference in output shape driven by a label
    that nothing else in the package reads.

    It now measures the filter's own negative-frequency energy, the per-filter
    form of the test ``filterbank_is_real`` applies to a bank.  That also makes
    it *correct* per filter rather than per designer: a below-Nyquist channel of
    a complex Gabor bank really does have no negative-frequency content, and
    reports one-sided, while its above-Nyquist channels report two-sided.
    """
    from cool_frames.numpy.filterbanks import magresp
    from cool_frames.numpy.filters import audfilters, cqtfilters, gabfilters

    def _sided(gm, L):
        freq, _db = magresp(gm, L=L, fs=FS)
        return "one" if freq[0] >= 0 else "two"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        banks = {
            "aud": audfilters(FS, 512),
            "cqt": cqtfilters(FS, 512),
            "gab_real": gabfilters(FS, 512, M=32, a=8, real=True),
        }
        sides = {name: _sided(r[0][5], r[3]) for name, r in banks.items()}

        assert len(set(sides.values())) == 1, (
            f"single-sided designers disagree about the default axis: {sides}"
        )
        assert sides["aud"] == "one"

        # The label really is what changed, not the filters: cqt still emits 1.
        assert banks["cqt"][0][5].get("realonly") == 1
        assert banks["aud"][0][5].get("realonly") == 0

        # Per-filter correctness on a genuinely two-sided bank.
        gc, _ac, fcc, Lc, _i = gabfilters(FS, 512, M=32, a=8, real=False)
        below = [m for m in range(len(gc)) if fcc[m] < FS / 2 * 0.6][2]
        above = next(m for m in range(len(gc)) if fcc[m] > FS / 2 * 1.2)
        assert _sided(gc[below], Lc) == "one"
        assert _sided(gc[above], Lc) == "two"


# ---------------------------------------------------------------------------
# Open item 10: stale doctests, and the defects they were hiding
# ---------------------------------------------------------------------------


@pytest.mark.requires_impl
def test_erb_scale_conversions_match_glasberg_moore():
    """Four doctests documented an ERB-rate scale the code does not implement.

    ``freqtoaud(1000.0)`` was documented as ``9.264...``; it returns 15.572.
    The implementation is the right one — Glasberg & Moore's ERB-rate is
    ``21.4 * log10(1 + 0.00437 f)``, which gives 15.62 at 1 kHz — and it
    round-trips through ``audtofreq`` to 1e-16.  The docs were simply wrong,
    and had been for long enough that the same wrong value appeared in four
    places.

    A doctest nobody runs is a claim nobody checked, which is why these are now
    in CI.  This test pins the *reference* value independently of the doctest,
    so the two would have to be wrong in the same way to agree.
    """
    from cool_frames.numpy.filters import audtofreq, erbtofreq, freqtoaud, freqtoerb

    expected = 21.4 * np.log10(1 + 0.00437 * 1000.0)
    got = float(freqtoaud(1000.0))
    assert got == pytest.approx(expected, rel=0.01), (
        f"freqtoaud(1000) = {got:.4f}, but Glasberg-Moore gives {expected:.4f}"
    )
    assert float(freqtoerb(1000.0)) == pytest.approx(got)

    for f0 in (100.0, 1000.0, 8000.0):
        assert float(audtofreq(freqtoaud(f0))) == pytest.approx(f0, rel=1e-9)
        assert float(erbtofreq(freqtoerb(f0))) == pytest.approx(f0, rel=1e-9)


@pytest.mark.requires_impl
def test_firwin_norms_normalise():
    """``norm='energy'`` multiplied by sqrt(M) instead of dividing by the norm.

    Found while fixing a doctest that asserted ``np.sqrt(np.sum(w**2)) == 256``
    for an energy-normalised window — a claim that is wrong whichever
    convention you pick, and which turned out to be hiding a real defect.

    ``_apply_norm`` scaled *up* by ``sqrt(M)`` for ``'energy'``/``'2'`` and by
    ``M`` for ``'1'``/``'area'``, so a "unit energy" Hann window of length 512
    had an L2 norm of 313.5.  That contradicted ``core._norm.normalize_window``,
    the public ``setnorm`` and ``_warpedfilters._setnorm`` — all of which
    divide — and LTFAT.  ``firwin``'s default is ``'inf'``, which was always
    right, so the damage was confined to explicit callers; ``gabfilters`` is
    the one in-tree caller that asks for energy, via ``_gabwin``.
    """
    from cool_frames.numpy.filters import firwin
    from cool_frames.numpy.filters._gabfilters import _gabwin

    for M in (32, 256, 512):
        assert float(np.linalg.norm(firwin("hann", M, norm="energy"))) == pytest.approx(1.0)
        assert float(np.linalg.norm(firwin("hann", M, norm="2"))) == pytest.approx(1.0)
        assert float(np.sum(np.abs(firwin("hann", M, norm="1")))) == pytest.approx(1.0)
        assert float(np.max(np.abs(firwin("hann", M, norm="inf")))) == pytest.approx(1.0)
        # The one in-tree caller that asks for energy.
        assert float(np.linalg.norm(_gabwin("hann", M, norm="energy"))) == pytest.approx(1.0)


@pytest.mark.requires_impl
def test_gabfilters_still_reconstructs_after_the_norm_fix():
    """Rescaling the window must not disturb the frame.

    ``norm='energy'`` is ``gabfilters``' default, so correcting it changes the
    coefficient scale of every Gabor bank in the package.  It must not change
    what matters: the dual scales inversely, so reconstruction stays exact and
    the conditioning is untouched.  Pinned because a scale change that *did*
    leak into the frame properties would be a much worse bug than the one being
    fixed.
    """
    from cool_frames.numpy.filterbanks import (
        filterbank,
        filterbankdual,
        filterbankresponse,
        ifilterbank,
    )
    from cool_frames.numpy.filters import gabfilters

    x = np.random.default_rng(0).standard_normal(512)
    ratios = {}
    for norm in ("energy", "inf"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g, a, _fc, L, _info = gabfilters(FS, 512, M=32, a=8, norm=norm)
            c = filterbank(x, g, a, L=L)
            xr = np.real(np.asarray(ifilterbank(c, filterbankdual(g, a, L), a, Ls=512)))[:512]
            resp = np.asarray(filterbankresponse(g, a, L, real=True))
        resp = resp[resp > 1e-12]
        assert np.linalg.norm(xr - x) / np.linalg.norm(x) < 1e-12, (
            f"norm={norm}: reconstruction broke"
        )
        ratios[norm] = float(resp.max() / resp.min())

    assert ratios["energy"] == pytest.approx(ratios["inf"], rel=1e-9), (
        f"the window norm changed the frame conditioning: {ratios}"
    )


@pytest.mark.requires_impl
def test_filter_transfer_functions_may_be_arrays_not_only_callables():
    """``gm["H"](L)`` assumed a callable; materialised banks are not.

    A filter descriptor's ``'H'`` may be a ``callable(L)`` or an already-
    evaluated array — the designers produce the former so a bank can be
    re-evaluated at any length, while ``prepare_filters`` and the torch
    wrappers produce the latter.

    Eight call sites across the phase modules called it unconditionally, which
    raised ``TypeError: 'numpy.ndarray' object is not callable`` (or ``'Tensor'
    object is not callable``) for every materialised bank — i.e. for the whole
    torch backend whenever ``fc`` was not passed explicitly.  Each was found
    only by tripping over it, so the check now lives in one helper.
    """
    from cool_frames.numpy.filters._hval import eval_foff, eval_H

    arr = np.arange(8, dtype=complex)
    assert np.array_equal(eval_H(arr, 16), arr), "an array H must pass through"
    assert np.array_equal(
        eval_H(lambda L: np.arange(L, dtype=complex), 8), np.arange(8, dtype=complex)
    ), "a callable H must be called"
    assert eval_foff(3, 16) == 3
    assert eval_foff(lambda L: L // 4, 16) == 4
    assert eval_foff(None, 16) == 0


@pytest.mark.requires_torch_impl
def test_torch_phase_api_works_on_materialised_filters():
    """The end-to-end consequence of the callable-H assumption.

    Every one of these raised ``TypeError`` before, because the torch wrappers
    hand on filters whose ``'H'`` is a tensor.  They are exercised together
    because they failed together and for one reason.
    """
    torch = pytest.importorskip("torch")
    from cool_frames.torch.filterbanks import filterbank
    from cool_frames.torch.filters import audfilters
    from cool_frames.torch.phase import (
        filterbankconstphase,
        filterbankphasederiv,
        filterbankphasegrad,
        gla,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g, a, _fc, L, _info = audfilters(16000, 8000)
        x = torch.randn(8000)
        c = filterbank(x, g, a, L=L)

        assert len(filterbankconstphase(x, g, a, L=L)) == 2
        assert gla([cm.abs() for cm in c], g, a, L=L, Ls=8000, maxit=2)[-1] == 2
        res, _cc = filterbankphasederiv(x, g, a, L=L, derivs=["tt", "tf"])
        assert sorted(res) == ["tf", "tt"] and len(res["tt"]) == len(g)
        tgrad, _fg, _s, _c2 = filterbankphasegrad(x, g, a, L=L)
        assert len(tgrad) == len(g)
