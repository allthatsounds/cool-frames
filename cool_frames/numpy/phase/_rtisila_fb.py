"""The RTISI-LA family on a filter bank (internal).

PHASERET defines RTISI-LA for a Gabor frame, where a frame is a column of
coefficients and a windowed segment of the signal.  On a filter bank the
frames are blocks of time: frame ``k`` holds every coefficient whose time
``n a_m`` falls in ``[k H, (k+1) H)``, with ``H`` the ``frame_hop`` (the
bank's hop when it is uniform, so a frame is a column as in PHASERET; else
the largest hop of the channels that contain neither DC nor Nyquist).  The algorithm is then PHASERET's:

- frames enter one at a time; the partial reconstruction contains the frames
  already committed and the ``lookahead + 1`` frames in the look-ahead
  window, and nothing later (the real-time constraint);
- each step updates the window's frames ``maxit`` times, newest to oldest,
  each update re-analysing the current partial reconstruction at that
  frame's coefficients and imposing the target magnitude;
- the oldest frame is then committed and the window moves on;
- the newest frame is analysed with Zhu's windows (``_HopClass.build_spec``),
  built from the bank's atoms; on a bank that is a Gabor frame these are
  PHASERET's ``specg1`` / ``specg2`` and the result is PHASERET's.

The partial reconstruction is held as its spectrum and updated exactly:
committing or changing a frame adds the synthesis of the change, and a frame
is analysed from the spectrum directly, so no step costs a full transform.
Both use each channel's frequency response as the bank's own
``filterbank`` / ``ifilterbank`` do (band-limited, full-length or FIR), with
the time shift of coefficient ``n`` applied as a phase, so an update is
exact to rounding (checked against ``filterbank`` / ``ifilterbank`` in the
tests).  Its cost is the number of non-zero bins of the channels in the frame,
of the bank and of its dual: for the benchmark's 513-channel Gabor bank,
whose exact dual is about 4800 bins per channel, about 6 ms per update.
"""

from __future__ import annotations

from math import ceil

import numpy as np

_TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# One channel's frequency response, as the bank's transforms use it
# ---------------------------------------------------------------------------
def _channel_spectrum(gm: dict, a_row: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """``(kappa, values)``: the signed DFT indices at which the channel is
    non-zero and its response there.  Coefficient ``n`` of the channel is
    ``scale * sum_i values[i] F[kappa[i] mod L] exp(2 pi i kappa[i] n / N_m)``
    and synthesises ``conj(values[i]) * c[n] * exp(-2 pi i kappa[i] n / N_m)``
    into bin ``kappa[i] mod L`` (``comp_filterbank_fftbl`` and its inverse;
    the index stays signed because ``N_m`` need not divide ``L``)."""
    if "H" in gm:
        H = np.asarray(gm["H"], dtype=complex).ravel()
        if H.size == 0:
            return np.zeros(0, dtype=np.int64), H
        if H.size == L and int(a_row[1]) == 1:
            kappa = np.arange(L, dtype=np.int64)  # full-length: foff unused
        else:
            kappa = int(gm["foff"]) + np.arange(H.size, dtype=np.int64)
        return kappa, H
    # Time-domain FIR filter: its full response, delayed by its offset (the
    # form `comp_ifilterbank` synthesises with; analysis is its adjoint).
    from ..core import postpad

    h = np.asarray(gm["h"]).ravel()
    skip = int(gm.get("offset", 0) or 0)
    k = np.arange(L, dtype=np.int64)
    H = np.fft.fft(postpad(h.astype(complex), L)) * np.exp(-2j * np.pi * k * skip / L)
    return k, H


class _HopClass:
    """The channels sharing one hop: their analysis and synthesis as sparse
    matrices over the bins they touch, and the phase table of the hop."""

    def __init__(self, chans, g_ready, gd_ready, a_norm, L, N):
        from scipy.sparse import csr_matrix

        self.chans = np.asarray(chans, dtype=np.int64)
        m0 = int(chans[0])
        num, den = int(a_norm[m0, 0]), int(a_norm[m0, 1])
        self.N = int(N)
        self.L = int(L)
        # With N | L the phase depends on the index mod L only: fold it, so
        # every bin appears once and the updates need no accumulation.
        self.folded = den == 1 and self.N * num == L
        scale = 1.0 / (self.N * (num / den))
        nch = len(chans)

        def assemble(bank, conj_values: bool):
            rows, kap, vals = [], [], []
            for i, m in enumerate(chans):
                kp, v = _channel_spectrum(bank[m], a_norm[m], L)
                if self.folded:
                    kp = np.mod(kp, L)
                rows.append(np.full(kp.size, i, dtype=np.int64))
                kap.append(kp)
                vals.append(v)
            rows = np.concatenate(rows)
            kap = np.concatenate(kap)
            vals = np.concatenate(vals)
            U, inv = np.unique(kap, return_inverse=True)
            if conj_values:
                mat = csr_matrix((np.conj(vals), (inv, rows)), shape=(U.size, nch))
            else:
                mat = csr_matrix((vals * scale, (rows, inv)), shape=(nch, U.size))
            return U, mat

        self.U, self.A = assemble(g_ready, False)
        self.Ud, self.S = assemble(gd_ready, True)
        self.hop = num if den == 1 else None
        self.scale = scale
        self._g_ready = [g_ready[int(m)] for m in chans]
        self._gd_ready = [gd_ready[int(m)] for m in chans]
        self._a_rows = [a_norm[int(m)] for m in chans]
        self.spec = None  # (base, W1, W2), built on demand
        self.FU = np.mod(self.U, L)
        self.FUd = np.mod(self.Ud, L)
        self.FUd_neg = np.mod(-self.FUd, L)
        self.dup = (not self.folded) and np.unique(self.FUd).size < self.FUd.size
        self.table = np.exp(2j * np.pi * np.arange(self.N) / self.N)

    def _phase(self, kappa: np.ndarray, n0: int, n1: int) -> np.ndarray:
        n = np.arange(n0, n1, dtype=np.int64)
        return self.table[np.mod(kappa[:, None] * n[None, :], self.N)]

    def analyse(self, F: np.ndarray, n0: int, n1: int) -> np.ndarray:
        """Coefficients ``n0..n1-1`` of every channel of the class."""
        v = F[self.FU][:, None] * self._phase(self.U, n0, n1)
        return np.asarray(self.A @ v)

    def build_spec(self, fc: np.ndarray) -> bool:
        """Zhu's analysis windows for the newest frame (PHASERET's
        ``comp_rtisilawins``), as time-domain atoms: for channel ``m`` with
        hop ``a``, synthesis atom ``psi`` and centre frequency ``fc``,

            spec2(t) = rect(t) * sum_{j=0}^{J} exp(2 pi i fc j a) psi(t - j a)

        and ``spec1`` the same sum from ``j = 1``, where ``rect`` is the
        main lobe of the channel's analysis atom (from its peak to the first
        rise of its modulus on either side: the whole support of a compact
        window, the window's length for ``gabfilters``) and
        ``J = ceil(len(rect) / a) - 1``.  On a
        Gabor frame this is PHASERET's ``specg2`` / ``specg1``: the sum of
        the frame's dual window and those of the frames after it, cut to the
        frame, with the modulation of the channel.  Needs an integer hop
        (a fractional hop puts the atoms off the sample grid); returns
        whether it was built."""
        if self.hop is None or self.spec is not None:
            return self.spec is not None
        L, a = self.L, self.hop
        rows = []
        for i in range(len(self.chans)):
            kp, v = _channel_spectrum(self._g_ready[i], self._a_rows[i], L)
            dense = np.zeros(L, dtype=complex)
            np.add.at(dense, np.mod(kp, L), np.conj(v))
            phi = np.fft.ifft(dense)
            kd, vd = _channel_spectrum(self._gd_ready[i], self._a_rows[i], L)
            dense[:] = 0
            np.add.at(dense, np.mod(kd, L), np.conj(vd))
            psi = np.fft.ifft(dense)
            start, width = _main_lobe(np.abs(phi))
            tau = np.mod(start + np.arange(width), L)
            J = max(0, ceil(width / a) - 1)
            m = int(self.chans[i])
            terms = [
                np.exp(2j * np.pi * fc[m] * j * a) * psi[np.mod(tau - j * a, L)]
                for j in range(J + 1)
            ]
            spec2 = np.sum(terms, axis=0)
            spec1 = spec2 - terms[0]
            rows.append((start, np.conj(spec1), np.conj(spec2)))
        D = max(r[1].size for r in rows)
        base = np.array([r[0] for r in rows], dtype=np.int64)
        W1 = np.zeros((len(rows), D), dtype=complex)
        W2 = np.zeros((len(rows), D), dtype=complex)
        for i, (_st, w1, w2) in enumerate(rows):
            W1[i, : w1.size] = w1
            W2[i, : w2.size] = w2
        self.spec = (base, W1, W2)
        return True

    def analyse_spec(self, f: np.ndarray, n0: int, n1: int, first: bool) -> np.ndarray:
        """Coefficients ``n0..n1-1`` analysed with the newest frame's windows
        from the time-domain partial reconstruction ``f`` (only their phase
        is used)."""
        base, W1, W2 = self.spec
        W = W1 if first else W2
        n = np.arange(n0, n1, dtype=np.int64)
        idx = np.mod(
            base[:, None, None]
            + (n * self.hop)[None, :, None]
            + np.arange(W.shape[1])[None, None, :],
            self.L,
        )
        return np.einsum("ird,id->ir", f[idx], W)

    def synthesise(self, F: np.ndarray, delta: np.ndarray, n0: int, n1: int, real: bool) -> None:
        """Add the synthesis of ``delta`` (channels x ``n1-n0``) to the
        spectrum ``F`` of the signal (``real``: of twice its real part)."""
        X = np.asarray(self.S @ delta)
        D = np.einsum("ij,ij->i", X, np.conj(self._phase(self.Ud, n0, n1)))
        if self.dup:
            np.add.at(F, self.FUd, D)
            if real:
                np.add.at(F, self.FUd_neg, np.conj(D))
        else:
            F[self.FUd] += D
            if real:
                F[self.FUd_neg] += np.conj(D)


# ---------------------------------------------------------------------------
# Frames, durations and the default look-ahead
# ---------------------------------------------------------------------------
def _inner_channels(g_ready: list[dict], a_norm: np.ndarray, L: int) -> list[int] | None:
    """The channels containing neither DC nor Nyquist, or ``None`` if that
    is none of them.  The DC and Nyquist complements of an auditory or
    wavelet bank are narrow bands with long atoms and large hops; left in,
    they would set a frame as long as the signal."""
    inner = [
        m for m in range(len(g_ready)) if not _touches_dc_or_nyquist(g_ready[m], a_norm[m], L)
    ]
    return inner or None


def default_frame_hop(g_ready: list[dict], a_norm: np.ndarray, L: int) -> int:
    """The bank's hop when uniform, else the largest hop of the channels
    containing neither DC nor Nyquist (rounded up)."""
    afrac = a_norm[:, 0] / a_norm[:, 1]
    inner = _inner_channels(g_ready, a_norm, L)
    sel = afrac if inner is None else afrac[inner]
    return max(1, int(ceil(float(np.max(sel)) - 1e-12)))


def _energy_width(h: np.ndarray, step: float, frac: float) -> float:
    """Length of the shortest circular run of ``h`` holding ``frac`` of its
    energy, in samples (``step`` samples per entry)."""
    e = np.abs(h) ** 2
    tot = float(e.sum())
    if tot == 0.0:
        return 0.0
    n = e.size
    cs = np.concatenate([[0.0], np.cumsum(np.concatenate([e, e]))])
    need = frac * tot
    # For every start, the first end with enough energy.
    ends = np.searchsorted(cs, cs[:n] + need, side="left")
    return float(np.min(ends - np.arange(n))) * step


def _main_lobe(mag: np.ndarray) -> tuple[int, int]:
    """``(start, width)`` of the circular run around the peak of ``mag``
    over which it does not rise, walking outwards (at most half the circle
    each way)."""
    n = mag.size
    i0 = int(np.argmax(mag))
    half = n // 2
    fwd = np.roll(mag, -i0)[: half + 1]
    up = np.flatnonzero(np.diff(fwd) > 0)
    right = int(up[0]) if up.size else half
    bwd = np.roll(mag[::-1], i0 + 1)[: half + 1]  # mag[i0], mag[i0-1], ...
    up = np.flatnonzero(np.diff(bwd) > 0)
    left = int(up[0]) if up.size else half
    width = min(n, left + right + 1)
    return (i0 - left) % n, width


def _energy_interval(e: np.ndarray, frac: float) -> tuple[int, int]:
    """``(start, width)`` of the shortest circular run of ``e`` (energies)
    holding ``frac`` of the total."""
    n = e.size
    cs = np.concatenate([[0.0], np.cumsum(np.concatenate([e, e]))])
    ends = np.searchsorted(cs, cs[:n] + frac * float(e.sum()), side="left")
    widths = ends - np.arange(n)
    i = int(np.argmin(widths))
    return i, int(max(1, min(widths[i], n)))


_WIDTH_FRACTION = 1.0 - 1e-3


def atom_duration(g_ready: list[dict], L: int, frame_hop: int, channels=None) -> float:
    """The longest analysis atom's duration among ``channels`` (default all):
    the shortest time interval holding all but 1e-3 of its energy, from its
    envelope sampled at a spacing of at most ``frame_hop / 8`` samples."""
    from ..core import postpad

    widths: dict[bytes, float] = {}
    best = 0.0
    for m in range(len(g_ready)) if channels is None else channels:
        gm = g_ready[m]
        if "H" not in gm:
            h = np.asarray(gm["h"]).ravel()
            key = b"h" + np.round(np.abs(h), 12).tobytes()
            if key not in widths:
                widths[key] = _energy_width(h, 1.0, _WIDTH_FRACTION)
        else:
            H = np.asarray(gm["H"], dtype=complex).ravel()
            if H.size == 0:
                continue
            key = b"H" + np.round(np.abs(H), 12).tobytes()
            if key not in widths:
                nfft = max(H.size, int(ceil(8 * L / max(frame_hop, 1))))
                nfft = min(L, 1 << int(ceil(np.log2(nfft))))
                if H.size > nfft:
                    nfft = L
                env = np.fft.ifft(postpad(H, nfft))
                widths[key] = _energy_width(env, L / nfft, _WIDTH_FRACTION)
        best = max(best, widths[key])
    return best


def _touches_dc_or_nyquist(gm: dict, a_row, L: int) -> bool:
    kappa, v = _channel_spectrum(gm, a_row, L)
    k = np.mod(kappa[v != 0], L)
    return bool(np.any(k == 0) or (L % 2 == 0 and np.any(k == L // 2)))


def default_lookahead(g_ready, a_norm, L, frame_hop, n_frames) -> int:
    """PHASERET's ``ceil(M/a) - 1`` -- the frames after a frame that its
    atoms reach -- with the window length ``M`` replaced by the duration of
    the longest atom among the channels containing neither DC nor Nyquist."""
    D = atom_duration(g_ready, L, frame_hop, _inner_channels(g_ready, a_norm, L))
    return int(min(max(0, ceil(D / frame_hop - 1e-9) - 1), max(n_frames - 1, 0)))


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------
class FbFrames:
    """Coefficients grouped by hop and by frame, with the spectrum of the
    current partial reconstruction."""

    def __init__(self, g, gd, a_norm, L, N, real, frame_hop):
        from ..filterbanks._utils import prepare_filters

        self.L = int(L)
        self.real = bool(real)
        self.M = len(g)
        self.Nm = [int(n) for n in N]
        self.a_norm = a_norm
        g_ready = prepare_filters(g, a_norm, L)[0]
        gd_ready = prepare_filters(gd, a_norm, L)[0]
        self.g_ready = g_ready
        keys: dict[tuple[int, int], list[int]] = {}
        for m in range(self.M):
            keys.setdefault((int(a_norm[m, 0]), int(a_norm[m, 1])), []).append(m)
        self.classes = [
            _HopClass(ch, g_ready, gd_ready, a_norm, L, self.Nm[ch[0]]) for ch in keys.values()
        ]
        H = int(frame_hop)
        if H < 1:
            raise ValueError(f"frame_hop must be a positive integer, got {frame_hop!r}")
        self.frame_hop = H
        # frames[k] = [(class index, n0, n1), ...]; coefficient n of a class
        # with hop num/den is in frame floor(n num / (den H)).
        nf = 0
        spans = []
        for cl in self.classes:
            m0 = int(cl.chans[0])
            num, den = int(a_norm[m0, 0]), int(a_norm[m0, 1])
            n = np.arange(cl.N, dtype=np.int64)
            fr = (n * num) // (den * H)
            starts = np.searchsorted(fr, np.arange(int(fr[-1]) + 2))
            spans.append(starts)
            nf = max(nf, int(fr[-1]) + 1)
        self.n_frames = nf
        self.frames = []
        for k in range(nf):
            fl = []
            for ci, starts in enumerate(spans):
                if k + 1 < starts.size and starts[k] < starts[k + 1]:
                    fl.append((ci, int(starts[k]), int(starts[k + 1])))
            self.frames.append(fl)
        self.F = np.zeros(self.L, dtype=complex)

    # -- coefficient storage, per class: (channels, N) ----------------------
    def stack(self, per_channel: list[np.ndarray], dtype=complex) -> list[np.ndarray]:
        return [
            np.stack([np.asarray(per_channel[m], dtype=dtype).ravel() for m in cl.chans])
            for cl in self.classes
        ]

    def unstack(self, per_class: list[np.ndarray]) -> list[np.ndarray]:
        out: list[np.ndarray] = [np.zeros(0)] * self.M
        for cl, X in zip(self.classes, per_class):
            for i, m in enumerate(cl.chans):
                out[int(m)] = X[i].copy()
        return out

    def analyse(self, k: int) -> list[np.ndarray]:
        return [self.classes[ci].analyse(self.F, n0, n1) for ci, n0, n1 in self.frames[k]]

    def enable_spec(self, fc: np.ndarray) -> None:
        """Build Zhu's newest-frame windows for every class with an integer
        hop (the others keep the plain analysis)."""
        self.spec_classes = {ci for ci, cl in enumerate(self.classes) if cl.build_spec(fc)}

    def analyse_newest(self, k: int, first: bool) -> list[np.ndarray]:
        spec = getattr(self, "spec_classes", set())
        if not spec:
            return self.analyse(k)
        f = np.fft.ifft(self.F)
        out = []
        for ci, n0, n1 in self.frames[k]:
            cl = self.classes[ci]
            if ci in spec:
                out.append(cl.analyse_spec(f, n0, n1, first))
            else:
                out.append(cl.analyse(self.F, n0, n1))
        return out

    def add(self, k: int, deltas: list[np.ndarray]) -> None:
        for (ci, n0, n1), d in zip(self.frames[k], deltas):
            self.classes[ci].synthesise(self.F, d, n0, n1, self.real)


def run_rtisi(n_frames: int, lookahead: int, maxit: int, enter, update, order=None) -> None:
    """PHASERET's RTISI-LA schedule over ``n_frames`` frames.

    ``enter(k)`` gives frame ``k`` its initial coefficients when it enters
    the look-ahead window; ``update(k)`` re-analyses the partial
    reconstruction at frame ``k`` and imposes the magnitude.  Frames
    ``0 .. lookahead`` enter together (PHASERET's buffers start with the
    first ``lookahead`` frames read); after that one frame enters per step.
    Each step updates the window ``maxit`` times, newest frame first unless
    ``order(k, window, it)`` says otherwise (``update(j, it, newest)`` is
    told the iteration and whether ``j`` is the frame that entered at this
    step), then commits frame ``k``.  At
    the end the window shrinks (PHASERET wraps round to the first frames
    instead, which a finite signal does not have)."""
    for k in range(min(lookahead, n_frames - 1) + 1):
        enter(k)
    for k in range(n_frames):
        newest = k + lookahead
        if k > 0 and newest < n_frames:
            enter(newest)
        window = list(range(min(newest, n_frames - 1), k - 1, -1))
        for it in range(maxit):
            for j in window if order is None else order(k, window, it):
                update(j, it, j == newest)


def _princarg(x: np.ndarray) -> np.ndarray:
    return x - _TWO_PI * np.round(x / _TWO_PI)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------
class _Problem:
    """Arguments, dual, frames and look-ahead shared by the three drivers."""

    def __init__(self, s_list, g, a, L, real, frame_hop, lookahead, maxit):
        from ..filterbanks._frame import filterbankdual
        from ..filterbanks._utils import normalise_a

        M = len(g)
        if len(s_list) != M:
            raise ValueError(f"s has {len(s_list)} channels but the bank has {M}")
        self.s_in = [np.asarray(sm).ravel() for sm in s_list]
        self.s_abs = [np.abs(sm).astype(float) for sm in self.s_in]
        self.a_norm = normalise_a(a, M)
        self.N = [sm.size for sm in self.s_abs]
        afrac = self.a_norm[:, 0] / self.a_norm[:, 1]
        self.L = int(round(self.N[0] * afrac[0])) if L is None else int(L)
        self.real = bool(real)
        m = int(maxit)
        if m != maxit or m < 1:
            raise ValueError(f"maxit must be a positive integer, got {maxit!r}")
        self.maxit = m
        self.g = g
        self.gd = filterbankdual(g, self.a_norm, self.L, real=self.real)
        if frame_hop is None:
            from ..filterbanks._utils import prepare_filters

            H = default_frame_hop(prepare_filters(g, self.a_norm, self.L)[0], self.a_norm, self.L)
        else:
            H = frame_hop
        if int(H) != H or H < 1:
            raise ValueError(f"frame_hop must be a positive integer, got {frame_hop!r}")
        self.eng = FbFrames(g, self.gd, self.a_norm, self.L, self.N, self.real, int(H))
        nf = self.eng.n_frames
        if lookahead is None:
            self.lookahead = default_lookahead(self.eng.g_ready, self.a_norm, self.L, int(H), nf)
        else:
            la = int(lookahead)
            if la != lookahead or la < 0 or la > nf - 1:
                raise ValueError(
                    f"lookahead must be an integer in [0, {nf - 1}] ({nf} frames of "
                    f"{int(H)} samples), got {lookahead!r}"
                )
            self.lookahead = la
        self.S = self.eng.stack(self.s_abs, dtype=float)

    def niter(self) -> int:
        return self.maxit * (self.lookahead + 1)

    def finish(self, c_list, Ls):
        """Synthesis, and ``relres = || |A f| - s || / ||s||`` for the
        signal the coefficients synthesise (before cropping to ``Ls``)."""
        from ..filterbanks._core import filterbank, ifilterbank

        f = ifilterbank(c_list, self.gd, self.a_norm, Ls=self.L, real=self.real)
        re = filterbank(np.real(f) if self.real else f, self.g, self.a_norm, L=self.L)
        num = np.sqrt(
            sum(
                float(np.sum((np.abs(np.asarray(r).ravel()) - s) ** 2))
                for r, s in zip(re, self.s_abs)
            )
        )
        den = np.sqrt(sum(float(np.sum(s**2)) for s in self.s_abs))
        relres = float(num / den) if den > 0 else 0.0
        if Ls is not None:
            f = (
                f[: int(Ls)]
                if int(Ls) <= f.shape[0]
                else ifilterbank(c_list, self.gd, self.a_norm, Ls=int(Ls), real=self.real)
            )
        return c_list, f, relres

    # -- the exact engine ---------------------------------------------------
    def run_exact(self, init=None, order=None, spec=False):
        """The exact engine's RTISI-LA; ``spec``: analyse the newest frame
        with Zhu's windows (:meth:`_HopClass.build_spec`)."""
        eng, S = self.eng, self.S
        if spec:
            from ._centerfreq import filter_center_frequencies

            eng.enable_spec(np.asarray(filter_center_frequencies(self.g, self.L), dtype=float))
        C = [np.zeros(sc.shape, dtype=complex) for sc in S]

        def enter(k):
            if init is None:
                return
            vals = init(k, C)
            if vals is None:
                return
            for (ci, n0, n1), v in zip(eng.frames[k], vals):
                C[ci][:, n0:n1] = v
            eng.add(k, vals)

        def update(k, it=0, newest=False):
            est = eng.analyse_newest(k, it == 0) if (newest and spec) else eng.analyse(k)
            deltas = []
            for (ci, n0, n1), e in zip(eng.frames[k], est):
                new = S[ci][:, n0:n1] * np.exp(1j * np.angle(e))
                deltas.append(new - C[ci][:, n0:n1])
                C[ci][:, n0:n1] = new
            eng.add(k, deltas)

        run_rtisi(
            eng.n_frames,
            self.lookahead,
            self.maxit,
            enter,
            update,
            None if order is None else (lambda k, w, it: order(k, w, it, C)),
        )
        return eng.unstack(C)

    # -- initialisations ------------------------------------------------------
    def frame_values(self, per_class, k):
        return [per_class[ci][:, n0:n1] for ci, n0, n1 in self.eng.frames[k]]

    def spsi_init(self):
        return _spsi_init(self)

    def unwrap_init(self, unwrappar: float):
        """Phase-vocoder start: each channel's phase advanced from its last
        two coefficients (PHASERET's ``'unwrap'``, with the channel's centre
        frequency for the Gabor channel's ``m / M``)."""
        from ._centerfreq import filter_center_frequencies

        fc = np.asarray(filter_center_frequencies(self.g, self.L), dtype=float)
        eng, S = self.eng, self.S
        omega = []
        for cl in eng.classes:
            m0 = int(cl.chans[0])
            hop = self.a_norm[m0, 0] / self.a_norm[m0, 1]
            omega.append(_TWO_PI * hop * fc[cl.chans])

        def init(k, C):
            vals = []
            for ci, n0, n1 in eng.frames[k]:
                out = np.empty((C[ci].shape[0], n1 - n0), dtype=complex)
                for j, n in enumerate(range(n0, n1)):
                    p1 = np.angle(C[ci][:, n - 1]) if n >= 1 else 0.0
                    p2 = np.angle(C[ci][:, n - 2]) if n >= 2 else 0.0
                    w = omega[ci]
                    p0 = p1 + w + _princarg(p1 - p2 - w)
                    out[:, j] = unwrappar * S[ci][:, n] * np.exp(1j * p0)
                    C[ci][:, n] = out[:, j]
                vals.append(out)
            return vals

        return init


def _spsi_init(pb: _Problem):
    """SPSI continued from the refined phase: each time step of the entering
    frame advances every channel's phase from its previous coefficient's
    current phase, as PHASERET's ``gsrtisila`` does with ``comp_spsi`` and
    the frame before (``_spsi.spsi``'s step, which works across channels in
    frequency order)."""
    from math import lcm

    from ._centerfreq import filter_center_frequencies
    from ._spsi import _spsi_group

    eng, S = pb.eng, pb.S
    fc = np.asarray(filter_center_frequencies(pb.g, pb.L), dtype=float)
    hops = pb.a_norm[:, 0] / pb.a_norm[:, 1]
    den = 1
    for cl in eng.classes:
        den = lcm(den, int(pb.a_norm[int(cl.chans[0]), 1]))
    phase = np.zeros(len(pb.g))

    def init(k, C):
        events: dict[int, list[tuple[int, int, int]]] = {}
        for ci, n0, n1 in eng.frames[k]:
            m0 = int(eng.classes[ci].chans[0])
            num, d = int(pb.a_norm[m0, 0]), int(pb.a_norm[m0, 1])
            for n in range(n0, n1):
                events.setdefault(n * num * (den // d), []).append((ci, n0, n))
        vals = [
            np.zeros((len(eng.classes[ci].chans), n1 - n0), dtype=complex)
            for ci, n0, n1 in eng.frames[k]
        ]
        slot = {ci: j for j, (ci, _n0, _n1) in enumerate(eng.frames[k])}
        for t in sorted(events):
            chans, where = [], []
            for ci, n0, n in events[t]:
                for i, m in enumerate(eng.classes[ci].chans):
                    chans.append(int(m))
                    where.append((ci, i, n0, n))
            order = np.argsort(chans, kind="stable")
            active = [chans[j] for j in order]
            for j in order:
                ci, i, _n0, n = where[j]
                phase[chans[j]] = np.angle(C[ci][i, n - 1]) if n >= 1 else 0.0
            sabs = np.array([S[where[j][0]][where[j][1], where[j][3]] for j in order])
            _spsi_group(active, sabs, phase, hops, fc)
            for j in order:
                ci, i, n0, n = where[j]
                v = S[ci][i, n] * np.exp(1j * phase[chans[j]])
                vals[slot[ci]][i, n - n0] = v
                C[ci][i, n] = v
        return vals

    return init


def _check_startphase(name: str, startphase: str, allowed: tuple[str, ...]) -> str:
    if startphase not in allowed:
        raise ValueError(
            f"{name}: startphase must be one of {allowed} for a filter bank, got {startphase!r}"
        )
    return startphase


def rtisila_fb(s_list, g, a, *, L, Ls, real, maxit, lookahead, frame_hop, startphase, seed):
    _check_startphase("rtisila", startphase, ("zhu", "zero", "rand"))
    pb = _Problem(s_list, g, a, L, real, frame_hop, lookahead, maxit)
    init = None
    if startphase == "zero":

        def init(k, C):
            return pb.frame_values(pb.S, k)
    elif startphase == "rand":
        rng = np.random.default_rng(seed)
        R = [sc * np.exp(_TWO_PI * 1j * rng.random(sc.shape)) for sc in pb.S]

        def init(k, C):
            return pb.frame_values(R, k)

    c = pb.run_exact(init, spec=True)
    return (*pb.finish(c, Ls), pb.niter())


def gsrtisila_fb(s_list, g, a, *, L, Ls, real, maxit, lookahead, frame_hop, startphase, unwrappar):
    _check_startphase("gsrtisila", startphase, ("zhu", "zeros", "zero", "input", "unwrap", "spsi"))
    pb = _Problem(s_list, g, a, L, real, frame_hop, lookahead, maxit)
    init = None
    if startphase == "zero":

        def init(k, C):
            return pb.frame_values(pb.S, k)
    elif startphase == "input":
        Sin = pb.eng.stack(pb.s_in)

        def init(k, C):
            return pb.frame_values(Sin, k)
    elif startphase == "unwrap":
        init = pb.unwrap_init(unwrappar)
    elif startphase == "spsi":
        init = pb.spsi_init()
    c = pb.run_exact(init, spec=True)
    return (*pb.finish(c, Ls), pb.niter())


def lertisila_fb(
    s_list,
    g,
    a,
    *,
    L,
    Ls,
    real,
    maxit,
    lookahead,
    frame_hop,
    startphase,
    seed,
    variant,
    energy_order,
    onthefly,
    relthr,
    unwrappar,
):
    _check_startphase("lertisila", startphase, ("zhu", "zero", "rand", "input", "unwrap"))
    if variant not in ("trunc", "modtrunc"):
        raise ValueError(f"variant must be 'trunc' or 'modtrunc', got {variant!r}")
    pb = _Problem(s_list, g, a, L, real, frame_hop, lookahead, maxit)
    eng = pb.eng
    afrac = pb.a_norm[:, 0] / pb.a_norm[:, 1]
    hops = np.round(afrac).astype(int)
    use_kernel = bool(np.allclose(afrac, hops))

    Sin = pb.eng.stack(pb.s_in) if startphase == "input" else None
    R = None
    if startphase == "rand":
        rng = np.random.default_rng(seed)
        R = [sc * np.exp(_TWO_PI * 1j * rng.random(sc.shape)) for sc in pb.S]

    def order_fn(k, window, it, energy):
        if it == 0 or not energy_order:
            return window
        oldest_first = window[::-1]
        e = np.array([energy(j) for j in oldest_first])
        return [oldest_first[i] for i in np.argsort(-e, kind="stable")]

    if not use_kernel:
        if variant == "modtrunc" or onthefly:
            raise ValueError(
                "lertisila: with fractional hop sizes there is no truncated kernel; "
                "variant='modtrunc' and onthefly=True need integer hops"
            )
        init = _le_init(pb, startphase, unwrappar, Sin, R)
        cache: dict = {}

        def order(k, window, it, C):
            if it == 1:
                cache[k] = order_fn(
                    k,
                    window,
                    it,
                    lambda j: sum(
                        float(np.sum(np.abs(C[ci][:, n0:n1]) ** 2)) for ci, n0, n1 in eng.frames[j]
                    ),
                )
            return window if it == 0 else cache[k]

        c = pb.run_exact(init, order if energy_order else None)
        return (*pb.finish(c, Ls), pb.niter())

    # Le Roux's truncated projection kernel over the flat coefficient vector.
    from ._leglakernel import cached_kernel

    kern = cached_kernel(
        g,
        pb.gd,
        hops,
        pb.N,
        pb.L,
        real=pb.real,
        relthr=relthr,
        zero_self_term=(variant == "modtrunc"),
    )
    offs = kern.offs
    rows = []
    for k in range(eng.n_frames):
        r = [
            (offs[eng.classes[ci].chans][:, None] + np.arange(n0, n1)[None, :]).ravel()
            for ci, n0, n1 in eng.frames[k]
        ]
        rows.append(np.concatenate(r) if r else np.zeros(0, dtype=np.int64))
    P = [kern._P[r] for r in rows]
    Q = [kern._Q[r] for r in rows] if kern._Q is not None else None
    s_flat = np.concatenate(pb.s_abs)
    x = np.zeros(s_flat.size, dtype=complex)

    def per_class_to_flat(per_class_vals):
        if not per_class_vals:
            return np.zeros(0, dtype=complex)
        return np.concatenate([v.ravel() for v in per_class_vals])

    init_pc = _le_init(pb, startphase, unwrappar, Sin, R, flat=x, offs=offs)

    def enter(k):
        if init_pc is None:
            return
        vals = init_pc(k, None)
        if vals is not None:
            x[rows[k]] = per_class_to_flat(vals)

    def update(k, it=0, newest=False):
        r = rows[k]
        if not onthefly:
            est = P[k] @ x
            if Q is not None:
                est = est + Q[k] @ np.conj(x)
            x[r] = s_flat[r] * np.exp(1j * np.angle(est))
            return
        Pk = P[k]
        Qk = Q[k] if Q is not None else None
        for i in range(r.size):
            lo, hi = Pk.indptr[i], Pk.indptr[i + 1]
            v = np.dot(Pk.data[lo:hi], x[Pk.indices[lo:hi]])
            if Qk is not None:
                lo, hi = Qk.indptr[i], Qk.indptr[i + 1]
                v += np.dot(Qk.data[lo:hi], np.conj(x[Qk.indices[lo:hi]]))
            x[r[i]] = s_flat[r[i]] * np.exp(1j * np.angle(v))

    cache: dict = {}

    def order(k, window, it):
        if it == 1:
            cache[k] = order_fn(k, window, it, lambda j: float(np.sum(np.abs(x[rows[j]]) ** 2)))
        return window if it == 0 else cache[k]

    run_rtisi(eng.n_frames, pb.lookahead, pb.maxit, enter, update, order if energy_order else None)
    c = [x[offs[m] : offs[m + 1]].copy() for m in range(len(g))]
    return (*pb.finish(c, Ls), pb.niter())


def _le_init(pb: _Problem, startphase, unwrappar, Sin, R, flat=None, offs=None):
    """The coefficients a frame enters with, per class (``None``: empty).
    With ``flat``/``offs`` (the kernel path), 'unwrap' reads the current
    coefficients from the flat vector."""
    if startphase == "zhu":
        return None
    if startphase == "zero":
        return lambda k, C: pb.frame_values(pb.S, k)
    if startphase == "input":
        return lambda k, C: pb.frame_values(Sin, k)
    if startphase == "rand":
        return lambda k, C: pb.frame_values(R, k)
    # unwrap
    if flat is None:
        return pb.unwrap_init(unwrappar)
    eng = pb.eng
    Cview = [np.zeros(sc.shape, dtype=complex) for sc in pb.S]
    base = pb.unwrap_init(unwrappar)

    def init(k, _C):
        for ci, cl in enumerate(eng.classes):
            Cview[ci][:] = flat[offs[cl.chans][:, None] + np.arange(cl.N)[None, :]]
        return base(k, Cview)

    return init
