# Changelog

## 0.1.1 (unreleased)

Correctness release. The first half covers the phase-retrieval family; the
second covers a systematic audit of the filter designers, the filterbank core,
the operators and the PyTorch backend, which turned up ~40 further reproduced
defects. `DEFECT_REGISTER.md` records all of them, fixed and still-open, with
the measurement that established each one.

Several of these are **behavioural changes** — code that ran before will now
produce different (and correct) numbers — and one is a **breaking API change**.

### Breaking

- **`spsi` now takes `fc` in Hz plus a required `fs`**: `spsi(s, a, fc, fs)`.

  It previously documented `fc` as normalised (cycles per sample) while every
  filterbank constructor in the package returns Hz. Passing the constructor's
  output straight through — the obvious thing to do, and what the tutorial
  showed — advanced the phase by a factor of `fs` too fast and produced a result
  *worse than leaving the phase at zero* (consistency 0.50 against a 0.38
  zero-phase baseline; the correct call gives 0.11).

  Callers who had already normalised should pass `fs=1.0`. A `fc` whose maximum
  exceeds `fs/2` is now rejected with an explanatory error rather than silently
  producing nonsense.

### Fixed

- **`decolbfgs` (NumPy) never ran its optimiser.** The analytic gradient
  contradicted the objective — it used `conj(c)` where the Wirtinger derivative
  calls for `c`, and mis-handled the factor implied by single-sided synthesis.
  A finite-difference check disagreed in sign as well as magnitude, so
  `scipy.optimize.minimize` failed its first line search and returned
  `status=2` ("ABNORMAL") with `nit == 0`. `niter` was 0 for every `maxit` and
  the output was bit-identical however long you asked it to run.

  The gradient now agrees with central finite differences to ~1e-8 in both the
  real and complex branches, and quality improves with `maxit` (consistency
  0.065 at 1 iteration, 0.0022 at 200, where it previously sat at 0.215
  regardless).

  Relatedly, the gradient's `|c|^(p-2)` term overflowed on an all-zero channel
  (`tiny**-1.33` → inf → `inf * 0` → nan); the magnitude is now floored.

- **`legla`/`flegla` were plain `gla`/`fgla`.** The documented truncated
  projection kernel did not exist: `relthr` was never read, and `'modtrunc'`
  computed `angle(c + (proj - c)) ≡ angle(proj)`, algebraically identical to
  `'trunc'`. Output was bit-identical to `gla` at every setting.

  LEGLA now builds the real projection kernel
  `k_{m,m'} = ifft(conj(G_m) · Gd_{m'})`, truncates it at `relthr`, and applies
  it as a sparse operator (see `cool_frames/numpy/phase/_leglakernel.py`).
  `relthr=0` reproduces GLA's exact projection to machine precision — the
  property the implementation is tested against — and larger values give a
  progressively cheaper, progressively different operator. `'modtrunc'`
  additionally zeroes the self-term `k_{m,m}[0]`.

  This is a trade: the kernel costs O(M²) FFTs to build, after which each
  iteration is cheaper than a full analysis-synthesis pass. On a 23-channel ERB
  bank at `Ls=512`, `legla(relthr=1e-2)` is slower than `gla` at 10 iterations
  (0.22 s vs 0.07 s) and 2.3× faster at 100 (0.29 s vs 0.66 s). For a handful
  of iterations, use `gla`.

- **`fgla`/`flegla` returned unprojected coefficients.** They applied the
  momentum step after the magnitude projection and returned the extrapolated
  point, so `|c_out| != s` — 16 % relative error at 20 iterations, 120 % at 2 —
  while every other member of the family guaranteed equality to ~1e-16. A
  caller applying the family-wide assumption got a silent gain error. They now
  re-project the extrapolate, which also measures at least as well on
  consistency as returning the last projected iterate would.

- **`gsrtisila`'s `'spsi'` and `'unwrap'` start phases used a bogus frequency
  ramp.** Both branches fell back to `fc[m] = m / M` — reaching ~0.96 cycles
  per sample, nearly twice Nyquist — because the filter dicts carry no `fc`
  entry. Centre frequencies are now recovered from the filters' own transfer
  functions (`_centerfreq.py`), accurate to well under one DFT bin. The
  "smarter" starts are now actually better than `'zero'`: consistency 0.074
  (`spsi`) and 0.103 (`unwrap`) against 0.118.

- **torch `legla` reported the wrong residual.** It built `relres` as
  `torch.abs(torch.as_tensor(cp, dtype=dtype))` with a *real* `dtype`, so the
  cast ran before `abs` and discarded the imaginary part — 0.185 at iteration 6
  where the NumPy backend reported 0.070. Coefficients were unaffected, but
  `relres` is what a caller plots and what `tol` is compared against, so the
  method also stopped at the wrong time.

- **torch `decolbfgs` could not be differentiated through.** It built its
  optimisation variable as `x0_flat.clone().requires_grad_(True)`; when
  `s_list` required grad, that was a non-leaf tensor and `torch.optim.LBFGS`
  refused it outright ("can't optimize a non-leaf Tensor"). The inner solve is
  now properly detached, and gradients reach `s` through the final magnitude
  projection — not through the L-BFGS trajectory, which is neither meaningful
  nor affordable to differentiate.

- **torch `decolbfgs` reported closure evaluations as `niter`.** L-BFGS calls
  its closure several times per iteration during the line search, so the count
  was incomparable with the NumPy backend (4, 10, 50 against 1, 5, 40). Both
  now report iterations.

### Added

- **`seed` argument** for `startphase='rand'` across `gla`, `legla`,
  `decolbfgs`, `rtisila` and `lertisila` in both backends. Random starts were
  drawn from unseeded generators and so were irreproducible; the default is
  unchanged (`seed=None` still draws fresh entropy).

- **`cool_frames.numpy.phase._leglakernel.LeglaKernel`** — the truncated
  projection kernel, documented and independently testable.

- **`cool_frames.numpy.phase._centerfreq.filter_center_frequencies`** —
  recovers normalised centre frequencies from filter transfer functions.

### Tests

- Phase-retrieval test suite: invariant properties for the whole family,
  structural unit tests, NumPy/torch parity and differentiability, plus a
  runner that executes every Python example in the README and docs.
- Coverage of `numpy/phase` + `torch/phase` rose from 38 % to 70 %; the
  phase-retrieval modules specifically went from ~8 % to 70–93 %.

### Documentation

- Fixed every non-running example: a four-element return unpacked into two
  names, an import of `rtpghifb` (never exported), `real=True` passed to torch
  `filterbank` (no such argument), `filterbankrealtight` (it is
  `filterbanktight`), four `cool_frames.torch.sigproc` helpers that do not
  exist, and several blocks referencing undefined names.

### Fixed — audit of the rest of the package

`DEFECT_REGISTER.md` has the full list with reproductions. The ones that
silently returned wrong numbers:

- **`filterbankdual`/`filterbanktight` returned an all-zero bank for any FIR
  filterbank**, so `ifilterbank` reconstructed exactly `0.0` while
  `filterbankbounds` still reported a valid frame. A diagonal dual cannot exist
  for a time-limited filter (it is full-band, so never painless), so this now
  raises with an explanation and points at `ifilterbankiter`.
- **FIR synthesis ignored the filter `offset`**, so synthesis was not the
  adjoint of analysis: the frame operator came out non-symmetric with negative
  eigenvalues and CG diverged (relres 4.6e12). Adjointness is now 8e-15. The
  same block also mis-broadcast 1-D coefficient input.
- **torch `ifilterbank` discarded every frequency-domain channel** whenever an
  FIR channel was present (125 % error), **`comp_ifilterbank_td` used zero
  instead of periodic padding** (31.5 %), **`comp_filterbank_td` truncated**
  before the offset slice (wrong coefficient count), and **FIR filters could
  not be used with a real signal at all** (`RuntimeError`). All four fixed;
  parity is now ~5e-16.
- **`torch.operators` forced float32 throughout**, so float64 parity was ~1.8e-07
  and `gradcheck` failed. Now 5.3e-16.
- **`reassigned_spectrogram` always returned zeros** — it called
  `filterbankphasegrad` with three wrong arguments and a bare `except` swallowed
  the failure. The two summaries were also mapped to the wrong names.
- **Spectrogram padding sat at 0 dB (magnitude 1.0)** and set the peak, so for a
  quiet input the dynamic-range window was anchored ~30 dB too high.
- **`gabfilters` omitted the 1/√2 edge scaling** every other designer applies
  (κ 1.667 → 1.000001) and claimed a painless condition it never satisfies (it
  now warns, like the others).
- **`framemuleigs` silently symmetrised the operator**, so a complex symbol gave
  a leading eigenvalue 10 % off and reported complex eigenvalues as real.
- **The `'pcg'` preconditioner applied a frequency-domain diagonal to a
  time-domain residual** — not a preconditioner, and it slowed convergence.
- **`ifilterbankiter` reported the residual of the complex iterate** while
  returning the real part ("converged" 3.2e-07 for a true residual of 0.226).
- **`filterbankwin`'s `'dual'`/`'tight'` were silent synonyms** for the real
  variants; the complex ones were unreachable.
- **`biquadfilter` decoded DC and Nyquist to fs/4** — the logit fallback used the
  sigmoid's midpoint rather than its limit.
- **`freqwin('butterworth')` was 14 % too wide** (half-power convention where the
  other three windows solve for `bwrelheight`).
- **`torch_additions.denoise` synthesised with the tight frame** after analysing
  with the original (residual 28.7 → 1.8e-11).
- Plus: `framemulinv(maxit=0)` raised `UnboundLocalError`; torch `framemulinv`
  always reported `'iter': 1`; `recommend_filterbank`'s "bins from f0"
  expression contained no `f0`.

- **The torch backend no longer upcasts float32 to float64** (the item left open
  in the first pass). A `float32` input now gives `complex64`/`float32` out,
  computed in single precision: memory halves exactly and accuracy lands at
  float32 epsilon rather than being a cosmetic cast. See
  `cool_frames/torch/_dtypes.py`.

### Also fixed

- **mypy** passes at the CI-pinned 1.20.2. Three errors were failing the gate on
  `main`; one was an inert `# type: ignore` that was not the first comment on its
  line and so suppressed nothing.
- **The docs build is warning-free.**

### Fixed — second pass

The fifteen items the first pass left open are now fixed too:

- **`framemul` took `np.real(...)` unconditionally**, so the operator was only
  R-linear even with `real=False` while `framemulappr`'s complex branch models a
  C-linear generator. The recovered symbol was 44.5 % off in Hilbert-Schmidt
  norm on an operator with a provably exact answer; the round trip is now
  5.1e-15.
- **torch `framemulappr` was a different, wrong algorithm** — HS error 1.033,
  worse than returning a zero symbol, with `real` never read. It now delegates
  to the validated NumPy implementation: exact parity, and `real`, `method`,
  `max_gram` and `rcond` all live.
- **torch `filterbankiter` sliced the CG iterate to `Ls`**, making the map a
  projection rather than `F*F`; it diverged (relres 399.7) whenever `Ls < L`.
- **`reconstruct`'s `'pghi'` and `'spsi'` were both random phase**, bitwise
  identical, both reporting `converged: True`. `'spsi'` is now really SPSI,
  `'legla'` is added, and `'pghi'` is wired to the repaired magnitude path (see
  below). Separately the GLA calls took the default `real=False` on a
  single-sided bank — fixing that alone moved GLA from −6.0 dB to −31.3 dB and
  fGLA to −52.8 dB.
- **`warpedfilters(freqrange='complex')` crashed** on its default sampling
  (`np.vstack` on a 1-D `a`).
- **`waveletfilters(redtar=...)` was inert** on every non-uniform mode (`N_new`
  computed, `N_old` used) and produced `a = 0` on `uniform`.
- **`analyze_*` omitted `L`** on every internal call, inflating redundancy
  (3.63 against a true 2.15) and firing a false "not painless" note.
- **`analyze_coefficients` read a stacked `(N, M)` array as `(M, N)`**,
  computing every per-channel statistic over the wrong axis.
- **The convention-mismatch detector measured the reconstructed spectrum**,
  where single- and two-sided banks overlap, so `audfilters` slipped past and
  reconstructed with 46 % error in silence. It now measures the filters, where
  the two families separate with a factor-of-five margin.
- **`filterbankdual`/`filterbanktight` never checked the painless condition**
  although `filterbankwin` had always computed `ispainless` and nothing read it.
- **`filterbankbounds_svd` could return a negative lower bound**, giving
  "condition numbers" like −3.2e+15.
- **`_winwidthatheight` assumed the peak at index 0**, so the time-frequency
  ratio's shape term was algebraically dead: a Hann, a rectangle, a triangle and
  a three-bin needle all returned the same `gamma`.
- Plus torch's misclassification of fractional-hop full-length filters,
  `waveletfilters(delay=...)` skipping the appended Nyquist channel, and
  `filterbankfreqz`'s undocumented-as-ignored `a`.

### PGHI from magnitude alone now works

`filterbankconstphase`'s *signal* path was always excellent (consistency 0.0047
on an ERB bank where the zero-phase baseline is 0.385). Its *magnitude* path —
the only one applicable when magnitudes are all you have — measured 0.385
without `sqtfr` and 0.393 with it, i.e. at or worse than doing nothing.

The cause was a single missing term. `comp_filterbankphasegradfrommag` returned
the instantaneous-frequency **deviation** from each channel's centre frequency,
while the heap integrator (and `filterbankphasegrad`) consume the **absolute**
normalised instantaneous frequency. The centre-frequency term is an order of
magnitude larger than the deviation it was carrying, so omitting it did not
degrade the estimate — it replaced it. Two smaller defects fell out of the same
investigation: `fc` was passed in the `fc/fs` convention where the integrator
wants `fc/fs·2`, and the edge channels were half-scaled and divided by `1.0`
instead of their frequency spacing.

Consistency against the zero-phase baseline, magnitude only:

| fixture | zero phase | PGHI | signal path |
|---|---|---|---|
| chirp 200–800 Hz | 0.711 | **0.044** | 0.044 |
| chirp 300–1500 Hz | 0.767 | **0.040** | 0.041 |
| two sines | 0.385 | **0.170** | 0.005 |
| white noise | 0.669 | **0.324** | 0.323 |

Every fixture improves (2.1×–19×). On frequency-modulated input a single
non-iterative pass now beats 100 iterations of GLA (−27.1 dB vs −19.6 dB) and
comes within a decibel of what the same integrator achieves from the *true*
gradients. It is weakest on stationary tones over a coarse filterbank, which is
inherent to the method and is now documented rather than hidden.

`filterbankconstphase` gains an `fs` parameter. Without it, an `fc` that looks
like Hz is still normalised by assuming the top channel sits at Nyquist — which
is exact for `audfilters`, `cqtfilters` and `gabfilters(real=True)`, and wrong
by about a factor of two for a two-sided bank — but that inference now **warns**
instead of happening silently. Passing `fs` skips it and is bitwise identical
where the assumption holds. This is not a breaking change; an earlier v0.1.1
build did raise here, which broke existing callers for no gain on the common
path.

- **`filterbankconstphase` drew its below-threshold random phase from NumPy's
  global state.** Randomising the phase of coefficients at the noise floor is
  correct — integrating through them propagates noise — but using
  `np.random.uniform` made four identical calls return four different answers,
  and silently advanced the caller's global random stream as a side effect of
  running a transform. It now uses a local `Generator` and accepts
  `rng=<seed or Generator>` for reproducible output.

### Also in this release

- **`reconstruct` derives `real=` from the filters** instead of hardcoding
  `real=True`, via the new public
  `cool_frames.numpy.filterbanks.filterbank_is_real(g, a, L)`. The hardcoded
  value was right for every bank the auditory and constant-Q designers produce
  and silently wrong (~30 dB, no exception) for a genuinely two-sided one. Pass
  `real=` explicitly to override the detection.
- **The lint scope is declared once**, in `[tool.ruff] include` in
  `pyproject.toml`, so a bare `ruff check .` is exactly the CI check.

### The remaining audit items are closed

All twelve items the audit had left open are fixed. The ones a caller will
notice:

- **`warpedfilters(freqrange='complex')` built its negative-frequency channels
  by evaluating the warp outside its domain.** Three defects, all silent: the
  computed `symmetry` flag was never forwarded (`warpedblfilter` took no such
  argument); MATLAB's `+1` in the mirrored offset was stripped as a 1-based
  artifact when it actually compensates for the `H[::-1]` reversal, putting
  every mirrored channel one bin low; and the deliberately wide mirroring
  window was never trimmed, leaving an aliased tail that gave one channel 4.5x
  its twin's energy. Every negative channel is now bitwise the mirror of its
  positive twin.
- **`warpedfilters(min_win=...)` was inert** — both edge builders were called
  with a literal `min_win=1`.
- **`filterbankiter` and `ifilterbankiter` defaulted to `real=False`**, which
  diverged on the package's flagship bank (100 iterations to a relative
  residual of 58) and reconstructed it with 23 % error respectively. Both now
  derive `real` from the filters, as `reconstruct` does; the torch
  `filterbankiter` is fixed in parity.
- **`firwin(norm='energy')` did not normalise** — it multiplied by `sqrt(M)`,
  so a "unit energy" Hann window of length 512 had an L2 norm of 313.5. This is
  `gabfilters`' default window norm, so Gabor coefficient *scales* change;
  reconstruction and conditioning do not (the dual scales inversely).
- **The phase modules assumed `filter['H']` was always a callable**, raising
  `TypeError` for every materialised bank — in effect the whole torch backend
  whenever `fc` was not passed explicitly.
- **`ifilterbank` silently ignored `Ls > L`**, returning fewer samples than
  asked for; it now warns.
- **`magresp` and `plotfft` drew their two-sided axes wrongly** — a `linspace`
  stretched by one bin against `fftshift`-ed data, and an unshifted axis that
  made the plotted line double back across the middle.
- **`hopfilters` is removed** from `cool_frames.torch.filters`; it raised
  unconditionally, as there is no NumPy implementation to wrap.
- **`torch.filters.firwin` takes `device` and `dtype`** like every sibling, and
  `filterbankresponse`, `filterbankfreqz` and `ifilterbankiter` take `dtype`
  (defaults unchanged).

**All 33 doctests are fixed and now run in CI.** Four were substantively wrong
— `cqtfilters` was documented as returning 14 channels against an actual 66,
and the ERB-rate of 1 kHz as 9.264 against 15.572 — and ten could never have
run at all. The two defects above (`firwin`'s norm and the callable-`H`
assumption) were found by fixing them.

One item in the register turned out to be a **misdiagnosis**:
`analyze_filterbank`'s probe signal was recorded as aliasing on low-rate banks,
but its `/8000` is a digital-frequency normalisation rather than a sampling
rate, so the tones were always below Nyquist. The code is unchanged and the
comment now says so.

## 0.1.0

First public release.
