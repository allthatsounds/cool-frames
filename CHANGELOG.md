# Changelog

## 0.1.1 (unreleased)

Correctness release. The first half covers the phase-retrieval family; the
second covers a systematic audit of the filter designers, the filterbank core,
the operators and the PyTorch backend, which turned up ~40 further reproduced
defects. `DEFECT_REGISTER.md` records all of them, fixed and still-open, with
the measurement that established each one.

Several of these are **behavioural changes** — code that ran before will now
produce different (and correct) numbers — and two are **breaking API changes**.

### Breaking

- **`filterbankconstphase` returns a 2-tuple**: `c, usedmask = filterbankconstphase(...)`.

  NumPy previously returned the coefficient list alone while the torch backend
  already returned `(coeffs, usedmask)`, so the same code could not drive both.
  NumPy's own annotation already claimed `-> tuple` and was silenced with a
  `# type: ignore`, and LTFAT returns the mask too — so of the three ways to
  resolve the disagreement, this is the only one that does not leave something
  else wrong.

  `usedmask` is a per-channel boolean array, True where the phase was
  *integrated* rather than drawn at random because the coefficient sat below
  `tol` of the peak. It was being computed and discarded since v0.1.0; it tells
  you which part of the reconstruction carries information.

  Also aligned, non-breaking: the torch `filterbankconstphase` now forwards
  `sqtfr`, `fs` and `rng`, so magnitude-path PGHI and reproducible phase are
  reachable from either backend, and `filterbankbounds` takes `return_kappa` on
  both.

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

### Frame admissibility is now predictable in closed form

A painless bank can be handed parameters for which it is not a frame: the lower
bound is exactly zero, a band of the spectrum is annihilated, and
`filterbankbounds` does not notice because it is evaluated on a response that
does not see the gap. `filterbankbounds_svd` does notice, but costs O(L^2) and
needs the bank built first.

`cool_frames.diagnostics.admissibility` answers the question before the bank
exists, from the design parameters alone. `A = 0` iff the filters leave a DFT
bin uncovered, and coverage is integer arithmetic: a prototype of odd length
`W_m` centred on bin `k_m = round(L*f_m/fs)` overlaps its neighbour iff
`k_{m+1} - k_m <= (W_m-1)/2 + (W_{m+1}-1)/2 - 1`.

Because every designer places its channels uniformly in a warping coordinate
*and* sizes its filters constant in that same coordinate, the warping
derivative cancels and the condition reduces to one dimensionless ratio,
`D_u < L_u - c*fs/(L*b_min)`, binding at the lowest channel. The designer
families then reach the floor by different routes: `warpedfilters` reduces to
the scale-independent `2*bwmul > 1/bins` and `audfilters` to
`spacing < bwmul/winbw`, whereas for constant-Q designs `fc` cancels and the
continuum condition `Qvar > t/(t+1)` (with `t = 2**(1/bins)`) holds for every
`Qvar >= 2/3` — so a constant-Q bank never fails from redundancy, only from
low-frequency resolution.

Verified against the measured frame response on **8,524 of 8,524
configurations**, no errors in either direction: `audfilters` 5400 (540 per
scale across all ten scales), `greenwoodfilters` 540, `cqtfilters` 1916,
`warpedfilters` 668 under two different warpings. The covering mechanism itself
was additionally confirmed against `filterbankbounds_svd` (66/66) and against
the measured response for `waveletfilters` and `gabfilters` (150/150).

Conditioning comes with it but is weaker: `ripple_curve` gives kappa as a
universal function of the overlap ratio alone (1 at the Hann-squared
partition-of-unity ratios 1/4 and 1/3, 2 at half overlap, divergent as the
ratio approaches 1), accurate to within a decade on 87 % of admissible banks.
The `floor23` hop quantisation is not modelled, so the predictor is **exact for
"is it a frame"** and **order-of-magnitude for "how well conditioned"**.

New API: `predict_admissible`, `ripple_curve`, `max_overlap_for_kappa`,
`min_channels`, `min_bins`. 16 regression tests pin the agreement so that a
change to any designer's geometry which breaks it fails the build.

**Every designer now runs the check itself.** `audfilters`, `cqtfilters`,
`greenwoodfilters`, `warpedfilters`, `waveletfilters` and `gabfilters` call
`check_admissible` before returning and publish the verdict as
`info["admissible"]`, alongside the geometry it used (`fsupp`, `fsupp_inner`,
`fsupp_dc`, `fsupp_nyq`; `scalevec` and `bwmul` for `warpedfilters`). A
geometry below the floor raises `NotAFrameWarning` naming the first uncovered
bin and its frequency. The bank is still built — analysis still works, and
studying the gap is a legitimate thing to want — but the failure is announced
where the parameters were chosen rather than later, when the missing band shows
up as an all-zero dual.

Extending the predictor to the two remaining designers needed a rule for each,
since neither is a painless frequency-domain design:

- `gabfilters` builds its prototype in the **time** domain, so its realised
  support is not a designed bandwidth but the width of the block it stores:
  every channel occupies exactly `gl = len(g)` DFT bins at
  `A_k = k*(L/M) - gl//2`, with no dead endpoints. For a named window
  (`gl == M`) this reduces to **frame iff `L/M <= M`**.
- `waveletfilters` are dilations, so each support edge is a fixed multiple of
  that channel's centre frequency — the dilation cancels — and the designer
  reads the interval straight off `freqwavelet`'s own `foff`/`fsupp`.

Validated at 100 % on 5,525 `gabfilters` and 4,259 `waveletfilters`
configurations, including randomised hold-out grids and grids deliberately
biased toward non-frames, plus an independent 720-configuration re-check.
Layouts the covering test cannot express report `info["admissible"] = None`
rather than an unvalidated verdict: `gabfilters(windowaxis='freq')`,
`gabfilters` given a window array whose length is not `M`, and
`waveletfilters` with `lowpass='none'`/`'repeat'`, `highpass='none'` or a
two-sided `freqrange`.

Note that this makes some existing configurations warn that did not before —
correctly. The package's own `waveletfilters` test fixtures build banks whose
measured lower bound is exactly zero.

### The three Colab notebooks were broken, and nothing could see it

`examples/colab/*.ipynb` imported `filterbankrealdual`,
`constphase_nonuniform`, `cool_frames.numpy.phase._admm` and
`cool_frames.numpy.phase._diff_admm` — none of which exist — unpacked the
designers' five-element return into four names, and called
`_causal_tgrad_tick` and `_fixed_order_phase_tick`, which no cell defines. They
had been that way since the June 2026 consolidation moved the Diff-RTPGHI
research code out of the package.

Nothing caught it because `[tool.ruff] include` is a whitelist and carried
`extend-exclude = ["examples/colab"]`, annotated "WIP Colab tutorial notebooks
(cross-cell refs / placeholder helpers) are not linted as library code". A
reviewer's first act is to open `02_reproduce_paper_results.ipynb` and press
Run, so this was the most visible code in the repository and the least tested.

All three are rebuilt against the shipped API and now execute end to end:

- **01** is a PGHI tutorial: filterbank geometry, the admissibility check, the
  signal path versus the magnitude path, heap integration, fGLA refinement.
- **02** benchmarks every phase-retrieval method the package ships over six
  signals on the 134-channel ERB bank. The Diff-RTPGHI, ADMM, RAAR and DM rows
  are gone — those algorithms live with the SPL paper's code, not here, and
  printing rows this package cannot compute was the wrong fix. The RTISI-LA
  family and `legla` are also absent, for a measured reason stated in the
  notebook: at 77x redundancy a two-iteration `rtisila` run takes minutes, and
  `legla` refuses to build its kernel below `relthr = 0.5`, which truncates so
  hard that what runs is no longer meaningfully LeGLA.
- **03** measures which torch paths carry a gradient (analysis, synthesis and
  `gla` do; `filterbankconstphase` and `ifilterbankiter` do not), gradchecks
  `gla` against central finite differences, optimises through it, and then
  shows why a waveform target is nevertheless the wrong objective downstream of
  Griffin-Lim from zero phase — and that freezing the phase restores a monotone
  magnitudes-to-waveform map without losing differentiability.

`tests/test_example_notebooks.py` compiles every cell and resolves every
`cool_frames` import in every notebook under `examples/`, and flags any name
that is called but never bound. It fails on all three of the old notebooks. The
ruff exclusion is gone and notebooks are linted (`extend-include = ["*.ipynb"]`).

### Two torch docstrings said the opposite of the truth

`cool_frames.torch.phase`'s module docstring claimed "every algorithm here is
differentiable with respect to the input magnitudes", and
`torch/phase/_constphase.py` twice directed readers to
`constphase_nonuniform` in `_diff_constphase.py` for a differentiable
alternative. That module does not exist anywhere in the package, and
`filterbankconstphase` is a NumPy shim that detaches. Both now state what is
actually differentiable and what is not, and point at `gla`, which is.

### `warpedfilters` was losing the lower sideband of every filter

Found by the above: the predictor was correct about the design and the
implementation was not. `comp_warpedfreqresponse` evaluates its prototype at
warped bin positions, producing an argument that runs symmetrically over
[-0.5, 0.5], and passes it to `firwin_eval` — which uses the whole-point-even
convention (`x` in [0, 1), peak at 0, even about 0.5) and returns zero for
every negative argument. Every warped channel therefore began at its own centre
bin instead of half a passband below it: on a log-warped bank at fs = 8 kHz,
`bins = 1`, `bwmul = 0.6`, channel `fc = 128 Hz` occupied bins 19..26 where its
design calls for 12..26.

Wrapping the argument into [0, 1) restores the symmetric window. Realised
supports now match the design to the bin, and the admissibility predictor's
accuracy on `warpedfilters` goes from 70.5 % to 100 %.

The same negative-argument pattern survives in the legacy `comp_zerofilt` and
`_comp_nyquistfilt` edge builders, which are dead code (the unified complement
is used instead) and should be deleted rather than repaired.

### Also fixed here

- **`comp_filterbankphasegradfrommag` below-branch time correction had the
  wrong sign.** Both branches adjust the neighbour to the centre coefficient's
  time instant; the upward branch forms `logs[neigh] - logs[w]` and subtracts
  the correction, the downward branch forms the difference the other way round
  and must therefore add it. It subtracted in both. The effect on reconstructed
  phase is small — `dist` is usually a fraction of a frame — but it biased
  every interior channel.
- **The `sqtfr` convention is now documented per designer** on
  `filterbankconstphase`: `ones(M)` for `audfilters` (and explicitly *not*
  `compute_tfr_from_filters`, which returns L/gamma), `g[m]['tfr'](L)` for
  `waveletfilters`, `Cg*gl**2` for `gabfilters`, and not established for
  `cqtfilters`. None of these has been validated side by side against MATLAB
  LTFAT; that check is what still limits magnitude-only phase retrieval.

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

- **`torch.ifilterbankiter` had the `real=False` default too.** It was the
  fourth place that defect lived — after the NumPy `filterbankiter`, the torch
  `filterbankiter` and the NumPy `ifilterbankiter` — and it outlived all three
  fixes, reconstructing the flagship bank with 23 % error while its siblings
  reached machine precision. Its output dtype was also hardcoded to float64;
  it now follows the coefficients', like the rest of the torch backend.

One item in the register turned out to be a **misdiagnosis**:
`analyze_filterbank`'s probe signal was recorded as aliasing on low-rate banks,
but its `/8000` is a digital-frequency normalisation rather than a sampling
rate, so the tones were always below Nyquist. The code is unchanged and the
comment now says so.

### Found by chasing the coverage number

Codecov read 76 %. Executing the code no test had executed turned up two more
defects, which is the useful thing to say about the metric: it was not hygiene,
it was a list of places nobody had checked.

- **`pghi_findgamma` returned a window constant 6-10x too large for a numeric
  window.** A 256-tap Hann gave `Cg = 2.195` against the tabulated 0.25645 for
  the same window — and the table *is* the precomputed answer to that search.
  Since `gamma = Cg * gl**2` scales the phase gradients PGHI integrates, this
  did not blur the phase estimate, it replaced it.

  The cause is the fifth occurrence in this audit of one defect fixed in one of
  two copies. `_winwidthatheight` finds the threshold crossing by scanning
  `g[:gl//2+1]`, which only tracks the falling flank if the window peaks at
  index 0; handed the centred ordering `scipy.signal.get_window` produces, it
  runs up the rising flank and measures roughly `gl` for any shape. The same
  helper in `cool_frames/numpy/filters/_gabfilters.py` had this fixed in the
  second pass; the copy under `numpy/phase/` kept it, because the two are
  private, near-identical and nothing compared them. Windows passed by *name*
  were never affected — they return the tabulated constant and never enter the
  search — so no existing caller could see it.

  Fixed, and `test_findgamma.py` now holds the two copies to each other across
  eight window shapes at four heights.

  Still open, and pinned in both directions: with the ordering corrected the
  search is 7-45 % above the table, because `_findbestgauss` returns the top of
  its own hardcoded search range for four of the five tabulated windows. The
  test asserts the ratio stays in `[1.0, 1.5)`, so neither a regression to the
  6-10x error nor a genuine improvement can pass silently.

- **Every regression test protecting a torch fix ran in neither CI job.** The
  torch cases in `tests/regressions/` are guarded with
  `pytest.importorskip("torch")`, so they skip in the numpy job for want of
  torch — and the torch job collected only `tests/torch_backend/`. Green
  locally, absent upstream, and their coverage reached codecov from nowhere.
  Together with `codecov-action`'s default of `fail_ci_if_error: false`, which
  turns a dropped upload into a warning inside a green job, that is why the
  project read 76 % — the numpy job alone, to within a point — while the union
  of the two jobs measures **85.04 %**. A `codecov.yml` now declares both flags
  with carryforward and an 80 % target on project and patch.

- **The iterative phase-retrieval family had no test coverage at all.**
  `rtisila`, `gsrtisila`, `lertisila`, `legla`, `decolbfgs` and `spsi` — twelve
  implementations across the two backends — sat at 7-11 %, meaning imports ran
  and algorithms did not. All twelve are correct; `test_phase_retrieval_family.py`
  now exercises them, asserting **consistency** rather than `magnitudeerr`,
  which every routine in this family satisfies by construction and on which zero
  phase scores a perfect `-inf dB`.

- Two reporting defects found there, **pinned rather than fixed**, because both
  are changes to a public return contract and the call is yours:
  `relres` from the RTISIL family is computed after the magnitude projection and
  is therefore ~1e-16 unconditionally — reported alongside an actual consistency
  of -13 dB, while GLA honestly reports 0.249 for the same quality — and
  `wpghi_findgamma` accepts a `tfr` argument it never reads. See
  `DEFECT_REGISTER.md`.

### `waveletfilters` shipped a bank that could not be inverted

`waveletfilters` defaulted to `painless=False`. The resulting bank sat 22x
over its painless limit, so its frame operator was not diagonal in frequency
and the diagonal dual that `filterbankdual` returns — the only dual this
package computes in closed form — was not a dual at all. A white-noise round
trip lost **75 %** of the signal. `filterbanktight` failed the same way, and
even `ifilterbankiter` left 12 % after 400 CG iterations.

The bank looked healthy while doing it. `filterbankbounds` reported
`A = 1.658`, `kappa = 2.276`; `filterbankbounds_svd`, the exact eigenvalue
oracle, reported `A = 0`. Both were right about their own question:
`filterbankresponse` computes the *diagonal entry* of the frame operator
correctly, and on this bank the operator carried 45 % of its mass off the
diagonal.

Changes:

- **`painless` now defaults to `True`.** The default bank round-trips at
  4.9e-16, matching `audfilters` (5.1e-16) and `cqtfilters` (5.5e-16). It
  costs redundancy — 1.39 to 8.98 at `fs = 8000`, `Ls = 4096`, 64 geometric
  scales — and `painless=False` still gives you the cheap analysis-only bank,
  now with a warning saying what it cannot do.

- **`painless` is honoured by every sampling mode.** It used to apply to
  `regsampling` only and be silently ignored by `uniform`, `fractional` and
  `fractionaluniform`, which failed to reconstruct (0.09, 0.82, 0.80). All
  four are now at 4.4e-16.

- **The DC and Nyquist complements get their hops capped too.** They are
  appended after the hops are chosen and simply inherited one. On a sparse
  scale set (24 scales at 6/octave) the Nyquist complement — wide by
  construction, because it spans everything the wavelets left uncovered —
  arrived 1113 bins wide on a hop of 6, `aW/L = 3.87`. One channel over the
  limit was enough to put the exact lower bound at 0 while the estimator
  reported `kappa = 4.4`. Lowering a hop is a safe local repair; the response
  is rescaled by `sqrt(a_new/a_old)` to keep the `scal = sqrt(a)` convention.

- **`info["painless"]` and `info["painless_ratio"]`** report the condition
  measured on the bank actually returned, after the complements are appended
  and after `redtar` has rewritten the hops — the two places where a cap
  applied mid-design stops being a guarantee. A violation warns at
  construction, where the parameters were chosen.

- **`info["admissible"]` no longer returns `None` for layouts the closed-form
  predictor cannot express** (`lowpass='none'`/`'repeat'`, complex banks).
  `lowpass='none'` leaves 79 DFT bins uncovered at the defaults and used to
  report `None`, which reads as "fine". Those layouts now fall back to
  measuring the realised response and report `source='measured'`.

Why no test caught it: `tests/layer1_filters/unit/test_waveletfilters.py`
had a `TestReconstruction` class whose docstring documented the defect as
expected behaviour and then declined to test it — "waveletfilters ... does NOT
form a painless frame. We therefore test reconstruction with cqtfilters and
audfilters." There was no reconstruction test for `waveletfilters` at all. The
`cqtfilters` case that stayed behind asserted `rel_err < 1.0`, so a 100 %
relative error passed; it measures 4.7e-16. And no test anywhere compared
`filterbankbounds` against `filterbankbounds_svd`, which is the one comparison
that makes a non-diagonal frame operator visible. All three are now fixed:
reconstruction tests for `waveletfilters` across all four sampling modes and
both entry points, a `< 1e-10` bound on the `cqtfilters` case, and an
estimator-vs-oracle test parametrised over four designers with the
non-painless bank kept as a negative control.


### LTFAT's `tfr` rule, recovered; and which phase-gradient path is right

**`info["tfr"]` on every painless designer.** LTFAT publishes `info.tfr` as an
opaque function handle and never states how it is built, so cool-frames'
designers exposed no equivalent and `filterbankconstphase`'s magnitude path had
no `sqtfr` to offer callers. Recovered from the exported references:

```
tfr[m] = 1/(2*winbw**2) * L / W[m]**2,    W[m] = fsupp[m] * L/fs
```

`W` is the channel's *designed* bandwidth in DFT bins and `winbw` the prototype
window's equivalent-bandwidth factor (3/8 for the Hann, so the constant is
32/9). The form is the strong part: `tfr * W**2 / L` is constant to **4.2e-16**
across all 29 audfilters channels — over tfr values spanning 0.0298 to 719 —
and all 78 cqtfilters channels. The formula then reproduces LTFAT's own numbers
to a *uniform* 4.5e-5 and 2.1e-5 (median equal to max, so one scale factor, not
a per-channel error; that residual is the small `fsupp` convention difference).

This is **not** `L / comp_tfrfromwin(realised |H|)`, which gives a Hann constant
of 3.5288 instead of 3.5556 and is off by ~1 % per channel, varying. LTFAT uses
the design bandwidth, not the realised response.

`audfilters`, `cqtfilters`, `greenwoodfilters` and `warpedfilters` now publish
`info["tfr"]` and `info["tfr_source"]`. `warpedfilters` is the case that
motivated this: LTFAT returns no info struct for it at all — no `fc`, no `tfr` —
so no reference value exists and no cross-language check is possible, but it
designs its bandwidths by the same warped rule and windows them with the same
Hann, so the value is derived rather than left blank.

**`edge_mode` now defaults to `'ltfat'`.** The DC and Nyquist complements have
one frequency neighbour rather than two, and the two implementations scaled that
case differently — this package averaged and restored the interior's two-sided
scaling, LTFAT sums, so those channels came out exactly 2x apart. Which was
right could not be adjudicated, because the derivative-filter path that would
otherwise arbitrate does not reproduce the magnitude path under any gamma.

It can be adjudicated now: the signal path turns out to be **exact** against a
known instantaneous frequency (0.0 % on every tone probe, both designers), so it
is legitimate ground truth. Median wrapped |magnitude − signal| on the DC
channel, in Hz:

| designer | probe | `rescaled` | `ltfat` | ratio |
|---|---|---|---|---|
| audfilters | white noise | 6.39 | **3.99** | 1.60 |
| audfilters | sweep | 8.33 | **5.98** | 1.39 |
| audfilters | 3 tones | 8.02 | **5.77** | 1.39 |
| greenwoodfilters | white noise | 36.07 | **24.70** | 1.46 |
| cqtfilters | white noise | 22.67 | **11.62** | 1.95 |

Five of five, three designers, and interior channels identical to the last bit.
The torch backend tracks the same choice. The Nyquist complement does not
arbitrate and is excluded: both modes land 1200–4000 Hz from the signal path
there, on a scale where fs/2 = 4000, so the magnitude estimate is worthless on
that channel regardless — a separate limitation, now recorded.

**The signal/magnitude gap is a resolution effect, not a fixed bias.** This
revises the previous "structural, not a parameter choice" framing, which was
measured on one bank. Median inner-channel |magnitude − signal|, cqtfilters:

| bins/octave | M | sweep | white noise | 3 tones |
|---|---|---|---|---|
| 3 | 21 | 139.03 | 36.06 | 391.07 |
| 6 | 40 | 16.13 | 17.75 | 136.69 |
| 12 | 78 | 1.35 | 9.33 | 3.75 |
| 24 | 154 | 0.52 | 4.93 | 0.62 |
| 48 | 305 | **0.44** | 2.05 | **0.44** |

On a well-resolved bank the two paths agree to well under 1 Hz on tonal
material. Broadband input converges more slowly and does not reach that floor,
which is what an estimator assuming one dominant component per cell should do.
Spectral crowding matters too but less, and not monotonically in spacing: at
bins=12, one tone 0.61 Hz, two an octave apart 1.77, three semitones 0.80, one
semitone 1.47, white noise 9.33.

The same resolution story explains the `sqtfr` sensitivity documented earlier.
On the coarse audfilters bank a global 2x beats LTFAT's own `sqrt(info.tfr(L))`
(24.0 vs 78.4 Hz on the sweep); on the finer cqtfilters bank the two are within
15 % (1.15 vs 1.35). Gamma matters when the bank is under-resolved and stops
mattering when it is not — so `2x` beating the derived value on
`greenwoodfilters` (20.6 vs 54.4) is the *same* effect seen where the tfr is
known-correct, not evidence that the derived value is wrong. That is the closest
thing to validation available for a designer the reference gives nothing for.

**Fixed while doing this:** `tfr` came out `nan` on exactly the DC and Nyquist
channels of `audfilters` and `greenwoodfilters`, whose complements carry their
bandwidth in `fsupp_dc`/`fsupp_nyq` and store `0` in `fsupp`. `sqrt(info["tfr"])`
would have poisoned every coefficient of the magnitude path. All four designers
now return finite `tfr` on every channel.

**What none of this licenses:** any claim that magnitude-only gradient
estimation is intrinsically limited. Exactly one algorithm was tested, and two
faithful implementations of one algorithm agreeing is evidence about the port,
not about the method class.


## 0.1.0

First public release.
