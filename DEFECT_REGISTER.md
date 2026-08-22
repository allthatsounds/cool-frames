# Defect register — cool-frames audit, 2026-08-21

A systematic audit of the four subsystems outside the phase family turned up
**~40 reproduced defects**. Everything below was verified numerically before
being recorded; nothing here is a code-reading guess.

This file tracks what was fixed in v0.1.1 and what is still open, so the
remainder does not have to be rediscovered.

Every defect the audit recorded is now fixed, as are the backend API
divergences found afterwards. Nothing is open.

Legend: **FIXED** — corrected and covered by the test suite · **OPEN** — verified,
not yet fixed.

---

## Fixed in v0.1.1

### Silently wrong results

| # | Area | Defect | Was → is |
|---|---|---|---|
| 1 | `filterbanks/_frame.py` | `filterbankdual`/`filterbanktight` returned an **all-zero bank** for any FIR filterbank (the `else` fallback emitted the zero filter), so `ifilterbank` reconstructed exactly `0.0` while `filterbankbounds` still reported a valid frame | silent 100 % error → explicit `ValueError` naming the channel, explaining that a time-limited filter is full-band and so can never satisfy the painless condition, and pointing at `ifilterbankiter` |
| 2 | `filterbanks/_utils.py` | FIR synthesis read the filter `offset` into `skip` and **never used it**, so synthesis was not the adjoint of analysis. The "frame operator" came out non-symmetric (symmetry error 1.33) with negative eigenvalues, and CG diverged (relres 4.6e12) | adjointness error 8e-15; CG on a κ=99 FIR bank now converges to 2e-4 |
| 3 | `filterbanks/_utils.py` | Same block: `np.atleast_2d` turned a 1-D `(N,)` coefficient array into a `(1, N)` **row**, so one scalar was broadcast across every bin | 1-D and 2-D input now agree exactly (was 59.3 apart) |
| 4 | `torch/filterbanks/_core.py` | `ifilterbank` **returned early** whenever any FIR channel was present, discarding every band-limited and full-length channel and skipping the real-mode fold | 125 % error vs NumPy → 5.4e-16 |
| 5 | `torch/core/_core.py` | `comp_ifilterbank_td` used **zero padding** where NumPy uses periodic extension | 31.5 % backend disagreement → 5.7e-16 |
| 6 | `torch/core/_core.py` | `comp_filterbank_td` truncated the convolution to `L` **before** the offset slice, returning too few coefficients for any filter with a non-zero offset (59 vs 64 at a=1) | lengths now match NumPy exactly |
| 7 | `torch/filterbanks/_core.py` | FIR filters were **unusable with a real signal** — `conv1d` rejects mixed real/complex operands, raising `RuntimeError` for every real input | signal promoted to the working complex dtype |
| 8 | `numpy/diagnostics/spectrogram.py` | `reassigned_spectrogram` called `filterbankphasegrad(c, a, fc)` — coefficients as signal, hops as filters, centre frequencies as hops — and unpacked a 4-tuple into two names. A bare `except Exception` swallowed it, so both documented outputs were **identically zero for every input**. The two summaries were also mapped to the wrong names | correct signature, no bare except, real gradients; `tgrad`/`fgrad` mapped the right way round |
| 9 | `numpy/diagnostics/spectrogram.py` | Channel padding used **0.0 dB (magnitude 1.0)**, and `peak_db` was taken over the padded array. On a typical ERB bank ~88 % of the image is padding, so for a quiet input the padding *was* the peak and the dynamic-range window sat ~30 dB too high | padded at the display floor; peak taken over real coefficients only. Scaling the input now shifts the image by a constant, as it must |
| 10 | `numpy/operators/_framemul.py` | `framemuleigs` **silently symmetrised** the operator (`0.5*(mat + mat.T)`, then `eigvalsh`/`eigsh`) regardless of the symbol, so a complex symbol gave a leading eigenvalue 10 % off and reported genuinely complex eigenvalues as real numbers of the wrong sign | self-adjointness detected; general solver used otherwise. Exact (0.0 error) on a complex symbol |
| 11 | `torch/operators/_framemul.py` | Every entry point forced **float32**, so float64 calls matched NumPy only to ~1.8e-07 and `torch.autograd.gradcheck` failed | dtype follows the caller; parity now 5.3e-16, gradients finite |
| 12 | `filters/_gabfilters.py` | Omitted the **1/√2 edge scaling** every other designer applies, so the DC and Nyquist channels were double-counted by the real-mode fold: a 4×-overlap Hann DGT — an exactly tight frame — read κ = 1.667 with a 67 % response spike | κ = 1.000001 |
| 13 | `filters/_gabfilters.py` | A code comment asserted the truncation "enables the painless dual condition". It does not — painlessness needs `M² ≤ 4L`, false for every default lattice — and unlike every other designer no warning was emitted | comment corrected; warns like the others, and points at `ifilterbankiter` (which reconstructs to 7.8e-14 in 3 iterations) |
| 14 | `filters/_filters.py` | `biquadfilter` fell back to `phi = 0.0` / `rho = 0.0` at DC, Nyquist and `r ≤ 0` — the sigmoid's **midpoint**, not its limit. Rebuilding a DC resonator from its own stored ML parameters moved it to fs/4 | logit clamped into the open interval; round trip faithful at all endpoints |
| 15 | `filters/_freqwin.py` | `freqwin('butterworth')` used the half-power convention while `gauss`/`roex`/`gammatone` in the same function solve for `bwrelheight`, making it **14 % too wide** | cutoff rescaled; all four now agree |
| 16 | `torch_additions/recipes/audio_denoising.py` | `denoise` analysed with `g` and synthesised with `filterbanktight(g)` — not a round trip | residual 28.7 (relative) → 1.8e-11 |
| 17 | `filterbanks/_core.py` | The `'pcg'` preconditioner applied a **frequency-domain** diagonal to a **time-domain** residual. Not a preconditioner — it consistently *slowed* convergence (9 → 15 iterations where the correct one takes 4) | applied in the Fourier domain, where the frame operator is actually diagonal |
| 18 | `filterbanks/_core.py` | `ifilterbankiter` reported the residual of the **complex CG iterate** while returning `np.real(x)` — "converged" at 3.2e-07 for a signal whose true residual was 0.226 | measured on the value actually returned |
| 19 | `filterbanks/_utils.py` | `filterbankwin`: `'dual'` ≡ `'realdual'` and `'tight'` ≡ `'realtight'` — all four branches called the same function with the default `real=True`, so the complex variants were **unreachable** | each branch passes the right `real=` |
| 20 | `numpy/operators/_framemul.py` | `framemulinv(maxit=0)` raised `UnboundLocalError` from its own return statement | `k` bound up front |
| 21 | `torch/operators/_framemul.py` | `framemulinv` always reported `'iter': 1` (the loop bound `_k`, the return read `k`) | reports the real count |
| 22 | `numpy/diagnostics/recommend_filterbank.py` | The "CQT bins **from f0**" expression contained no `f0` and evaluated to 12 for every input, making the `f0 > 500` branch dead and a detected f0 *coarser* than the no-f0 default of 48 | resolution derived from the harmonic-spacing argument the comment describes |

### Also fixed

- **mypy**: three errors at the CI-pinned version (1.20.2) — the repo's `main`
  was failing its own gate. One was an inert `# type: ignore` that was not the
  first comment on its line, so it suppressed nothing; the `Optional` is now
  narrowed properly instead.
- **Sphinx**: the docs build is warning-free. `FilterbankRecommendation`
  documented its fields both in an `Attributes` block and as dataclass fields,
  producing a duplicate-object warning for each.
- **torch dtype propagation** (the item left open in the previous pass): a
  float32 input now gives complex64/float32 out, computed in single precision —
  memory exactly halves and accuracy lands at float32 epsilon, not a cosmetic
  cast. See `cool_frames/torch/_dtypes.py`.

---

## Fixed in the second pass

Everything the first pass recorded as open, except where noted.

| # | Area | Defect | Was → is |
|---|---|---|---|
| 23 | `numpy/operators/_framemul.py` | `framemul` applied `np.real(...)` unconditionally, so the operator was only R-linear even with `real=False`, while `framemulappr`'s `real=False` branch models a C-linear generator | symbol 44.5 % off in HS norm → HS round trip 5.1e-15 |
| 24 | `torch/operators/_framemul.py` | `framemulappr` was a different algorithm: built its synthesis matrix by *analysing* with `g_synthesis`, only ever did the diagonal approximation, and never read `real` | HS error 1.033 (worse than a zero symbol) → delegates to the validated NumPy implementation; exact parity, `real`/`method`/`max_gram`/`rcond` all live |
| 25 | `torch/filterbanks/_frame.py` | `filterbankiter` sliced the CG iterate to `Ls`, making the map a projection rather than `F*F` | diverged at relres 399.7 when `Ls < L` → matches NumPy to 5e-16 at both `Ls < L` and `Ls == L` |
| 26 | `torch_additions/recipes/magnitude_to_audio.py` | `reconstruct`'s `'pghi'` and `'spsi'` were both random phase, bitwise identical, both reporting `converged: True`; the GLA calls took the default `real=False` on a single-sided bank | `'spsi'` is now really SPSI, `'legla'` added, `'pghi'` wired to the fixed magnitude path (see J1); the `real=` fix alone moved GLA from −6.0 dB to −31.3 dB and fGLA to −52.8 dB, and `real=` is now derived from the filters rather than hardcoded (J2) |
| 27 | `filters/_warpedfilters_design.py` | `freqrange='complex'` crashed on `regsampling`/`uniform` (`np.vstack` on a 1-D `a`) | all three sampling modes run |
| 28 | `filters/_waveletfilters.py` | `redtar` was inert on every non-uniform mode (`N_new` computed, `N_old` used) and produced `a = 0` on `uniform` | redundancy now responds monotonically on both modes; hops clamped at 1 |
| 29 | `filterbanks/_analysis.py` | `analyze_*` omitted `L` on every internal call, inflating redundancy by `L_actual/L_requested` and firing a false "not painless" note | 3.63 → 2.15 (the true value), note gone |
| 30 | `filterbanks/_analysis.py` | `analyze_coefficients` read a stacked `(N, M)` array as `(M, N)`, computing every per-channel statistic over the wrong axis | M reported as 16 for a 23-channel bank → correct |
| 31 | `filterbanks/_core.py`, torch equivalent | The convention-mismatch detector measured the *reconstructed spectrum*, where single- and two-sided banks overlap (0.383 vs 0.28–1.0), so `audfilters` slipped past and reconstructed with 46 % error in silence | measured on the *filters*, where the families separate 0.007–0.122 vs 0.590–1.000 — a factor-of-five margin |
| 32 | `filterbanks/_frame.py` | `filterbankdual`/`filterbanktight` never checked the painless condition, although `filterbankwin` had always computed `ispainless` and nothing read it | warns, naming the offending channels; silent for banks that do satisfy it |
| 33 | `filterbanks/_frame.py` | `filterbankbounds_svd` returned raw eigenvalue extremes, so `A` could be negative (−9.7e-16), giving "condition numbers" like −3.2e+15 | clamped at 0 |
| 34 | `filters/_gabfilters.py` | `_winwidthatheight` assumed the peak at index 0, but the designers store peak-*centred* responses, so the shape term collapsed to a constant: a Hann, a rect, a triangle and a 3-bin needle all gave `gamma = 11597.8` | rolls the peak to 0 first; needle 4.5 < hann 2834 < rect 11598, and DFT-ordered input is unchanged |
| 35 | `torch/filterbanks/_core.py` | Full-length filters with fractional hops were misclassified (torch tested only `len(H) == L`; NumPy also requires an integer hop) | raised `RuntimeError` → matches NumPy |
| 36 | `filters/_waveletfilters.py` | The `delay` loop stopped at `lp_num`, excluding the appended complement highpass | `[5,5,5,5,5,5,5,0]` → every channel delayed |
| 37 | `filterbanks/_frame.py` | `filterbankfreqz`'s `a` was documented as a parameter and never read | now explicitly optional and documented as ignored |

## Closed after the first pass — the three judgement calls

These were recorded as open (or worked around) in the first v0.1.1 pass and
have since been closed properly. They are kept here because the *reasoning*
matters more than the diffs.

### J1. The PGHI magnitude path — root-caused and fixed

`filterbankconstphase`'s *signal* path was always excellent (consistency
0.0047 on an ERB bank where the zero-phase baseline is 0.385). Its *magnitude*
path returned 0.385 without `sqtfr` and 0.393 with it — at or worse than doing
nothing — so `reconstruct(method='pghi')` was made to raise.

Diagnosis, in the order the evidence arrived:

- Feeding the **true** gradients into the magnitude path's integrator gave
  0.0031–0.0047, matching the signal path → **the heap integrator was never
  broken**.
- A least-squares fit of a single scalar α against the true gradients gave
  residuals of 0.906/0.759 *at every* `tfr` scale → the estimate was
  **shape**-wrong, not scale-wrong.
- Per-channel comparison against a pure tone settled it: the true `tgrad` for a
  440 Hz tone at fs = 4000 is 0.2201 ≈ 440/4000·2, i.e. the **absolute**
  normalised instantaneous frequency, while the estimator was returning values
  around ∓0.002–0.17 — the *deviation* from each channel's centre frequency.

**Root cause:** `comp_filterbankphasegradfrommag` returned the deviation; the
integrator (and `filterbankphasegrad`) use the absolute value. The
centre-frequency term is an order of magnitude larger than the deviation it
was carrying, so omitting it did not degrade the estimate — it replaced it.

Two smaller defects fell out of the same investigation:

- `fc` was being passed in the `fc/fs` convention where the integrator wants
  `fc/fs·2`, a silent factor of two on the estimated gradient.
- Edge channels summed their one available side against a denominator that
  defaulted to `1.0` instead of the frequency spacing, leaving channel 0 and
  channel M−1 both half-scaled and divided by the wrong quantity.

**A correction to an earlier entry in this file.** The first pass recorded that
the magnitude path's fallback — inferring the sampling rate as `max(fc) * 2`
when given Hz without `fs` — was "true of no filterbank in the package", and
replaced it with a hard `ValueError`. That was **backwards**, and measuring it
afterwards showed why: the inference amounts to assuming the top channel sits
at Nyquist, and that is *exact* for every single-sided designer here —
`audfilters`, `cqtfilters` (which appends a Nyquist channel whatever `fmax` is
set to) and `gabfilters(real=True)` all put their last channel at fs/2. It is
wrong by about a factor of two only for a two-sided bank, whose channels run
past Nyquist, and `fc` alone cannot distinguish the two cases.

So the hard error was the wrong trade: it broke every existing caller to buy
nothing on the path they were actually on. The inference is restored, and what
was actually wrong with it — the silence — is fixed instead: it now warns,
naming the assumption and the case it fails in. Passing `fs` skips it entirely
and is bitwise identical on a bank where the assumption holds.

**A hypothesis that was wrong, recorded so it is not re-tried:** the two
one-sided difference quotients are *summed*, which looks like a bug — they each
estimate the same derivative, so averaging seems right. It is not. Averaging
was tested directly and, on the three chirp fixtures (the only ones with a
large, smooth, well-resolved frequency deviation, i.e. the case this
correction exists for), the consistency optimum sits sharply at the summed
scale. The stationary-tone fixtures do prefer a halved correction, but that is
noise-shrinkage on signals whose true deviation is ≈ 0 and carries no
information about the correct scale. The estimator is left unbiased.

Result — consistency, magnitude-only, against the zero-phase baseline:

| fixture | zero phase | PGHI from magnitude | signal path (true gradients) |
|---|---|---|---|
| chirp 200–800 Hz | 0.711 | **0.044** (16.1×) | 0.044 |
| chirp 300–1500 Hz | 0.767 | **0.040** (19.3×) | 0.041 |
| 8 kHz chirp | 0.795 | **0.063** (12.6×) | 0.036 |
| two sines | 0.385 | **0.170** (2.3×) | 0.005 |
| impulse + tone | 0.447 | **0.163** (2.7×) | 0.148 |
| white noise | 0.669 | **0.324** (2.1×) | 0.323 |

Every fixture improves. On frequency-modulated input the magnitude-only
estimate is within a decibel of what the same integrator achieves from the
*true* gradients, and a single non-iterative pass beats 100 iterations of GLA
(−27.1 dB vs −19.6 dB). Magnitude-only PGHI remains weakest on stationary
tones over a coarse (23-channel ERB) bank, which is expected and is now
documented rather than hidden. `reconstruct(method='pghi')` is wired up.

### J2. `real=` is derived from the filters, not assumed

`reconstruct` hardcoded `real=True` at five call sites. Correct for every bank
the auditory and constant-Q designers produce, silently wrong (~30 dB, no
exception) for a genuinely two-sided one. It now calls the new public
`filterbanks.filterbank_is_real(g, a, L)` — a promotion of the detector that
already backed `ifilterbank`'s mismatch warning, whose two families separate by
a factor of five — with an explicit `real=` parameter to override it.

### J3. The lint scope is pinned

`ruff --fix` was run from the repo root rather than over the CI-linted set,
touching ~14 files nobody had agreed to change and mixing cosmetic import
shuffles into a behavioural patch. All of them are reverted and verified clean
(this pass also caught and restored `.github/workflows/docs.yml`, deleted in
the working tree, and four stray import reshuffles missed the first time).

The scope now lives in `[tool.ruff] include` in `pyproject.toml` and nowhere
else; CI runs bare `ruff check .` / `ruff format --check .` and inherits it. A
bare local invocation is now exactly the CI check, so the overreach is not
reachable by accident. A regression test asserts both halves.

## Closed in the third pass — the carried-over items

All twelve carried-over items are now closed. Three of them were not what the
register said they were, and those corrections are recorded here rather than
quietly dropped.

| # | Item | Outcome |
|---|---|---|
| 1 | `warpedfilters(min_win=)` inert | **Fixed** — both edge builders were called with a literal `min_win=1`. |
| 2 | `warpedfilters(freqrange='complex')` negative channels | **Fixed** — three separate defects; see below. |
| 3 | `analyze_filterbank` hardcodes fs = 8000 | **Misdiagnosis** — see below. Comment corrected, code unchanged. |
| 4 | `ifilterbank` ignores `Ls > L` | **Fixed** — now warns instead of silently returning `L` samples. |
| 5 | `filterbankiter(real=False)` diverges | **Fixed** — `real` is derived from the filters; same fix applied to the torch twin, and to `ifilterbankiter`, which had the identical defect and was not filed. |
| 6 | `torch.filters.hopfilters` always raises | **Fixed** — removed; there is no NumPy `hopfilters` to wrap. |
| 7 | `torch.filters.firwin` has no `device` | **Fixed** — takes `device` and `dtype` like every sibling. |
| 8 | Fixed output dtypes | **Fixed** — `dtype=` added to `filterbankresponse`, `filterbankfreqz`, `ifilterbankiter`; defaults unchanged. |
| 9 | `magresp`/`plotfft` two-sided axes | **Fixed** — both. |
| 10 | Stale doctests | **Fixed** — all 33, and they now run in CI. Two real defects fell out. |
| 11 | Dead code | **Fixed** — six pieces removed. |
| 12 | `realonly` inconsistency | **Fixed at the root** — `magresp` no longer keys off it. |

### Item 2 was three defects, not one

Each is silent, and each leaves the peak amplitudes intact, which is why
"does it look like a filter" checks never caught any of them:

1. `warpedfilters` computed a `symmetry` flag and dropped it. `warpedblfilter`
   took no such argument, so it could not be forwarded to
   `comp_warpedfreqresponse`/`comp_warpedfoff` — which have always accepted it.
   Every negative-fc channel was built by evaluating the warp below zero,
   outside its domain.
2. The mirrored branch of `comp_warpedfoff` had MATLAB's `+1` stripped from it
   as a 1-based indexing artifact. It is not one — it compensates for the `n-1`
   in the `H[::-1]` reversal — so every mirrored channel landed exactly one bin
   low. A shift search found a uniform −1 offset across all 32 mirror pairs.
3. The mirrored branch takes a deliberately wide window (~2B rather than the
   filter's own B−A) because the roll-and-reverse arithmetic needs it, and never
   trimmed back. The surplus is the aliased `win_hi` term the positive twin
   discards: channel −2321.6 Hz came out with 3334 nonzero bins and 4.5× the
   energy of its +2321.6 Hz twin.

With all three fixed, every negative channel is now **bitwise** the mirror of
its positive twin (relative L2 difference 0.0 across all 32 pairs, against a
median of 0.18 before).

*A blind alley worth recording:* narrowing the mirrored truncation window to
B−A looks obviously right and is wrong — it truncates the energy away entirely
and takes the mirror error from 0.10 to 1.00. The wide window is load-bearing.

### Item 3 was a misdiagnosis

The register recorded that `analyze_filterbank` "hard-codes fs = 8000 for its
probe signal … so on a 4 kHz bank its 2500 Hz tone is above Nyquist". That is
wrong. `t` is a *sample index*, so `440 * t / 8000` is a digital frequency of
0.055 cycles/sample: the three tones sit at 0.110, 0.250 and 0.625 of Nyquist
whatever the bank's real sampling rate is, and the highest is at 0.3125
cycles/sample, comfortably below the 0.5 that would alias. The probe is
deliberately scale-invariant.

Rewriting it in Hz against the filters' own `fs` was tried and is numerically
identical to 5e-14, so the code is unchanged and only the comment is — plus a
regression test pinning the property, because the expression *reads* like a bug
and the next person to reach for the obvious fix should be told why not.

### Two defects found while fixing item 10

- **`firwin`'s `norm='energy'` did not normalise.** `_apply_norm` multiplied by
  `sqrt(M)` for `'energy'`/`'2'` and by `M` for `'1'`/`'area'`, so a "unit
  energy" Hann window of length 512 had an L2 norm of 313.5. This contradicted
  `core._norm.normalize_window`, the public `setnorm` and
  `_warpedfilters._setnorm` — all of which divide — and LTFAT. `firwin`'s
  default is `'inf'`, which was always right, so the damage was confined to
  explicit callers; `gabfilters` is the one in-tree caller that asks for energy.
  Reconstruction and conditioning are unaffected (the dual scales inversely).

- **`gm["H"](L)` assumed a callable.** A descriptor's `'H'` may be a
  `callable(L)` or an already-materialised array; the designers produce the
  former, `prepare_filters` and the torch wrappers the latter. Eight call sites
  across the phase modules called it unconditionally, raising `TypeError:
  'numpy.ndarray' object is not callable` for every materialised bank — in
  effect for the whole torch backend whenever `fc` was not passed explicitly.
  The check now lives in one helper, `filters._hval`.

### And one on the analysis side

**`ifilterbankiter` had `filterbankiter`'s defect too**, on the synthesis side,
and was not filed: the documented `real=False` default reconstructed the
flagship `audfilters` bank with **23 % error** where the correct mode reaches
4.5e-16. Found by reading the two signatures next to each other.

## Closed in the fourth pass — backend API divergences

Found by diffing the two backends' signatures against each other rather than by
anything failing. That comparison was worth doing on its own account: it turned
up a **fourth** copy of the `real=False` default defect, in
`torch.ifilterbankiter`, which had survived three separate rounds of fixing its
siblings and was still reconstructing the flagship bank with 23 % error.

| Function | Was | Now |
|---|---|---|
| `filterbankconstphase` | NumPy returned `list[array]`; torch returned `(list[Tensor], Tensor)` | Both return `(coeffs, usedmask)`, with `usedmask` a per-channel boolean list in both backends |
| `filterbankconstphase` | `sqtfr`, `fs`, `rng` NumPy-only | forwarded by the torch wrapper, so magnitude-path PGHI and reproducible phase work from either backend |
| `filterbankbounds` | `return_kappa` NumPy-only | added to torch |
| `torch.ifilterbankiter` | `real=False`, output hardcoded to float64 | `real` derived from the filters; output follows the coefficients' dtype |

The return-type change is **breaking** and is the fifth such change recorded in
this release. It was chosen over the alternatives because it is the only option
that makes the NumPy backend agree with three things it already disagreed with:
the torch backend, LTFAT (which returns `usedmask` too), and its own
`-> tuple` annotation, which was a lie silenced by a `# type: ignore`. It also
restores the `usedmask` that was being computed and thrown away — the record of
which coefficients received integrated phase rather than random fill, i.e. of
which part of the answer means anything.

**Deliberately *not* aligned**, so nobody "fixes" them:

- `device`/`dtype` on `filterbankdual`, `filterbanktight`, `filterbankscale`,
  and the `L` that `filterbankscale` needs to materialise filters, are
  torch-only because only torch has devices to place things on and lengths to
  materialise at.
- `dtype` on `filterbankresponse`, `filterbankfreqz` and `ifilterbankiter` is
  NumPy-only because the NumPy backend is a float64 reference implementation
  and needs an explicit opt-out, whereas the torch backend already follows its
  input's dtype. A redundant `dtype=` on torch would be noise.

## Minor, recorded for completeness

- `filterbankconstphase`'s `usedmask` — computed and discarded since v0.1.0 —
  is now returned, as part of the return-type alignment above.

## Verified clean

Worth recording, so the same ground is not re-covered:

- Perfect reconstruction on the painless path — `audfilters` 5.2e-16,
  `cqtfilters` 4.5e-16, `greenwoodfilters` 4.8e-16 — and tight-frame round trips
  at the same level.
- Analysis linearity (5.9e-14), shift covariance (3.7e-15), Parseval/energy
  relations, `pack`/`unpack` round trip (exact), `filterbankscale`.
- Adjointness of the frequency-domain path (ratio 1.0 + 1e-15j).
- `partial_tighten` reaches κ = 1.000000 exactly for `audfilters` and `cqtfilters`.
- `sigproc.thresh`: all three modes exact against their definitions, real and
  complex, including the `largest` fraction and count modes.
- `framemul`/`framemuladj` adjointness (1.3e-15); `framemulinv` residual honest;
  `framemulappr(real=True)` provably HS-optimal against a brute-force `lstsq`.
- No inert parameters in `audfilters`, `cqtfilters`, `greenwoodfilters` —
  every documented keyword measurably changes the output.
- torch/NumPy parity ≤ 1.7e-14 for the frequency-domain paths across five
  designers × four sampling modes, mono and multichannel, `stack=True`;
  `gradcheck` passes.
- `torch.filters._biquad` matches NumPy to ≤ 1.4e-14 for every norm, and
  gradients reach the `rho`/`phi` pole parameters.
