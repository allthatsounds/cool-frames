# cool-frames — Mathematical Reference

## 0  Mission and Scope

**cool-frames** is the reference toolbox
for audio time-frequency analysis with mathematically guaranteed
properties. Every representation is invertible, well-conditioned, and
differentiable, so it can serve as the foundation for both classical
signal processing and learned audio models.

The mathematical properties (frame bounds, conditioning, gradient flow)
are not ends in themselves — they exist because they make audio analysis
and synthesis work reliably. The goal is computational listening: the
ability to analyse, process, and reconstruct audio better than any other
toolbox in this field. At a later stage, the same machinery should
support computational orchestration (sound generation).

**Three-layer architecture:**

1. **Representation** — Build the TF representation. Filterbanks,
   frame theory, analysis/synthesis, frame bounds, tight/dual frames,
   arbitrary filter specification (FIR, band-limited, and future IIR
   pole-zero). The representation is always invertible and its
   conditioning is always known.

2. **TF operations** — Transform, refine, or invert TF representations.
   Phase retrieval, reassignment, synchrosqueezing, Gabor/frame
   multipliers. These are operators in the TF domain whose properties
   derive from the frame theory.

3. **Recipes** — Specific audio tasks built on layers 1 and 2:
   tonal/transient separation, denoising, feature extraction.

**In scope:** anything that serves invertible, well-conditioned TF
analysis of audio — including multiwindow/multitaper frames, adaptive
representations, pole-zero filters, Cohen's-class distributions (when
invertible), and differentiable pipelines for learned models.

**Out of scope:** scattering transforms (a cascade of modulus operators —
different paradigm, already served by kymatio), empirical mode
decomposition (data-driven, not frame-theoretic). Both can use cool-frames
filterbanks as a substrate.

---

## 0a  Frame Operator, Hessian, and Optimisation Geometry

The frame operator S = D*D (analysis followed by synthesis) has the
same structure as the Gram matrix W^T W of a linear layer in a neural
network. The frame bounds A and B are the extremal eigenvalues of S.
This structural identity has practical consequences for any
differentiable pipeline that uses a cool-frames filterbank as its front-end.

**Condition number and gradient flow.** The condition number kappa = B/A
of the frame operator controls gradient propagation. When kappa = 1
(tight frame), the analysis operator preserves all directions equally —
no gradients are amplified or suppressed during backpropagation. This
is the signal-processing equivalent of orthogonal initialisation.

**Hessian of reconstruction loss.** When differentiating through an
analysis-synthesis pipeline, the Jacobian of the analysis operator is D,
and the Hessian of the reconstruction loss involves S. Frame bounds
directly characterise the optimisation landscape: A > 0 guarantees no
spurious local minima; B/A close to 1 means well-conditioned gradients.

**Tight frames as preconditioning.** Applying `filterbankrealtight`
before backpropagation is equivalent to preconditioning the optimiser —
it makes S = I so the Hessian of the reconstruction term is isotropic.

**Phase retrieval Hessian.** The Hessian of the phase retrieval
objective involves both S and the current point in the search space.
Good frame bounds are necessary for PGHI convergence. The second-order
phase derivatives (tt, ff, tf) computed by `filterbankphasederiv` are
curvature of the phase surface in the TF plane — another Hessian, but
in signal space rather than parameter space.

**Gabor multipliers.** A Gabor multiplier M_sigma = D* diag(sigma) D
(analyse, multiply by a TF symbol sigma, synthesise) is the fundamental
operator for time-varying filtering. Its approximation quality is
controlled by the frame bounds: for a tight frame, M_sigma is
optimally localised. The eigenvalues of M_sigma are bounded by
A * min(sigma) <= lambda <= B * max(sigma). This makes frame
multipliers the natural bridge between frame theory and audio
processing — and their well-posedness depends on the same quantities
that control gradient flow.

---

A concise summary of the mathematics, conventions, and algorithmic tricks
used in the NumPy (and PyTorch) implementations. Each section gives the
key formulas and points to the standard references; the goal is that a
reader can look up any term they need.

---

## 1  DFT and Signal Conventions

**DFT definition.** We use the NumPy/FFTW convention (unitary up to
scaling):

    X[k] = sum_{n=0}^{L-1} x[n] exp(-j 2 pi n k / L),    k = 0, ..., L-1

Inverse:

    x[n] = (1/L) sum_{k=0}^{L-1} X[k] exp(+j 2 pi n k / L)

**Frequency bin layout.** Bin 0 is DC; bins 1 to L/2 are positive
frequencies; bins L/2+1 to L-1 are negative frequencies (aliased).
`fftindex(L)` returns the signed index vector [0, 1, ..., L/2, -L/2+1,
..., -1].

**Centred modulo.** `modcent(x, m) = ((x + m/2) mod m) - m/2`, mapping
into (-m/2, m/2]. Used throughout for wrapping frequencies to the
principal range.

**Involution.** For Hermitian-symmetric (real-signal) handling:

    involute(f)[0] = conj(f[0])
    involute(f)[n] = conj(f[L - n])   for n = 1, ..., L-1

This mirrors the spectrum for real-valued filterbanks.

**Signal shape.** Mono signals have shape (L,); multichannel signals have
shape (L, W). Internally, mono is promoted to (L, 1) and squeezed back
on output.

---

## 2  Filterbank Analysis and Synthesis

**Analysis.** Given M filters g_m with hop sizes a_m, the filterbank
coefficients of a signal f of length L are:

    c_m[n] = sum_{l=0}^{L-1} f[l] conj(g_m[l - n a_m]) exp(-j 2 pi l k / M)

In the frequency domain this becomes pointwise multiplication:

    C_m = IFFT( F . G_m ) / a_m

where F = FFT(f) and G_m = FFT(g_m), followed by decimation (keep every
a_m-th sample). The division by a_m maintains energy scaling.

**Synthesis.** Reconstruction accumulates each channel's contribution:

    F_hat[k] += sum_m  tile(FFT(c_m), a_m)[k] . conj(G_m[k])

where `tile` periodically repeats the short FFT by a factor of a_m.

**Three computation paths.** Filters are classified at runtime:

- **Full-length FFT** (`comp_filterbank_fft`): filter stored as length-L
  array. Direct pointwise multiply + IFFT + subsample.
- **Band-limited FFT** (`comp_filterbank_fftbl`): filter stored as short
  array with frequency offset `foff`. Only the nonzero DFT bins are
  touched. Key trick: overlap-add via reshape-and-sum instead of explicit
  circular convolution.
- **Time-domain FIR** (`comp_filterbank_td`): short impulse response h.
  Circular convolution via FFT, subsample with skip offset.

**Rational hop sizes.** Represented as (M, 2) integer arrays
[numerator, denominator]. The effective hop is a_frac = num/denom. Number
of time frames: N_m = round(L / a_frac_m). Transform length L is chosen
as the smallest multiple of lcm(all denominators) that is >= Ls.

---

## 3  Filter Representations

**Filter dictionary format.** Each filter g_m is a Python dict:

    { 'H':        callable(L) -> ndarray  or  ndarray,
      'foff':     callable(L) -> int      or  int,
      'realonly': 0 or 1,
      'delay':    int,
      'fs':       float or None }

For FIR filters, the alternative keys are:

    { 'h': ndarray, 'offset': int, 'delay': int, 'fs': float }

**Band-limited storage convention.** The transfer function H is stored
un-shifted (peak at index len(H)//2). The offset foff points to the DFT
bin where the array starts:

    foff(L) = round(L * fc_hz / fs) - len(H) // 2

This places the window peak exactly at the centre-frequency bin. Window
lengths are always odd to guarantee an unambiguous centre sample.

**Callable vs. precomputed.** H and foff may be callables taking L as
argument (lazy evaluation for parametric filters) or precomputed arrays
(for filters materialised at a specific L).

---

## 4  Frame Theory

**Frames.** A filterbank {g_m, a_m} forms a frame for C^L if there exist
constants 0 < A <= B < infinity such that for all f in C^L:

    A ||f||^2  <=  sum_m  sum_n |<f, T_{n a_m} g_m>|^2  <=  B ||f||^2

A is the lower frame bound, B the upper frame bound, kappa = B/A is the
condition number.

**Frame operator.** S f = sum_m sum_n <f, T_{n a_m} g_m> T_{n a_m} g_m.
For painless frames, S is diagonal in the DFT domain.

**Painless condition.** The frame operator is diagonal (and frame
bounds are trivially computable) when:

    supp(g_m) <= L / a_m    for all m

i.e., each filter's support in DFT bins does not exceed the number of
frequency bins per hop. Equivalently, bwmul <= 1 in our parameterisation.

**Frame response (diagonal of S).** Under the painless condition:

    resp[k] = sum_m |G_m(k)|^2 / a_m

Frame bounds: A = min_k resp[k], B = max_k resp[k].

**Dual frame.** The canonical dual filters satisfy g_d_m = S^{-1} g_m.
Under painless conditions:

    G_d_m[k] = G_m[k] / resp[k]

**Tight frame.** The canonical tight filters satisfy g_t_m = S^{-1/2} g_m:

    G_t_m[k] = G_m[k] / sqrt(resp[k])

A tight frame has A = B, giving energy-preserving analysis and trivial
reconstruction (synthesis with the same filters, no dual needed).

**Partial tightening.** For auditory/CQT filterbanks where edge filters
cause poor conditioning, partial tightening modifies only the edge
filters while leaving inner channels intact, interpolated by a parameter
alpha in [0, 1].

---

## 4a  Real-Valued Filterbanks and Folded Frame Response

Auditory filterbanks (`audfilters`, `cqtfilters`, etc.) analyse
real-valued signals. Their filters cover only the positive frequencies
DC to Nyquist; the negative-frequency half of the DFT is intentionally
empty. This distinction between the complex and real cases propagates
through every frame-theoretic operation and is a common source of
confusion.

**Complex (full-spectrum) frame response.** As defined in Section 4:

    resp[k] = sum_m |G_m(k)|^2 / a_m,   k = 0, ..., L-1

For a real filterbank the negative-frequency bins (L/2+1 to L-1) have
no filter coverage, so min_k resp[k] = 0 and A = 0. This does not
indicate a defect; it simply means the full-spectrum frame operator is
not invertible — which is expected, since the filterbank was never
designed to represent complex signals.

**Folded (real) frame response.** For real signals, every DFT bin k and
its mirror L-k carry conjugate-redundant information. The appropriate
frame response folds the two halves together:

    resp_real[k] = resp[k] + involute(resp)[k]
                 = resp[k] + resp[(L - k) mod L]

This is what `filterbankresponse(g, a, L, real=True)` computes, and it
is the diagonal of the *real* frame operator. For a well-designed
auditory filterbank with DC and Nyquist bookend filters, resp_real > 0
everywhere, giving A > 0 and a valid frame for real signals.

**Three bounds functions and when to use each:**

| Function               | Response used           | Bins examined  | Use case                          |
|------------------------|-------------------------|----------------|-----------------------------------|
| `filterbankbounds`     | real=False (unfolded)   | all L          | complex-valued filterbanks        |
| `filterbankrealbounds` | real=False, then [:L//2+1] | DC..Nyquist | quick check of positive-freq coverage |
| resp_real min/max      | real=True (folded)      | all L          | true real frame bounds            |

After `filterbankrealtight` normalisation, the *folded* response is
uniformly 1.0, confirming a tight frame for real signals. The *unfolded*
response at DC and Nyquist is 0.5 (because those bins have no
negative-frequency partner to fold with), and the full-spectrum lower
bound A remains 0 (empty negative-frequency region). Both are correct;
they answer different questions.

**Why `filterbankrealtight` uses the folded response.** The tight-frame
normalisation divides each filter by sqrt(resp_real):

    G_tight_m[k] = G_m[k] / sqrt(resp_real[k])

This ensures perfect reconstruction for *real* input signals. Using the
unfolded response would over-normalise by a factor of sqrt(2) at most
frequencies (where resp_real ≈ 2 * resp) and under-normalise at DC and
Nyquist (where resp_real = resp because the mirror bin is the same bin).

**Edge filters guarantee A > 0.** The `audfilters` function constructs a
DC lowpass (centred at 0 Hz) and a Nyquist highpass (centred at fs/2) as
complements of the inner-channel response (Section 7). These bookend
filters ensure that resp_real[k] > 0 at every bin, which is necessary
and sufficient for the real frame bounds A > 0.

---

## 5  Auditory Frequency Scales

All scale conversions are implemented as bijections Hz <-> scale units.

**ERB (Glasberg & Moore 1990):**

    erb(f)  = 9.2645 sign(f) ln(1 + |f| * 0.00437)
    f(erb)  = sign(erb) (exp(|erb| / 9.2645) - 1) / 0.00437

**ERB bandwidth at centre frequency (ANSI S1.11):**

    bw(fc) = 24.7 + fc / 9.265   [Hz]

**Bark (Zwicker & Terhardt):**

    bark(f) = 26.81 / (1 + 1960/|f|) - 0.53
    f(bark) = 1960 / (26.81/(bark + 0.53) - 1)

**Mel:**

    mel(f) = 2595 log10(1 + |f|/700)
    f(mel) = 700 (10^{mel/2595} - 1)

**Mel-1000:**

    mel1000(f) = (1000/ln 2) ln(1 + |f|/1000)
    f(mel1000) = 1000 (exp(mel1000 * ln 2 / 1000) - 1)

**Auditory spacing.** `audspace(fmin, fmax, n, scale)` generates n
frequencies linearly spaced on the chosen auditory scale: convert
endpoints to scale units, linspace, convert back to Hz.

---

## 6  Window Functions

All windows use DFT-symmetric convention: evaluated at
x = (n - M/2) / M for n = 0, ..., M-1 (range [-0.5, 0.5)), stored in
FFT order (DC at index 0; apply fftshift to centre).

**Hann:**          g[x] = 0.5 + 0.5 cos(2 pi x)
**Cosine/SqrtHann:** g[x] = max(0, cos(pi x))
**Hamming:**       g[x] = 0.54 + 0.46 cos(2 pi x)
**Blackman:**      g[x] = 0.42 + 0.5 cos(2 pi x) + 0.08 cos(4 pi x)

**Hann bandwidth.** The main-lobe bandwidth of a Hann window of length
M in normalised frequency is hann_winbw = 2/M. This is used to relate
window length to filter bandwidth.

**Un-shifted Hann (for direct filter constructor):**

    h[n] = 0.5 (1 - cos(2 pi n / (n-1))),   n = 0, ..., N-1

Peak at index N//2. Used in _make_direct_filter to avoid the fftshift
ambiguity for even-length windows.

---

## 7  Auditory Filterbank Design (audfilters)

1. **Frequency range:** fmin from first auditory spacing unit above 0;
   fmax = fs/2.
2. **Inner channels:** M centre frequencies linearly spaced on the
   chosen auditory scale.
3. **Bandwidth:** bw_m = bwmul * audfiltbw(fc_m, scale).
4. **Hop sizes:** determined by redundancy and bandwidth via
   a_m = floor23(L / (redmul * bw_bins_m)), where floor23 finds the
   largest integer <= n with only factors 2 and 3 (for FFT efficiency).
5. **Edge filters (DC, Nyquist):** complement strategy — designed so the
   total frame response including edges is flat:

       H_edge(k) = P(k) sqrt(S_max - S_inner(k))

   where S_inner is the frame response of the inner channels alone,
   S_max = max(S_inner), and P(k) is a Hann taper for smooth roll-off.

---

## 8  CQT Filterbank Design (cqtfilters)

**Constant-Q property:** the ratio fc/bw is constant across channels.
Centre frequencies are exponentially spaced (linear on log-frequency
scale) with `bins` channels per octave.

**Frequency spacing:**

    fc[m] = fmin * 2^{m / bins},   m = 0, ..., M-1

**Q factor:** Q = fc / (Qvar * bw), where Qvar depends on the design
variant.

**Sampling modes:** 'regsampling' (floor23-quantised hops),
'uniform' (single hop for all channels), 'fractional' (rational hops),
'fractionaluniform' (uniform rational).

Edge filters use the same complement strategy as audfilters.

---

## 9  Hop-Driven Filterbank Design (hopfilters)

Inverts the standard design flow: hop sizes a_m are the primary input;
bandwidths and constraints are derived.

**Bandwidth from hop:** B_m = bwmul * fs / a_m  [Hz], giving
supp_bins = round(L * bwmul / a_m) DFT bins.

**Five frame constraints on centre frequencies fc:**

1. *Adjacent overlap (necessary for A > 0):*
   fc[m+1] - fc[m] < (B[m] + B[m+1]) / 2

2. *Well-conditioned overlap (50% Hann overlap):*
   fc[m+1] - fc[m] <= (B[m] + B[m+1]) / 4

3. *DC edge coverage:* B_dc = 2 fc[1] (automatic by design)

4. *Nyquist edge coverage:* analogous

5. *Edge filter painless condition:*
   a_dc <= fs / (2 fc[1]),  a_nyq <= fs / (2 (fs/2 - fc[M]))

6. *Inner painless condition (sufficient):* bwmul <= 1

**Automatic edge hop reduction:** if the painless condition is violated
for edge filters, a_dc and a_nyq are reduced to
max(1, floor(fs / (2 fc[1]))).

**Frequency decimation factors b:** dual of hop sizes a. Conversion:
fc = cumsum(b); b[0] = fc[0], b[m] = fc[m] - fc[m-1]. The (a, b) pair
parameterises the NSGT lattice directly.

---

## 10  Wavelet Filterbank Design (waveletfilters)

Wavelets are constructed in the frequency domain via `freqwavelet` and
assembled into a filterbank by `waveletfilters`.

**Supported wavelet types:**

- **Cauchy/Morse:** H(y) = exp(-2 pi y^gamma + (order - j beta) ln y + C),
  where order = (alpha-1)/(2 gamma), C is a log-normalisation constant.
  Analytic (one-sided in frequency). Support boundaries found via
  Lambert W function.

- **Morlet:** H(y) = [exp(-0.5 (sigma - |y|)^2) -
  exp(-0.5 (sigma^2 + |y|^2))] / D, where D normalises the peak to 1
  and sigma controls the trade-off. Peak position found by fixed-point
  iteration.

- **Frequency B-spline (fbsp):** H(y) = B_n((|y| - 1) fb n/2 + n/2),
  where B_n is the cardinal B-spline of order n (1-5) and fb >= 2
  controls bandwidth.

- **Analytic spline, Complex spline:** variants of fbsp with analytic
  signal construction.

**Scale-to-frequency mapping:** centre frequency = basefc / scale.
Scales > 1 give wider (lower-frequency) wavelets.

---

## 11  Warped Filterbank Design (warpedfilters)

Achieves non-uniform frequency resolution by warping the frequency axis
through an auditory scale before applying a uniform window.

**Core idea:** evaluate a standard window function (e.g., Hann) not at
linearly-spaced DFT bins but at positions warped through
freqtoscale/scaletofreq:

    bins_warped = (freqtoscale(f) - freqtoscale(fc)) / bw

Then: H[k] = firwin_eval(wintype, bins_warped[k]).

**Nyquist mirror:** a second copy of the window is evaluated at
bins_hi = 2 * freqtoscale(fs/2) + bins_lo to handle wrap-around near
Nyquist.

**Frequency offset:** computed in the warped domain:

    foff = floor(scaletofreq(fcscale - 0.5 * bw) / fs * L)

---

## 12  Normalisation Conventions

Three normalisation modes are used consistently throughout:

- **Energy (L2, default):** scale so ||g||_2 = 1. For DFT-domain
  filters: multiply by sqrt(L). Preserves signal energy across the
  transform.

- **Area (L1):** scale so sum |g| = 1. Multiply by L.

- **Peak (Linf):** scale so max |g| = 1. No additional factor.

The convention `norm="energy"` is the default for all filter design
functions.

---

## 13  Phase Gradient Computation

**Instantaneous frequency (time gradient):**

    tgrad_m[n] = Re(c_d_m[n] conj(c_m[n])) / s_m[n] * 2/L

where c_d_m is the analysis with the frequency-derivative filter
H_cd[k] = H[k] * (-j 2 pi k / L), and
s_m[n] = max(|c_m[n]|^2, minlvl * max|c|^2) is the spectrogram with a
reliability floor. Clipped to [-2, 2].

**Group delay (frequency gradient):**

    fgrad_m[n] = Im(c_h_m[n] conj(c_m[n])) / s_m[n]

where c_h_m uses the time-derivative filter H_ch[k] = H[k] * (-j n),
with n being the absolute bin index.

**Phase gradient filters.** For each analysis filter g_m, two derivative
filters are constructed:

- g_cd: frequency-domain multiplication by -j 2 pi k / L (time
  derivative → instantaneous frequency)
- g_ch: frequency-domain multiplication by -j n (frequency derivative
  → group delay)

---

## 14  Phase Reconstruction (PGHI)

**Phase Gradient Heap Integration.** Reconstructs phase from magnitude-
only coefficients using the phase gradients tgrad, fgrad.

Algorithm:

1. Sort all time-frequency coefficients by magnitude (descending).
2. Process in order: for each (m, n), propagate phase to its 4
   neighbours (time +/- 1 in same channel, frequency +/- 1 to adjacent
   channels) if the neighbour hasn't been visited yet.
3. Phase propagation rules:
   - Time step: phi[m, n+1] = phi[m, n] + pi * tgrad[m, n]
   - Frequency step: phi[m+1, n] = phi[m, n] + fgrad[m, n]

**Cross-rate neighbour mapping.** For non-uniform hop sizes, the time
index in an adjacent channel is:

    n_adj = (n * a_m / a_{m+1}) mod N_{m+1}

**Gamma estimation.** The PGHI algorithm requires a window-specific
parameter gamma that relates the time and frequency spread of the
window. `pghi_findgamma` / `wpghi_findgamma` compute this from the
filter's frequency response.

---

## 15  Spectral Reassignment

Each coefficient (m, n) is relocated to a new position based on its
local phase gradients:

    target_freq = cfreq_m + tgrad_m[n]
    target_time = a_m * n + fgrad_m[n]

The magnitude |c_m[n]|^2 is accumulated at the nearest bin to
(target_freq, target_time). Centre frequencies are normalised to [0, 2)
and wrapped.

**Synchrosqueezing** is the frequency-only variant: reassign only along
the frequency axis (using tgrad), keeping time position fixed. This
preserves invertibility.

---

## 15a  Gabor Multipliers and Frame Multipliers

**Gabor multiplier.** Given analysis filters {g_m, a_m} forming a frame,
a dual or tight set {g_d_m}, and a time-frequency symbol sigma_m[n],
the Gabor multiplier is:

    (M_sigma f)[l] = sum_m sum_n sigma_m[n] <f, T_{n a_m} g_m> T_{n a_m} g_d_m[l]

This is the fundamental operator for time-varying filtering: analyse,
multiply each coefficient by the symbol, synthesise. For time-invariant
systems (sigma independent of n), it reduces to a frequency-domain
multiplication. For a diagonal symbol (sigma independent of m), it
reduces to pointwise amplitude modulation.

**Eigenvalue bounds.** For a tight frame with bound A:

    A * min(sigma) <= <M_sigma f, f> / ||f||^2 <= A * max(sigma)

When A = B = 1, the multiplier's eigenvalues equal the symbol values —
no frame-induced distortion.

**Approximation quality.** A general linear operator T can be
approximated by a Gabor multiplier M_sigma that minimises
||T - M_sigma||_HS. The quality of this approximation depends on the
frame redundancy and the time-frequency localisation of the window.
Higher redundancy and better-localised windows give better
approximations of slowly-varying operators.

**Invertibility.** If sigma_m[n] > 0 everywhere, the multiplier is
invertible (positive-definite). The condition number of M_sigma is
bounded by (B/A) * max(sigma)/min(sigma). A tight frame minimises the
frame contribution to the condition number.

**Frame multiplier (generalisation).** Allows different analysis and
synthesis frames: M = D_synth* diag(sigma) D_analysis. This is needed
for multiwindow decompositions where analysis and synthesis windows
differ.

**MATLAB reference implementation.** The LTFAT operator framework in
``ltfat_2.0/inst/operators/`` provides the porting reference:

- ``framemul(f, Fa, Fs, s)`` — apply frame multiplier
- ``iframemul(f, Fa, Fs, s)`` — invert (conjugate gradient on M_sigma)
- ``framemuladj(f, Fa, Fs, s)`` — adjoint operator
- ``framemulappr(T, Fa, Fs)`` — best Hilbert-Schmidt approximation of
  a general operator T by a frame multiplier
- ``framemuleigs(Fa, Fs, s, K)`` — K largest eigenvalues

The deprecated ``gabmul`` was Gabor-specific; ``framemul`` generalises
to arbitrary frame pairs, which is what cool-frames should implement (our
filterbanks are already general frames).

**MulAcLab.** The interactive TF audio editor ``mulaclab.m`` (52k lines)
is the primary application of frame multipliers — the user draws a TF
mask (the symbol sigma), and the engine applies the multiplier to the
audio. A Python modernisation (DSP engine + pluggable inpainting
backends + web GUI) is planned as a separate package.

**Implementation note.** cool-frames does not yet have an explicit multiplier
API, but the operation is straightforward given the existing analysis
and synthesis functions:

    c = filterbank(f, g, a, L)          # analyse
    c_masked = [c[m] * sigma[m] for m in range(M)]  # multiply
    f_hat = ifilterbank(c_masked, g_d, a, L)         # synthesise

A dedicated API would add: frame selection (dual vs tight), symbol
validation, efficient inversion (CG on M_sigma), eigenvalue estimation,
and best-approximation of arbitrary operators.

---

## 16  Iterative Phase Retrieval Algorithms

All operate in the filterbank domain (non-uniform hop sizes supported).

- **GLA (Griffin-Lim):** alternate between constraining magnitude and
  projecting onto consistent STFT space. c_{k+1} = A(|target| . phase(S c_k)).

- **Le-GLA (Accelerated GLA):** adds momentum:
  c_{k+1} = GLA(c_k) + alpha * (GLA(c_k) - GLA(c_{k-1})).

- **SPSI (Single-Pass Spectrogram Inversion):** propagates phase
  frame-by-frame using peak-channel logic. O(1) per coefficient.

- **RTISILA (Real-Time Iterative Spectrogram Inversion with Look-Ahead):**
  combines SPSI initialisation with iterative refinement over a sliding
  window.

- **GSRTISILA / LERTISILA:** generalised and line-exponential variants
  with improved convergence.

- **DECOLBFGS:** L-BFGS optimisation of the phase-only objective
  min_phi ||S(|target| exp(j phi)) - |target| exp(j phi)||^2.

**Magnitude error metric:**

    err = ||  |c_hat| - |c_target|  ||_F  /  || |c_target| ||_F

---

## 17  Real-Time Streaming Phase Reconstruction

**Tick-based processing.** One tick = one frame from the coarsest channel
(max hop). For channel m, K_m = a_max / a_m new frames arrive per tick.

**State.** Per-channel ring buffers (depth 3) of log-magnitudes, frame
counters, previous phase and gradient values. Causal phase gradients via
backward finite differences:

    fd = (3 f[n] - 4 f[n-1] + f[n-2]) / 2     (3 frames available)
    fd = f[n] - f[n-1]                          (2 frames)
    fd = 0                                       (1 frame)

**Phase propagation (trapezoidal rule):**

    phi_m[n] = phi_m[n-1] + dt * (tgrad_prev + tgrad_curr) / 2

Cross-channel:

    phi_{m+1}[n] = phi_m[n] + dt * (tgrad_m + tgrad_{m+1}) / 2
                            + dfc * (fgrad_m + fgrad_{m+1}) / 2

**Differentiable variant.** `constphase_nonuniform` replaces the heap
with a descending-magnitude sort and straight-through estimator for the
argsort, enabling gradient flow through the entire phase reconstruction
pipeline.

---

## 18  Algorithmic Tricks and Implementation Notes

**Overlap-add via reshape-and-sum.** Band-limited FFT analysis avoids
explicit circular convolution by zero-padding to a multiple of N_m,
reshaping to (-1, N_m), and summing along axis 0. This is the core
efficiency trick for sparse filters.

**floor23 for FFT-friendly sizes.** `floor23(n)` returns the largest
integer <= n of the form 2^i * 3^j. Pre-builds a lookup table up to
2^20, then binary-searches. Used for hop sizes and transform lengths to
ensure efficient FFT execution.

**Callable filter descriptors.** H and foff can be callables taking L
as argument, allowing compact parametric storage. The filter is only
materialised at the specific L needed for a given signal length.

**Odd window lengths.** The direct filter constructor always uses odd
window lengths (round to odd if even) to guarantee that the centre
sample is unambiguous and the peak aligns exactly with the target
frequency bin.

**Edge filter complement.** Instead of designing DC and Nyquist filters
independently, they are derived as the complement of the inner-channel
frame response: H_edge = sqrt(S_max - S_inner). This automatically
fills any spectral gaps and guarantees a flat total response.

**Real-valued signal optimisation.** For filters marked `realonly=1`,
only positive frequencies are stored; the negative-frequency contribution
is reconstructed via the involution (Hermitian mirror). Analysis averages
with the conjugate mirror: c = (c + c_conj) / 2.

**Phase gradient reliability floor.** Division by the spectrogram in
phase gradient computation uses a floor:
s = max(|c|^2, minlvl * max|c|^2) to prevent amplification of noise in
low-energy regions.

**Lambert W for wavelet support.** The Cauchy/Morse wavelet's effective
support boundaries are found analytically via the Lambert W function
(both branches b=0 and b=-1), avoiding iterative root-finding.

**B-spline evaluation.** Cardinal B-splines of orders 1-5 are evaluated
via piecewise polynomial formulas (no recursion), enabling vectorised
NumPy computation.

**Frequency-domain warping.** Warped filterbanks evaluate standard
window functions at non-uniformly spaced positions (warped through an
auditory scale). This achieves non-uniform frequency resolution without
explicitly designing non-uniform filters — the warping does the work.

---

## 19  Key References

For the reader who wants to go deeper, the main references behind
this codebase are:

- Søndergaard, P. L. (2007). *An efficient algorithm for the discrete
  Gabor transform using full-length windows.* Preprint, arXiv:0709.3259.
  [Sondergaard factorization, comp_wfac, comp_dgt_walnut]

- Holighaus, N., Dörfler, M., Velasco, G. A., & Grill, T. (2013).
  *A framework for invertible, real-time constant-Q transforms.*
  IEEE Trans. Audio, Speech, Language Process., 21(4), 775-785.
  [Non-stationary Gabor transform, non-uniform filterbanks]

- Průša, Z., Balazs, P., & Søndergaard, P. L. (2017). *A noniterative
  method for reconstruction of phase from STFT magnitude.* IEEE/ACM
  Trans. Audio, Speech, Language Process., 25(5), 1154-1164.
  [PGHI — Phase Gradient Heap Integration]

- Balazs, P., Dörfler, M., Jaillet, F., Holighaus, N., & Velasco, G.
  (2011). *Theory, implementation and applications of nonstationary
  Gabor frames.* J. Comput. Appl. Math., 236(6), 1481-1496.
  [Frame theory for non-uniform filterbanks]

- Glasberg, B. R. & Moore, B. C. J. (1990). *Derivation of auditory
  filter shapes from notched-noise data.* Hearing Research, 47, 103-138.
  [ERB scale]

- Griffin, D. W. & Lim, J. S. (1984). *Signal estimation from modified
  short-time Fourier transform.* IEEE Trans. ASSP, 32(2), 236-243.
  [Griffin-Lim algorithm]

- Balazs, P. (2007). *Basic definition and properties of Bessel
  multipliers.* J. Math. Anal. Appl., 325(1), 571-585.
  [Frame multipliers — eigenvalue bounds, approximation theory]

- Dörfler, M. & Torrésani, B. (2010). *Representation of operators in
  the time-frequency domain and generalized Gabor multipliers.*
  J. Fourier Anal. Appl., 16(2), 261-293.
  [Gabor multiplier approximation of linear operators]

- Auger, F. & Flandrin, P. (1995). *Improving the readability of
  time-frequency and time-scale representations by the reassignment
  method.* IEEE Trans. Signal Process., 43(5), 1068-1089.
  [Spectral reassignment]

- Ltfat toolbox: P. L. Søndergaard, B. Torrésani, and P. Balazs.
  *The Linear Time-Frequency Analysis Toolbox.*
  Int. J. Wavelets Multiresolut. Inf. Process., 10(4), 2012.
  [LTFAT — the MATLAB toolbox this port is based on]
