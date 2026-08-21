Changelog
=========

0.1.0 (2026-08)
---------------

First public release of the cool-frames core: the invertible filterbank stack in a
NumPy reference backend and a differentiable PyTorch backend.

**Filter design** (``cool_frames.filters``)

- ``audfilters`` (ERB/gammatone, Bark, Mel, and further auditory scales),
  ``greenwoodfilters``, ``cqtfilters``, ``gabfilters``, ``waveletfilters``,
  ``warpedfilters``, ``freqwavelet``, ``firwin``/``firkaiser``/``freqwin``
  window constructors, and biquad IIR resonators (``comp_biquad``,
  ``biquadfilter``).
- Fractional sampling (``L = Ls``, exact redundancy) supported by all
  designers; ``fs``-first parameter convention with center frequencies in Hz.

**Analysis, synthesis, and frame theory** (``cool_frames.filterbanks``,
``cool_frames.operators``)

- ``filterbank`` / ``ifilterbank`` (direct and iterative variants) with
  full-length FFT, band-limited FFT, and time-domain FIR computation paths.
- Painless reconstruction at machine precision; ``filterbankdual``,
  ``filterbanktight``, ``filterbankbounds`` (``real=True`` by default — audio
  is real), SVD ground-truth bounds, ``painlessfilterbank``.
- Frame multipliers: ``framemul``, ``framemuladj``, ``framemulinv`` (CG
  inversion), ``framemulappr``, ``framemuleigs``.
- Spectral analysis helpers: ``analyze_filterbank``,
  ``analyze_frame_operator``, ``analyze_coefficients``.

**Phase** (``cool_frames.phase``)

- Phase gradients (``filterbankphasegrad``, ``filterbankphasederiv``) and
  reconstruction from magnitude: PGHI (``filterbankconstphase``), GLA/LeGLA,
  SPSI, RTISILA/LERTISILA/GSRTISILA, DECOLBFGS, and reassignment/
  synchrosqueezing.

**PyTorch backend** (``cool_frames.torch``)

- Mirrors the NumPy API surface; all operations differentiable with respect to
  the input signal, with standard device placement for GPU execution.
  Filter design delegates to the NumPy backend at setup time.
- Differentiable biquad IIR (``comp_biquad``, ``biquadfilter``) in rho/phi.

**Inspection and primitives**

- ``cool_frames.diagnostics``: filterbank and reassigned spectrograms, centre-frequency
  estimation, and ``recommend_filterbank`` (signal-driven designer choice).
- ``cool_frames.sigproc``: coefficient-domain sparsity primitives (``thresh``,
  ``largest``).

Feature extraction (chroma, mel/MFCC), source separation and audio effects are
deliberately **out of scope** — see the README for how cool-frames composes with
librosa/torchaudio for those.

**Validation and infrastructure**

- Property-based test suite (algebraic, reconstruction, bounds, energy,
  consistency, ML/spectral categories) for both backends, run in CI on
  Python 3.10–3.13.
- Cross-language regression harness diffing Python output against committed
  MATLAB LTFAT reference data (``tests/crosslang/``).
- Benchmarks with committed baselines (pytest-benchmark), Sphinx docs, ruff +
  mypy linting.
