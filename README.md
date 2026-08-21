# cool-frames — invertible time-frequency analysis

*The frame-theory layer of COOL — Computational Orchestration and Listening.*

[![CI](https://github.com/allthatsounds/cool-frames/actions/workflows/ci.yml/badge.svg)](https://github.com/allthatsounds/cool-frames/actions/workflows/ci.yml)
[![License: EUPL-1.2](https://img.shields.io/badge/license-EUPL--1.2-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://allthatsounds.github.io/cool-frames/)

Invertible time-frequency analysis for audio, in Python. cool-frames is a
port and redesign of the [LTFAT](http://ltfat.org/) filterbank module:
auditory, wavelet, constant-Q, Gabor, and warped filter design with exact
analysis/synthesis, frame theory, and phase retrieval — in a **NumPy**
reference backend and a differentiable, GPU-ready **PyTorch** backend.
Correctness is guarded by a property-based test suite grounded in frame
theory and by cross-language validation against MATLAB LTFAT.

## Features

- **Filter design** — ERB (gammatone), Bark, Mel, Greenwood, constant-Q,
  wavelet, Gabor, and warped filterbanks, plus biquad IIR resonators.
- **Analysis & synthesis** — `filterbank` / `ifilterbank` with painless
  reconstruction at machine precision.
- **Frame theory** — frame bounds, dual/tight frames, condition number,
  real-filterbank support, frame multipliers.
- **Phase retrieval** — PGHI, RTISILA, Griffin-Lim (GLA/LeGLA), SPSI, and
  reassignment/synchrosqueezing.
- **NumPy** reference backend + **PyTorch** backend: same API, differentiable
  end-to-end, standard device placement for GPU execution.
- **Diagnostics** — frame health at a glance, filterbank and reassigned
  spectrograms, and a signal-driven `recommend_filterbank`.

## Scope — and what to use instead

cool-frames is a reference implementation of **invertible time-frequency analysis**,
not a general audio-feature toolbox. The boundary is deliberate:

| If you want | Use |
|---|---|
| Auditory / constant-Q / wavelet analysis you can invert to machine precision | **cool_frames** |
| Frame bounds, dual and tight frames, condition numbers, frame multipliers | **cool_frames** |
| Phase retrieval on a *non-uniform* filterbank (PGHI, RTISILA, LeGLA, SPSI) | **cool_frames** |
| All of the above, differentiable, for training a learnable front end | **cool_frames.torch** |
| MFCC, chroma, mel spectrograms, beat tracking, HPSS, onset detection | [librosa](https://librosa.org) |
| Batched STFT features, dataset pipelines, audio I/O and resampling | [torchaudio](https://pytorch.org/audio) |

cool-frames ships no feature-extraction layer on purpose. Those problems are solved,
well tested, and heavily used elsewhere; a second-best copy inside cool-frames would
help nobody. The two compose cleanly — cool-frames gives you coefficients you can
edit and invert, librosa gives you features computed from them.

**Closest relatives.** [LTFAT](http://ltfat.org/) (MATLAB) is the origin of
this design and the reference cool-frames is validated against;
[nsgt](https://github.com/grrrr/nsgt) implements non-stationary Gabor
transforms in Python; phase retrieval methods such as PGHI have until now
existed mainly in MATLAB (`phaseret`). cool-frames's contribution is to bring the
frame-theoretic guarantees, the non-uniform designers and the phase-retrieval
family into one Python package with a differentiable backend.

## Installation

```bash
pip install cool-frames
```

For the differentiable PyTorch backend:

```bash
pip install "cool-frames[torch]"
```

The distribution is `cool-frames`; the import name is `cool_frames`:

```python
import cool_frames
# or, for a shorter local name:
import cool_frames as cf
```

Until the first PyPI release, install directly from GitHub:

```bash
pip install "cool-frames @ git+https://github.com/allthatsounds/cool-frames.git"
```

## Quick start

```python
import numpy as np
from cool_frames.filters import audfilters
from cool_frames.filterbanks import filterbank, filterbankdual, ifilterbank

fs = 16000
Ls = fs * 2  # 2 seconds of signal
x = np.random.randn(Ls)

# Design an ERB filterbank (fs first; fc returned in Hz)
g, a, fc, L, info = audfilters(fs, Ls)
gd = filterbankdual(g, a, L)  # real=True is the default — audio is real

# Analyse
c = filterbank(x, g, a, L=L)

# Synthesise (perfect reconstruction)
x_hat = ifilterbank(c, gd, a, Ls=Ls)
print(f"Reconstruction error: {np.max(np.abs(x - x_hat)):.2e}")
```

The PyTorch backend mirrors the NumPy API and is differentiable with respect
to the input signal:

```python
import torch
from cool_frames.torch.filters import audfilters
from cool_frames.torch.filterbanks import filterbank, filterbankdual, ifilterbank

g, a, fc, L, info = audfilters(16000, 4096)
gd = filterbankdual(g, a, L)
x = torch.randn(L, requires_grad=True)
c = filterbank(x, g, a)         # differentiable analysis
x_hat = ifilterbank(c, gd, a)   # differentiable synthesis (perfect reconstruction)
```

See [`examples/`](examples/) for runnable demos and the
[tutorials](https://allthatsounds.github.io/cool-frames/tutorials/index.html) for
worked examples.

## Package layout

| Subpackage    | Contents                                              |
|---------------|-------------------------------------------------------|
| `core`        | Low-level FFT kernels and math utilities              |
| `filters`     | Filter design: auditory scales, windows, wavelets     |
| `filterbanks` | Analysis, synthesis, and frame theory                 |
| `operators`   | Frame multipliers and TF-domain operators             |
| `phase`       | Phase gradients, reconstruction, and retrieval        |
| `diagnostics` | Spectrograms, centre frequencies, designer choice     |
| `sigproc`     | Coefficient-domain sparsity (`thresh`, `largest`)     |

Each subpackage exists in both backends: `cool_frames.numpy.*` (also re-exported at
the top level, e.g. `cool_frames.filters`) and `cool_frames.torch.*`.

## Validation

- A property-based test suite (algebraic identities, perfect reconstruction,
  frame bounds, energy conservation, consistency) runs on every commit for
  both backends.
- Cross-language regression tests diff cool-frames's output against committed MATLAB
  LTFAT reference data (`tests/crosslang/`), pinning down the reconciled
  MATLAB↔Python divergences (lowpass hop size, firwin centering, Nyquist
  convention).
- The mathematics behind the implementation is documented in
  [`MATH_REFERENCE.md`](MATH_REFERENCE.md).

## Citing

If you use cool-frames in your research, please cite it (see
[`CITATION.cff`](CITATION.cff)):

> C. Hollomey, "cool-frames: Computational Listening Algorithms — A Python Toolbox
> for Auditory Filterbanks with Frame-Theoretic Foundations," 2026.

Companion papers on the toolbox, its property-based validation methodology,
and a frame-theoretic benchmark of time-frequency representations are in
preparation; reproduction notebooks live in
[`examples/colab/`](examples/colab/).

## Development

```bash
git clone https://github.com/allthatsounds/cool-frames.git
cd cool-frames
pip install -e ".[dev]"
pytest -m "not slow and not requires_ref"
```

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Licensed under the [European Union Public Licence v1.2 (EUPL-1.2)](LICENSE).

## Acknowledgements

Based on the MATLAB [LTFAT](http://ltfat.org/) filterbank module by Peter L.
Søndergaard, Zdeněk Průša, and contributors.
