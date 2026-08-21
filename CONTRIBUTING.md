# Contributing to cool-frames

Thank you for your interest in contributing! This document covers how to
set up your development environment, run the test suite, and submit changes.

## Table of contents

- [Development setup](#development-setup)
- [Running tests](#running-tests)
- [Code style](#code-style)
- [Submitting changes](#submitting-changes)
- [Reporting bugs](#reporting-bugs)
- [Requesting features](#requesting-features)
- [Architecture overview](#architecture-overview)

---

## Development setup

```bash
git clone https://github.com/allthatsounds/cool-frames.git
cd cool-frames

# Editable install with all dev dependencies
pip install -e ".[dev,torch]"

# Install docs dependencies (optional)
pip install -e ".[docs]"
```

---

## Running tests

```bash
# NumPy backend (fast, no torch required)
pytest tests/ --ignore=tests/torch_backend -m "not slow and not requires_ref"

# Torch backend (CPU)
pytest tests/torch_backend/ -m "not slow"

# Full suite (skips anything that needs reference .mat files or CUDA)
pytest tests/ -m "not slow and not requires_ref"
```

**Reference data** (for `requires_ref` tests) is generated from MATLAB:

```matlab
cd filterbank/
export_reference_data()
```

This writes `.mat` files into `tests/reference_data/`. These tests are
skipped automatically when the directory is absent.

---

## Code style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
# Check
ruff check cool_frames/
ruff format --check cool_frames/

# Auto-fix
ruff check --fix cool_frames/
ruff format cool_frames/
```

Type annotations follow Python 3.10+ syntax (`X | Y` unions, `list[T]`
rather than `List[T]`).  Run mypy in advisory mode:

```bash
mypy cool-frames --ignore-missing-imports
```

---

## Submitting changes

1. **Fork** the repository and create a branch from `main`.
2. Make your changes, including tests for any new behaviour.
3. Ensure all tests pass and ruff reports no errors.
4. Open a **Pull Request** against `main`. Include:
   - A short description of what changed and why.
   - Any relevant issue numbers (`Fixes #123`).
   - Numerical results or plots for algorithm changes.

PRs that affect the public API must update the docstring for the affected
functions and, where applicable, the `docs/` pages.

---

## Reporting bugs

Use the **Bug report** issue template. Please include:

- Python version and OS
- Minimal reproducible example
- Full traceback
- Expected vs. actual behaviour

---

## Requesting features

Use the **Feature request** issue template. Describe the use case and, if
possible, sketch the proposed API.

---

## Architecture overview

```
cool_frames/
├── numpy/          # Reference NumPy implementation
│   ├── core/       # Low-level FFT kernels
│   ├── filters/    # Filter design (auditory, CQT, wavelet, ...)
│   ├── filterbanks/# Analysis, synthesis, frame theory
│   └── phase/      # Phase gradients, retrieval, real-time variants
├── torch/          # PyTorch backend (differentiable, GPU-ready)
│   ├── core/       # FFT kernels (mirrors numpy/core)
│   ├── filters/    # Thin wrappers — filter design stays in NumPy
│   ├── filterbanks/# Analysis, synthesis, frame theory
│   ├── phase/      # GLA, PGHI, phase gradients
│   └── sigproc/    # RMS, gain, thresholding, range compression
├── diagnostics/ sigproc/  # Inspection + coefficient-domain primitives
└── (top-level cool_frames.filters, cool_frames.filterbanks, ... re-export cool_frames.numpy.*)
```

**Key design principles:**

- *Filter design is setup-time.* The torch backend calls numpy for filter
  design and converts the results to tensors. Gradients only flow through
  analysis/synthesis, not filter construction.
- *Dual frames for perfect reconstruction.* Call `filterbankdual` at setup
  (real=True by default) to obtain the synthesis frame.
- *Fixed-iteration unrolling for differentiability.* Phase retrieval
  algorithms (`gla`, etc.) use a fixed iteration count rather
  than early stopping so that autograd can unroll through them.

### Adding a new filter design

1. Implement in `numpy/filters/_yourfilter.py` following the existing
   filter-dict conventions (`H`, `foff`, `realonly` keys).
2. Export from `numpy/filters/__init__.py`.
3. Add a thin wrapper in `torch/filters/_wrappers.py` that calls the numpy
   version and passes through `numpy_filters_to_torch`.
4. Write tests in `tests/layer1_filters/unit/`.

### Adding a new phase retrieval algorithm

1. Implement in `numpy/phase/_youralgorithm.py`.
2. Port the differentiable version to `torch/phase/_youralgorithm.py`,
   using `filterbank`/`ifilterbank` from `torch.filterbanks`.
3. Export from both `__init__.py` files.
4. Add tests in `tests/torch_backend/test_phase_retrieval.py`.
