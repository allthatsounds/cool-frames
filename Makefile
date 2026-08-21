# cool-frames — development task runner
# Usage: make <target>
# Run `make help` for a list of available targets.

.DEFAULT_GOAL := help
SHELL := /bin/bash

PYTHON ?= python
PIP    ?= pip

# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Install package in editable mode with dev + docs extras
	$(PIP) install -e ".[dev,docs]"

.PHONY: install-torch
install-torch: ## Install with torch extra (CPU-only)
	$(PIP) install -e ".[dev,docs,torch]"

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------

.PHONY: lint
lint: ## Run ruff linter
	ruff check .

.PHONY: format
format: ## Run ruff formatter (in-place)
	ruff format .
	ruff check --fix .

.PHONY: format-check
format-check: ## Check formatting without changes
	ruff format --check .

.PHONY: typecheck
typecheck: ## Run mypy type checker
	mypy cool_frames/torch cool_frames/numpy/core/_math.py cool_frames/numpy/core/_fourier.py cool_frames/numpy/core/_norm.py --ignore-missing-imports --follow-imports=silent

.PHONY: check
check: lint format-check typecheck ## Run all quality checks

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run NumPy backend tests (fast)
	pytest tests/ \
		--ignore=tests/torch_backend \
		-m "not slow and not requires_ref" \
		--tb=short -q

.PHONY: test-torch
test-torch: ## Run Torch backend tests
	pytest tests/torch_backend/ -m "not slow" --tb=short -q

.PHONY: test-all
test-all: ## Run all tests (NumPy + Torch, including slow)
	pytest tests/ --tb=short -q

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	pytest tests/ \
		--ignore=tests/torch_backend \
		-m "not slow and not requires_ref" \
		--cov=cool_frames \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--tb=short -q
	@echo "Coverage report: htmlcov/index.html"

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

.PHONY: bench
bench: ## Run benchmarks (NumPy)
	pytest benchmarks/bench_filterbank.py benchmarks/bench_phase.py \
		--benchmark-only --benchmark-autosave -q

.PHONY: bench-torch
bench-torch: ## Run benchmarks (Torch)
	pytest benchmarks/bench_torch.py \
		--benchmark-only --benchmark-autosave -q

.PHONY: bench-compare
bench-compare: ## Compare latest two benchmark runs
	pytest-benchmark compare --group-by=name

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

.PHONY: docs
docs: ## Build Sphinx HTML documentation
	sphinx-build -b html docs/ docs/_build/html
	@echo "Docs: docs/_build/html/index.html"

.PHONY: docs-live
docs-live: ## Start live-reloading docs server
	sphinx-autobuild docs/ docs/_build/html --open-browser

.PHONY: docs-clean
docs-clean: ## Remove built docs
	rm -rf docs/_build

# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

.PHONY: build
build: ## Build sdist and wheel
	$(PYTHON) -m build

.PHONY: build-check
build-check: build ## Build and verify with twine
	twine check dist/*

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info htmlcov .mypy_cache .pytest_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
