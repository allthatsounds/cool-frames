# Deprecation Policy

This document describes how `cool-frames` handles API deprecation
and removal.

## Versioning

cool-frames follows [Semantic Versioning 2.0.0](https://semver.org/):

- **Patch** releases (0.1.x) contain bug fixes and documentation changes only.
- **Minor** releases (0.x.0) may add new features and deprecate existing APIs.
- **Major** releases (x.0.0) may remove previously deprecated APIs.

While the project is in the 0.x series (pre-1.0), minor releases may contain
breaking changes, but we will always provide a deprecation period first.

## Deprecation lifecycle

1. **Deprecation announced.** A `DeprecationWarning` is added to the affected
   function, class, or import path. The warning message names the replacement
   and the version in which the deprecated API will be removed. The deprecation
   is documented in the changelog.

2. **Grace period.** The deprecated API continues to work for at least **two
   minor releases** (or 6 months, whichever is longer) after the deprecation
   warning is introduced.

3. **Removal.** After the grace period the deprecated API is removed in the
   next minor or major release. The removal is documented in the changelog.

## Currently deprecated APIs

There are no deprecated APIs in 0.1.0. The top-level convenience paths
(`cool_frames.filters`, `cool_frames.filterbanks`, `cool_frames.phase`, ...) are stable re-exports
of the NumPy backend (`cool_frames.numpy.*`); both spellings are supported.

## Backport policy

- Security fixes are backported to the latest patch release of the current
  minor series.
- Bug fixes are generally not backported unless they affect correctness of
  numerical results.

## How to detect deprecations in your code

Run your test suite with warnings turned into errors:

```bash
python -W error::DeprecationWarning -m pytest
```

Or filter for cool-frames warnings:

```python
import warnings
warnings.filterwarnings("error", category=DeprecationWarning,
                        module=r"cool_frames\..*")
```
