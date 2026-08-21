# cool-frames examples

Runnable demos and tutorials for the `cool-frames` core. Every script is
self-contained (it synthesises its own test signal) and can be run directly:

```bash
python examples/01_quickstart.py
```

## Scope: publication-safe only

These examples deliberately use **only textbook / reference technique**. Each
one makes a "why cool-frames" argument concrete: perfect reconstruction, the
counterfactual edit, frame control, perceptual resolution with inversion, one
substrate for many tasks, and LTFAT reproducibility. Research extensions
(restoration, source separation, coherence/modulation analysis, learnable
front ends, streaming overlap-add, time-adaptive overlap) are developed in
separate packages and are not demonstrated here.

## Contents

| # | file | shows |
|---|------|-------|
| 01 | `01_quickstart.py` | analyse → look → synthesise, ~1e-16 round-trip |
| 02 | `02_designing_filterbanks.py` | designers, the `scale=` families, custom filters, checking A>0 |
| 03 | `03_frame_theory.py` | bounds A,B,κ; dual & tight; `partial_tighten`; SVD ground-truth check |
| 04 | `04_perfect_reconstruction.py` | machine-precision inverse across painless designers |
| 05 | `05_counterfactual_edit.py` | TF-domain edit via `framemul` on a tight frame |
| 06 | `06_auditory_gallery.py` | ERB/Bark/Mel/CQT spectrograms + the scale tour |
| 08 | `08_phase_retrieval.py` | classical GLA / fGLA / SPSI from magnitude |
| 10 | `10_frame_health.py` | one-call frame diagnostics |
| 11 | `11_differentiable_toy.py` | back-prop through analyse→synthesise (toy; needs torch) |

Scripts that plot save PNGs to `examples/_output/` when matplotlib is
available; otherwise they print the numeric results. `11` skips cleanly if the
PyTorch backend is not installed.

The property-based tests under `tests/` are the executable specification of the
guarantees these demos illustrate (perfect reconstruction, frame bounds, energy
conservation).
