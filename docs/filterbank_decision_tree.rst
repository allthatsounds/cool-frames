Choosing a Filterbank
=====================

cool-frames provides several filterbank designers, each optimised for different
signal types and use cases. This guide helps you pick the right one.

.. contents:: On this page
   :local:
   :depth: 2


Quick-start: automatic recommendation
--------------------------------------

If you have an audio signal and want a data-driven suggestion::

    from cool_frames.diagnostics import recommend_filterbank

    rec = recommend_filterbank(signal, fs)
    print(rec.designer)   # e.g. "cqtfilters"
    print(rec.params)     # suggested keyword arguments
    print(rec.rationale)  # why this designer was chosen

The function analyses harmonicity, onset rate, bandwidth occupancy, and
stationarity, then returns a structured recommendation.  It covers
``audfilters`` and ``cqtfilters``.  The remaining designers
(``waveletfilters``, ``warpedfilters``) are methodology-driven and must be
chosen explicitly.  Two designers described below
(``timeadaptivefilters``, ``polezerofilters``) belong to the extension
packages and are not shipped in the cool-frames core.


Decision tree
-------------

The diagram below encodes the same logic as ``recommend_filterbank``.
Follow the questions from top to bottom.

.. code-block:: text

   ┌─────────────────────────────────┐
   │   What is the application?      │
   └───────────┬─────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
   Real-time?      Offline analysis,
       │           modification, or
       │           phase retrieval
       │               │
       ▼               ▼
   ┌────────┐    Is the signal harmonic?
   │audfilters│   (tonal / pitched)
   │(ERB)    │        │
   └────────┘   ┌─────┴─────┐
                ▼           ▼
              Yes           No
                │           │
                ▼           ▼
           ┌─────────┐  High transient density
           │cqtfilters│  AND non-stationary?
           └─────────┘       │
                        ┌────┴────┐
                        ▼        ▼
                      Yes        No
                        │        │
                        ▼        ▼
               ┌──────────────┐  ┌──────────┐
               │timeadaptive- │  │audfilters │
               │filters       │  │(ERB/Bark) │
               └──────────────┘  └──────────┘

**Parameter adjustments by purpose:**

- *Modification* (EQ, denoising, masking): increase ``redmul`` to 2 for
  robustness to coefficient manipulation.
- *Phase retrieval*: increase ``redmul`` to 4 for faster convergence.
- *Feature extraction*: default redundancy is fine; use ``ufilterbank``
  for uniform-size output.


Designer reference
------------------

audfilters
^^^^^^^^^^

General-purpose auditory filterbank with perceptually motivated spacing
(ERB, Bark, or Mel scale). Best all-round choice when no strong prior
knowledge about the signal exists.

**When to use:**

- Broadband content (environmental sound, noise, effects)
- General audio analysis and visualisation
- Real-time or streaming applications
- When tight frame bounds matter (audfilters typically has the lowest
  condition number)

**Key parameters:** ``scale`` (``"erb"``, ``"bark"``, ``"mel"``),
``fmin``, ``fmax``, ``bwmul``, ``redmul``.

::

    g, a, fc, L, info = audfilters(fs, Ls, scale="erb")


cqtfilters
^^^^^^^^^^

Constant-Q filterbank with logarithmic frequency spacing. Each filter
has the same Q factor (centre frequency / bandwidth ratio), giving equal
resolution on a log-frequency axis.

**When to use:**

- Music analysis (pitch, chords, melody)
- Speech fundamental frequency tracking
- Any signal with strong harmonic structure
- When you need resolution proportional to pitch

**Key parameters:** ``fmin``, ``fmax``, ``bins`` (bins per octave).

::

    g, a, fc, L, info = cqtfilters(fs, Ls, fmin=50, bins=48)


timeadaptivefilters (not included in the core package)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Time-varying filterbank that segments the signal at detected transient
boundaries and applies a different filterbank to each segment.

**When to use:**

- Signals with distinct transient and sustained regions
  (e.g. percussive music, speech with plosives, impulsive noise)
- When a single time-frequency resolution is insufficient
- Offline analysis only (requires the full signal for onset detection)

**Key parameters:** ``onset_threshold``, ``transient_bwmul``,
``sustained_bwmul``, ``overlap_samples``.

::

    fb = timeadaptivefilters(fs, Ls, f=signal)
    coeffs = fb.analyse(signal)
    y = fb.synthesise(coeffs)


waveletfilters
^^^^^^^^^^^^^^

Wavelet filterbank following wavelet-theory conventions (dyadic or
user-specified scales). Produces wavelet coefficients compatible with
standard wavelet analysis frameworks.

**When to use (methodology-driven):**

- Geophysics, seismology, or fields with established wavelet conventions
- Coherence analysis across scales
- When interoperability with wavelet toolboxes is required
- Wavelet-packet or multiresolution analysis

This designer is *not* recommended automatically by
``recommend_filterbank`` because the choice to use wavelets is typically
driven by domain convention rather than signal properties.

::

    g, a, fc, L, info = waveletfilters(fs, Ls)


warpedfilters
^^^^^^^^^^^^^

Filterbank on a user-defined warped frequency axis. Allows arbitrary
frequency-to-position mappings.

**When to use (user-driven):**

- Custom perceptual scales not covered by ERB/Bark/Mel
- Cochlear models with specific frequency-place maps
- Research comparing different frequency warping strategies
- When the frequency axis must match a specific publication

::

    g, a, fc, L, info = warpedfilters(fs, Ls, warp_func, inv_warp_func)


polezerofilters (not included in the core package)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IIR (infinite impulse response) pole-zero filterbank. Filters are
parameterised by pole and zero locations, providing minimal-latency
filtering suitable for real-time sample-by-sample processing.
``polezerofilters`` is part of the extension packages and is not shipped in
the cool-frames core; single biquad IIR resonators (``biquadfilter``) are.

**When to use:**

- Real-time applications requiring minimal latency
- When FIR filter length would be impractically long
- Analog filter emulation
- Learnable filterbanks with gradient-based optimisation

Note: IIR frames have weaker theoretical guarantees than FIR frames.
Reconstruction quality depends on optimisation convergence.

::

    g, a, fc = polezerofilters(fs, Ls, order=2)


Channel count and the frame property
------------------------------------

Every designer has a redundancy floor below which the filters no longer cover
the spectrum: the response has gaps, the frame lower bound ``A`` is 0, and
there is no perfect reconstruction. The designers do **not** currently warn
when you land there (``gabfilters`` is the exception — it warns when the hop
exceeds the channel count), so check after designing:

.. code-block:: python

    from cool_frames.filters import audfilters
    from cool_frames.filterbanks import filterbankbounds

    fs, Ls, M = 16_000, 4096, 24
    g, a, fc, L, info = audfilters(fs, Ls, M=M)
    A, B, kappa = filterbankbounds(g, a, L, return_kappa=True)
    assert A > 0            # it is a frame at all
    print(kappa)            # ... and how well conditioned it is

There are really two thresholds, and they are far apart. For ``audfilters``
on the ERB scale, measured:

.. list-table::
   :header-rows: 1

   * - fs / Ls
     - largest M with A = 0
     - first M with A > 0
     - first M with κ < 100
   * - 8 kHz / 2048
     - 11
     - 12 (κ ≈ 1.4·10³)
     - 13
   * - 16 kHz / 4096
     - 13
     - 14 (κ ≈ 1.7·10⁴)
     - 17
   * - 48 kHz / 4096
     - 16
     - 18 (κ ≈ 7.5·10³)
     - 20

Crossing into "is a frame" therefore does not mean "is usable": the first few
admissible channel counts give condition numbers in the thousands, where
reconstruction is numerically fragile even though it is formally exact.

.. note::

   cool-frames has no closed-form predictor for the admissible
   ``(M, fs, Ls, bwmul)`` region of a given designer — neither for existence
   (``A > 0``) nor for a conditioning target (``κ`` below some bound). The
   numbers above are empirical. Deriving such a predictor per designer is
   open work; until then, measure.

Post-filterbank processing
--------------------------

Regardless of which filterbank you choose, cool-frames provides a consistent
processing pipeline:

**Refining the representation:**

- ``filterbankphasegrad`` / ``filterbankphasederiv``: compute
  instantaneous frequency and group delay for reassignment.
- ``filterbankreassign``: sharpen the time-frequency representation by
  moving energy to its instantaneous time-frequency position.

**Phase retrieval (magnitude → audio):**

- ``gla``: Griffin-Lim Algorithm (iterative, general).
- ``legla``: Le Roux's accelerated GLA.
- ``spsi``: Single-Pass Spectrogram Inversion (fast, non-iterative).
- ``rtisila`` / ``gsrtisila`` / ``lertisila``: real-time iterative
  methods with look-ahead.
- ``decolbfgs``: L-BFGS optimisation for phase retrieval.
- ``admm`` / ``raar`` / ``dm``: convex splitting methods.

**Operators:**

- ``framemul``: frame multiplier (time-frequency masking with a symbol).
- ``framemulinv``: inversion of a frame multiplier.
- ``framemuleigs``: eigenvalues of a frame multiplier.

**Diagnostics:**

- ``filterbank_spectrogram`` / ``reassigned_spectrogram``: inspection.
- ``recommend_filterbank``: automatic filterbank selection.
