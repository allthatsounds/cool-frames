Filterbank basics: analysis, synthesis, and perfect reconstruction
===================================================================

The fundamental operation of ``cool-frames`` is the
*analysis-modify-synthesis* workflow:

1. **Design** a filterbank (choose a frequency scale and density).
2. **Analyse** an input signal into per-channel coefficient arrays.
3. **Modify** the coefficients (denoise, EQ, separate, …).
4. **Synthesise** back to a time-domain signal.

When the filterbank forms a *frame*, perfect reconstruction is
guaranteed: synthesising the unmodified coefficients recovers the
original signal to floating-point precision.

.. rubric:: Companion script

``examples/01_erb_filterbank.py``

Designing the filterbank
------------------------

:func:`cool_frames.numpy.filters.audfilters` designs an
ERB (Equivalent Rectangular Bandwidth) auditory filterbank – the
default choice for audio work.  It returns four objects:

.. code-block:: python

   from cool_frames.numpy.filters import audfilters

   fs = 16_000    # sample rate (Hz)
   Ls = fs * 2    # signal length: 2 s

   g, a, fc, L, info = audfilters(fs, Ls)

``g``
    List of filter impulse responses, one ``ndarray`` per channel.
    Channels are ordered from low to high frequency.

``a``
    List of integer hop sizes – the downsampling factor applied after
    each channel's convolution.  Non-uniform hop sizes give a
    near-constant time-frequency resolution across the auditory scale.

``fc``
    Centre frequencies of each channel in Hz, useful for labelling
    plots or building frequency-selective masks.

``L``
    The next valid DFT length for this filterbank.  Always pass this
    to :func:`~cool_frames.numpy.filterbanks.filterbank` so
    circular convolution is computed correctly.

Computing the dual frame
------------------------

To synthesise with perfect reconstruction you need the *dual frame*
:math:`\tilde{g}`, computed from the analysis filters:

.. code-block:: python

   from cool_frames.numpy.filterbanks import filterbankdual

   gd = filterbankdual(g, a, L)

For a *tight frame* the dual equals the analysis filters scaled by
:math:`1/A`, which can be computed more cheaply with
:func:`~cool_frames.numpy.filterbanks.filterbankrealtight`.

Analysis
--------

.. code-block:: python

   import numpy as np
   from cool_frames.numpy.filterbanks import filterbank

   t = np.arange(Ls) / fs
   x = np.sin(2 * np.pi * 440 * t)   # 440 Hz test tone

   c = filterbank(x, g, a, L=L)

``c`` is a list of complex ``ndarray`` objects.  ``c[k]`` has length
``ceil(L / a[k])`` – the number of time frames in channel *k*.
Lower-frequency channels have larger hop sizes and therefore fewer
frames, reflecting the coarser time resolution at low frequencies.

Synthesis
---------

.. code-block:: python

   from cool_frames.numpy.filterbanks import ifilterbank

   x_rec = ifilterbank(c, gd, a, Ls=Ls, real=True)

   err = np.max(np.abs(x - x_rec))
   print(f"Reconstruction error: {err:.2e}")   # < 1e-10

Pass ``real=True`` for real-valued signals (the common case) to avoid
complex arithmetic in the overlap-add step.

Frequency-band processing
--------------------------

Because each element of ``c`` corresponds to a specific frequency band,
processing is as simple as modifying the relevant entries:

.. code-block:: python

   # Suppress everything above 4 kHz
   c_lp = [
       ci if fc[i] <= 4000 else np.zeros_like(ci)
       for i, ci in enumerate(c)
   ]
   x_lp = ifilterbank(c_lp, gd, a, Ls=Ls, real=True)

This coefficient-domain edit pattern — analyse, modify, resynthesise with
the dual frame — is what :mod:`cool_frames.operators` generalises: a frame
multiplier applies an arbitrary time-frequency symbol in one step, and can
be inverted or approximated.

Other frequency scales
----------------------

The package ships additional filter-design functions:

.. code-block:: python

   from cool_frames.numpy.filters import (
       cqtfilters,      # constant-Q (geometric frequency spacing)
       waveletfilters,  # dyadic wavelet filterbank
       gabfilters,      # Gabor / STFT filterbank (uniform hop)
   )

All of them return the same ``(g, a, fc, L)`` tuple and are therefore
interchangeable in the analysis-synthesis loop shown above.
