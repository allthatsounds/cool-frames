Audio denoising via spectral thresholding
=========================================

Noise often occupies different filterbank channels from the desired
signal.  By applying a threshold in the coefficient domain – keeping
large-magnitude coefficients and suppressing small ones – we can
attenuate noise without distorting the primary audio content.

.. rubric:: Companion script

``examples/02_audio_denoising.py``

Theory
------

After analysis the coefficient in channel *k* at time frame *n* is:

.. math::

   c_k[n] = \langle f, g_k(\cdot - n \cdot a_k) \rangle

For a noisy observation :math:`f = f_0 + \epsilon` we want to estimate
:math:`f_0` by attenuating channels and frames where :math:`|c_k[n]|`
is small relative to noise.

Three strategies are available:

**Hard thresholding**
    Zero out any coefficient whose magnitude falls below
    :math:`\tau = \tau_{\text{dB}} \cdot \max_n|c_k[n]|`.
    Aggressive suppression, may cause musical noise artefacts.

**Soft thresholding**
    Shrink magnitudes by :math:`\tau`, floored at zero:
    :math:`\hat{c}_k[n] = \max(|c_k[n]| - \tau, 0) \cdot e^{j\angle c_k[n]}`.
    Smoother but reduces overall energy.

**Wiener filtering**
    Estimate per-frame SNR and apply a gain that approaches 1 where
    signal dominates and 0 where noise dominates.  Requires a noise
    power estimate, provided via the ``threshold_db`` parameter.

Denoising with core primitives
------------------------------

Thresholding in the coefficient domain takes a handful of lines with the
core analysis/synthesis functions and :func:`cool_frames.sigproc.thresh`:

.. code-block:: python

   import numpy as np
   from cool_frames.filters import audfilters
   from cool_frames.filterbanks import filterbank, filterbankdual, ifilterbank
   from cool_frames.sigproc import thresh

   # noisy_signal: 1D float64 array, fs: sample rate
   fs, Ls = 16_000, len(noisy_signal)
   g, a, fc, L, info = audfilters(fs, Ls)
   gd = filterbankdual(g, a, L)

   c = filterbank(noisy_signal, g, a, L=L)

   threshold_db = -30      # relative to the in-channel peak magnitude
   c_dn = [
       thresh(ci, 10 ** (threshold_db / 20) * np.max(np.abs(ci)),
              mode='hard')[0]    # or 'soft' / 'wiener'
       for ci in c
   ]

   denoised = ifilterbank(c_dn, gd, a, Ls=Ls)

Comparing per-channel energy before and after
(``[np.sum(np.abs(ci) ** 2) for ci in c]``) shows which frequency bands
were suppressed most aggressively.

Tuning the threshold
--------------------

``threshold_db`` is relative to the *in-channel peak magnitude* for
hard/soft, and is used as an absolute noise power estimate for Wiener.
A value around -25 dB to -35 dB is typical for moderate noise.

For strong noise, lower the threshold (e.g. ``-20``); for very quiet
noise, raise it (e.g. ``-40``) to preserve more signal.

Building it manually
--------------------

The recipe is thin wrapper around the public API.  The same pipeline
in full:

.. code-block:: python

   from cool_frames.numpy.filters import audfilters
   from cool_frames.numpy.filterbanks import (
       filterbank, ifilterbank, filterbankrealtight,
   )

   g, a, fc, L, info = audfilters(fs, len(noisy_signal))
   c = filterbank(noisy_signal, g, a)

   # --- your thresholding logic here ---
   c_clean = [np.where(np.abs(ci) > tau, ci, 0) for ci in c]
   # ------------------------------------

   gd = filterbankrealtight(g, a, L)
   denoised = ifilterbank(c_clean, gd, a, len(noisy_signal))
