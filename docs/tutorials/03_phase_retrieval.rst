Phase retrieval from filterbank magnitudes
==========================================

Many audio processing pipelines – vocoders, neural audio codecs, and
spectral editors – work exclusively with *magnitude* representations.
Phase retrieval is the problem of estimating a consistent phase from a
target magnitude spectrogram so that the result can be synthesised back
to audio.

.. rubric:: Companion script

``examples/03_phase_retrieval.py``

Background
----------

Given a magnitude coefficient array :math:`|c_k[n]|` the goal is to
find a signal :math:`f` whose filterbank coefficients satisfy:

.. math::

   \bigl| \langle f, g_k(\cdot - n\cdot a_k) \rangle \bigr| \approx |c_k[n]|

Different algorithms offer different trade-offs between quality and
speed:

PGHI – Phase Gradient Heap Integration
    Propagates phase using the *time-frequency phase gradient* implicit
    in the filterbank geometry.  Fast (single-pass) and produces
    spectrally coherent results.  Recommended as a default.
    Reference: Průša et al. (2017).

GLA – Griffin-Lim Algorithm
    Iterative projection-onto-convex-sets method.  Each iteration
    enforces magnitude consistency and filterbank consistency
    alternately.  Converges monotonically; quality improves with more
    iterations.  Reference: Griffin & Lim (1984).

fGLA – Fast Griffin-Lim
    Accelerated GLA using a momentum term (``alpha`` parameter).
    Typically 2–3× faster convergence than GLA for the same target
    quality.  Reference: Perraudin et al. (2013).

SPSI – Single-Pass Spectrogram Inversion
    A lightweight deterministic alternative to GLA: propagates phase
    horizontally across time frames.  Very fast but lower quality than
    GLA for non-harmonic sounds.  Reference: Beauregard et al. (2015).

Reconstructing from magnitudes
------------------------------

The phase-retrieval functions in :mod:`cool_frames.phase` share a common
call pattern — pass the magnitude coefficients together with the
filterbank that produced them:

.. code-block:: python

   from cool_frames.filters import audfilters
   from cool_frames.filterbanks import filterbank
   from cool_frames.phase import gla, spsi

   # 1. Analyse original signal
   g, a, fc, L, info = audfilters(fs, Ls)
   c = filterbank(f, g, a, L=L)
   s_mag = [abs(ci) for ci in c]   # discard phase

   # 2. Reconstruct from magnitudes (iterative Griffin-Lim / fast GLA)
   c_rec, f_rec, relres, niter = gla(
       s_mag, g, a, L=L, Ls=Ls, real=True, method='fgla',  # or method='gla'
   )

   # Single-pass alternative: SPSI phase estimate + dual-frame synthesis
   from cool_frames.filterbanks import filterbankdual, ifilterbank

   # spsi takes fc in Hz plus the sampling rate, so audfilters' output goes
   # straight in.  (Pass fs=1.0 if your fc is already in cycles per sample.)
   c_spsi, _phase = spsi(s_mag, a, fc, fs)
   gd = filterbankdual(g, a, L)
   f_spsi = ifilterbank(c_spsi, gd, a, Ls=Ls, real=True)

Calling the phase-retrieval functions directly
----------------------------------------------

For more control (e.g. tracking convergence or setting tolerance):

.. code-block:: python

   from cool_frames.numpy.phase import gla, filterbankconstphase

   # GLA – 50 iterations.  The return is always a 4-tuple.
   c_gla, f_gla, relres, niter = gla(
       s_mag, g, a, L=L, Ls=Ls,
       real=True,
       maxit=50,
       tol=1e-6,           # stop early if the residual drops below this
   )
   print(niter, relres[-1])   # iterations actually run, final residual

   # PGHI.  Returns (coefficients, usedmask); `usedmask` is True where the
   # phase was integrated rather than filled in at random below the threshold.
   c_pghi, usedmask = filterbankconstphase(f, g, a, L=L, fc=fc)
   f_pghi = ifilterbank(c_pghi, gd, a, Ls=Ls, real=True)
   print(f"{100 * sum(m.mean() for m in usedmask) / len(usedmask):.0f}% integrated")

Choosing an algorithm
---------------------

.. list-table::
   :header-rows: 1
   :widths: 12 12 12 64

   * - Method
     - Speed
     - Quality
     - Best for
   * - PGHI
     - ★★★★
     - ★★★
     - Fast single-pass reconstruction; tonal signals
   * - fGLA
     - ★★★
     - ★★★★
     - Iterative; balanced quality/speed trade-off
   * - GLA
     - ★★
     - ★★★
     - Reference baseline; guaranteed convergence
   * - SPSI
     - ★★★★★
     - ★★
     - Minimal-overhead pipelines; strongly harmonic content

Real-time phase reconstruction
-------------------------------

For streaming applications use the causal, look-ahead members of the family —
``rtisila`` and its enhanced variants — which advance frame by frame rather
than iterating over the whole signal:

.. code-block:: python

   from cool_frames.numpy.phase import gsrtisila, lertisila, rtisila

   c_rt, f_rt, relres, niter = rtisila(
       s_mag, g, a, L=L, Ls=Ls, real=True,
       maxit=5,          # inner iterations per frame
       lookahead=None,   # None selects a default from the window length
   )

``lertisila`` and ``gsrtisila`` take the same arguments and trade more work per
frame for better consistency.  All three cost seconds rather than milliseconds
on signals of a few thousand samples — they are designed for low *latency*, not
low total cost.
