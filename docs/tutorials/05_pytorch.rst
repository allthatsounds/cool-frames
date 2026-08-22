PyTorch differentiable filterbank
==================================

The ``cool_frames.torch`` subpackage provides a fully
differentiable filterbank that runs on CPU or GPU and integrates
seamlessly with PyTorch's automatic differentiation engine.

.. rubric:: Companion script

``examples/05_pytorch_differentiable.py``

.. note::

   Install the PyTorch extra::

       pip install "cool-frames[torch]"

   Filter *design* still uses the NumPy backend (a setup-time
   computation).  Only analysis and synthesis are differentiable.

Basic usage
-----------

.. code-block:: python

   import torch
   import cool_frames.torch as lfb_t
   from cool_frames.numpy.filters import audfilters
   from cool_frames.numpy.filterbanks import filterbankdual

   # Filter design (NumPy, done once)
   fs, Ls = 16_000, 32_000
   g, a, fc, L, info = audfilters(fs, Ls)
   gd = filterbankdual(g, a, L)

   # Input signal as a torch tensor
   x = torch.randn(Ls, dtype=torch.float64, requires_grad=True)

   # Analysis
   c = lfb_t.filterbanks.filterbank(x, g, a, L=L)

   # Synthesis
   x_rec = lfb_t.filterbanks.ifilterbank(c, gd, a, Ls=Ls, real=True)

   # Perfect reconstruction
   print((x - x_rec).abs().max())   # < 1e-10

Backpropagation
---------------

Gradients flow through both analysis and synthesis, enabling end-to-end
training of audio models that include filterbank processing:

.. code-block:: python

   loss = x_rec.abs().sum()
   loss.backward()
   print(x.grad.norm())   # ∂loss / ∂x

Channel-selective loss functions
---------------------------------

Because each element of ``c`` is a separate tensor, frequency-selective
objectives are straightforward:

.. code-block:: python

   import torch
   import numpy as np

   c = lfb_t.filterbanks.filterbank(x, g, a, L=L)

   # Penalise energy in channels above 4 kHz
   high_mask = torch.tensor(fc > 4000, dtype=torch.float64)
   energies  = torch.stack([ci.abs().pow(2).mean() for ci in c])
   loss_hf   = (high_mask * energies).sum()
   loss_hf.backward()

Phase retrieval via gradient descent
--------------------------------------

The differentiable filterbank supports gradient-based phase retrieval
(see :mod:`cool_frames.torch.phase` for the dedicated algorithms):

.. code-block:: python

   # Reference magnitudes
   s_mag = [ci.detach().abs() for ci in c]

   x_est = torch.zeros(Ls, dtype=torch.float64, requires_grad=True)
   opt   = torch.optim.Adam([x_est], lr=1e-2)

   for _ in range(200):
       opt.zero_grad()
       c_est = lfb_t.filterbanks.filterbank(x_est, g, a, L=L)
       loss  = sum(
           (ci.abs() - s_ref).pow(2).mean()
           for ci, s_ref in zip(c_est, s_mag)
       )
       loss.backward()
       opt.step()

Dedicated phase-retrieval algorithms (GLA, ADMM, RAAR, Difference Map)
are available in :mod:`cool_frames.torch.phase` and converge much
faster than vanilla gradient descent.

GPU acceleration
----------------

Move filters and signal to CUDA with the standard PyTorch API:

.. code-block:: python

   device = "cuda" if torch.cuda.is_available() else "cpu"
   x_gpu  = x.to(device)

   # g and gd are lists of NumPy arrays; the filterbank
   # automatically moves operations to the same device as x.
   c_gpu = lfb_t.filterbanks.filterbank(x_gpu, g, a, L=L)

Signal-processing utilities
----------------------------

:mod:`cool_frames.torch.sigproc` provides ``thresh``, a differentiable
coefficient-thresholding helper with hard, soft and Wiener modes:

.. code-block:: python

   import torch
   from cool_frames.torch.sigproc import thresh

   c_dn, n_kept = thresh(torch.randn(64, dtype=torch.float64), 0.5, mode='soft')
