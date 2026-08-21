Quickstart
==========

ERB filterbank analysis/synthesis
----------------------------------

.. code-block:: python

   import numpy as np
   from cool_frames.numpy.filters import audfilters
   from cool_frames.numpy.filterbanks import (
       filterbank, ifilterbank, filterbankdual,
   )

   # Design an ERB-scale filterbank
   fs = 16000
   Ls = fs * 2  # 2 seconds of audio
   g, a, fc, L, info = audfilters(fs, Ls)

   # Compute the dual window for perfect reconstruction
   gd = filterbankdual(g, a, L)

   # Analyse a test signal
   x = np.random.randn(Ls)
   c = filterbank(x, g, a, L=L)

   # Synthesise back
   x_rec = ifilterbank(c, gd, a, Ls=Ls, real=True)

   # Verify perfect reconstruction
   print(f"Reconstruction error: {np.max(np.abs(x - x_rec)):.2e}")

PyTorch backend
---------------

.. code-block:: python

   import torch
   import cool_frames.torch as lfb_t
   from cool_frames.numpy.filters import audfilters
   from cool_frames.numpy.filterbanks import filterbankdual

   # Filter design still uses NumPy (setup-time)
   fs, Ls = 16000, 32000
   g, a, fc, L, info = audfilters(fs, Ls)
   gd = filterbankdual(g, a, L)

   # Convert signal to torch tensor
   x = torch.randn(Ls, dtype=torch.float64, requires_grad=True)

   # Analysis and synthesis are differentiable
   c = lfb_t.filterbanks.filterbank(x, g, a, L=L)
   x_rec = lfb_t.filterbanks.ifilterbank(c, gd, a, Ls=Ls)  # real=True default

   # Backpropagate through the filterbank
   loss = x_rec.abs().sum()
   loss.backward()
   print(f"Gradient norm: {x.grad.norm():.4f}")
