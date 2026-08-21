"""numpy.sigproc – coefficient-domain sparsity primitives (LTFAT parity).

``thresh`` and ``largest`` operate on filterbank coefficients and are the
building blocks of the sparsity-based methods (denoising, sparse solvers).
Level/companding helpers were dropped in favour of NumPy one-liners.
Mirrors ``cool_frames.torch.sigproc``.
"""
from ._sigproc import largest, thresh

__all__ = ["thresh", "largest"]
