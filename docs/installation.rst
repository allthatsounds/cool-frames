Installation
============

Requirements
------------

- Python 3.10+
- NumPy >= 1.24
- SciPy >= 1.10

Install from source
-------------------

.. code-block:: bash

   git clone https://github.com/allthatsounds/cool-frames.git
   cd cool-frames
   pip install -e ".[dev]"

To include the PyTorch backend:

.. code-block:: bash

   pip install -e ".[dev,torch]"

Optional: PyTorch backend
-------------------------

The ``cool_frames.torch`` module provides GPU-accelerated,
differentiable filterbank operations. It requires PyTorch >= 2.0:

.. code-block:: bash

   pip install torch>=2.0

The torch backend uses NumPy for filter design (setup-time) and
PyTorch tensors for analysis/synthesis (run-time).
