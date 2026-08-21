"""Sphinx configuration for cool_frames documentation."""

import os
import sys
from unittest.mock import MagicMock

# -- Path setup ---------------------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# Mock torch so docs build without PyTorch installed
autodoc_mock_imports = ["torch"]

# -- Project information ------------------------------------------------------
project = "cool-frames"
author = "Clara Hollomey"
copyright = "2024–2026, Clara Hollomey / allthatsounds"
release = "0.1.0"

# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",       # Google/NumPy-style docstrings
    "sphinx.ext.intersphinx",    # link to NumPy/SciPy/PyTorch docs
    "sphinx.ext.viewcode",       # [source] links
    "sphinx.ext.mathjax",        # math rendering
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- autodoc / autosummary ----------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autosummary_generate = True
autodoc_typehints = "description"

# Suppress warnings for harmless duplications
# - ref.citation: duplicate citations (same reference in numpy and torch docstrings)
# - (duplicate object warnings are harder to suppress via config)
suppress_warnings = [
    "ref.python",  # Suppress Python reference warnings
    "ref.citation",  # Suppress duplicate citation warnings (same refs in numpy + torch)
]

# -- Napoleon (NumPy docstring support) ----------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# Don't create cross-references for attribute types (to avoid duplicate warnings)
napoleon_attr_annotations = False

# -- Intersphinx mappings -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}

# -- HTML output --------------------------------------------------------------
html_theme = "furo"
html_title = "cool_frames — Computational Listening Algorithms"
html_static_path = ["_static"]

# Furo options
html_theme_options = {
    "source_repository": "https://github.com/allthatsounds/cool-frames",
    "source_branch": "main",
    "source_directory": "filterbank/docs/",
}
