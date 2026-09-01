"""
test_example_notebooks.py
=========================
Checks that every notebook under ``examples/`` still parses and still imports
things that exist.

Why this exists
---------------
All three Colab notebooks shipped in ``examples/colab/`` were broken for months
without a single test noticing.  They imported ``filterbankrealdual``,
``constphase_nonuniform``, ``cool_frames.numpy.phase._admm`` and
``cool_frames.numpy.phase._diff_admm`` -- none of which exist -- unpacked the
designers' five-element return into four names, and called helper functions
that were never defined in the notebook at all.  Nothing caught it because
``[tool.ruff] include`` is a whitelist that never mentioned ``examples/``, and
notebooks are not collected by pytest.

A reviewer's first act is to open ``02_reproduce_paper_results.ipynb`` and press
Run, so this is the most visible code in the repository and was the least
tested.

What this does and does not check
---------------------------------
It compiles every code cell (catching syntax errors and Python-version drift)
and resolves every ``cool_frames`` import (catching the exact rot above).  It
deliberately does **not** execute the notebooks: notebook 3 trains a network and
notebook 2 runs ten phase-retrieval methods over ten signals, which is minutes
of CPU, not a unit test.  Execution is a nightly job's business; this runs in
under a second and catches the failure mode that actually occurred.
"""

from __future__ import annotations

import ast
import importlib
import json
import pathlib
import warnings

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_NOTEBOOKS = sorted((_ROOT / "examples").rglob("*.ipynb"))

# If examples/ ever empties out, that is a mistake rather than a pass.
MIN_NOTEBOOKS = 3


def _cells(path: pathlib.Path) -> list[str]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        out.append("".join(cell.get("source", [])))
    return out


def _strip_magics(src: str) -> str:
    """Remove IPython line/cell magics and shell escapes, which are not Python."""
    kept = []
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("%", "!", "?")):
            kept.append(" " * (len(line) - len(stripped)) + "pass")
        else:
            kept.append(line)
    return "\n".join(kept)


def test_there_are_notebooks_to_check():
    assert len(_NOTEBOOKS) >= MIN_NOTEBOOKS, (
        f"expected at least {MIN_NOTEBOOKS} notebooks under examples/, "
        f"found {[p.name for p in _NOTEBOOKS]}")


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_and_compiles(path):
    for i, src in enumerate(_cells(path)):
        try:
            compile(_strip_magics(src), f"{path.name}:cell{i}", "exec")
        except SyntaxError as exc:
            pytest.fail(f"{path.name} cell {i} does not compile: {exc}")


def _imports(src: str):
    """Yield (module, name) pairs for every cool_frames import in ``src``."""
    tree = ast.parse(_strip_magics(src))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] == "cool_frames":
                for alias in node.names:
                    yield node.module, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "cool_frames":
                    yield alias.name, None


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_imports_resolve(path):
    """Every ``cool_frames`` name a notebook imports must actually exist.

    Two different things raise ImportError here and only one of them is the rot
    this test was written for:

    * ``cool_frames.numpy.phase._admm`` does not exist. That is the bug.
    * ``cool_frames.torch.filterbanks`` does exist, but importing it imports
      torch, and the ``test-numpy`` job installs ``.[dev]`` without torch. That
      is the environment. ``ci.yml`` already splits the doctests along exactly
      this line, for exactly this reason; this test was not given the same
      treatment and so failed on notebook 3 in all four numpy matrix entries.

    They are told apart by which module the error names: a ModuleNotFoundError
    naming something outside ``cool_frames`` is a missing optional dependency,
    and is deferred to the ``test-torch`` job, which runs this file with torch
    installed and therefore defers nothing. A bare ImportError naming no module
    counts as rot -- guessing the other way is how a real breakage goes quiet,
    which is the failure this file exists to prevent.
    """
    missing: list[str] = []
    deferred: list[str] = []
    for i, src in enumerate(_cells(path)):
        try:
            pairs = list(_imports(src))
        except SyntaxError:
            continue  # reported by the compile test above
        for module, name in pairs:
            try:
                mod = importlib.import_module(module)
            except ImportError as exc:
                absent = getattr(exc, "name", None)
                if absent is not None and absent.split(".")[0] != "cool_frames":
                    deferred.append(f"{module} (needs {absent!r})")
                else:
                    missing.append(f"cell {i}: no module {module!r} ({exc})")
                continue
            if name is not None and not hasattr(mod, name):
                missing.append(f"cell {i}: {module} has no attribute {name!r}")

    if deferred:
        # Visible in the warnings summary rather than silent: a deferral that
        # nobody can see is indistinguishable from a check that never runs.
        warnings.warn(
            f"{path.name}: {len(deferred)} import(s) not checked in this "
            "environment because an optional dependency is absent; the "
            "test-torch job checks them -- "
            + ", ".join(sorted(set(deferred))),
            stacklevel=1,
        )

    assert not missing, (
        f"{path.name} imports things that do not exist:\n  "
        + "\n  ".join(missing))


@pytest.mark.parametrize("path", _NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_uses_no_undefined_helper(path):
    """Catch calls to helpers that no cell defines and no import provides.

    Notebook 1 called ``_causal_tgrad_tick`` and notebook 2 called it too;
    neither defined it, because it left with the research code.  A name that is
    called but never bound anywhere in the notebook is that bug.
    """
    bound: set[str] = set()
    called: dict[str, int] = {}

    for i, src in enumerate(_cells(path)):
        try:
            tree = ast.parse(_strip_magics(src))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    bound.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called.setdefault(node.func.id, i)

    builtins_ = set(dir(__builtins__)) if isinstance(__builtins__, dict) is False \
        else set(__builtins__)  # type: ignore[arg-type]
    import builtins as _b
    builtins_ |= set(dir(_b))

    unresolved = {n: i for n, i in called.items()
                  if n not in bound and n not in builtins_}
    assert not unresolved, (
        f"{path.name} calls names that are never defined or imported: "
        + ", ".join(f"{n} (cell {i})" for n, i in sorted(unresolved.items())))
