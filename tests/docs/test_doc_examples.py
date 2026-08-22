"""
test_doc_examples.py
====================
Executes every Python example in ``README.md`` and ``docs/**/*.rst``.

Why this exists
---------------
Documentation examples are the first code a new user runs, and they rot
silently: a function gains a return value, a keyword is renamed, a module
moves, and the tutorial keeps rendering perfectly while no longer working.
Before this file existed, ``docs/tutorials/03_phase_retrieval.rst`` unpacked a
four-element return into two names, imported a function that had never been
exported, and fed ``spsi`` centre frequencies in the wrong unit — none of which
any test could see.

How it works
------------
For each document, the Python blocks are extracted **in order** and executed
one after another in a single shared namespace, so a later block can use names
that an earlier one defined — which is how a reader consumes them.  Each
document is seeded with a small preamble (``fs``, ``Ls``, a test signal ``f``)
standing in for the "assume you have a signal" sentence the prose opens with.

Opting a block out
------------------
Put ``# doctest: +SKIP`` on a line of a block that cannot run in CI — one that
reads a sound file from disk, plots to a window, or trains for ten minutes.
Use it sparingly: a skipped block is an untested block, and the skip count is
asserted below so that opting out is a visible decision rather than a quiet
one.
"""

from __future__ import annotations

import pathlib
import re

import pytest

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Blocks that legitimately cannot run under pytest, by (document, reason).
# Keep this list short; every entry is documentation nobody is checking.
MAX_SKIPPED_BLOCKS = 8

_RST_BLOCK = re.compile(
    r"^\.\. code-block:: python\s*\n"      # directive
    r"(?:^[ \t]*:\S+:.*\n)*"               # optional directive options
    r"\s*?\n"                              # blank line
    r"((?:^(?:[ \t]+.*)?\n)+)",            # the indented body
    re.MULTILINE,
)
_MD_BLOCK = re.compile(r"^```python\s*\n(.*?)^```", re.MULTILINE | re.DOTALL)


def _dedent(block: str) -> str:
    lines = [ln for ln in block.splitlines() if ln.strip()]
    if not lines:
        return ""
    indent = min(len(ln) - len(ln.lstrip()) for ln in lines)
    return "\n".join(ln[indent:] if len(ln) >= indent else ln for ln in block.splitlines())


def _blocks(path: pathlib.Path) -> list[str]:
    text = path.read_text()
    pattern = _MD_BLOCK if path.suffix == ".md" else _RST_BLOCK
    return [_dedent(m.group(1)).strip("\n") for m in pattern.finditer(text)]


def _documents() -> list[pathlib.Path]:
    docs = sorted((_ROOT / "docs").rglob("*.rst"))
    readme = _ROOT / "README.md"
    out = [p for p in docs if _blocks(p)]
    if readme.exists() and _blocks(readme):
        out.append(readme)
    return out


_PREAMBLE = """
import numpy as np
fs = 8000
Ls = 2048
_t = np.arange(Ls) / fs
f = (np.sin(2 * np.pi * 440 * _t) + 0.5 * np.sin(2 * np.pi * 1320 * _t)) * np.hanning(Ls)
x = f
signal = f
audio = f
"""

_DOCS = _documents()

try:  # torch is an optional extra; the numpy CI job does not install it.
    import torch  # noqa: F401

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _needs_torch(block: str) -> bool:
    return "import torch" in block or "cool_frames.torch" in block


@pytest.mark.requires_impl
@pytest.mark.parametrize("doc", _DOCS, ids=lambda p: str(p.relative_to(_ROOT)))
def test_documentation_examples_run(doc):
    """Every non-skipped Python block in ``doc`` executes without raising."""
    import matplotlib

    matplotlib.use("Agg")

    blocks = _blocks(doc)
    assert blocks, f"{doc}: no Python blocks found — has the markup changed?"

    ns: dict = {"__name__": "__doc_example__"}
    exec(compile(_PREAMBLE, f"<preamble:{doc.name}>", "exec"), ns)

    for i, block in enumerate(blocks, start=1):
        if "doctest: +SKIP" in block:
            continue
        if _needs_torch(block) and not _HAS_TORCH:
            pytest.skip("torch is not installed; the torch examples cannot run")
        try:
            exec(compile(block, f"{doc}#block{i}", "exec"), ns)
        except Exception as exc:
            pytest.fail(
                f"{doc.relative_to(_ROOT)} block {i} failed with "
                f"{type(exc).__name__}: {exc}\n"
                f"--- block ---\n{block}\n-------------"
            )


@pytest.mark.requires_impl
def test_the_number_of_skipped_blocks_does_not_creep():
    """Opting a block out of execution must stay a rare, deliberate act."""
    skipped = [
        (doc, i)
        for doc in _DOCS
        for i, block in enumerate(_blocks(doc), start=1)
        if "doctest: +SKIP" in block
    ]
    assert len(skipped) <= MAX_SKIPPED_BLOCKS, (
        f"{len(skipped)} documentation blocks are marked +SKIP "
        f"(limit {MAX_SKIPPED_BLOCKS}): "
        + ", ".join(f"{d.relative_to(_ROOT)}#{i}" for d, i in skipped)
    )


def _module_exists_on_disk(mod_name: str) -> bool:
    """Does this module exist in the source tree, without importing anything?

    ``importlib`` cannot answer this question in a job where an optional
    dependency is absent: importing ``cool_frames.torch.filters`` raises
    ``ImportError`` both when the module does not exist *and* when it exists
    but torch is not installed, and even ``find_spec`` has to import the parent
    package to look inside it.  Checking the filesystem separates the two
    cleanly and costs nothing.
    """
    base = _ROOT.joinpath(*mod_name.split("."))
    return (base / "__init__.py").exists() or base.with_suffix(".py").exists()


@pytest.mark.requires_impl
def test_documented_import_paths_exist():
    """Every ``from cool_frames... import ...`` in the docs resolves.

    Catches the failure mode where a document imports a name that was never
    exported (``rtpghifb`` once was) — cheaper and clearer than waiting for the
    block that uses it to blow up.

    The module path is checked on disk and the *names* only when the module
    actually imports.  Both halves matter and they fail differently: a wrong
    path is always wrong, whereas a name can only be verified where the
    module's dependencies are installed.  Conflating them made this test fail
    CI's NumPy-only job for three ``cool_frames.torch.*`` paths that were
    perfectly correct — torch simply is not installed there — which is a false
    alarm about the documentation and, worse, would have trained the next
    person to ignore it.
    """
    import importlib

    pattern = re.compile(r"^\s*from\s+(cool_frames[\w.]*)\s+import\s+(.+)$", re.MULTILINE)
    missing: list[str] = []
    unverified: set[str] = set()
    for doc in _DOCS:
        for block in _blocks(doc):
            for mod_name, names in pattern.findall(block):
                where = doc.relative_to(_ROOT)
                if not _module_exists_on_disk(mod_name):
                    missing.append(f"{where}: no module {mod_name}")
                    continue
                try:
                    mod = importlib.import_module(mod_name)
                except ImportError:
                    # The path is right; an optional dependency is missing in
                    # this environment, so the names cannot be checked here.
                    unverified.add(mod_name)
                    continue
                for name in (n.strip() for n in names.split(",")):
                    name = name.split(" as ")[0].strip("() \t")
                    if name and not hasattr(mod, name):
                        missing.append(f"{where}: {mod_name} has no {name!r}")

    assert not missing, "documented imports that do not resolve:\n  " + "\n  ".join(missing)

    if unverified:
        # Reported rather than silent: if this ever lists a module that should
        # have been importable, the skip is hiding something.
        print(
            "\nimport names not verified here (module present, dependency absent): "
            + ", ".join(sorted(unverified))
        )


def test_every_document_with_prose_examples_is_collected():
    """Guard against the extraction regex silently matching nothing."""
    assert len(_DOCS) >= 5, f"only {len(_DOCS)} documents yielded Python blocks"
    total = sum(len(_blocks(d)) for d in _DOCS)
    assert total >= 15, f"only {total} Python blocks extracted in total"
    assert np is not None
