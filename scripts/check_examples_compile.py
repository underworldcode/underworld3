#!/usr/bin/env python3
"""Check that every example script and notebook under docs/ parses (#944).

No CI ran the examples, so an example could stop compiling without anyone
noticing (Ex_NS_Couette_Flow.py had an IndentationError on development).
This gate is the cheap half of #944: it compiles each example, and runs none.

- every ``*.py`` under ``docs``: the whole file is compiled from its bytes, so
  a UTF-8 BOM or a PEP 263 coding line is honoured;
- every ``*.ipynb`` under ``docs``: each code cell is compiled on its own.
  - A cell that is valid Python as it stands passes unchanged.
  - Otherwise IPython syntax is removed and the cell is compiled again:
    - A cell magic whose body is Python (``%%time``, ``%%timeit``,
      ``%%capture``, ``%%prun``) has its first line removed and its body
      checked. Any other cell magic (``%%bash``, ``%%writefile`` ...) skips
      the cell.
    - ``%time <statement>`` and ``%timeit <statement>`` are replaced by the
      statement, which is checked.
    - Any other line magic (``%...``), a shell escape (``!...``) or a help
      query (``x?``) becomes ``pass`` at the same indent, so the line numbers
      still match the cell and an enclosing block stays valid.
  - Top-level ``await`` is allowed, as in a kernel.

Files under ``.ipynb_checkpoints`` are skipped. Standard library only, so it
runs in the style-gates job without installing Underworld. CI runs it on the
oldest Python the package supports, because a newer parser accepts syntax an
older one rejects. Exit status: 0 if everything parses, 1 otherwise, with each
failure listed as ``path[:cell N]:line: message``.

    python scripts/check_examples_compile.py [ROOT]
"""
import ast
import json
import re
import sys
import warnings
from pathlib import Path

# Invalid escape sequences and the like are still accepted, so they are no
# reason to fail the gate.
warnings.filterwarnings("ignore", category=SyntaxWarning)

# The whole stripped line is an IPython line magic, shell escape or help query.
_IPYTHON_LINE = re.compile(r"^(\s*)([%!].*|[\w.]+\?{1,2}|\?{1,2}[\w.]+)\s*$")

# `%time stmt` / `%timeit stmt` with no options: the statement is Python.
_TIMED_LINE = re.compile(r"^(\s*)%time(?:it)?\s+(?!-)(.+?)\s*$")

# Cell magics whose body is Python, and is checked.
_PYTHON_CELL_MAGICS = {"time", "timeit", "capture", "prun"}

_CELL_FLAGS = getattr(ast, "PyCF_ALLOW_TOP_LEVEL_AWAIT", 0)


def _compile(source, label, flags=0):
    try:
        compile(source, label, "exec", flags=flags, dont_inherit=True)
    except SyntaxError as exc:  # IndentationError and TabError are subclasses
        return f"{label}:{exc.lineno}: {type(exc).__name__}: {exc.msg}"
    except (ValueError, UnicodeDecodeError) as exc:  # null bytes, bad encoding
        return f"{label}: {type(exc).__name__}: {exc}"
    return None


def check_script(path):
    """The failure message for one example script, or None."""
    return _compile(path.read_bytes(), str(path))


def _cell_source(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def _strip_ipython(lines):
    """The cell's lines with IPython syntax replaced, or None to skip the cell."""
    head = next((i for i, line in enumerate(lines) if line.strip()), None)
    if head is not None and lines[head].lstrip().startswith("%%"):
        name = (lines[head].lstrip()[2:].split() or [""])[0]
        if name not in _PYTHON_CELL_MAGICS:
            return None  # a cell magic whose body is not Python
        lines = list(lines)
        lines[head] = ""  # the body is Python; keep the line numbers
    out = []
    for line in lines:
        timed = _TIMED_LINE.match(line)
        if timed:
            out.append(timed.group(1) + timed.group(2))
            continue
        magic = _IPYTHON_LINE.match(line)
        out.append(magic.group(1) + "pass" if magic else line)
    return out


def check_notebook(path):
    """The failure messages for one notebook's code cells."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return [f"{path}: not a readable notebook: {exc}"]
    if not isinstance(nb, dict) or not isinstance(nb.get("cells"), list):
        return [f"{path}: not an nbformat 4 notebook (no cell list)"]
    failures = []
    code_cells = (c for c in nb["cells"]
                  if isinstance(c, dict) and c.get("cell_type") == "code")
    for n, cell in enumerate(code_cells, start=1):
        source = _cell_source(cell)
        label = f"{path}:cell {n}"
        if _compile(source, label, _CELL_FLAGS) is None:
            continue  # valid Python as it stands
        lines = _strip_ipython(source.split("\n"))
        if lines is None:
            continue
        msg = _compile("\n".join(lines) + "\n", label, _CELL_FLAGS)
        if msg:
            failures.append(msg)
    return failures


def _files(root, pattern):
    return sorted(p for p in (root / "docs").rglob(pattern)
                  if ".ipynb_checkpoints" not in p.parts)


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent
    scripts = _files(root, "*.py")
    notebooks = _files(root, "*.ipynb")
    failures = []
    for path in scripts:
        msg = check_script(path)
        if msg:
            failures.append(msg)
    for path in notebooks:
        failures.extend(check_notebook(path))
    rel = lambda m: m.replace(str(root) + "/", "")
    for msg in failures:
        print(rel(msg))
    print(f"{len(scripts)} scripts and {len(notebooks)} notebooks under docs/ checked; "
          f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
