#!/usr/bin/env python3
"""Check that every example script and notebook under docs/ parses (#944).

No CI ran the examples, so an example could stop compiling without anyone
noticing (Ex_NS_Couette_Flow.py had an IndentationError on development).
This gate is the cheap half of #944: it compiles each example, and runs none.

- every ``*.py`` under ``docs/examples``: the whole file is compiled;
- every ``*.ipynb`` under ``docs``: each code cell is compiled on its own,
  after IPython syntax is removed. A cell magic whose body is Python
  (``%%time``, ``%%timeit``, ``%%capture``, ``%%prun``) has its first line
  removed and its body checked; any other cell magic (``%%bash``,
  ``%%writefile`` ...) skips the cell. A line magic (``%...``), a shell escape
  (``!...``) or a help query (``x?``) becomes ``pass`` at the same indent, so
  the line numbers still match the cell and an enclosing block stays valid.

Standard library only, so it runs in the style-gates job without installing
Underworld. Exit status: 0 if everything parses, 1 otherwise, with each
failure listed as ``path[:cell N]:line: message``.

    python scripts/check_examples_compile.py [ROOT]
"""
import json
import re
import sys
import warnings
from pathlib import Path

# Python 3.12 still accepts these, so they are no reason to fail the gate.
warnings.filterwarnings("ignore", category=SyntaxWarning)

# The whole stripped line is an IPython line magic, shell escape or help query.
_IPYTHON_LINE = re.compile(r"^(\s*)([%!].*|[\w.]+\?{1,2}|\?{1,2}[\w.]+)\s*$")

# Cell magics whose body is Python, and is checked.
_PYTHON_CELL_MAGICS = {"time", "timeit", "capture", "prun"}


def _compile(source, label):
    try:
        compile(source, label, "exec", dont_inherit=True)
    except SyntaxError as exc:  # IndentationError and TabError are subclasses
        return f"{label}:{exc.lineno}: {type(exc).__name__}: {exc.msg}"
    return None


def check_script(path):
    """The failure message for one example script, or None."""
    return _compile(path.read_text(encoding="utf-8"), str(path))


def _cell_source(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


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
        lines = _cell_source(cell).splitlines()
        head = next((i for i, line in enumerate(lines) if line.strip()), None)
        if head is not None and lines[head].lstrip().startswith("%%"):
            name = (lines[head].lstrip()[2:].split() or [""])[0]
            if name not in _PYTHON_CELL_MAGICS:
                continue  # a cell magic whose body is not Python
            lines[head] = ""  # the body is Python; keep the line numbers
        lines = [m.group(1) + "pass" if (m := _IPYTHON_LINE.match(line)) else line
                 for line in lines]
        msg = _compile("\n".join(lines) + "\n", f"{path}:cell {n}")
        if msg:
            failures.append(msg)
    return failures


def main(argv):
    root = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent
    scripts = sorted((root / "docs" / "examples").rglob("*.py"))
    notebooks = sorted(p for p in (root / "docs").rglob("*.ipynb")
                       if ".ipynb_checkpoints" not in p.parts)
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
    print(f"{len(scripts)} example scripts and {len(notebooks)} notebooks checked; "
          f"{len(failures)} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
