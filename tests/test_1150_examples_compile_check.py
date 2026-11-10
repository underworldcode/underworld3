"""The examples parse gate (scripts/check_examples_compile.py) catches what broke
the examples, and passes the IPython syntax a notebook may carry (#944).

The breakages it was written for were found on development a6eebb61:
- a rank guard, ``if uw.mpi.rank == 0:``, turned into ``uw.pprint(...)`` with the
  rest of its block left indented. Ex_NS_Couette_Flow.py and six other examples
  stopped compiling.
- a scratch line, ``uw.`` or ``Pm = ``, left in a cell.
- IPython automagic, ``ls ...`` with no ``!``, in a notebook cell.

The gate itself runs in the style-gates CI job and imports nothing from
underworld3, so this test needs no build either.
"""
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_examples_compile.py"


def _gate():
    spec = importlib.util.spec_from_file_location("check_examples_compile", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(tmp_path, scripts=(), notebooks=()):
    examples = tmp_path / "docs" / "examples"
    examples.mkdir(parents=True)
    for name, text in scripts:
        (examples / name).write_text(text)
    for name, cells in notebooks:
        nb = {"cells": [{"cell_type": "code", "source": src, "metadata": {},
                         "outputs": [], "execution_count": None} for src in cells],
              "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        (examples / name).write_text(json.dumps(nb))
    return tmp_path


def test_an_orphaned_rank_guard_block_fails_the_gate(tmp_path, capsys):
    broken = ('import underworld3 as uw\n'
              'uw.pprint("min X, max X:", 0, 1)\n'
              '    print("min Y, max Y:", 0, 1)\n')
    root = _tree(tmp_path, scripts=[("Ex_Broken.py", broken)])
    assert _gate().main(["gate", str(root)]) == 1
    out = capsys.readouterr().out
    assert "docs/examples/Ex_Broken.py:3: IndentationError" in out


def test_a_scratch_line_fails_the_gate(tmp_path, capsys):
    root = _tree(tmp_path, scripts=[("Ex_Scratch.py", "# %%\nPm = \n\nCv\n")])
    assert _gate().main(["gate", str(root)]) == 1
    assert "Ex_Scratch.py:2: SyntaxError" in capsys.readouterr().out


def test_notebook_automagic_fails_and_an_explicit_shell_escape_passes(tmp_path, capsys):
    root = _tree(tmp_path, notebooks=[("auto.ipynb", ["x = 1\n", "ls -trl output/\n"])])
    assert _gate().main(["gate", str(root)]) == 1
    assert "auto.ipynb:cell 2:1: SyntaxError" in capsys.readouterr().out

    root = _tree(tmp_path / "fixed",
                 notebooks=[("shell.ipynb", ["x = 1\n", "!ls -trl output/\n"])])
    assert _gate().main(["gate", str(root)]) == 0


def test_ipython_syntax_in_notebooks_passes(tmp_path):
    cells = [
        "%matplotlib inline\nimport math\n",        # a line magic
        "%%bash\nthis is not python at all\n",      # a shell cell magic skips the cell
        "!pip list | head\nx = math.pi\n",          # a shell escape
        "math.sqrt?\n",                            # a help query
        "def f(a):\n    return a\n",
    ]
    root = _tree(tmp_path, notebooks=[("ok.ipynb", cells)])
    assert _gate().main(["gate", str(root)]) == 0


def test_a_magic_keeps_the_reported_line_number(tmp_path, capsys):
    root = _tree(tmp_path, notebooks=[("lines.ipynb", ["%matplotlib inline\nx = (\n"])])
    assert _gate().main(["gate", str(root)]) == 1
    assert "lines.ipynb:cell 1:2:" in capsys.readouterr().out


def test_a_python_cell_magic_body_is_checked_and_a_shell_one_is_not(tmp_path, capsys):
    """%%time and %%timeit run Python, so their bodies are compiled; %%bash is
    skipped, also when the cell opens with a blank line."""
    root = _tree(tmp_path, notebooks=[("timed.ipynb", ["%%timeit\nx = (\n"])])
    assert _gate().main(["gate", str(root)]) == 1
    assert "timed.ipynb:cell 1:2:" in capsys.readouterr().out

    root = _tree(tmp_path / "shell",
                 notebooks=[("bash.ipynb", ["\n%%bash\nls -l | wc -l\nfor f in *; do echo $f; done\n"])])
    assert _gate().main(["gate", str(root)]) == 0


def test_an_indented_magic_keeps_its_block_valid(tmp_path):
    """A magic or shell escape alone in a block becomes `pass`, not a blank line,
    so the block it sits in still parses."""
    cells = ["if True:\n    !echo hello\n", "for i in range(2):\n    %time i\n"]
    root = _tree(tmp_path, notebooks=[("blocks.ipynb", cells)])
    assert _gate().main(["gate", str(root)]) == 0


def test_a_malformed_notebook_fails_without_crashing(tmp_path, capsys):
    examples = tmp_path / "docs" / "examples"
    examples.mkdir(parents=True)
    (examples / "list.ipynb").write_text("[]")
    (examples / "nocells.ipynb").write_text('{"cells": null}')
    assert _gate().main(["gate", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "list.ipynb: not an nbformat 4 notebook" in out
    assert "nocells.ipynb: not an nbformat 4 notebook" in out


def test_the_repository_examples_parse(capsys):
    """Every example script and notebook in this tree parses: what CI checks."""
    rc = _gate().main(["gate"])
    out = capsys.readouterr().out
    assert rc == 0, out
