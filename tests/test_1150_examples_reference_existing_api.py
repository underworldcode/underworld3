"""Every ``uw.<module>.<name>`` an example names still exists (#944).

``uw.function.evalf`` was removed in July 2025, replaced by
``uw.function.evaluate(..., rbf=True)``, but 25 scripts under ``docs/examples``
still called it, so they parsed and then failed at run time. The parse gate
(test_1150_examples_compile_check.py) cannot see that. This scan imports
underworld3 and checks every attribute an example reaches through one of the
public submodules.

``WIP/`` is excluded, as it is from the parse gate. ``_KNOWN_MISSING`` lists the
references that have no direct successor and need the example rewritten; it is
a ratchet, so delete an entry when its example is fixed.
"""
import ast
import json
from pathlib import Path

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

_DOCS = Path(__file__).resolve().parent.parent / "docs"
_SUBMODULES = ("function", "systems", "meshing", "discretisation", "swarm", "maths",
               "constitutive_models", "visualisation", "scaling", "mpi")

_KNOWN_MISSING = {
    # SLCN became AdvDiffusionSLCN with a different constructor.
    "examples/heat_transfer/intermediate/diffusion_slcn_cartesian.py: uw.systems.SLCN",
    # No successor: SegmentofSphere is a different mesh.
    "examples/utilities/advanced/Manifold_S2.5-SphCoords.py: uw.meshing.SegmentedSphere",
    "examples/utilities/advanced/Manifold_S2.5.py: uw.meshing.SegmentedSphere",
}


def _code(path):
    if path.suffix == ".py":
        return path.read_text(errors="replace")
    cells = json.loads(path.read_text())["cells"]
    source = ("".join(c["source"]) if isinstance(c["source"], list) else c["source"]
              for c in cells if c["cell_type"] == "code")
    # IPython magics and shell escapes are not Python.
    return "\n".join(line for line in "\n".join(source).splitlines()
                     if not line.lstrip().startswith(("%", "!")))


def _references():
    found = set()
    for path in sorted([*_DOCS.rglob("*.py"), *_DOCS.rglob("*.ipynb")]):
        if "WIP" in path.parts or ".ipynb_checkpoints" in path.parts:
            continue
        try:
            tree = ast.parse(_code(path))
        except SyntaxError:
            continue  # the parse gate reports these
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
                    and isinstance(node.value.value, ast.Name) and node.value.value.id == "uw"
                    and node.value.attr in _SUBMODULES):
                found.add((path.relative_to(_DOCS).as_posix(), node.value.attr, node.attr))
    return found


def test_examples_reference_only_existing_api():
    missing = {f"{where}: uw.{module}.{name}" for where, module, name in _references()
               if not hasattr(getattr(uw, module), name)}
    assert missing - _KNOWN_MISSING == set(), sorted(missing - _KNOWN_MISSING)


def test_the_ratchet_holds_no_fixed_entry():
    missing = {f"{where}: uw.{module}.{name}" for where, module, name in _references()
               if not hasattr(getattr(uw, module), name)}
    assert _KNOWN_MISSING - missing == set(), sorted(_KNOWN_MISSING - missing)
