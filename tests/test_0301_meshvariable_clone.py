r"""MeshVariable.clone — issue #498.

`EnhancedMeshVariable.clone` forwarded no arguments to a base that requires two,
so it raised whichever way it was called. Six shipped examples aborted on that
line and no test touched it.

Run: pixi run python -m pytest tests/test_0301_meshvariable_clone.py -v
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import underworld3 as uw


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.StructuredQuadBox(elementRes=(4, 4), qdegree=2)


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("vtype", ["scalar", "vector"])
def test_clone_copies_the_structure(mesh, degree, vtype):
    if vtype == "scalar":
        original = uw.discretisation.MeshVariable(
            f"s{degree}", mesh, 1, degree=degree
        )
    else:
        original = uw.discretisation.MeshVariable(
            f"v{degree}", mesh, mesh.dim, degree=degree
        )

    copy = original.clone(f"{vtype}{degree}_copy", r"{c}")

    assert copy.shape == original.shape
    assert copy.degree == original.degree
    assert copy.continuous == original.continuous
    assert copy.vtype == original.vtype
    assert copy.data.shape == original.data.shape


def test_clone_is_independent(mesh):
    """A clone shares structure, not storage."""

    original = uw.discretisation.MeshVariable("orig", mesh, 1, degree=2)
    original.data[...] = 3.0

    copy = original.clone("orig_copy", r"{o_c}")
    copy.data[...] = 7.0

    assert np.allclose(original.data, 3.0), "writing the clone changed the original"
    assert np.allclose(copy.data, 7.0)


def test_clone_takes_the_name_and_symbol_it_is_given(mesh):
    """The two arguments the wrapper used to swallow."""

    original = uw.discretisation.MeshVariable("named", mesh, 1, degree=1)
    copy = original.clone("renamed", r"{q_0}")

    assert copy.name != original.name
    assert "renamed" in copy.name


def test_clone_requires_both_arguments(mesh):
    """It is not an optional-argument API, and should not silently look like one."""

    original = uw.discretisation.MeshVariable("needsargs", mesh, 1, degree=1)

    with pytest.raises(TypeError):
        original.clone()
