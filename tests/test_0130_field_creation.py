"""Successive mesh variables on one mesh get distinct field ids, and `.array` is reachable.

This file used to be a converted debug script: the mesh and the three variables
were created at module level during pytest COLLECTION, and each step sat inside
a `try/except` that printed the exception. A duplicate field id or an
unreachable `.array` printed a cross and the run stayed green.
"""

import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = pytest.mark.level_1


@pytest.fixture(scope="module")
def mesh():
    return UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2
    )


def test_successive_variables_get_distinct_field_ids(mesh):
    """Three variables of mixed rank and degree, three different field ids.

    A repeated id is the failure this file was written to catch: the second
    variable would then address the first one's DOFs.
    """

    u = uw.discretisation.MeshVariable("u", mesh, 2, vtype=uw.VarType.VECTOR, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)
    s = uw.discretisation.MeshVariable("s", mesh, 1, vtype=uw.VarType.SCALAR, degree=1)

    ids = [u.field_id, p.field_id, s.field_id]

    assert len(set(ids)) == 3, f"field ids collide: {ids}"


def test_array_is_reachable_and_shaped_by_the_variable(mesh):
    """`.array` returns storage matching the variable's own component count."""

    scalar = uw.discretisation.MeshVariable(
        "s_array", mesh, 1, vtype=uw.VarType.SCALAR, degree=1
    )
    vector = uw.discretisation.MeshVariable(
        "v_array", mesh, 2, vtype=uw.VarType.VECTOR, degree=2
    )

    assert scalar.array.shape[-1] == 1
    assert vector.array.shape[-1] == mesh.dim
    assert scalar.array.shape[0] == scalar.coords.shape[0]
