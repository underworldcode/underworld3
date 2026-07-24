"""MeshVariable.stats() over the variable types.

Regression coverage for:

- issue #400: ``_tensor_stats`` indexed the flat component count (d*d)
  against the structured ``.array`` axis of size d — ``stats()`` on any
  genuine TENSOR variable raised IndexError;
- issue #384 lifecycle: the stats temporaries are created and cleaned up
  by name — repeated calls must work and leave no variable behind.
"""

import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture(scope="module")
def mesh():
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)


def test_tensor_stats_computes_frobenius(mesh):
    T = uw.discretisation.MeshVariable(
        "T400", mesh, (mesh.dim, mesh.dim), vtype=uw.VarType.TENSOR, degree=1)
    # Identity everywhere: ||I||_F = sqrt(d)
    with uw.synchronised_array_update():
        T.array[:, :, :] = np.eye(mesh.dim)

    result = T.stats()
    assert result["type"] == "tensor"
    expected = np.sqrt(mesh.dim)
    assert np.isclose(result["mean"], expected, rtol=1e-12)
    assert np.isclose(result["max"], expected, rtol=1e-12)
    assert np.isclose(result["min"], expected, rtol=1e-12)


def test_vector_stats_repeat_and_cleanup(mesh):
    V = uw.discretisation.MeshVariable("V400", mesh, mesh.dim, degree=1)
    with uw.synchronised_array_update():
        V.array[:, 0, 0] = 3.0
        V.array[:, 0, 1] = 4.0

    for _ in range(2):  # repeated calls: temp is recreated by name each time
        result = V.stats()
        assert np.isclose(result["magnitude_mean"], 5.0, rtol=1e-12)

    leftovers = [name for name in mesh.vars if name.startswith("_temp_")]
    assert leftovers == [], f"stats temporaries left behind: {leftovers}"
