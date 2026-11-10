"""Full-DM integrals retain retired field slots; solver sub-DMs stay compact."""

import numpy as np
import pytest
import underworld3 as uw

from underworld3.utilities import _jitextension as jit

pytestmark = pytest.mark.level_1


@pytest.fixture
def retired_field_mesh():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0., 0.), maxCoords=(1., 1.), cellSize=.5,
        regular=True, qdegree=3,
    )
    retired = uw.discretisation.MeshVariable("RetiredPrimaryVector", mesh, 2, degree=1)
    scalar = uw.discretisation.MeshVariable("RemainingPrimaryScalar", mesh, 1, degree=1)
    vector = uw.discretisation.MeshVariable("RemainingPrimaryVector", mesh, 2, degree=1)
    retired.array[:, 0, :] = 99
    x, y = np.asarray(scalar.coords).T
    scalar.array[:, 0, 0] = 2+3*x-4*y
    x, y = np.asarray(vector.coords).T
    vector.array[:, 0, :] = np.column_stack((x+y, 2*x-y))
    field_count = mesh.dm.getNumFields()
    # Controlled retirement models an orphaned DM field without relying on GC
    # timing. Keep the Python object alive, and preserve the actual DM layout.
    mesh.vars.pop("RetiredPrimaryVector")
    mesh._stale_lvec = True
    assert mesh.dm.getNumFields() == field_count == 3
    assert mesh.dm.getNumFields() > len(mesh.vars)
    return mesh, scalar, vector, retired


@pytest.mark.tier_b
@pytest.mark.parametrize("route", ("graph", "expanded"))
def test_integrals_use_full_primary_dm_offsets(retired_field_mesh, monkeypatch, route):
    mesh, scalar, vector, retired = retired_field_mesh
    monkeypatch.setattr(jit, "_jit_route_default", route)
    x, _ = mesh.X
    b = scalar.sym[0]
    for expression, exact in ((b, 1.5), (b*b, 13/3), (b.diff(x), 3.),
                              (vector.sym.dot(vector.sym), 11/6)):
        np.testing.assert_allclose(uw.maths.Integral(mesh, expression).evaluate(),
                                   exact, rtol=2e-12, atol=2e-12)
    for expression, exact in ((b, -.5), (b.diff(x), 3.)):
        np.testing.assert_allclose(uw.maths.BdIntegral(mesh, expression, "Top").evaluate(),
                                   exact, rtol=2e-12, atol=2e-12)
    measured = uw.maths.CellWiseIntegral(mesh, b).evaluate()
    owned = mesh._get_owned_cells_mask()
    cstart, _ = mesh.dm.getHeightStratum(0)
    expected = []
    for cell in np.flatnonzero(owned):
        volume, centroid, _ = mesh.dm.computeCellGeometryFVM(int(cell)+cstart)
        expected.append(volume*(2+3*centroid[0]-4*centroid[1]))
    np.testing.assert_allclose(measured[owned], expected, rtol=2e-12, atol=2e-12)
    assert mesh.dm.getNumFields() == 3 and len(mesh.vars) == 2


@pytest.mark.tier_b
def test_solver_primary_subset_keeps_compact_default(retired_field_mesh):
    mesh, scalar, vector, retired = retired_field_mesh
    primary = (scalar, vector)
    compact = jit._leaf_spellings(mesh, primary)
    full = jit._leaf_spellings(mesh, primary, jit._aux_component_offsets(mesh))
    assert compact[type(scalar.sym[0])] == "petsc_u[0]"
    assert compact[type(vector.sym[0])] == "petsc_u[1]"
    assert compact[type(vector.sym[1])] == "petsc_u[2]"
    assert full[type(scalar.sym[0])] == "petsc_u[2]"
    assert full[type(vector.sym[0])] == "petsc_u[3]"
    assert full[type(vector.sym[1])] == "petsc_u[4]"
