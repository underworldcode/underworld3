"""The remap of a smooth field at a slipped wall does not depend on the rank count:
the same accuracy as serial, and a ghost holds the same value as its owner.
See tests/test_0768_remap_at_a_slipped_wall.py.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(120)]


@pytest.mark.mpi(min_size=2)
def test_the_remap_is_partition_independent():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    field = (r - 0.5) * 2 + 0.5 * sympy.cos(4 * sympy.atan2(y, x)) * r
    T = uw.discretisation.MeshVariable("T0768", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(field, T.coords).reshape(-1)

    coords = np.array(mesh.X.coords)
    radius = np.linalg.norm(coords, axis=1)
    wall = (np.abs(radius - 1.0) < 1.0e-9) | (np.abs(radius - 0.5) < 1.0e-9)
    angle = 0.2 * (1 / 8) / 0.5
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    coords[wall] = coords[wall] @ rotation
    mesh.deform(coords)

    largest = lambda values: uw.mpi.comm.allreduce(
        float(values.max()) if values.size else 0.0, op=uw.MPI.MAX)
    dof = np.asarray(T.coords).reshape(-1, 2)
    value = np.asarray(T.array).reshape(-1)
    error = np.abs(value - uw.function.evaluate(field, dof).reshape(-1))
    dof_radius = np.linalg.norm(dof, axis=1)
    on_wall = (dof_radius > 0.99) | (dof_radius < 0.51)
    assert largest(error[on_wall]) < 1.0e-2      # serial 4.4e-3; under the clamp 7e-2 to 1e-1
    assert largest(error[~on_wall]) < 2.0e-3     # serial 8.3e-4; under the clamp 4e-2 to 5e-2

    iset, subdm = mesh.dm.createSubDM(T.field_id)
    gvec = subdm.createGlobalVector()
    lvec = subdm.createLocalVector()
    subdm.localToGlobal(T.vec, gvec, addv=False)
    subdm.globalToLocal(gvec, lvec, addv=False)
    assert largest(np.abs(value - lvec.array)) < 1.0e-12     # 8e-2 under the clamp


@pytest.mark.mpi(min_size=2)
def test_a_simplex_mesh_with_an_empty_rank_joins_the_transfer():
    """dm.isSimplex() is False on a rank with no cells, so a test on it would leave that
    rank outside the facet gather. The box is small enough to leave a rank empty."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=2.0, qdegree=3)
    start, end = mesh.dm.getHeightStratum(0)
    if 0 not in uw.mpi.comm.allgather(end - start):
        pytest.skip("every rank has cells at this rank count")
    T = uw.discretisation.MeshVariable("T0768e", mesh, 1, degree=1)
    dof = np.asarray(T.coords)
    T.array[:, 0, 0] = dof[:, 0] + 2.0 * dof[:, 1]
    coords = np.array(mesh.X.coords)
    coords[:, 1] *= 1.0 + 0.05 * coords[:, 0]
    assert all(uw.mpi.comm.allgather(mesh.deform(coords)))
