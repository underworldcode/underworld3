"""A move is decided over all ranks, and the mesh vector is marked stale on every
rank, so a solve after a move sees the remapped field at any rank count, including
when some ranks' nodes did not move. See tests/test_0766_mover_refreshes_mesh_vector.py.
"""
import numpy as np
from mpi4py import MPI
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_a, pytest.mark.timeout(120)]


def _annulus_with_a_layer(name):
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    T = uw.discretisation.MeshVariable(name, mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(
        0.5 * (1 - sympy.tanh((r - 0.9) / 0.03)), T.coords).reshape(-1)
    mesh.update_lvec()
    return mesh, T, r


def _largest_change_over_ranks(array, start):
    change = np.abs(np.array(array) - start)
    return uw.mpi.comm.allreduce(float(change.max()) if change.size else 0.0, op=MPI.MAX)


@pytest.mark.mpi(min_size=2)
def test_integral_after_a_move_sees_the_remapped_field():
    mesh, T, r = _annulus_with_a_layer("T0766a")
    start = np.array(T.array)
    density = 1 + 24 * sympy.exp(-(((r - 0.9) / 0.06) ** 2))
    uw.meshing.node_redistribution(mesh, density, slip_surfaces=True, skip_threshold=None,
                                   method_kwargs=dict(accel="none", n_outer=20))
    assert _largest_change_over_ranks(T.array, start) > 1.0e-3

    assert all(uw.mpi.comm.allgather(bool(mesh._stale_lvec)))
    seen_by_a_solve = float(uw.maths.Integral(mesh, T.sym[0]).evaluate())
    T.array[:, 0, 0] = np.array(T.array[:, 0, 0])
    remapped = float(uw.maths.Integral(mesh, T.sym[0]).evaluate())
    assert seen_by_a_solve == pytest.approx(remapped, rel=1.0e-12)


@pytest.mark.mpi(min_size=2)
def test_a_rank_whose_nodes_stay_put_joins_the_transfer():
    """Only rank 0 moves its nodes. Every other rank must still enter the collective
    remap and mark its mesh vector stale; deciding the move per rank left them outside
    while rank 0 waited for them."""
    mesh, T, _r = _annulus_with_a_layer("T0766b")
    coords = np.array(mesh.X.coords)
    if uw.mpi.rank == 0:
        radius = np.linalg.norm(coords, axis=1)
        inside = (radius > 0.6) & (radius < 0.9)
        coords[inside] *= 1.01
    assert mesh.deform(coords) is True
    assert all(uw.mpi.comm.allgather(bool(mesh._stale_lvec)))
    float(uw.maths.Integral(mesh, T.sym[0]).evaluate())     # a collective read completes
