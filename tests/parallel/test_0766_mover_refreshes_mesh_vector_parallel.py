"""A mesh move is agreed over all ranks and a solve after it sees the remapped field,
including when some ranks' nodes did not move or a rank holds nothing at all.
See tests/test_0766_mover_refreshes_mesh_vector.py.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(120)]


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
    return uw.mpi.comm.allreduce(float(change.max()) if change.size else 0.0, op=uw.MPI.MAX)


def _assert_a_solve_sees_the_remapped_field(mesh, T):
    seen_by_a_solve = float(uw.maths.Integral(mesh, T.sym[0]).evaluate())
    T.array[:, 0, 0] = np.array(T.array[:, 0, 0])       # a write the mesh vector does notice
    remapped = float(uw.maths.Integral(mesh, T.sym[0]).evaluate())
    assert seen_by_a_solve == pytest.approx(remapped, rel=1.0e-12)


@pytest.mark.mpi(min_size=2)
def test_integral_after_a_move_sees_the_remapped_field():
    mesh, T, r = _annulus_with_a_layer("T0766a")
    start = np.array(T.array)
    density = 1 + 24 * sympy.exp(-(((r - 0.9) / 0.06) ** 2))
    uw.meshing.node_redistribution(mesh, density, slip_surfaces=True, skip_threshold=None,
                                   method_kwargs=dict(accel="none", n_outer=20))
    assert _largest_change_over_ranks(T.array, start) > 1.0e-3
    _assert_a_solve_sees_the_remapped_field(mesh, T)


@pytest.mark.mpi(min_size=2)
def test_a_rank_whose_nodes_stay_put_joins_the_transfer():
    """Only vertices no other rank holds move, and only on rank 0, so on every other
    rank the coordinates are bitwise unchanged. Those ranks must still enter the
    collective remap; deciding the move per rank left them outside while rank 0 waited."""
    mesh, T, _r = _annulus_with_a_layer("T0766b")
    coords = np.array(mesh.X.coords)
    held_elsewhere = uw.mpi.comm.allgather(coords.round(12))
    moved_here = np.zeros(coords.shape[0], dtype=bool)
    if uw.mpi.rank == 0:
        others = {tuple(p) for rank, held in enumerate(held_elsewhere) if rank != 0 for p in held}
        moved_here = np.array([tuple(p) not in others for p in coords.round(12)])
    assert uw.mpi.comm.allreduce(int(moved_here.sum())) > 0
    coords[moved_here] *= 1.0 + 1.0e-3
    assert all(uw.mpi.comm.allgather(mesh.deform(coords)))
    _assert_a_solve_sees_the_remapped_field(mesh, T)


@pytest.mark.mpi(min_size=2)
def test_a_rank_with_no_cells_joins_the_transfer():
    """A 1 x 2 box leaves a rank without cells from np 3, and at np 2 the partition can
    do the same. That rank still enters global_evaluate: skipping it for an empty
    target left the populated ranks waiting."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(1, 2))
    T = uw.discretisation.MeshVariable("T0766c", mesh, 1, degree=1)
    T.array[...] = 1.0
    coords = np.array(mesh.X.coords)
    coords[:, 1] *= 1.0 + 0.05 * coords[:, 0]
    assert all(uw.mpi.comm.allgather(mesh.deform(coords)))
    _assert_a_solve_sees_the_remapped_field(mesh, T)


@pytest.mark.mpi(min_size=2)
def test_follow_metric_agrees_whether_the_mesh_moved():
    """A front confined to one sector moves some ranks' nodes by less than allclose
    can see. The polish behind the answer is collective, so the answer is too."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T0766d", mesh, 1, degree=1)
    T.array[:, 0, 0] = uw.function.evaluate(
        sympy.tanh((sympy.atan2(y, x) - 0.2) / 0.02), T.coords).reshape(-1)
    moved = uw.mpi.comm.allgather(uw.meshing.follow_metric(mesh, T, refinement=2.0, skip_threshold=None))
    assert len(set(moved)) == 1
