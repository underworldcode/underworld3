"""A semi-Lagrangian history leaves a static field where it is, on any number of
ranks (#1237).

The nodal history used to evaluate its start velocity and, serially, record the
field at nodes nudged 0.1 % toward the centroid of their closest cell. The
nudge moved a static field at every record: two SLCN steps of x + y, with that
value on the boundary, no flow and unit diffusivity, left it 7.498e-5 from
x + y at np 1. Parallel took the copy route and drifted by 2.533e-5. np 3
against np 1 was 4.965e-5, so serial and parallel differed by design. These
figures were measured on this triangle by the parallel reviewer. Without the
nudge both routes are one route, and the field stays at x + y to the solve
tolerance: the bound is 100 times it. The same check runs serially in
tests/test_1244_semi_lagrangian_static_field.py, so np 1 and np 3 are held to
one answer.
"""
import numpy as np
import pytest

import underworld3 as uw
from empty_rank_fixtures import (ALL_BOUNDARIES_VALUE, cells_along_a_side_leaving_one_rank_empty,
                                 cells_per_rank, right_triangle_of_cells)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]

TOLERANCE = 1.0e-10


def two_static_steps(ranks_with_cells, tag, size=None):
    """x + y after two SLCN steps with no flow, and its node coordinates, on this rank."""
    size = uw.mpi.size if size is None else size
    dm = right_triangle_of_cells(cells_along_a_side_leaving_one_rank_empty(size), ranks_with_cells)
    dm.createLabel("All_Boundaries")
    dm.markBoundaryFaces("All_Boundaries", ALL_BOUNDARIES_VALUE)     # collective
    mesh = uw.discretisation.Mesh(dm, qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable(f"T1244{tag}", mesh, 1, degree=2)
    X = np.asarray(T.coords)
    T.array[:, 0, 0] = X[:, 0] + X[:, 1] if X.size else np.empty(0)
    V = uw.discretisation.MeshVariable(f"V1244{tag}", mesh, 2, degree=1)   # no flow
    adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V.sym, order=1)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0
    adv.add_dirichlet_bc([x + y], "All_Boundaries")
    adv.tolerance = TOLERANCE
    for _ in range(2):
        adv.solve(timestep=0.01)
    return mesh, np.asarray(T.array).reshape(-1), X


def drift_from_x_plus_y(values, X):
    """The largest distance of the field from x + y over every rank."""
    local = float(np.abs(values - X[:, 0] - X[:, 1]).max()) if X.size else 0.0
    return uw.mpi.comm.allreduce(local, op=uw.MPI.MAX)


@pytest.mark.mpi(min_size=3)
@pytest.mark.skipif(uw.mpi.size < 3, reason="at np 2 the fixture is one cell, redistributed "
                    "so that a rank is empty whatever is asked")
@pytest.mark.parametrize("empty_rank", [False, True])
def test_a_static_field_stays_put_on_every_rank(empty_rank):
    ranks = uw.mpi.size - 1 if empty_rank else uw.mpi.size
    mesh, values, X = two_static_steps(ranks, "e" if empty_rank else "f")
    assert (0 in cells_per_rank(mesh)) == empty_rank
    assert drift_from_x_plus_y(values, X) < 100 * TOLERANCE
