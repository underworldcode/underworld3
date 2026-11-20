"""An SLCN step with a rank that holds no cells (review of #1234).

In parallel the semi-Lagrangian history records the current field by copying
the tracked variable's nodal data. On a rank with no nodes the copy's
``reshape(0, -1)`` raised, the rank fell back to evaluating ``psi_fn``, and the
two routes enter different collectives: the step hung at np 3 with an empty
rank. The ranks now agree on the route, and the copy reads its column count
from the array's shape.

Two SLCN steps of the field x + y, with that value on the boundary, no flow
and unit diffusivity, on the triangle with one rank empty, are compared with
the same steps on the same triangle shared among every rank. Both take the
parallel route, so they differ by what the solve tolerance leaves on two
partitions: 2.4e-9 at a tolerance of 1e-10 (measured by the reviewer at np 3).
The bound is 100 times the tolerance, far below the 5e-5 that separates the
serial route from the parallel one. A loose check against x + y holds the
answer; neither route keeps it exactly (in two steps the field moves 7.5e-5
from x + y serially and 2.5e-5 at np 3, measured by the reviewer).
"""

import numpy as np
import pytest

import underworld3 as uw
from empty_rank_fixtures import (ALL_BOUNDARIES_VALUE, cells_along_a_side_leaving_one_rank_empty,
                                 cells_per_rank, right_triangle_of_cells)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]

TOLERANCE = 1.0e-10


def _two_steps(ranks_with_cells, tag):
    """The field after two steps, and its node coordinates, on this rank."""
    size = uw.mpi.size
    dm = right_triangle_of_cells(cells_along_a_side_leaving_one_rank_empty(size),
                                 ranks_with_cells)
    dm.createLabel("All_Boundaries")
    dm.markBoundaryFaces("All_Boundaries", ALL_BOUNDARIES_VALUE)     # collective
    mesh = uw.discretisation.Mesh(dm, qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable(f"T1236{tag}", mesh, 1, degree=2)
    X = np.asarray(T.coords)
    T.array[:, 0, 0] = X[:, 0] + X[:, 1] if X.size else np.empty(0)
    V = uw.discretisation.MeshVariable(f"V1236{tag}", mesh, 2, degree=1)   # no flow
    adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V.sym, order=1)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0
    adv.add_dirichlet_bc([x + y], "All_Boundaries")
    adv.tolerance = TOLERANCE
    for _ in range(2):
        adv.solve(timestep=0.01)
    return mesh, np.asarray(T.array).reshape(-1), X


def _gathered(values, X):
    """Every rank's (x, y, value), sorted by position, on every rank."""
    rows = np.column_stack([X, values]) if X.size else np.empty((0, 3))
    table = np.vstack(uw.mpi.comm.allgather(rows))
    table = np.unique(np.round(table, 12), axis=0)       # shared nodes appear once
    return table[np.lexsort((table[:, 1], table[:, 0]))]


@pytest.mark.skipif(uw.mpi.size < 3, reason="at np 2 the fixture's one cell is redistributed "
                    "onto both ranks, so no rank is empty")
def test_an_slcn_step_with_an_empty_rank_matches_one_without():
    empty_mesh, empty_values, empty_X = _two_steps(uw.mpi.size - 1, "e")
    assert 0 in cells_per_rank(empty_mesh)
    full_mesh, full_values, full_X = _two_steps(uw.mpi.size, "f")
    assert 0 not in cells_per_rank(full_mesh)
    with_empty, without = _gathered(empty_values, empty_X), _gathered(full_values, full_X)
    assert with_empty.shape == without.shape
    np.testing.assert_allclose(with_empty[:, :2], without[:, :2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(with_empty[:, 2], without[:, 2], rtol=0, atol=100 * TOLERANCE)
    assert np.abs(with_empty[:, 2] - with_empty[:, 0] - with_empty[:, 1]).max() < 1e-4
