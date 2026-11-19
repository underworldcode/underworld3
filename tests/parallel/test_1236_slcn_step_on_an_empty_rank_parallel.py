"""An SLCN step with a rank that holds no cells (review of #1234).

In parallel the semi-Lagrangian history records the current field by copying
the tracked variable's nodal data. On a rank with no nodes the copy's
``reshape(0, -1)`` raised, the rank fell back to evaluating ``psi_fn``, and the
two routes enter different collectives: the step hung at np 3 with an empty
rank. The ranks now agree on the route, and the copy reads its column count
from the array's shape.

The field x + y with that value on the boundary, no flow and unit diffusivity
is a steady solution. In parallel the record is the nodal copy, so the field
is held to round-off. Serially it is not: the serial record writes each node
the value at a point nudged towards its cell centroid, which moved this field
by 7.5e-5 in two steps (measured serially on the same fixture). That is why
the test runs from np 3 only.
"""

import numpy as np
import pytest

import underworld3 as uw
from empty_rank_fixtures import (ALL_BOUNDARIES_VALUE, cells_along_a_side_leaving_one_rank_empty,
                                 cells_per_rank, right_triangle_of_cells)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]


@pytest.mark.skipif(uw.mpi.size < 3, reason="at np 2 the fixture's one cell is redistributed "
                    "onto both ranks, so no rank is empty")
def test_an_slcn_step_with_an_empty_rank_completes():
    size = uw.mpi.size
    dm = right_triangle_of_cells(cells_along_a_side_leaving_one_rank_empty(size),
                                 max(size - 1, 1))
    dm.createLabel("All_Boundaries")
    dm.markBoundaryFaces("All_Boundaries", ALL_BOUNDARIES_VALUE)     # collective
    mesh = uw.discretisation.Mesh(dm, qdegree=3)
    assert 0 in cells_per_rank(mesh)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T1236", mesh, 1, degree=2)
    X = np.asarray(T.coords)
    T.array[:, 0, 0] = X[:, 0] + X[:, 1] if X.size else np.empty(0)
    V = uw.discretisation.MeshVariable("V1236", mesh, 2, degree=1)   # no flow
    adv = uw.systems.AdvDiffusionSLCN(mesh, u_Field=T, V_fn=V.sym, order=1)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0
    adv.add_dirichlet_bc([x + y], "All_Boundaries")
    adv.tolerance = 1.0e-10
    for _ in range(2):
        adv.solve(timestep=0.01)
    error = float(np.abs(np.asarray(T.array).ravel() - (X[:, 0] + X[:, 1])).max()) if X.size else 0.0
    assert uw.mpi.comm.allreduce(error, op=uw.MPI.MAX) < 1.0e-9
