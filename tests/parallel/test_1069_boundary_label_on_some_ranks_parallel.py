"""An essential condition on a label only some ranks hold (#1138, #1211 review).

A label made after distribution exists only on the ranks that hold its points.
``PetscDSAddBoundary_UW`` then passed a null label to ``DMAddBoundary`` on the
others, and the solve ended in a segmentation fault (exit 59, measured by H1
at np 2 and 3 for Poisson and Stokes). The ranks now agree that the label
exists, and a rank without it gets it empty.

The mesh is the right triangle of ``empty_rank_fixtures`` with one rank empty,
and the label is made on the ranks that hold faces only. The test runs from
np 3: at np 2 the fixture's single cell is redistributed onto both ranks. A Poisson problem with
no source and the linear value x + y on the boundary has the exact solution
x + y, which a degree-2 field holds to round-off.
"""

import numpy as np
import pytest

import underworld3 as uw
from empty_rank_fixtures import (ALL_BOUNDARIES_VALUE, cells_along_a_side_leaving_one_rank_empty,
                                 cells_per_rank, right_triangle_of_cells)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]


@pytest.mark.skipif(uw.mpi.size < 3, reason="at np 2 the fixture has one cell, which Mesh "
                    "redistributes onto every rank, so no rank lacks the label")
def test_a_dirichlet_value_on_a_label_some_ranks_lack():
    size = uw.mpi.size
    dm = right_triangle_of_cells(cells_along_a_side_leaving_one_rank_empty(size),
                                 max(size - 1, 1))
    # markBoundaryFaces is collective, so every rank marks a scratch label; the
    # real label is then made only on the ranks that hold faces.
    c_start, c_end = dm.getHeightStratum(0)
    dm.createLabel("faces_scratch")
    dm.markBoundaryFaces("faces_scratch", ALL_BOUNDARIES_VALUE)
    faces = dm.getLabel("faces_scratch").getStratumIS(ALL_BOUNDARIES_VALUE)
    if c_end > c_start and faces is not None and faces.getSize():
        dm.createLabel("All_Boundaries")
        for f in faces.getIndices():
            dm.setLabelValue("All_Boundaries", int(f), ALL_BOUNDARIES_VALUE)
    dm.removeLabel("faces_scratch")
    mesh = uw.discretisation.Mesh(dm, qdegree=3)
    assert 0 in cells_per_rank(mesh)
    assert not uw.mpi.comm.allreduce(mesh.dm.hasLabel("All_Boundaries"), op=uw.MPI.LAND)
    T = uw.discretisation.MeshVariable("T1211", mesh, 1, degree=2)
    x, y = mesh.X
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    poisson.add_dirichlet_bc([x + y], "All_Boundaries")
    poisson.tolerance = 1.0e-10
    poisson.solve()
    X = np.asarray(T.coords)
    error = float(np.abs(np.asarray(T.array).ravel() - (X[:, 0] + X[:, 1])).max()) if X.size else 0.0
    assert uw.mpi.comm.allreduce(error, op=uw.MPI.MAX) < 1.0e-9
