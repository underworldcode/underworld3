"""Injection must reach parent owners that have no local extracted-region node.

Check P1/P2 vector fields, nonzero untouched parent values and both transfer
operations. Global coordinate gathering belongs only to this small independent
test oracle; the production transfer communicates through the parent PETSc DM.
"""

import numpy as np
from mpi4py import MPI
import pytest
from scipy.spatial import cKDTree
import underworld3 as uw


@pytest.mark.parametrize("degree", (1, 2))
@pytest.mark.parametrize("mode", ("replace", "add"))
def test_submesh_injection_preserves_all_parent_copies(degree, mode):
    parent = uw.meshing.BoxInternalBoundary(
        elementRes=(12, 12), zelementRes=(10, 2), zintCoord=.85,
        minCoords=(0., 0.), maxCoords=(1., 1.), simplex=True, qdegree=2,
    )
    child = parent.extract_region("Outer")
    source = uw.discretisation.MeshVariable("Child", child, 2, degree=degree)
    target = uw.discretisation.MeshVariable("Parent", parent, 2, degree=degree)
    x = np.asarray(source.coords)
    source.data[...] = np.column_stack((1 + x[:, 0] + 2*x[:, 1], 3 - x[:, 0]))
    target.data[...] = 7.
    child.prolongate(source, target, mode=mode)

    global_coords = np.concatenate(uw.mpi.comm.allgather(x))
    distance, _ = cKDTree(global_coords).query(np.asarray(target.coords))
    selected = distance < 1e-10
    expected = np.full(target.data.shape, 7.)
    points = np.asarray(target.coords)[selected]
    values = np.column_stack((1 + points[:, 0] + 2*points[:, 1], 3 - points[:, 0]))
    expected[selected] = values + (7. if mode == "add" else 0.)
    local = float(np.max(np.abs(target.data - expected), initial=0.))
    assert uw.mpi.comm.allreduce(local, op=MPI.MAX) < 1e-12
