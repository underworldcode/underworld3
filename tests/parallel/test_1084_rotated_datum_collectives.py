"""Prescribed FE wall data must be evaluated in the same order on empty ranks.

Run serial or with mpiexec. This exercises the datum schedule/evaluator only;
no Stokes model or spherical mesh is run.
"""

import numpy as np
import underworld3 as uw
from mpi4py import MPI

from underworld3.utilities.rotated_bc import _datum_batches, _eval_datum


def test_registered_boundaries_determine_collective_schedule_not_local_nodes():
    rank = uw.mpi.rank
    boundaries = ["Upper", "Lower"]
    # All ranks have the same registration, but zero, one or two local groups.
    owned = {0: [(10, 1, "Upper", 1.0), (11, 2, "Lower", -1.0)],
             1: [(20, 2, "Lower", 1.0)]}.get(rank, [])
    coordinates = {1: [2.0, 0.0], 2: [1.0, 0.0]}
    batches = list(_datum_batches(boundaries, {"Upper": 2.0, "Lower": 3.0}, owned, coordinates, 2))
    assert len(batches) == 2
    for index, (datum, coords, items) in enumerate(batches):
        assert uw.mpi.comm.allgather(datum) == [2.0+index]*uw.mpi.size
        assert coords.shape == (len(items), 2)
        if rank > 1:
            assert coords.shape == (0, 2)


def test_fe_datum_evaluation_accepts_empty_coordinate_batches_on_other_ranks():
    mesh = uw.meshing.Annulus(radiusInner=1.22, radiusOuter=2.22, cellSize=1/32, qdegree=2)
    field = uw.discretisation.MeshVariable("DatumCollectiveP2", mesh, 1, degree=2, continuous=True)
    field.data[:, 0] = 0.13*field.coords[:, 0]-0.2*field.coords[:, 1]
    # Only one rank queries nodes; every rank must still enter FE evaluation.
    coords = np.asarray(field.coords[:3]) if uw.mpi.rank == 0 else np.empty((0, 2))
    for _ in range(2):
        actual = _eval_datum(field.sym[0], coords, None)
        expected = 0.13*coords[:, 0]-0.2*coords[:, 1]
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert uw.mpi.comm.allreduce(1, op=MPI.SUM) == uw.mpi.size
