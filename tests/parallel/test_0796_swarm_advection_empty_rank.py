"""MPI regression test for swarm.advection with empty ranks.

A passive swarm whose particles are confined to a subset of ranks (e.g. the
user's crust-only tracers) leaves the remaining ranks holding zero particles.
Advection on such a swarm must not deadlock or crash:

1. The substep loop's ``global_evaluate(V_fn_matrix, particle_data)`` on an
   empty rank used to block inside the collective point-location machinery.
   The root cause (an empty rank taking a divergent DMLocatePoints branch) is
   fixed upstream (issue #611 / PR #656), which this test guards against
   regressing.
2. ``estimate_dt()`` reshaped the (empty) velocity array with
   ``reshape(n, -1)``, which NumPy cannot infer from zero elements ->
   ``ValueError: cannot reshape array of size 0 into shape (0,newaxis)``.
   Guarded in ``Swarm.estimate_dt``; this test exercises the default
   (non-``evalf``) path in ``order=2`` which runs ``estimate_dt``.
"""

import numpy as np
import pytest

import underworld3 as uw


pytestmark = [
    pytest.mark.level_2,
    pytest.mark.mpi(min_size=2),
    pytest.mark.timeout(60),
]


def test_advection_empty_rank_default(tmp_path_factory):
    """``swarm.advection(v.sym, ...)`` completes with empty ranks (default path).

    Only rank 0 holds particles; the other rank(s) hold zero. Exercises the
    default FE-interpolation ``global_evaluate`` path inside the ``order=2``
    substep loop plus the ``estimate_dt`` empty-rank guard.
    """
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25,
    )

    # Simple shear velocity: u = y, v = 0
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=1)
    with mesh.access(v):
        v.data[:, 0] = mesh.X.coords[:, 1]
        v.data[:, 1] = 0.0

    swarm = uw.swarm.Swarm(mesh=mesh)

    # Particles only on rank 0 (collective call)
    if uw.mpi.rank == 0:
        coords = (np.random.rand(100, mesh.dim) * 0.8 + 0.1)
    else:
        coords = np.empty((0, mesh.dim))
    swarm.add_particles_with_coordinates(coords)

    uw.mpi.comm.barrier()

    # Sanity: the empty-rank precondition must actually hold (all 100 points
    # are added on rank 0 and no migration happens here, so the other ranks
    # hold zero particles).
    sizes = uw.mpi.comm.allgather(swarm.local_size)
    assert sum(sizes) > 0, "No particles at all - test is vacuous"
    assert 0 in sizes, (
        f"Expected at least one empty rank, got local sizes {sizes} "
        "(test would no longer exercise the empty-rank path)"
    )

    # Exercise the empty-rank path with the DEFAULT (non-evalf) FE
    # interpolation, which rides global_evaluate. Pre-#611 this deadlocked;
    # pre (estimate_dt) guard the order=2 estimate_dt reshape crashed.
    swarm.advection(v.sym, delta_t=1.0, order=2)

    uw.mpi.comm.barrier()
    del swarm, mesh
