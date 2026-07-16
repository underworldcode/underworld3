"""Parallel regression tests for ``Swarm.read_timestep`` (issue #324).

``Swarm.read_timestep`` used to read the full coordinate dataset on every
rank and then call ``add_particles_with_global_coordinates(migrate=True)``.
Migration is a scatter with no deduplication, so a parallel restore
produced one copy of the saved swarm per rank (2x at np2, 4x at np4).
The fix reads on rank 0 only, stages empty arrays elsewhere, and lets
migration route each point to its owner — the same rank-0 routed-read
design already used by ``SwarmVariable.read_timestep``.

Run under MPI, e.g.::

    mpirun -np 2 python -m pytest --with-mpi tests/parallel/test_0757_swarm_read_timestep_mpi.py
    mpirun -np 4 python -m pytest --with-mpi tests/parallel/test_0757_swarm_read_timestep_mpi.py
"""

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a, pytest.mark.timeout(180)]


def _shared_tmp_path(tmp_path):
    """One rank-0 pytest tmp path, broadcast so all ranks share the file."""
    shared = str(tmp_path) if uw.mpi.rank == 0 else None
    shared = uw.mpi.comm.bcast(shared, root=0)
    uw.mpi.barrier()
    return shared


def _global_count(swarm):
    return uw.mpi.comm.allreduce(max(swarm.dm.getLocalSize(), 0))


def _gathered_sorted_coords(swarm):
    """All particle coordinates, gathered and lexicographically sorted.

    Sorting makes the comparison independent of the partition and of
    particle order within each rank.
    """
    local = np.ascontiguousarray(swarm._particle_coordinates.data[:].copy())
    stacked = uw.mpi.comm.allgather(local)
    coords = np.vstack(stacked)
    order = np.lexsort(np.round(coords, 12).T)
    return coords[order]


@pytest.mark.mpi(min_size=2)
def test_swarm_read_timestep_restores_each_particle_once(tmp_path):
    """Restored global particle count equals the saved count (no np-fold copies)."""
    tmp_path = _shared_tmp_path(tmp_path)

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0)

    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=2)

    saved_count = _global_count(swarm)
    saved_coords = _gathered_sorted_coords(swarm)

    swarm.write_timestep("t0757", "swarm", swarmVars=[], outputPath=tmp_path, index=0)
    # write_timestep returns on non-zero ranks while rank 0 is still
    # appending metadata; reopening the file before it is quiescent hits
    # HDF5 file locking (BlockingIOError, errno 35).
    uw.mpi.barrier()

    restored_swarm = uw.swarm.Swarm(mesh)
    restored_swarm.read_timestep("t0757", "swarm", 0, outputPath=tmp_path)

    restored_count = _global_count(restored_swarm)
    assert restored_count == saved_count, (
        f"read_timestep restored {restored_count} particles from a "
        f"{saved_count}-particle checkpoint at np={uw.mpi.size} "
        f"({restored_count / saved_count:.1f}x duplication)"
    )

    restored_coords = _gathered_sorted_coords(restored_swarm)
    np.testing.assert_allclose(restored_coords, saved_coords, atol=1e-12)


@pytest.mark.mpi(min_size=2)
def test_swarmvariable_read_on_restored_swarm(tmp_path):
    """A SwarmVariable read back on top of a restored swarm is consistent."""
    tmp_path = _shared_tmp_path(tmp_path)

    mesh = UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0)

    swarm = uw.swarm.Swarm(mesh)
    var = swarm.add_variable(name="X0757", size=1)
    swarm.populate(fill_param=2)

    # Each particle stores an analytic function of its own position, so a
    # duplicated or mis-routed restore cannot pass by accident.
    var.array[:, 0, 0] = (
        swarm._particle_coordinates.data[:, 0] + 0.5 * swarm._particle_coordinates.data[:, 1]
    )

    swarm.write_timestep("t0757v", "swarm", swarmVars=[var], outputPath=tmp_path, index=0)
    # See above: let rank 0 finish writing before any rank reopens the file.
    uw.mpi.barrier()

    restored_swarm = uw.swarm.Swarm(mesh)
    restored_var = restored_swarm.add_variable(name="X0757r", size=1)
    restored_swarm.read_timestep("t0757v", "swarm", 0, outputPath=tmp_path)
    restored_var.read_timestep("t0757v", "swarm", "X0757", 0, outputPath=tmp_path)

    coords = restored_swarm._particle_coordinates.data
    expected = coords[:, 0] + 0.5 * coords[:, 1]
    np.testing.assert_allclose(
        np.asarray(restored_var.array)[:, 0, 0], expected, atol=1e-8
    )
