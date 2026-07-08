"""Parallel regression test for the swarm-routed read_timestep path.

The new ``MeshVariable.read_timestep`` uses a transient swarm to ship saved
``(coord, value)`` pairs to the rank that owns each location. This test
verifies the parallel round-trip: write a known field on N ranks, read it
back on N ranks, assert allclose.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/ptest_0762_read_timestep_swarm_routed.py
"""

import shutil
import tempfile

import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing import UnstructuredSimplexBox

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.timeout(120)]


def _shared_tmpdir(prefix):
    """Make rank 0 a tmp directory and broadcast the path to every rank."""
    if uw.mpi.rank == 0:
        tmpdir = tempfile.mkdtemp(prefix=prefix)
    else:
        tmpdir = None
    return uw.mpi.comm.bcast(tmpdir, root=0)


def _cleanup_shared_tmpdir(tmpdir):
    """Drop the shared tmp directory once all ranks are done with it."""
    uw.mpi.comm.barrier()
    if uw.mpi.rank == 0:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.mpi(min_size=2)
@pytest.mark.level_1
@pytest.mark.tier_a
def test_meshvariable_read_timestep_parallel_round_trip():
    """Write field on N ranks, read back on same N ranks, expect allclose."""
    tmpdir = _shared_tmpdir("uw3_read_timestep_")
    try:
        mesh = UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=1.0 / 16.0,
        )
        X = uw.discretisation.MeshVariable("X", mesh, 1, degree=2)
        X2 = uw.discretisation.MeshVariable("X2", mesh, 1, degree=2)

        # Analytic field — varies in both dims so a faulty rank-local
        # read-everywhere implementation could not silently pass on a
        # constant field.
        X.array[:, 0, 0] = X.coords[:, 0] + 0.5 * X.coords[:, 1]

        mesh.write_timestep(
            "test", meshUpdates=False, meshVars=[X],
            outputPath=tmpdir, index=0,
        )

        X2.read_timestep("test", "X", 0, outputPath=tmpdir)

        assert np.allclose(X.array, X2.array, atol=1e-8), (
            f"[rank {uw.mpi.rank}] read_timestep round-trip failed:\n"
            f"max diff = {np.abs(np.asarray(X.array) - np.asarray(X2.array)).max():.3e}"
        )
    finally:
        _cleanup_shared_tmpdir(tmpdir)


@pytest.mark.skip(
    reason="Pre-existing: Swarm.write_timestep hangs on N>1 with parallel h5py "
    "(unrelated to read_timestep refactor). Serial round-trip is covered by "
    "tests/test_0003_save_load.py::test_swarmvariable_save_and_load."
)
@pytest.mark.mpi(min_size=2)
@pytest.mark.level_1
@pytest.mark.tier_a
def test_swarmvariable_read_timestep_parallel_round_trip():
    """Same round-trip check for SwarmVariable.read_timestep."""
    tmpdir = _shared_tmpdir("uw3_read_timestep_swarm_")
    try:
        mesh = UnstructuredSimplexBox(
            minCoords=(0.0, 0.0),
            maxCoords=(1.0, 1.0),
            cellSize=1.0 / 16.0,
        )
        swarm = uw.swarm.Swarm(mesh)
        var = swarm.add_variable(name="X", size=1)
        var2 = swarm.add_variable(name="X2", size=1)
        swarm.populate(fill_param=2)

        # Each particle stores its own x-coord
        var.array[:, 0, 0] = swarm._particle_coordinates.data[:, 0]

        swarm.write_timestep(
            "test", "swarm", swarmVars=[var],
            outputPath=tmpdir, index=0,
        )

        var2.read_timestep("test", "swarm", "X", 0, outputPath=tmpdir)

        # nnn=1 → exact match (nearest-neighbour) for the round-trip
        assert np.allclose(var.array, var2.array, atol=1e-8), (
            f"[rank {uw.mpi.rank}] swarm read_timestep round-trip failed:\n"
            f"max diff = {np.abs(np.asarray(var.array) - np.asarray(var2.array)).max():.3e}"
        )
    finally:
        _cleanup_shared_tmpdir(tmpdir)
