"""A disk snapshot after a dropped mesh writes the same meshes on every rank.

A dropped mesh with a variable is a reference cycle (the mesh owns its
variables, each variable holds its mesh), so it stays in the model's weak
registry until the cyclic collector runs, and the collector runs at a
different moment on each rank. ``write_snapshot`` makes a collective
checkpoint write per registered mesh: a rank that had collected the dropped
mesh and a rank that had not entered different numbers of collective writes
and hung (review 3 of the faulted-rebuild leak fix, 2026-10-27). The
snapshot now collects first on every rank. Here one rank collects and the
other is held off.

Run with at least two ranks.
"""
import gc

import pytest
from mpi4py import MPI

import underworld3 as uw

# The thread method: a rank blocked in MPI does not see the signal one.
pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(120, method="thread")]


def _kept_mesh_and_a_dropped_one_one_rank_collected():
    """A kept mesh with a variable, and a dropped one that rank 0 alone has
    collected; the collector is held off until the caller enables it."""
    uw.reset_default_model()
    kept = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    uw.discretisation.MeshVariable("T", kept, 1, degree=1)

    def make():
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
        uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

    gc.disable()
    make()
    with uw.selective_ranks(0) as collects:
        if collects:
            gc.collect()
    return kept


def test_a_disk_snapshot_after_a_dropped_mesh_is_rank_uniform(tmp_path):
    path = MPI.COMM_WORLD.bcast(str(tmp_path / "state.h5"), root=0)
    try:
        kept = _kept_mesh_and_a_dropped_one_one_rank_collected()
        uw.get_default_model().save_state(file=path)
    finally:
        gc.enable()
    metadata = uw.checkpoint.read_snapshot_metadata(path)
    assert metadata["mesh_names"] == [kept.name]


def test_an_in_memory_snapshot_after_a_dropped_mesh_is_rank_uniform():
    """The in-memory capture syncs each mesh's vectors (local to global, a
    collective) as it goes, so it has the same hazard."""
    try:
        kept = _kept_mesh_and_a_dropped_one_one_rank_collected()
        state = uw.get_default_model().save_state()
    finally:
        gc.enable()
    names = MPI.COMM_WORLD.allgather(state.mesh_names)
    assert all(n == [kept.name] for n in names), names
