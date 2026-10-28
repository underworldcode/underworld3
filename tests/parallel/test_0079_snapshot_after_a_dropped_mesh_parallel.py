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

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(120)]


def test_a_disk_snapshot_after_a_dropped_mesh_is_rank_uniform(tmp_path):
    uw.reset_default_model()
    kept = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
    uw.discretisation.MeshVariable("T", kept, 1, degree=1)

    def make():
        mesh = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25)
        uw.discretisation.MeshVariable("T", mesh, 1, degree=1)

    path = MPI.COMM_WORLD.bcast(str(tmp_path / "state.h5"), root=0)
    gc.disable()
    try:
        make()
        with uw.selective_ranks(0) as collects:
            if collects:
                gc.collect()
        uw.get_default_model().save_state(file=path)
    finally:
        gc.enable()

    metadata = uw.checkpoint.read_snapshot_metadata(path)
    assert metadata["mesh_names"] == [kept.name]
