"""A bad swarm sidecar on one rank is refused on every rank, before anything is
restored (#974).

A disk snapshot's swarm data is a per-rank sidecar, and its checks (the rank
count, the variable set, each variable's shape) were made by each rank on its
own sidecar, after the mesh variables had been restored and after the rank had
cleared its particles. A sidecar damaged on one rank raised there, while the
others returned from ``load_state`` as if the restore had succeeded. The
sidecars are now read and checked with the other restore checks, before
anything is restored, and the verdict is agreed: every rank raises the same
error, and the model is left as it was.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0974_swarm_sidecar_refused_everywhere_parallel.py
"""
import os

import h5py
import numpy as np
import pytest

import underworld3 as uw
from underworld3.checkpoint import disk_snapshot

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.timeout(300)]


def _model():
    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0 / 4.0)
    T = uw.discretisation.MeshVariable("T974", mesh, 1, degree=1)
    swarm = uw.swarm.Swarm(mesh)
    s = swarm.add_variable("s", 1, dtype=float)
    swarm.populate(fill_param=2)
    X = np.asarray(swarm._particle_coordinates.array).reshape(-1, 2)
    s.array[:, 0, 0] = 1.0 + X[:, 0] + 2.0 * X[:, 1]
    T.array[...] = 1.0
    return model, T, swarm, s


def _shared_path(tmp_path):
    return uw.mpi.comm.bcast(str(tmp_path / "s974.snap.h5") if uw.mpi.rank == 0 else None,
                             root=0)


def _drop_a_row_of_this_ranks_sidecar(path):
    with h5py.File(path, "r") as f:
        group = f["swarms"]
        name = list(group.keys())[0]
        pattern = str(group[name].attrs["sidecar_pattern"])
    sidecar = os.path.join(disk_snapshot._bulk_dir_for(path),
                           pattern.format(rank=uw.mpi.rank, size=uw.mpi.size))
    with h5py.File(sidecar, "r+") as f:
        data = f["variables"]["s"][...]
        del f["variables"]["s"]
        f["variables"].create_dataset("s", data=data[:-1])


@pytest.mark.tier_b
def test_a_bad_sidecar_on_one_rank_is_refused_on_every_rank(tmp_path):
    model, T, swarm, s = _model()
    path = _shared_path(tmp_path)
    model.save_state(file=path)
    uw.mpi.barrier()

    # the model moves on after the save; a refused restore must leave it here
    T.array[...] = 2.0
    s.array[...] = 5.0
    particles = swarm.dm.getLocalSize()
    if uw.mpi.rank == uw.mpi.size - 1:
        _drop_a_row_of_this_ranks_sidecar(path)
    uw.mpi.barrier()

    message = None
    try:
        model.load_state(path)
    except ValueError as err:
        message = str(err)
    messages = uw.mpi.comm.allgather(message)
    assert all(m is not None and "shape mismatch" in m for m in messages), messages
    assert len(set(messages)) == 1, messages

    untouched = (np.allclose(np.asarray(T.array), 2.0)
                 and swarm.dm.getLocalSize() == particles
                 and np.allclose(np.asarray(s.array), 5.0))
    assert all(uw.mpi.comm.allgather(untouched))
