"""A disk snapshot restores swarm-variable values into the PETSc field, on every rank (#971).

The serial file, tests/test_0020_snapshot_disk_swarm_values.py, carries the contract. The
sidecar is read per rank, so here, on every rank:

- after ``load_state(file)`` the PETSc field holds the saved values;
- they are still there after the canonical cache is dropped and re-read, as ``migrate()``
  does after advection.

On development eddab707 a Lagrangian-stress restart at np 2 gave 0.43553 against
0.68137 for the uninterrupted run; with the fix they agree to about 1e-16 at np 1, 2 and
4 (hyperion).

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_0020_snapshot_disk_swarm_values_parallel.py
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.timeout(300)]


def _model_with_swarm_values():
    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1.0 / 4.0)
    swarm = uw.swarm.Swarm(mesh)
    s = swarm.add_variable("s", 1, dtype=float)
    swarm.populate(fill_param=2)
    X = np.asarray(swarm._particle_coordinates.array).reshape(-1, 2)
    s.array[:, 0, 0] = 1.0 + X[:, 0] + 2.0 * X[:, 1]
    return model, swarm, s


def _shared_path(tmp_path):
    """One snapshot file for every rank: rank 0's temporary directory."""
    return uw.mpi.comm.bcast(str(tmp_path / "swarm.snap.h5") if uw.mpi.rank == 0 else None,
                             root=0)


def test_disk_restore_reaches_the_petsc_field_on_every_rank(tmp_path):
    model, swarm, s = _model_with_swarm_values()
    saved = np.array(s.array).copy()
    path = _shared_path(tmp_path)
    model.save_state(file=path)

    s.array[...] = 0.0
    model.load_state(path)

    raw = np.asarray(s.unpack_raw_data_from_petsc(squeeze=False)).reshape(saved.shape)
    restored = np.allclose(raw, saved)
    swarm._invalidate_canonical_data()
    reread = np.allclose(np.asarray(s.array), saved)
    verdicts = uw.mpi.comm.allgather((restored, reread))
    assert set(verdicts) == {(True, True)}, verdicts
