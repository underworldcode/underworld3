"""A disk snapshot restores swarm-variable values into the PETSc field.

The in-memory restore was fixed to write through ``var.data`` (#313); the
disk restore wrote into a detached numpy view, so the DMSwarm field kept
whatever the re-added particles were given. Anything that reads the field
directly (the ``cells`` proxy of the particle Lagrangian stress history) or
re-reads it after a cache invalidation (migrate) then saw zeros: a run
restarted with ``stress_transport="lagrangian"`` lost its stress history.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _model_with_swarm_values():
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 4.0
    )
    swarm = uw.swarm.Swarm(mesh)
    s = swarm.add_variable("s", 1, dtype=float)
    v = swarm.add_variable("v", 2, dtype=float)
    swarm.populate(fill_param=2)
    X = np.asarray(swarm._particle_coordinates.data)
    s.data[:, 0] = 1.0 + X[:, 0] + 2.0 * X[:, 1]
    v.data[:, 0] = X[:, 1]
    v.data[:, 1] = -X[:, 0]
    return uw, model, swarm, s, v


def test_disk_restore_reaches_petsc_field(tmp_path):
    """After load_state(file), the PETSc field holds the saved values."""
    uw, model, swarm, s, v = _model_with_swarm_values()
    saved_s = np.array(s.data).copy()
    saved_v = np.array(v.data).copy()
    path = str(tmp_path / "swarm.snap.h5")
    model.save_state(file=path)

    s.data[...] = 0.0
    v.data[...] = 0.0
    model.load_state(path)

    raw_s = np.asarray(s.unpack_raw_data_from_petsc(squeeze=False)).reshape(saved_s.shape)
    raw_v = np.asarray(v.unpack_raw_data_from_petsc(squeeze=False)).reshape(saved_v.shape)
    assert np.allclose(raw_s, saved_s)
    assert np.allclose(raw_v, saved_v)


def test_disk_restore_survives_cache_invalidation(tmp_path):
    """The restored values are still there after the canonical cache is
    dropped and re-read from PETSc (what migrate() does after advection)."""
    uw, model, swarm, s, v = _model_with_swarm_values()
    saved_s = np.array(s.data).copy()
    path = str(tmp_path / "swarm.snap.h5")
    model.save_state(file=path)

    s.data[...] = 0.0
    model.load_state(path)
    swarm._invalidate_canonical_data()

    assert np.allclose(np.asarray(s.data), saved_s)
