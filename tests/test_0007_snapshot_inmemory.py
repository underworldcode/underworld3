import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np


def _fresh_model_and_mesh():
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0
    )
    return uw, model, mesh


def test_meshvariable_in_memory_roundtrip():
    """Snapshot, scribble, restore: the MV global vector is recovered exactly."""
    uw, model, mesh = _fresh_model_and_mesh()
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    T.array[:, 0, 0] = T.coords[:, 0] + 2.0 * T.coords[:, 1]
    pre_array = np.asarray(T.array[...]).copy()

    snap = model.snapshot()

    T.array[...] = -42.0
    assert not np.allclose(np.asarray(T.array[...]), pre_array), "scribble didn't take"

    model.restore(snap)

    assert np.allclose(np.asarray(T.array[...]), pre_array, atol=0.0, rtol=0.0), (
        "MeshVariable.array is not bit-equivalent after restore"
    )


def test_multiple_meshvariables_roundtrip():
    """All MVs on a mesh are captured and restored, not just the first."""
    uw, model, mesh = _fresh_model_and_mesh()
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    V = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)

    T.array[:, 0, 0] = 1.5 * T.coords[:, 0]
    V.array[:, 0, 0] = 3.0 * V.coords[:, 0]
    V.array[:, 0, 1] = 7.0 * V.coords[:, 1]

    T_pre = np.asarray(T.array[...]).copy()
    V_pre = np.asarray(V.array[...]).copy()

    snap = model.snapshot()

    T.array[...] = 0.0
    V.array[...] = 0.0

    model.restore(snap)

    assert np.allclose(np.asarray(T.array[...]), T_pre)
    assert np.allclose(np.asarray(V.array[...]), V_pre)


def test_snapshot_is_independent_of_subsequent_writes():
    """Captured array is a copy; writes to the live MV don't leak into the snapshot."""
    uw, model, mesh = _fresh_model_and_mesh()
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = 5.0

    snap = model.snapshot()
    T.array[...] = -1.0

    # The backend still holds the captured value, not the post-write value.
    keys = snap.backend.list_vectors()
    var_key = next(k for k in keys if "var:T" in k)
    captured = snap.backend.load_vector(var_key)
    assert np.allclose(captured, 5.0), (
        "in-memory backend did not isolate the captured array from later writes"
    )


def test_mesh_version_invalidates_restore():
    """A bumped _mesh_version makes restore refuse rather than silently corrupt."""
    import underworld3 as uw
    from underworld3.checkpoint import SnapshotInvalidatedError

    uw_, model, mesh = _fresh_model_and_mesh()
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    T.array[:, 0, 0] = 1.0

    snap = model.snapshot()

    # Simulate a mesh-mutation event (e.g. adapt(), or any deformation
    # routed through the high-level callback that bumps _mesh_version).
    mesh._mesh_version += 1

    with pytest.raises(SnapshotInvalidatedError, match="_mesh_version"):
        model.restore(snap)


def test_restore_rejects_non_snapshot():
    """A bare dict / array is not a Snapshot; restore raises TypeError."""
    uw, model, mesh = _fresh_model_and_mesh()
    _ = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    with pytest.raises(TypeError):
        model.restore({"not": "a snapshot"})


def test_snapshot_path_is_v1_1_scope():
    """Passing path= raises NotImplementedError until the on-disk backend lands."""
    uw, model, mesh = _fresh_model_and_mesh()
    _ = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    with pytest.raises(NotImplementedError):
        model.snapshot(path="/tmp/should_not_be_written.h5")


# ----- Swarm coverage -----


def _fresh_model_mesh_and_swarm(with_material=True):
    """Create a fresh model + mesh + swarm. Swarm-variable creation must
    happen before populate(), so we build everything in one place.
    """
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 4.0
    )
    swarm = uw.swarm.Swarm(mesh)
    material = None
    if with_material:
        material = swarm.add_variable("material", 1, dtype=float)
    swarm.populate(fill_param=2)
    return uw, model, mesh, swarm, material


def test_swarm_positions_and_variable_roundtrip():
    """Snapshot, scramble swarm positions + svar, restore: both come back."""
    uw, model, mesh, swarm, material = _fresh_model_mesh_and_swarm()

    coords = swarm._particle_coordinates.data
    material.data[:, 0] = 0.5 * coords[:, 0] + coords[:, 1]
    coords_pre = coords.copy()
    material_pre = np.asarray(material.data).copy()

    snap = model.snapshot()

    coord_field = swarm.dm.getField("DMSwarmPIC_coor").reshape((-1, swarm.dim))
    coord_field[...] = -99.0
    swarm.dm.restoreField("DMSwarmPIC_coor")
    material.data[...] = -99.0

    model.restore(snap)

    assert np.allclose(swarm._particle_coordinates.data, coords_pre)
    assert np.allclose(np.asarray(material.data), material_pre)


def test_swarm_population_generation_starts_at_zero_and_bumps():
    """Sanity-check the counter bumps on each mutation category."""
    uw, model, mesh, swarm, _ = _fresh_model_mesh_and_swarm(with_material=False)
    after_populate = swarm._population_generation
    swarm.migrate(remove_sent_points=True)
    after_migrate = swarm._population_generation
    swarm.add_particles_with_coordinates(np.array([[0.5, 0.5]]))
    after_add_local = swarm._population_generation
    swarm.add_particles_with_global_coordinates(np.array([[0.25, 0.25]]))
    after_add_global = swarm._population_generation

    assert after_populate >= 1
    assert after_migrate > after_populate
    assert after_add_local > after_migrate
    assert after_add_global > after_add_local


def test_swarm_migrate_invalidates_restore():
    """A migrate() call between snapshot and restore makes restore refuse."""
    from underworld3.checkpoint import SnapshotInvalidatedError

    uw, model, mesh, swarm, _ = _fresh_model_mesh_and_swarm()

    snap = model.snapshot()
    swarm.migrate(remove_sent_points=True)

    with pytest.raises(SnapshotInvalidatedError, match="_population_generation"):
        model.restore(snap)


def test_swarm_add_particles_invalidates_restore():
    """add_particles_with_coordinates between snapshot and restore raises."""
    from underworld3.checkpoint import SnapshotInvalidatedError

    uw, model, mesh, swarm, _ = _fresh_model_mesh_and_swarm()

    snap = model.snapshot()
    swarm.add_particles_with_coordinates(np.array([[0.5, 0.5]]))

    with pytest.raises(SnapshotInvalidatedError, match="_population_generation"):
        model.restore(snap)


def test_swarm_internal_variables_are_not_captured():
    """Internal DMSwarm_* variables stay out of the snapshot key list."""
    uw, model, mesh, swarm, _ = _fresh_model_mesh_and_swarm()

    snap = model.snapshot()
    keys = snap.backend.list_vectors()
    swarmvar_keys = [k for k in keys if k.startswith(f"swarm:{id(swarm)}:var:")]
    captured_names = {k.split(":var:")[1].split(":data")[0] for k in swarmvar_keys}

    # User variable present, PETSc-internal ones absent.
    assert "material" in captured_names
    assert not any(n.startswith("DMSwarm") for n in captured_names)
