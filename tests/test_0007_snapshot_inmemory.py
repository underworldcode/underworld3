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


# ----- Swarm coverage (rebuild-on-restore semantics) -----


def _fresh_model_mesh_and_swarm(with_material=True):
    """Fresh model + mesh + populated swarm. svar must be added pre-populate."""
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


def _swarm_coords(swarm):
    """Return a copy of the current per-rank particle coords."""
    field = swarm.dm.getField("DMSwarmPIC_coor").reshape((-1, swarm.dim))
    out = np.asarray(field).copy()
    swarm.dm.restoreField("DMSwarmPIC_coor")
    return out


def test_swarm_no_change_roundtrip():
    """Trivial case: snapshot, scribble, restore — both coords and svar recovered."""
    uw, model, mesh, swarm, material = _fresh_model_mesh_and_swarm()
    material.data[:, 0] = 0.5 * _swarm_coords(swarm)[:, 0]
    coords_pre = _swarm_coords(swarm)
    material_pre = np.asarray(material.data).copy()

    snap = model.snapshot()

    coord_field = swarm.dm.getField("DMSwarmPIC_coor").reshape((-1, swarm.dim))
    coord_field[...] = -99.0
    swarm.dm.restoreField("DMSwarmPIC_coor")
    material.data[...] = -99.0

    model.restore(snap)

    assert np.allclose(_swarm_coords(swarm), coords_pre)
    assert np.allclose(np.asarray(material.data), material_pre)


def test_swarm_restore_after_migrate():
    """Migrate between snapshot and restore: restore puts the swarm back. This
    is the case my earlier counter-as-gate design wrongly refused."""
    uw, model, mesh, swarm, material = _fresh_model_mesh_and_swarm()
    material.data[:, 0] = 1.0
    coords_pre = _swarm_coords(swarm)
    material_pre = np.asarray(material.data).copy()
    pop_gen_pre = swarm._population_generation

    snap = model.snapshot()

    # Mutate: migrate() will bump the counter regardless of whether
    # particles actually moved. Restore must succeed anyway.
    swarm.migrate(remove_sent_points=True)
    assert swarm._population_generation > pop_gen_pre, "migrate didn't bump counter"

    model.restore(snap)

    assert np.allclose(_swarm_coords(swarm), coords_pre), (
        "restore did not recover particle coords across a migrate event"
    )
    assert np.allclose(np.asarray(material.data), material_pre), (
        "restore did not recover svar data across a migrate event"
    )


def test_swarm_restore_after_add_particles():
    """Particles added between snapshot and restore: restore *removes* them."""
    uw, model, mesh, swarm, material = _fresh_model_mesh_and_swarm()
    material.data[:, 0] = 2.0
    coords_pre = _swarm_coords(swarm)
    material_pre = np.asarray(material.data).copy()
    npre = swarm.dm.getLocalSize()

    snap = model.snapshot()

    swarm.add_particles_with_coordinates(
        np.array([[0.5, 0.5], [0.25, 0.75]])
    )
    assert swarm.dm.getLocalSize() != npre, "add_particles didn't grow swarm"

    model.restore(snap)

    assert swarm.dm.getLocalSize() == npre, (
        "restore did not roll back to the captured particle count"
    )
    assert np.allclose(_swarm_coords(swarm), coords_pre)
    assert np.allclose(np.asarray(material.data), material_pre)


def test_swarm_population_generation_is_informational_not_a_gate():
    """The counter ticks up across mutations but does NOT block restore."""
    uw, model, mesh, swarm, _ = _fresh_model_mesh_and_swarm()
    gen_at_capture = swarm._population_generation

    snap = model.snapshot()

    swarm.migrate(remove_sent_points=True)
    swarm.add_particles_with_coordinates(np.array([[0.5, 0.5]]))
    gen_during = swarm._population_generation
    assert gen_during > gen_at_capture

    # Restore is expected to *succeed*, not raise.
    model.restore(snap)

    # And the counter has moved on from where it was at capture,
    # because restore itself counts as a population change.
    assert swarm._population_generation > gen_at_capture


def test_swarm_internal_variables_are_not_captured():
    """Internal DMSwarm_* variables stay out of the snapshot key list."""
    uw, model, mesh, swarm, material = _fresh_model_mesh_and_swarm()

    snap = model.snapshot()
    keys = snap.backend.list_vectors()
    swarm_name = swarm._snapshot_stable_name()
    svar_keys = [k for k in keys if k.startswith(f"swarm:{swarm_name}:var:")]
    captured_names = {k.split(":var:")[1].split(":data")[0] for k in svar_keys}

    assert "material" in captured_names
    assert not any(n.startswith("DMSwarm") for n in captured_names)


# ----- State-as-dataclass contract: Symbolic DDt -----


def _fresh_model_mesh_and_symbolic_ddt(order=2):
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 4.0
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    ddt = uw.systems.ddt.Symbolic(T.sym, order=order)
    return uw, model, mesh, T, ddt


def test_symbolic_ddt_registers_with_model():
    """A fresh DDt auto-registers in Model._state_bearers."""
    uw, model, mesh, T, ddt = _fresh_model_mesh_and_symbolic_ddt()
    assert ddt in model._state_bearers


def test_symbolic_ddt_state_is_a_snapshottable_dataclass():
    """``.state`` returns a SnapshottableState (DDtSymbolicState) with the
    expected schema version."""
    from underworld3.checkpoint import SnapshottableState

    uw, model, mesh, T, ddt = _fresh_model_mesh_and_symbolic_ddt(order=2)
    state = ddt.state
    assert isinstance(state, SnapshottableState)
    assert state._schema_version == 1
    # Fresh DDt: order-sized dt_history, not initialised, zero solves.
    assert state.dt_history == [None, None]
    assert state.history_initialised is False
    assert state.n_solves_completed == 0


def test_symbolic_ddt_roundtrip_recovers_state():
    """Snapshot mid-trajectory, advance, restore, state equals captured."""
    uw, model, mesh, T, ddt = _fresh_model_mesh_and_symbolic_ddt(order=2)

    # Advance two solves so dt_history fills.
    ddt.update_pre_solve(dt=0.1)
    ddt.update_post_solve(dt=0.1)
    ddt.update_pre_solve(dt=0.2)
    ddt.update_post_solve(dt=0.2)
    state_pre = ddt.state
    # Sanity: history is populated.
    assert state_pre.history_initialised is True
    assert state_pre.n_solves_completed == 2
    assert state_pre.dt_history == [0.2, 0.1]

    snap = model.snapshot()

    # Mutate: take another solve, dt_history changes.
    ddt.update_pre_solve(dt=0.5)
    ddt.update_post_solve(dt=0.5)
    assert ddt.state.dt_history == [0.5, 0.2]

    model.restore(snap)

    # Primary state is back to captured.
    state_post = ddt.state
    assert state_post.dt_history == state_pre.dt_history
    assert state_post.history_initialised == state_pre.history_initialised
    assert state_post.n_solves_completed == state_pre.n_solves_completed
    assert state_post.dt == state_pre.dt


def test_symbolic_ddt_restore_rejects_wrong_schema_version():
    """Hand-built state with wrong _schema_version is refused on apply."""
    uw, model, mesh, T, ddt = _fresh_model_mesh_and_symbolic_ddt(order=2)
    bad_state = ddt.state
    bad_state._schema_version = 999

    with pytest.raises(ValueError, match="schema version"):
        ddt.state = bad_state


def test_symbolic_ddt_restore_rejects_order_mismatch():
    """Restoring a state captured at a different order raises (programming-
    error guard; in practice this shouldn't happen within a single run)."""
    uw, model, mesh, T, ddt = _fresh_model_mesh_and_symbolic_ddt(order=2)
    bad_state = ddt.state
    bad_state.dt_history = [0.1, 0.2, 0.3]  # length 3 != order 2

    with pytest.raises(ValueError, match="dt_history length mismatch"):
        ddt.state = bad_state


def test_symbolic_ddt_snapshot_is_deep_copy():
    """Mutating the live DDt after snapshot doesn't leak into the
    captured state-bearer payload."""
    uw, model, mesh, T, ddt = _fresh_model_mesh_and_symbolic_ddt(order=2)
    ddt.update_pre_solve(dt=0.1)
    ddt.update_post_solve(dt=0.1)

    snap = model.snapshot()
    captured_state = snap.state_bearers[0][1]  # (key, state)
    captured_dt_history = list(captured_state.dt_history)

    # Scribble the live DDt's internal state — must not leak into snapshot.
    ddt._dt_history[0] = -999.0
    ddt._n_solves_completed = 42

    assert captured_state.dt_history == captured_dt_history
    assert captured_state.n_solves_completed != 42


# ----- State-as-dataclass: other DDt flavors -----
#
# Construction-side smoke tests + roundtrip. We exercise the .state /
# .state.setter mechanics directly rather than running full solves;
# the BDF/AM coefficient re-derivation happens in the setter, so a
# manual primary-state mutation is enough to validate the retrofit.


def test_eulerian_ddt_roundtrip():
    import underworld3 as uw
    from underworld3.systems.ddt import DDtEulerianState

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 4.0
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    ddt = uw.systems.ddt.Eulerian(
        mesh, T.sym, uw.VarType.SCALAR, degree=1, continuous=True, order=2
    )
    assert ddt in model._state_bearers
    assert isinstance(ddt.state, DDtEulerianState)

    # Manually advance state (avoid running real projections).
    ddt._dt_history = [0.2, 0.1]
    ddt._history_initialised = True
    ddt._n_solves_completed = 2
    ddt._dt = 0.2
    state_pre = ddt.state

    snap = model.snapshot()

    ddt._dt_history = [0.99, 0.99]
    ddt._history_initialised = False
    ddt._n_solves_completed = 0
    ddt._dt = None

    model.restore(snap)

    assert ddt.state.dt_history == state_pre.dt_history
    assert ddt.state.history_initialised == state_pre.history_initialised
    assert ddt.state.n_solves_completed == state_pre.n_solves_completed
    assert ddt.state.dt == state_pre.dt
    # psi_star names are stable bindings — must match.
    assert ddt.state.psi_star_var_names == state_pre.psi_star_var_names


def test_semilagrangian_ddt_roundtrip():
    import underworld3 as uw
    from underworld3.systems.ddt import DDtSemiLagrangianState

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 4.0
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    V = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
    ddt = uw.systems.ddt.SemiLagrangian(
        mesh, T.sym, V.sym, uw.VarType.SCALAR, degree=1, continuous=True, order=2
    )
    assert ddt in model._state_bearers
    state = ddt.state
    assert isinstance(state, DDtSemiLagrangianState)
    assert state.with_forcing_history is False
    assert state.forcing_star_var_name is None

    ddt._dt_history = [0.3, 0.1]
    ddt._history_initialised = True
    ddt._n_solves_completed = 2
    ddt._dt = 0.3
    state_pre = ddt.state

    snap = model.snapshot()
    ddt._dt_history = [None, None]
    ddt._history_initialised = False
    ddt._n_solves_completed = 0
    model.restore(snap)

    assert ddt.state.dt_history == state_pre.dt_history
    assert ddt.state.history_initialised is True
    assert ddt.state.n_solves_completed == 2


def test_lagrangian_swarm_ddt_registers_and_state_type():
    """Lagrangian_Swarm must be constructed before swarm.populate; the
    retrofit registers it and exposes a typed state. Roundtrip is not
    exercised here because advection requires a velocity-field setup
    beyond the scope of a core unit test."""
    import underworld3 as uw
    from underworld3.systems.ddt import DDtLagrangianSwarmState

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 4.0
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    swarm = uw.swarm.Swarm(mesh)
    ddt = uw.systems.ddt.Lagrangian_Swarm(
        swarm=swarm,
        psi_fn=T.sym,
        vtype=uw.VarType.SCALAR,
        degree=1,
        continuous=True,
        order=2,
    )
    swarm.populate(fill_param=2)

    assert ddt in model._state_bearers
    assert isinstance(ddt.state, DDtLagrangianSwarmState)
    assert len(ddt.state.dt_history) == 2
    assert ddt.state.psi_star_var_names  # non-empty


# Note: uw.systems.ddt.Lagrangian has a pre-existing bug
# (references uw.swarm.UWSwarm which does not exist), so we cannot
# directly construct one for testing. The retrofit code is in place
# and follows the same pattern as the other flavors; consumers that
# construct Lagrangian via the higher-level solver pathways will get
# the .state / .state.setter / registration automatically.
