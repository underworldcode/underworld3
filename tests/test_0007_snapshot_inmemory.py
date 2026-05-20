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
    # Find the DDt's captured state by type — state_bearers is
    # unordered and now also contains the model tracker.
    from underworld3.systems.ddt import DDtSymbolicState

    captured_state = next(
        st for _key, st in snap.state_bearers
        if isinstance(st, DDtSymbolicState)
    )
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


def test_lagrangian_ddt_roundtrip():
    """Lagrangian creates its own internal swarm; the fix in this PR
    restored uw.swarm.Swarm in __init__ (was a typo'd UWSwarm)."""
    import underworld3 as uw
    from underworld3.systems.ddt import DDtLagrangianState

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 4.0
    )
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    V = uw.discretisation.MeshVariable("V", mesh, 2, degree=2)
    ddt = uw.systems.ddt.Lagrangian(
        mesh=mesh, psi_fn=T.sym, V_fn=V.sym,
        vtype=uw.VarType.SCALAR, degree=1, continuous=True, order=2,
    )
    assert ddt in model._state_bearers
    assert isinstance(ddt.state, DDtLagrangianState)

    ddt._dt_history = [0.2, 0.1]
    ddt._history_initialised = True
    ddt._n_solves_completed = 2
    ddt._dt = 0.2
    state_pre = ddt.state

    snap = model.snapshot()
    ddt._dt_history = [None, None]
    ddt._history_initialised = False
    ddt._n_solves_completed = 0
    model.restore(snap)

    assert ddt.state.dt_history == state_pre.dt_history
    assert ddt.state.history_initialised is True
    assert ddt.state.n_solves_completed == 2
    assert ddt.state.psi_star_var_names == state_pre.psi_star_var_names


# ----- End-to-end back-stepping demonstration -----
#
# Everything above this comment is unit-style: build a thing, snapshot,
# scribble, restore, check equality. This block exercises the toolkit's
# actual reason for existing: a *real* time-stepping use case where the
# consumer takes a step, detects it was bad, snapshots back, and retries
# with smaller Δt. The pattern is canonical adaptive-Δt CFL control;
# the snapshot mechanism is the thing that makes "snap back" possible
# without manually unwinding mesh / swarm / DDt state.


def test_backstepping_cfl_recovery_end_to_end():
    """Canonical adaptive-Δt back-step demonstration.

    Set up a swarm advecting in a known velocity field, with a
    material variable carried along and a Symbolic DDt accumulating
    BDF history. Take one too-large Δt step → CFL violation
    (max-particle-displacement exceeds the mesh cell radius). Detect
    it. Restore the snapshot. Retry with a smaller Δt → CFL satisfied,
    state evolves cleanly. The full triple of state (swarm positions,
    material variable, DDt history) is recovered on restore.
    """
    import underworld3 as uw
    import sympy
    import numpy as np

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0
    )

    # Outward-radial velocity from the box centre. |V| ranges from 0
    # at the centre to ~0.71 at the corners — pick Δt to give a
    # genuine CFL violation rather than tweak parameters to fit.
    x, y = mesh.X
    V_fn = sympy.Matrix([[x - 0.5, y - 0.5]]).T

    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable("material", 1, dtype=float)
    swarm.populate(fill_param=2)
    coords_initial = swarm._particle_coordinates.data.copy()
    material.data[:, 0] = coords_initial[:, 0]  # carry x as marker
    material_initial = np.asarray(material.data).copy()

    # A separate DDt manages BDF history for a scalar field on the
    # mesh. Advance it manually past startup so its captured state is
    # non-trivial.
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    ddt = uw.systems.ddt.Symbolic(T.sym, order=2)
    ddt._dt_history = [0.05, 0.05]
    ddt._history_initialised = True
    ddt._n_solves_completed = 2
    ddt._dt = 0.05
    ddt_state_initial = ddt.state

    # The user's CFL threshold: a particle moving more than one cell
    # radius in a single step has crossed an element. min_radius is
    # the standard UW3 cell-size proxy.
    cfl_threshold = mesh.get_min_radius()

    # Take the snapshot *before* the speculative step. Everything that
    # will be touched gets captured.
    snap = model.snapshot()

    # Speculative step at the candidate Δt. Bigger than the user
    # thinks is safe — they'll check after and back-step if it isn't.
    candidate_dt = 0.5
    swarm.advection(V_fn, delta_t=candidate_dt, step_limit=False)

    # CFL check: max displacement among local particles.
    coords_after_bad = swarm._particle_coordinates.data
    max_disp_bad = np.max(
        np.linalg.norm(coords_after_bad - coords_initial, axis=1)
    )
    assert max_disp_bad > cfl_threshold, (
        f"speculative step at dt={candidate_dt} should violate CFL "
        f"(max_disp={max_disp_bad:.4f} vs threshold {cfl_threshold:.4f})"
    )

    # Back-step. Everything captured is brought back to the snapshot
    # point — swarm positions, the material variable carried with the
    # swarm, and the DDt's BDF history.
    model.restore(snap)

    assert np.allclose(swarm._particle_coordinates.data, coords_initial), (
        "particle positions did not roll back after restore"
    )
    assert np.allclose(np.asarray(material.data), material_initial), (
        "swarm-variable data did not roll back after restore"
    )
    assert ddt.state.dt_history == ddt_state_initial.dt_history, (
        "DDt history did not roll back after restore"
    )
    assert ddt.state.n_solves_completed == ddt_state_initial.n_solves_completed

    # Retry with a smaller Δt. CFL now satisfied.
    retry_dt = candidate_dt / 10.0
    swarm.advection(V_fn, delta_t=retry_dt, step_limit=False)

    coords_after_good = swarm._particle_coordinates.data
    max_disp_good = np.max(
        np.linalg.norm(coords_after_good - coords_initial, axis=1)
    )
    assert max_disp_good < cfl_threshold, (
        f"retry at dt={retry_dt} should satisfy CFL "
        f"(max_disp={max_disp_good:.4f} vs threshold {cfl_threshold:.4f})"
    )
    # Sanity: smaller dt produced strictly smaller displacement.
    assert max_disp_good < max_disp_bad / 5.0


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


# ----- Bit-identical continuation (the core production guarantee) -----
#
# Everything above proves *state equality after restore*. That is
# necessary but not the actual guarantee a backtracking consumer
# relies on. The guarantee is: after restore, *continuing the
# simulation* reproduces the trajectory of a run that never took the
# discarded step. These two tests assert that, bit-for-bit
# (np.array_equal — no tolerance), with a swarm + mesh variable +
# Symbolic DDt all live so the mesh -> swarm -> state-bearer restore
# ordering is exercised together.


def _build_continuation_fixture():
    """Mesh + swarm(+material) + a driven mesh variable + Symbolic DDt.

    Returns everything needed to run a deterministic step loop.
    """
    import underworld3 as uw
    import sympy

    uw.reset_default_model()
    model = uw.get_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 6.0
    )
    x_sym, y_sym = mesh.X
    V_fn = sympy.Matrix([[x_sym - 0.5, y_sym - 0.5]]).T

    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
    T.array[:, 0, 0] = 0.0

    swarm = uw.swarm.Swarm(mesh)
    material = swarm.add_variable("material", 1, dtype=float)
    swarm.populate(fill_param=2)
    material.data[:, 0] = np.linalg.norm(
        swarm._particle_coordinates.data - 0.5, axis=1
    )

    ddt = uw.systems.ddt.Symbolic(T.sym, order=2)

    return uw, model, mesh, V_fn, T, swarm, material, ddt


def _step(uw, V_fn, T, swarm, ddt, dt):
    """One deterministic step: advect swarm, evolve T by a fixed rule,
    advance the DDt history. No solver, no randomness."""
    ddt.update_pre_solve(dt)
    swarm.advection(V_fn, delta_t=dt, step_limit=False)
    # Deterministic, history-free field update so T carries evolving
    # state through the mesh-variable snapshot path.
    T.array[:, 0, 0] = T.array[:, 0, 0] + dt
    ddt.update_post_solve(dt)


def _capture_full_state(T, swarm, material, ddt):
    """Everything that must match for bit-identical continuation."""
    return {
        "T": np.asarray(T.array[...]).copy(),
        "coords": swarm._particle_coordinates.data.copy(),
        "material": np.asarray(material.data).copy(),
        "dt_history": list(ddt.state.dt_history),
        "n_solves": ddt.state.n_solves_completed,
        "ddt_dt": ddt.state.dt,
    }


def _assert_bit_identical(a, b, label):
    assert np.array_equal(a["T"], b["T"]), f"{label}: T differs"
    assert np.array_equal(a["coords"], b["coords"]), (
        f"{label}: swarm coords differ"
    )
    assert np.array_equal(a["material"], b["material"]), (
        f"{label}: swarm material differs"
    )
    assert a["dt_history"] == b["dt_history"], (
        f"{label}: DDt dt_history differs ({a['dt_history']} vs "
        f"{b['dt_history']})"
    )
    assert a["n_solves"] == b["n_solves"], f"{label}: DDt n_solves differs"
    assert a["ddt_dt"] == b["ddt_dt"], f"{label}: DDt dt differs"


def test_continuation_deterministic_after_restore():
    """snapshot S -> K steps -> A; restore(S) -> K steps -> B.
    A and B must be bit-identical. Proves restore leaves no residual
    state that perturbs subsequent evolution."""
    uw, model, mesh, V_fn, T, swarm, material, ddt = (
        _build_continuation_fixture()
    )

    # Advance to a non-trivial state before snapshotting (fill DDt
    # history, move particles off their lattice).
    for _ in range(3):
        _step(uw, V_fn, T, swarm, ddt, 0.05)

    snap = model.snapshot()

    # Branch A: K steps straight from S.
    for _ in range(5):
        _step(uw, V_fn, T, swarm, ddt, 0.05)
    state_A = _capture_full_state(T, swarm, material, ddt)

    # Branch B: restore S, then the identical K steps.
    model.restore(snap)
    for _ in range(5):
        _step(uw, V_fn, T, swarm, ddt, 0.05)
    state_B = _capture_full_state(T, swarm, material, ddt)

    _assert_bit_identical(state_A, state_B, "deterministic-continuation")


def test_continuation_bit_identical_across_stash_and_recover():
    """The real 'git stash for steps' guarantee:

      control:  S -> K good steps                       -> ctrl
      stash:    S -> bad disruptive step -> restore(S)
                  -> same K good steps                  -> stash

    ctrl and stash must be bit-identical: the discarded step must
    leave no trace whatsoever after restore + continuation."""
    uw, model, mesh, V_fn, T, swarm, material, ddt = (
        _build_continuation_fixture()
    )

    for _ in range(3):
        _step(uw, V_fn, T, swarm, ddt, 0.05)

    snap = model.snapshot()

    # Control: K good steps from S.
    for _ in range(5):
        _step(uw, V_fn, T, swarm, ddt, 0.05)
    ctrl = _capture_full_state(T, swarm, material, ddt)

    # Stash scenario: back to S, take a deliberately disruptive step
    # (10x Δt — large advection, big T jump, DDt history shift), then
    # discard it via restore and run the intended K good steps.
    model.restore(snap)
    _step(uw, V_fn, T, swarm, ddt, 0.5)  # the regretted step
    model.restore(snap)
    for _ in range(5):
        _step(uw, V_fn, T, swarm, ddt, 0.05)
    stash = _capture_full_state(T, swarm, material, ddt)

    _assert_bit_identical(ctrl, stash, "stash-and-recover")


