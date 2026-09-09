import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np


def _fresh_model():
    import underworld3 as uw

    uw.reset_default_model()
    return uw, uw.get_default_model()


def test_tracker_exists_with_default_conventions():
    """A fresh model has a tracker pre-seeded with time/step/dt."""
    uw, model = _fresh_model()
    assert model.tracker.time == 0.0
    assert model.tracker.step == 0
    assert model.tracker.dt is None
    assert set(model.tracker.keys()) == {"time", "step", "dt"}


def test_tracker_is_registered_state_bearer():
    """The tracker auto-registers so snapshot/restore see it."""
    uw, model = _fresh_model()
    assert model.tracker in model._state_bearers


def test_tracker_builtins_revert_on_restore():
    """time/step/dt are managed entries — they roll back."""
    uw, model = _fresh_model()
    model.tracker.time = 3.14
    model.tracker.step = 7
    model.tracker.dt = 0.05

    snap = model.save_state()

    model.tracker.time = 99.0
    model.tracker.step = 999
    model.tracker.dt = 1.0

    model.load_state(snap)

    assert model.tracker.time == 3.14
    assert model.tracker.step == 7
    assert model.tracker.dt == 0.05


def test_tracker_user_quantity_reverts():
    """A user-added scalar is managed automatically — no dataclass,
    no special status — and reverts on restore."""
    uw, model = _fresh_model()
    model.tracker.my_diagnostic = 42.0

    snap = model.save_state()
    model.tracker.my_diagnostic = -1.0
    model.load_state(snap)

    assert model.tracker.my_diagnostic == 42.0


def test_tracker_numpy_quantity_reverts_by_value():
    """A numpy array on the tracker is deep-copied into the snapshot,
    so post-snapshot in-place mutation doesn't leak, and it reverts."""
    uw, model = _fresh_model()
    arr = np.array([1.0, 2.0, 3.0])
    model.tracker.history = arr

    snap = model.save_state()
    model.tracker.history[:] = -9.0  # in-place mutation
    assert np.allclose(model.tracker.history, -9.0)

    model.load_state(snap)
    assert np.allclose(model.tracker.history, [1.0, 2.0, 3.0])


def test_tracker_quantity_added_after_snapshot_is_dropped_on_restore():
    """git-stash semantics: restore returns to exactly the captured
    point, so a quantity created after the snapshot disappears."""
    uw, model = _fresh_model()
    model.tracker.a = 1.0

    snap = model.save_state()
    model.tracker.b = 2.0  # created after snapshot
    assert "b" in model.tracker

    model.load_state(snap)
    assert "a" in model.tracker
    assert "b" not in model.tracker


def test_tracker_is_what_makes_state_revertible():
    """The contrast that motivates the tracker: a loose Python
    variable is NOT reverted by restore; the same value parked on the
    tracker IS. This is the whole point."""
    uw, model = _fresh_model()

    loose_time = 0.0
    model.tracker.time = 0.0

    snap = model.save_state()

    # Advance both the loose variable and the tracked one.
    loose_time = 5.0
    model.tracker.time = 5.0

    model.load_state(snap)

    # The loose variable is untouched by restore (the language can't
    # know about it); the tracked one rolled back.
    assert loose_time == 5.0          # NOT reverted
    assert model.tracker.time == 0.0  # reverted automatically


def test_tracker_state_roundtrip_is_bit_identical():
    """snapshot S -> mutate -> restore: tracker.state equals the
    captured state exactly (dataclass equality)."""
    uw, model = _fresh_model()
    model.tracker.time = 1.0
    model.tracker.step = 2
    model.tracker.payload = np.arange(5).astype(float)
    state_pre = model.tracker.state

    snap = model.save_state()
    model.tracker.time = 12345.0
    model.tracker.payload[:] = 0.0
    model.load_state(snap)

    state_post = model.tracker.state
    assert state_post.managed["time"] == state_pre.managed["time"]
    assert state_post.managed["step"] == state_pre.managed["step"]
    assert np.array_equal(
        state_post.managed["payload"], state_pre.managed["payload"]
    )


def test_tracker_continuation_with_solver_loop():
    """Realistic: drive time/step on the tracker through a stepping
    loop, snapshot mid-run, take a regretted step, restore, continue;
    the tracker is exactly back and continues correctly."""
    uw, model = _fresh_model()
    import sympy

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 6.0
    )
    x, y = mesh.X
    V_fn = sympy.Matrix([[x - 0.5, y - 0.5]]).T
    swarm = uw.swarm.Swarm(mesh)
    swarm.populate(fill_param=2)

    def do_step(dt):
        swarm.advection(V_fn, delta_t=dt, step_limit=False)
        model.tracker.time = model.tracker.time + dt
        model.tracker.step = model.tracker.step + 1

    for _ in range(3):
        do_step(0.05)

    snap = model.save_state()
    t_snap, s_snap = model.tracker.time, model.tracker.step

    # Regretted big step.
    do_step(0.5)
    assert model.tracker.step == s_snap + 1
    assert model.tracker.time != t_snap

    model.load_state(snap)
    assert model.tracker.time == t_snap
    assert model.tracker.step == s_snap

    # Continue cleanly.
    for _ in range(2):
        do_step(0.05)
    assert model.tracker.step == s_snap + 2
    assert abs(model.tracker.time - (t_snap + 0.10)) < 1e-12


@pytest.mark.xfail(
    reason="mesh.t is a symbolic atom bound to PETSc's petsc_t, which the "
    "high-level solve() wrappers never set, so an expression containing it is "
    "silently ZERO inside a solve and solve(time=...) is accepted and ignored. "
    "Remove this xfail when mesh.t resolves to the model clock (#410 ruling: "
    "time is model-owned, mesh.t becomes a back-compat accessor onto it).",
    strict=False,
)
def test_mesh_t_resolves_to_the_model_clock():
    """The other clock. `mesh.t` is what users reach for in a time-dependent
    boundary condition, and it is NOT `model.tracker.time` — so a source term
    proportional to it should scale with the clock, and today does not.

    The constant-source control is what makes the assertion meaningful: it
    proves the Poisson problem produces a non-trivial solution at all, so a
    zero answer with `mesh.t` is the clock's fault and not the setup's.
    """
    uw, model = _fresh_model()
    import sympy

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0, qdegree=2
    )
    T = uw.discretisation.MeshVariable("T_clock", mesh, 1, degree=2)
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    for boundary in ("Top", "Bottom", "Left", "Right"):
        poisson.add_dirichlet_bc(0.0, boundary)
    poisson.petsc_options.delValue("ksp_monitor")

    poisson.f = sympy.sympify(1.0)
    poisson.solve()
    control = np.abs(np.asarray(T.array)).max()
    assert control > 1.0e-3, "control failed: the Poisson setup itself is trivial"

    poisson.f = mesh.t
    model.tracker.time = 5.0
    poisson.solve()
    assert np.abs(np.asarray(T.array)).max() > 0.1 * control


def test_a_dimensional_clock_survives_a_disk_snapshot(tmp_path):
    """The pattern asks scripts to define units, which makes the clock
    dimensional. That clock must survive a restart.

    The plain-float control is what makes this specific: it shows the disk
    path works for ordinary values, so a dropped quantity would be about
    units and not about the tracker or the file.
    """
    uw, model = _fresh_model()

    model.set_reference_quantities(
        domain_depth=uw.quantity(500, "km"),
        material_density=uw.quantity(3300, "kg/m**3"),
        material_viscosity=uw.quantity(1e21, "Pa*s"),
    )
    uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0, qdegree=2
    )

    model.tracker.plain_control = 3.25
    model.tracker.time = uw.quantity(4.5, "Myr")

    path = str(tmp_path / "units.snap.h5")
    model.save_state(file=path)

    model.tracker.plain_control = -1.0
    model.tracker.time = uw.quantity(-1.0, "Myr")
    model.load_state(path)

    assert model.tracker.plain_control == pytest.approx(3.25), (
        "control failed: the disk snapshot lost an ordinary float too"
    )
    assert model.tracker.time.magnitude == pytest.approx(4.5)
    assert str(model.tracker.time.units) == "megayear"


def test_a_dimensional_array_survives_a_disk_snapshot(tmp_path):
    """The magnitude may be an array, which is stored as a dataset rather
    than an attribute — the other half of the quantity round-trip."""
    uw, model = _fresh_model()

    model.set_reference_quantities(
        domain_depth=uw.quantity(500, "km"),
        material_density=uw.quantity(3300, "kg/m**3"),
        material_viscosity=uw.quantity(1e21, "Pa*s"),
    )
    uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8.0, qdegree=2
    )

    model.tracker.depths = uw.quantity(np.array([10.0, 20.0, 30.0]), "km")

    path = str(tmp_path / "arr.snap.h5")
    model.save_state(file=path)
    model.tracker.depths = uw.quantity(np.array([0.0]), "km")
    model.load_state(path)

    assert np.allclose(model.tracker.depths.magnitude, [10.0, 20.0, 30.0])
    assert str(model.tracker.depths.units) == "kilometer"
