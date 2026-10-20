"""A regretted step leaves no trace.

Restoring a saved state and continuing gives the same fields to the last bit
whatever was solved between the save and the restore. A restore brings back everything the next solve reads and discards
each preconditioner's setup, so the steps after it are those of a run started
cold from the saved state.

This is what makes the start of a step a clean place to reconstruct a run
from — a step rejected for a Newton failure and retaken in two halves, or a
replay from a recorded step. The viscoelastic Navier-Stokes case exercises every kind of
state a step reads: the unknowns, the extrapolation level of the momentum
advection, the log-conformation stress history and its time-scheme
coefficients, and the mesh geometry.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


@pytest.fixture(autouse=True)
def _reset():
    import underworld3 as uw

    uw.reset_default_model()
    yield
    uw.reset_default_model()


def _channel():
    """Oldroyd-B channel flow: parabolic inflow, no-slip walls, SUPG momentum
    and an Eulerian log-conformation stress history."""
    import underworld3 as uw

    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0), cellSize=0.15, qdegree=3)
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1)
    ns.stress_transport = "eulerian"
    ns.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        ns.Unknowns, order=1, integrator="etd", objective_rate="upper_convected",
        stress_history="log_conformation")
    cm = ns.constitutive_model
    eta, relaxation_time = 0.01, 0.05
    cm.Parameters.shear_viscosity_0 = 0.5 * eta
    cm.Parameters.shear_modulus = 0.5 * eta / relaxation_time
    cm.Parameters.solvent_viscosity = 0.5 * eta
    ns.devss_viscosity = eta
    x, y = mesh.X
    ns.add_dirichlet_bc((4.0 * y * (1 - y), 0.0), "Left")
    ns.add_dirichlet_bc((0.0, 0.0), "Top")
    ns.add_dirichlet_bc((0.0, 0.0), "Bottom")
    return uw, mesh, ns, cm, v, p


def _step(ns, cm, dt):
    cm.Parameters.dt_elastic = dt
    ns.solve(timestep=dt, zero_init_guess=False)


def _state(ns, v, p):
    return [np.asarray(v.array[...]).copy(), np.asarray(p.array[...]).copy()] + [
        np.asarray(ps.array[...]).copy() for ps in ns.DFDt.psi_star]


def test_a_regretted_step_leaves_no_trace():
    uw, mesh, ns, cm, v, p = _channel()
    orchestration_model = uw.get_default_model()
    dt = 0.02

    for _ in range(3):
        _step(ns, cm, dt)
    token = orchestration_model.save_state()

    # Restore, then three steps.
    orchestration_model.load_state(token)
    for _ in range(3):
        _step(ns, cm, dt)
    direct = _state(ns, v, p)

    # Restore, a regretted step ten times too long, restore, the same steps.
    orchestration_model.load_state(token)
    _step(ns, cm, 10 * dt)
    orchestration_model.load_state(token)
    for _ in range(3):
        _step(ns, cm, dt)
    replay = _state(ns, v, p)

    names = ["velocity", "pressure"] + [f"stress history level {k}" for k in range(len(ns.DFDt.psi_star))]
    for name, a, b in zip(names, direct, replay):
        assert np.array_equal(a, b), (
            f"{name} after a regretted step differs from the run without it: "
            f"max abs diff {np.max(np.abs(a - b)):.3e} (expected exactly 0)")


def test_a_restore_brings_back_what_the_solver_carries_between_steps():
    """A step leaves state on the solver that steers the next one: the
    warm-start flag (a diverged solve clears it, and it decides how the next
    solve starts), the step length a solve without ``timestep`` reuses, and the
    rate of change ``estimate_dt`` reads. A restore puts all three back."""
    uw, mesh, ns, cm, v, p = _channel()
    orchestration_model = uw.get_default_model()
    dt = 0.02
    for _ in range(2):
        _step(ns, cm, dt)
    assert ns.has_solution
    suggested = ns.estimate_dt()
    token = orchestration_model.save_state()

    _step(ns, cm, 10 * dt)
    ns._has_solution = False          # what a diverged regretted step leaves behind
    assert ns.estimate_dt() != suggested

    orchestration_model.load_state(token)
    assert ns.has_solution
    assert ns.estimate_dt() == suggested
    assert ns.delta_t.sym == dt


def test_a_restore_of_an_undeformed_mesh_leaves_the_solver_built():
    """The restore must not move a mesh that did not move: a move rebuilds
    every solver on the mesh, and a rebuilt solver starts its next solve from
    a different place."""
    uw, mesh, ns, cm, v, p = _channel()
    orchestration_model = uw.get_default_model()
    _step(ns, cm, 0.02)
    version = mesh._geometry_version
    token = orchestration_model.save_state()
    _step(ns, cm, 0.02)
    orchestration_model.load_state(token)
    assert mesh._geometry_version == version
    assert not ns._needs_dm_rebuild


def test_an_advection_diffusion_solver_gets_its_step_state_back():
    """The Eulerian advection-diffusion solver carries the same kind of step
    state (the step length a solve without ``timestep`` reuses, the rate
    ``estimate_dt`` reads); a restore puts it back."""
    import sympy
    import underworld3 as uw

    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8), minCoords=(0.0, 0.0),
                                        maxCoords=(1.0, 1.0), qdegree=3)
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=1)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
    adv_diff = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v)
    adv_diff.constitutive_model = uw.constitutive_models.DiffusionModel
    adv_diff.constitutive_model.Parameters.diffusivity = 0.01
    adv_diff.add_dirichlet_bc(0.0, "Left")
    v.array[:, 0, 0] = 1.0
    x, y = mesh.X
    T.array = uw.function.evaluate(sympy.sin(sympy.pi * x) * sympy.sin(sympy.pi * y), T.coords)

    orchestration_model = uw.get_default_model()
    adv_diff.solve(timestep=0.01)
    suggested = adv_diff.estimate_dt()
    token = orchestration_model.save_state()
    adv_diff.solve(timestep=0.2)
    assert adv_diff.estimate_dt() != suggested
    orchestration_model.load_state(token)
    assert adv_diff.estimate_dt() == suggested
    assert adv_diff.state.last_timestep == 0.01


def _cavity_stokes(uw, mesh, rotated=False):
    import sympy

    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    amplitude = uw.function.expression(r"A", 1.0, "forcing amplitude")
    x, y = mesh.X
    stokes.bodyforce = sympy.Matrix([0, amplitude * sympy.sin(sympy.pi * x) * sympy.cos(sympy.pi * y)])
    stokes.tolerance = 1.0e-8
    stokes.petsc_use_pressure_nullspace = True
    for wall in ("Left", "Right", "Bottom", "Top"):
        if rotated:
            stokes.add_rotated_freeslip_bc(0, wall)
    if not rotated:
        stokes.add_dirichlet_bc((0.0, None), "Left")
        stokes.add_dirichlet_bc((0.0, None), "Right")
        stokes.add_dirichlet_bc((None, 0.0), "Bottom")
        stokes.add_dirichlet_bc((None, 0.0), "Top")
    return stokes, v, p, amplitude


def _regretted_and_direct(uw, stokes, v, amplitude):
    """Velocity after save -> restore -> solve, and after save -> regretted
    solve -> restore -> solve."""
    orchestration_model = uw.get_default_model()
    stokes.solve(zero_init_guess=True)
    token = orchestration_model.save_state()
    orchestration_model.load_state(token)
    amplitude.sym = 2.0
    stokes.solve(zero_init_guess=True)
    direct = np.asarray(v.array[...]).copy()
    orchestration_model.load_state(token)
    amplitude.sym = 7.0
    stokes.solve(zero_init_guess=True)
    orchestration_model.load_state(token)
    amplitude.sym = 2.0
    stokes.solve(zero_init_guess=True)
    return direct, np.asarray(v.array[...]).copy()


def test_a_restore_keeps_multigrid_transfers_set_in_code():
    """A velocity multigrid on transfers set in code (a custom hierarchy)
    must survive a restore -- a reset would drop the transfers and the next
    setup would try to coarsen the mesh itself -- and a regretted solve must
    leave no trace through it."""
    import underworld3 as uw
    from underworld3.utilities import custom_mg

    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25, regular=True, qdegree=3)

    def level(dm):
        return uw.discretisation.Mesh(dm.clone(), simplex=True, qdegree=3,
                                      coordinate_system_type=base.CoordinateSystem.coordinate_type,
                                      boundaries=base.boundaries)

    dm1 = base.dm.refine()
    coarse = [level(base.dm), level(dm1)]
    mesh = level(dm1.refine())
    for k, m in enumerate([base] + coarse + [mesh]):
        m.name = f"level{k}"
    stokes, v, p, amplitude = _cavity_stokes(uw, mesh)
    custom_mg.set_custom_fmg(stokes, coarse, builder="barycentric", field_id=0)

    direct, replay = _regretted_and_direct(uw, stokes, v, amplitude)
    assert stokes.snes.getConvergedReason() > 0
    assert np.array_equal(direct, replay), (
        f"regretted solve left a trace: {np.abs(direct - replay).max():.3e}")


def test_a_restore_discards_the_rotated_free_slip_workspace():
    """The rotated free-slip solve keeps its own operator, preconditioner and
    a hint that licenses a shortcut on the next solve. A restore must drop
    them, as it does the solver's own preconditioner, so the next solve builds
    them from the restored state rather than from a regretted one."""
    import underworld3 as uw

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=2)
    stokes, v, p, amplitude = _cavity_stokes(uw, mesh, rotated=True)
    orchestration_model = uw.get_default_model()
    stokes.solve(zero_init_guess=True)
    token = orchestration_model.save_state()
    amplitude.sym = 7.0
    stokes.solve(zero_init_guess=True)
    assert stokes._rotated_linear_cache is not None
    orchestration_model.load_state(token)
    assert stokes._rotated_linear_cache is None
    amplitude.sym = 2.0
    stokes.solve(zero_init_guess=True)
    assert stokes.snes is not None and stokes._rotated_linear_cache is not None


def test_a_disk_snapshot_carries_the_solver_step_state(tmp_path):
    """A disk snapshot carries what each solver carries between steps, so a
    restore from a file is the same line as a restore from memory."""
    uw, mesh, ns, cm, v, p = _channel()
    orchestration_model = uw.get_default_model()
    dt = 0.02
    for _ in range(2):
        _step(ns, cm, dt)
    saved = ns.state
    path = str(tmp_path / "line.snap.h5")
    orchestration_model.save_state(file=path)
    _step(ns, cm, 10 * dt)
    ns._has_solution = False
    orchestration_model.load_state(path)
    assert ns.state == saved

