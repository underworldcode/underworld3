"""What the step transcript claims must be what happened.

Two ways it was over- or under-reporting, both found by writing a real
annulus convection run in the timestepping pattern.

1. The transcript counted one operator as two. The hook lives in
   ``_update_constants``, which is the single point every solver passes on its
   way to a solve — except that the rotated free-slip loop pushes constants a
   second time for its own assembly, after the public ``solve()`` has already
   announced the solver. So a Stokes solve on a curved boundary recorded twice.

2. ``estimate_dt()`` lost its units under the pattern's own idiom.
   ``dt = fraction * solver.estimate_dt()`` came back as a bare float whenever
   the estimate happened to be a Python float rather than a numpy scalar,
   because ``np.squeeze`` promoted it to a 0-d array and a dimensionalised
   array is a ``UnitAwareArray``, which drops units under arithmetic.
"""

import pytest

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import numpy as np
import sympy


def _annulus_model(units=True):
    import underworld3 as uw

    uw.reset_default_model()
    model = uw.get_default_model()
    if units:
        model.set_reference_quantities(
            shell_thickness=uw.quantity(2200, "km"),
            thermal_diffusivity=uw.quantity(1e-6, "m**2/s"),
            mantle_viscosity=uw.quantity(1e22, "Pa*s"),
            temperature_contrast=uw.quantity(2500, "K"),
        )
    mesh = uw.meshing.Annulus(
        radiusInner=0.55, radiusOuter=1.0, cellSize=0.25, degree=1, qdegree=3
    )
    v = uw.discretisation.MeshVariable("v", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = (
        uw.quantity(1e22, "Pa*s") if units else 1.0
    )
    stokes.tolerance = 1.0e-6
    stokes.petsc_options.delValue("ksp_monitor")
    stokes.add_rotated_freeslip_bc(0.0, "Upper")
    stokes.add_rotated_freeslip_bc(0.0, "Lower")

    radius = sympy.sqrt(mesh.X.dot(mesh.X))
    if units:
        stokes.bodyforce = (
            -uw.quantity(3300, "kg/m**3")
            * uw.quantity(3e-5, "1/K")
            * uw.quantity(9.81, "m/s**2")
            * T.sym[0]
            * mesh.X
            / radius
        )
    else:
        stokes.bodyforce = -1.0e5 * T.sym[0] * mesh.X / radius

    adv = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v.sym)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = (
        uw.quantity(1e-6, "m**2/s") if units else 1.0
    )
    adv.add_dirichlet_bc(1.0, "Lower")
    adv.add_dirichlet_bc(0.0, "Upper")
    adv.tolerance = 1.0e-6
    adv.petsc_options.delValue("ksp_monitor")

    scale = 1.0
    if units:
        scale = float(model.get_fundamental_scales()["length"].to("m").magnitude)
    X = np.asarray(T.coords)[:, :2] / scale
    r = np.sqrt((X**2).sum(axis=1))
    th = np.arctan2(X[:, 1], X[:, 0])
    shell = (r - 0.55) / (1.0 - 0.55)
    T.array[:, 0, 0] = (1.0 - shell) + 0.1 * np.sin(5.0 * th) * np.sin(np.pi * shell)
    adv.Unknowns.DuDt.initialise_history()

    return uw, model, mesh, stokes, adv, T


def _names(entry, kind="solve"):
    return [e["name"] for e in entry.events if e["kind"] == kind]


def test_rotated_freeslip_solve_is_recorded_once():
    """A curved-boundary Stokes solve is one operator, not two."""
    uw, model, mesh, stokes, adv, T = _annulus_model(units=False)
    stokes.solve(zero_init_guess=True)

    model.tracker.time = 0.0
    model.tracker.step = 0

    with model.step(0.01, label="convect"):
        adv.solve(timestep=0.01, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)

    entry = model.transcript[-1]
    solves = _names(entry)
    assert sum(1 for n in solves if "Stokes" in n) == 1, (
        f"the rotated free-slip dispatch recorded more than one Stokes solve: {solves}"
    )
    assert sum(1 for n in solves if "AdvectionDiffusion" in n) == 1, solves
    assert len(_names(entry, "history_shift")) == 1


def test_a_solver_called_twice_is_still_recorded_twice():
    """The de-duplication must not hide a genuinely repeated solve."""
    uw, model, mesh, stokes, adv, T = _annulus_model(units=False)
    stokes.solve(zero_init_guess=True)

    model.tracker.time = 0.0
    model.tracker.step = 0

    with model.step(0.01):
        adv.solve(timestep=0.01, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)
        adv.solve(timestep=0.01, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)

    solves = _names(model.transcript[-1])
    assert sum(1 for n in solves if "Stokes" in n) == 2, solves


@pytest.mark.parametrize("solver_name", ["stokes", "adv"])
def test_estimate_dt_survives_being_scaled(solver_name):
    """`dt = fraction * solver.estimate_dt()` must keep its units."""
    uw, model, mesh, stokes, adv, T = _annulus_model(units=True)
    stokes.solve(zero_init_guess=True)

    solver = stokes if solver_name == "stokes" else adv
    dt = solver.estimate_dt()
    assert hasattr(dt, "to"), f"{solver_name}.estimate_dt() returned {type(dt).__name__}"

    scaled = 0.5 * dt
    assert hasattr(scaled, "to"), (
        f"0.5 * {solver_name}.estimate_dt() dropped its units "
        f"({type(dt).__name__} -> {type(scaled).__name__})"
    )
    assert float(scaled.to("s").magnitude) == pytest.approx(
        0.5 * float(dt.to("s").magnitude), rel=1e-12
    )


def test_a_scaled_estimate_drives_the_clock():
    """The whole idiom, end to end: the scaled estimate must reach the clock."""
    uw, model, mesh, stokes, adv, T = _annulus_model(units=True)
    stokes.solve(zero_init_guess=True)

    model.tracker.time = uw.quantity(0.0, "Myr")
    model.tracker.step = 0

    dt = 0.5 * adv.estimate_dt()
    with model.step(dt, label="convect"):
        adv.solve(timestep=dt, zero_init_guess=False)
        stokes.solve(zero_init_guess=False)

    elapsed = model.tracker.time
    assert hasattr(elapsed, "to")
    assert float(elapsed.to("s").magnitude) == pytest.approx(
        float(dt.to("s").magnitude), rel=1e-12
    )
