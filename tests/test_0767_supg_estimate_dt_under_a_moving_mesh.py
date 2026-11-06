"""The accuracy-based estimate_dt must not decay under a moving mesh.

A move leaves wall nodes away from their Dirichlet values, and the next solve
restores them by an amount that does not shrink with the timestep. estimate_dt
counted that as a rate of change, which makes the next step proportional to the
last. On this case the step falls from 8.9e-5 to 1.8e-5 over six moves when the
constrained nodes count, and rises from 3.2e-4 to 1.7e-2 when they don't.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_estimate_dt_holds_across_repeated_moves():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 16, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    theta = sympy.atan2(y, x)
    T = uw.discretisation.MeshVariable("T0767", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(
        0.5 * (1 - sympy.tanh((r - 0.9) / 0.03)) * (1 + 0.3 * sympy.cos(4 * theta)),
        T.coords).reshape(-1)
    velocity = sympy.Matrix([[-y, x]])
    adv = uw.systems.AdvDiffusion(mesh, T, velocity, order=1, theta=1.0)
    adv.constitutive_model = uw.constitutive_models.DiffusionModel
    adv.constitutive_model.Parameters.diffusivity = 1.0e-3
    adv.add_dirichlet_bc(0.0, "Upper")
    adv.add_dirichlet_bc(1.0, "Lower")

    adv.solve(timestep=1.0e-3, zero_init_guess=False)   # the walls take their data here
    steps = []
    for _ in range(6):
        density = uw.meshing.metric_density_from_gradient(
            mesh, T, refinement=5, coarsening="auto", metric_choice="front-following")
        uw.meshing.node_redistribution(
            mesh, density, slip_surfaces=True, skip_threshold=None,
            method_kwargs=dict(step_frac=0.2, accel="cg", momentum=0.0))
        dt = float(adv.estimate_dt())
        steps.append(dt)
        adv.solve(timestep=dt, zero_init_guess=False)
    steps.append(float(adv.estimate_dt()))
    assert steps[-1] > steps[0], steps
