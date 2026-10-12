"""Velocity extrapolations hold when the timestep changes.

Two schemes extrapolate the velocity from the levels they hold:

* the semi-Lagrangian trace-back takes its mid-point stage at the mid TIME of
  the step, :math:`v^{n+1/2} = v^n + \\tfrac{r}{2}(v^n - v^{n-1})`;
* the Eulerian Navier-Stokes solver advects with the velocity extrapolated to
  the new level, :math:`\\mathbf{a} = \\mathbf{u}^n + r(\\mathbf{u}^n - \\mathbf{u}^{n-1})`;

with :math:`r = \\Delta t_n / \\Delta t_{n-1}`, the ratio of this step to the
last. Both are linear extrapolation in time, exact for a velocity that varies
linearly in time whatever the steps. Written for equal steps (:math:`r = 1`),
they lose that exactness, and the scheme drops to first order, wherever the
step changes: a step rejected and retaken in halves, a step cut to land on an
output time, an adaptive controller.
"""

import numpy as np
import pytest
import sympy

pytestmark = [pytest.mark.tier_b]


@pytest.fixture(autouse=True)
def _reset():
    import underworld3 as uw

    uw.reset_default_model()
    yield
    uw.reset_default_model()


@pytest.mark.level_1
def test_the_trace_back_mid_time_velocity_is_exact_for_a_linear_history():
    """A velocity that grows linearly in time, a step of 0.1 then one of 0.05:
    the mid-time velocity is the velocity at the middle of the second step."""
    import underworld3 as uw
    from underworld3.systems.ddt import CharacteristicTrace

    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4), minCoords=(0.0, 0.0),
                                        maxCoords=(1.0, 1.0))
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=1)
    trace = CharacteristicTrace(mesh, v)
    speed, rate, dt_previous, dt = 1.0, 2.0, 0.1, 0.05

    trace.ensure_levels(2)                         # one recorded level, as a history asks for
    v.array[:, 0, 0] = speed                       # v at t = 0
    trace.begin_step(dt_previous)
    trace.finish_step()                            # recorded as v^{n-1}
    v.array[:, 0, 0] = speed + rate * dt_previous  # v at t = 0.1

    middle = np.asarray(uw.function.evaluate(trace.midtime_expr(dt)[0, 0], v.coords)).reshape(-1)
    expected = speed + rate * (dt_previous + 0.5 * dt)
    assert np.allclose(middle, expected, rtol=1.0e-12, atol=0.0), (
        f"mid-time velocity {middle.max():.6f}, expected {expected:.6f} "
        "(linear extrapolation to t + dt/2 with the step ratio)")


@pytest.mark.level_2
def test_the_navier_stokes_advecting_velocity_is_the_linear_extrapolation():
    """After a step of 0.02, a step of 0.01: the advecting velocity is the
    linear extrapolation of the last two velocity levels to the new time."""
    import underworld3 as uw

    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(2.0, 1.0),
                                             cellSize=0.2, qdegree=3)
    v = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
    ns = uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1)
    ns.constitutive_model = uw.constitutive_models.ViscousFlowModel
    ns.constitutive_model.Parameters.shear_viscosity_0 = 0.01
    ramp = uw.function.expression(r"r_{\mathrm{in}}", 0.25, "inflow ramp")
    x, y = mesh.X
    ns.add_dirichlet_bc((ramp * 4.0 * y * (1 - y), 0.0), "Left")
    ns.add_dirichlet_bc((0.0, 0.0), "Top")
    ns.add_dirichlet_bc((0.0, 0.0), "Bottom")

    for level, dt in ((0.25, 0.02), (0.5, 0.02)):
        ramp.sym = level
        ns.solve(timestep=dt, zero_init_guess=False)
    u_previous = np.asarray(ns._u_prev.array[...]).copy()
    u_now = np.asarray(v.array[...]).copy()

    ramp.sym = 0.75
    ns.solve(timestep=0.01, zero_init_guess=False)
    advecting = np.asarray(ns._a_var.array[...])
    expected = u_now + 0.5 * (u_now - u_previous)
    scale = np.abs(u_now).max()
    assert np.allclose(advecting, expected, rtol=0.0, atol=1.0e-12 * scale), (
        f"advecting velocity departs from the extrapolation with the step ratio by "
        f"{np.abs(advecting - expected).max():.3e}")


@pytest.mark.level_1
def test_swarm_advection_takes_the_mid_time_velocity_with_the_step_ratio():
    """Through the real path: a swarm advected through a uniform velocity
    that grows linearly in time, with steps of 0.02, 0.01, 0.02 and 0.005.
    Once a velocity level is recorded, each step moves a particle by exactly
    dt times the velocity at the middle of that step."""
    import underworld3 as uw

    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8), minCoords=(0.0, 0.0),
                                        maxCoords=(1.0, 1.0))
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=1)
    swarm = uw.swarm.Swarm(mesh)
    swarm.add_particles_with_coordinates(np.array([[0.3, 0.5]]))
    speed, rate = 1.0, 2.0
    t = 0.0
    errors = []
    for dt in (0.02, 0.01, 0.02, 0.005):
        v.array[:, 0, 0] = speed + rate * t
        x0 = np.asarray(swarm._particle_coordinates.data).copy()
        swarm.advection(v.sym, dt, order=2, midtime_velocity=True)
        x1 = np.asarray(swarm._particle_coordinates.data)
        moved = float(x1[0, 0] - x0[0, 0]) if x1.size else 0.0
        errors.append(abs(moved - dt * (speed + rate * (t + 0.5 * dt))) / dt)
        t += dt
    # the first step has no recorded level and takes v^n alone
    assert max(errors[1:]) < 1.0e-12, (
        f"mid-time displacement error per unit step {errors[1:]} "
        "(expected round-off once a level is recorded)")


@pytest.mark.level_1
def test_a_restore_brings_back_the_step_length_of_a_swarm_trace():
    """A swarm's own mid-time trace records the length of the step its
    velocity level belongs to; a regretted step of another length must not
    leave that length behind."""
    import underworld3 as uw

    mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8), minCoords=(0.0, 0.0),
                                        maxCoords=(1.0, 1.0))
    v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=1)
    swarm = uw.swarm.Swarm(mesh)
    swarm.add_particles_with_coordinates(np.array([[0.3, 0.5]]))
    v.array[:, 0, 0] = 1.0
    orchestration_model = uw.get_default_model()
    for _ in range(2):
        swarm.advection(v.sym, 0.02, order=2, midtime_velocity=True)
    token = orchestration_model.save_state()
    swarm.advection(v.sym, 0.04, order=2, midtime_velocity=True)
    orchestration_model.load_state(token)
    assert swarm._characteristics._recorded_dt == 0.02
