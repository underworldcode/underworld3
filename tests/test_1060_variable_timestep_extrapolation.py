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

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(autouse=True)
def _reset():
    import underworld3 as uw

    uw.reset_default_model()
    yield
    uw.reset_default_model()


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
