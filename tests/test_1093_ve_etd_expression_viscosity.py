"""The ETD Maxwell integrator with a spatially varying viscosity.

The exponential (ETD) coefficients alpha = exp(-dt/tau) and phi are formed
from the Maxwell time tau = eta/mu. When the viscosity is a sympy EXPRESSION
of position (a damage band, a temperature-dependent viscosity) rather than a
number, the coefficient update could not take its float, fell back to the
viscous limit (alpha = phi = 0) and the model silently lost its elastic
response: a block under a step traction moved at the steady viscous rate
instead of taking the elastic jump (measured on the box fault study,
2026-10-17: first-step end velocity 7e-3 against 6e-2).

The check: a viscosity written as an expression whose VALUE is uniform must
give the same first step as the constant it equals; and a genuinely varying
viscosity must not collapse onto the viscous limit.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _first_step_end_velocity(eta_expr, dt=0.125, tag="a"):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(3.0, 1.0),
                                              cellSize=0.25, regular=False, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable(f"v_{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"p_{tag}", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
        stokes.Unknowns, order=1, integrator="etd", objective_rate="none")
    stokes.constitutive_model.Parameters.shear_modulus = 1.0
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_expr(x, y)
    stokes.bodyforce = sympy.Matrix([0.0, 0.0])
    stokes.tolerance = 1e-8
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.add_dirichlet_bc((None, 0.0), "Bottom")
    stokes.add_natural_bc(sympy.Matrix([[-0.01, 0.0]]), "Right")    # tension on the free end
    stokes.solve(timestep=dt, zero_init_guess=True)
    pts = np.array([[2.95, 0.5]])
    return float(np.asarray(uw.function.evaluate(v.sym, pts)).reshape(-1, 2)[0, 0])


def test_uniform_expression_matches_the_constant():
    u_const = _first_step_end_velocity(lambda x, y: 1.0, tag="c")
    # numerically uniform, but a sympy expression of position: float() of it fails
    u_expr = _first_step_end_velocity(lambda x, y: 1.0 + 1.0e-12 * x, tag="e")
    assert abs(u_expr - u_const) < 1e-6 * abs(u_const), (u_const, u_expr)


def test_varying_viscosity_keeps_the_elastic_response():
    # a weak band in the middle third: the response must stay near the elastic
    # jump of the constant case (the band is small), not fall to the viscous rate,
    # which is an order of magnitude smaller at this dt (tau = 1 period, dt = 1/8)
    u_const = _first_step_end_velocity(lambda x, y: 1.0, tag="c2")
    u_visc = _first_step_end_velocity(lambda x, y: 1.0, dt=1.0e6, tag="v")       # dt >> tau: viscous limit
    u_band = _first_step_end_velocity(
        lambda x, y: 1.0 - 0.99 * sympy.exp(-((x - 1.5) / 0.1) ** 2) * sympy.exp(-((y - 0.5) / 0.1) ** 2), tag="b")
    assert u_visc < 0.3 * u_const                                 # the two limits are distinct here
    assert u_band > 0.7 * u_const, (u_const, u_visc, u_band)      # the band did not make it viscous
