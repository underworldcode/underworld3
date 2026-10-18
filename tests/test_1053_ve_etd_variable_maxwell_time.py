"""The ETD Maxwell integrator with a viscosity that varies in space.

The exponential (ETD) coefficients alpha = exp(-dt/tau) and phi = (1 - alpha)/x
come from the Maxwell time tau = eta/mu. A viscosity written as an expression of
position (a damage band, a temperature-dependent law) has no float, so the
coefficients are formed symbolically and evaluated pointwise; reading such a
viscosity as the viscous limit (alpha = phi = 0) would remove the elastic response
everywhere.

Checked here against the constant the expression equals (a numerically uniform
expression, 1 + 1e-12 x, has no float but the same value): the first step at
order 1 and 2; a run whose step changes, since the step must reach the kernels as
a constant and not as a number frozen into them; the transversely isotropic model.
And a weak band is more compliant than the uniform block but far from the viscous
limit of the same banded model.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

UNIFORM = lambda x, y: 1.0 + 1.0e-12 * x


@pytest.fixture(autouse=True)
def _reset():
    uw.reset_default_model()
    yield
    uw.reset_default_model()


def _end_velocities(eta_expr, dts, order=1, model="isotropic"):
    """A block clamped on the left under a tension on its right end: the end
    velocity after each step of ``dts``."""
    uw.reset_default_model()
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(3.0, 1.0),
                                              cellSize=0.25, regular=False, qdegree=3)
    x, y = mesh.X
    v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    if model == "isotropic":
        stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
            stokes.Unknowns, order=order, integrator="etd", objective_rate="none")
        stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_expr(x, y)
    else:
        stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
            stokes.Unknowns, order=order, integrator="etd")
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
        stokes.constitutive_model.Parameters.shear_viscosity_1 = eta_expr(x, y)
        stokes.constitutive_model.Parameters.director = sympy.Matrix([0, 1])
    stokes.constitutive_model.Parameters.shear_modulus = 1.0
    stokes.bodyforce = sympy.Matrix([0.0, 0.0])
    stokes.tolerance = 1e-10
    stokes.add_dirichlet_bc((0.0, 0.0), "Left")
    stokes.add_dirichlet_bc((None, 0.0), "Bottom")
    stokes.add_natural_bc(sympy.Matrix([[-0.01, 0.0]]), "Right")
    out = []
    for k, dt in enumerate(dts):
        stokes.solve(timestep=dt, zero_init_guess=(k == 0))
        out.append(float(np.asarray(uw.function.evaluate(v.sym, np.array([[2.95, 0.5]]))).reshape(-1, 2)[0, 0]))
    return np.array(out)


def _close(a, b):
    return np.abs(a - b).max() <= 1e-10 * np.abs(b).max()


@pytest.mark.parametrize("order", [1, 2])
def test_a_uniform_expression_is_the_constant(order):
    dts = [0.125, 0.125, 0.125]
    const = _end_velocities(lambda x, y: 1.0, dts, order)
    expr = _end_velocities(UNIFORM, dts, order)
    assert _close(expr, const), (const, expr)


def test_a_change_of_step_reaches_the_kernels():
    dts = [0.125, 0.5, 0.05, 1.0e6]
    const = _end_velocities(lambda x, y: 1.0, dts)
    expr = _end_velocities(UNIFORM, dts)
    assert _close(expr, const), (const, expr)


def test_the_transversely_isotropic_model():
    dts = [0.125, 0.5]
    const = _end_velocities(lambda x, y: 1.0, dts, model="ti")
    expr = _end_velocities(UNIFORM, dts, model="ti")
    assert _close(expr, const), (const, expr)


def test_a_weak_band_keeps_the_elastic_response():
    band = lambda x, y: 1.0 - 0.9 * sympy.exp(-((x - 1.5) / 0.1) ** 2)
    u_const = _end_velocities(lambda x, y: 1.0, [0.125])[0]
    u_band = _end_velocities(band, [0.125])[0]
    u_band_viscous = _end_velocities(band, [1.0e6])[0]
    assert u_band > u_const                      # a weak band only adds compliance
    assert u_band > 3.0 * u_band_viscous, (u_const, u_band, u_band_viscous)


@pytest.mark.parametrize("model", ["isotropic", "ti"])
def test_dimensional_parameters_give_the_nondimensional_maxwell_time(model):
    """eta, mu and dt stated with units: the coefficients come from their scaled
    values (dt/tau = 0.1 here), not their magnitudes (1e21 Pa s / 3.2e7 Pa is a
    Maxwell time in seconds against a step of 100 kyr)."""
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(1.0, "km"), viscosity=uw.quantity(1.0e21, "Pa*s"),
        time=uw.quantity(1.0, "Myr"))
    mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, units="km/Myr")
    p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, units="Pa")
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    if model == "isotropic":
        stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel(
            stokes.Unknowns, order=1, integrator="etd")
        stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1.0e21, "Pa*s")
    else:
        stokes.constitutive_model = uw.constitutive_models.TransverseIsotropicVEPFlowModel(
            stokes.Unknowns, order=1, integrator="etd")
        stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1.0e21, "Pa*s")
        stokes.constitutive_model.Parameters.shear_viscosity_1 = uw.quantity(1.0e21, "Pa*s")
        stokes.constitutive_model.Parameters.director = sympy.Matrix([0, 1])
    stokes.constitutive_model.Parameters.shear_modulus = uw.quantity(1.0e21 / 3.15576e13, "Pa")
    stokes.constitutive_model.Parameters.dt_elastic = uw.quantity(100.0, "kyr")
    stokes.constitutive_model._update_history_coefficients()
    alpha = float(stokes.DFDt._exp_alpha.sym)
    assert abs(alpha - np.exp(-0.1)) < 1.0e-6, alpha
