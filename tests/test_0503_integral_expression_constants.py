"""A `uw.function.expression` inside an integrand must reach the integral kernel.

The JIT routes every expression constant to PETSc's constants array (so that a
changed value does not recompile). The integral classes compiled through that
path but never set the values on the DS they integrate with, so the kernel read
zeros: any integrand carrying a viscosity, a time, or any other expression
integrated to nothing, and a fresh Integral returned the same zero from the
cache. Found on the cylinder drag (the viscous traction vanished), 2026-09-05.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(scope="module")
def setup():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25, regular=True, qdegree=3)
    x, y = mesh.X
    T = uw.discretisation.MeshVariable("T_c", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(y ** 2, T.coords).reshape(-1)     # dT/dy = 2y
    return mesh, x, y, T


def test_volume_integral_carries_the_expression_value(setup):
    mesh, x, y, T = setup
    c = uw.function.expression(r"c_{v}", 2.0, "probe constant")
    integral = uw.maths.Integral(mesh, c * T.sym[0].diff(y))               # 2 * int 2y = 2
    assert np.isclose(integral.evaluate(), 2.0, rtol=1e-8)
    c.sym = 3.0                                                            # a changed value, no recompile
    assert np.isclose(integral.evaluate(), 3.0, rtol=1e-8)
    assert np.isclose(uw.maths.Integral(mesh, c * T.sym[0].diff(y)).evaluate(), 3.0, rtol=1e-8)


def test_boundary_integral_carries_the_expression_value(setup):
    mesh, x, y, T = setup
    c = uw.function.expression(r"c_{b}", 2.0, "probe constant")
    integral = uw.maths.BdIntegral(mesh, c * T.sym[0].diff(y), "Top")       # 2 * 2 * length 1
    assert np.isclose(integral.evaluate(), 4.0, rtol=1e-8)
    c.sym = 0.5
    assert np.isclose(integral.evaluate(), 1.0, rtol=1e-8)


def test_cellwise_integral_carries_the_expression_value(setup):
    mesh, x, y, T = setup
    c = uw.function.expression(r"c_{c}", 2.0, "probe constant")
    cells = uw.maths.CellWiseIntegral(mesh, c * T.sym[0].diff(y)).evaluate()
    assert np.isclose(np.asarray(cells).sum(), 2.0, rtol=1e-8)


def test_constitutive_flux_in_a_boundary_integral(setup):
    """The case that found it: the viscous traction on a wall."""
    mesh, x, y, T = setup
    v = uw.discretisation.MeshVariable("U_c", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P_c", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, v, p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 2.0
    v.array[:, 0, :] = uw.function.evaluate(sympy.Matrix([[y ** 2, 0.0]]), v.coords).reshape(-1, 2)
    sigma_xy = stokes.constitutive_model.flux[0, 1]                        # 2 eta (du/dy)/2 = 2y * 2 / ... = eta * 2y
    assert np.isclose(uw.maths.BdIntegral(mesh, sigma_xy, "Top").evaluate(), 4.0, rtol=1e-8)


def test_a_constant_created_at_zero_still_reaches_the_kernel(setup):
    """#696: a runtime constant whose value is zero at construction must not be
    folded away by sympy (exp(c) with c.is_zero became 1 at construction, so a
    ramp that started at t = 0 stayed frozen); setting it later must change the
    value."""
    mesh, x, y, T = setup
    c0 = uw.function.expression(r"c_{0}", 0.0, "starts at zero")
    integrand = sympy.exp(c0) * T.sym[0].diff(y)
    assert c0 in integrand.atoms(sympy.Symbol), "sympy folded exp(c) at construction"
    integral = uw.maths.Integral(mesh, integrand)
    assert abs(float(integral.evaluate()) - 1.0) < 1e-10      # int dT/dy = 1 on this box
    c0.sym = 1.0
    assert abs(float(uw.maths.Integral(mesh, integrand).evaluate()) - np.e) < 1e-9
