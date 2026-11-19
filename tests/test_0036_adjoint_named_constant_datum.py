r"""An essential condition whose value is one named constant (``V_w``).

``unwrap_expression`` returns a Python float for a bare constant expression in
model units (``V_w`` with value 5 mm/yr gives 0.5), and the adjoint's scan of
the boundary reads called ``.atoms`` on that float: ``gradient()`` raised
``AttributeError: 'float' object has no attribute 'atoms'``. ``-V_w`` came back
as a sympy ``Float`` and passed, so the side walls of a shortening box failed on
the left and not on the right. Found converting the fault_friction example to
explicit quantities.

The gradient is checked against central differences of the solve, with
reference quantities and without.
"""
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _box(references):
    if references:
        uw.get_default_model().set_reference_quantities(
            length=uw.quantity(10, "km"), viscosity=uw.quantity(1e21, "Pa*s"),
            velocity=uw.quantity(10, "mm/yr"))
        wall = uw.expression(r"V_w", uw.quantity(5, "mm/yr"), "wall speed")
        eta = uw.expression(r"\eta_0", uw.quantity(1e21, "Pa*s"), "viscosity")
        L = uw.expression("L", uw.quantity(10, "km"), "box width")
    else:
        wall = uw.expression(r"V_w", 0.5, "wall speed")
        eta = uw.expression(r"\eta_0", 1.0, "viscosity")
        L = 1
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1 / 8)
    x, y = mesh.X
    alpha = uw.expression(r"\alpha", 2.0, "dimensionless knob")
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta * (1 + alpha * x / L)     # varies across the walls: the flow is not pure shear
    stokes.add_dirichlet_bc((wall, None), "Left")          # exactly the named constant
    stokes.add_dirichlet_bc((-wall, None), "Right")
    stokes.add_dirichlet_bc((None, 0.0), "Bottom")
    stokes.tolerance = 1.0e-11
    stokes.solve()
    u = stokes.Unknowns.u
    misfit = u.sym[1] ** 2
    return stokes, misfit, alpha


@pytest.mark.parametrize("references", [False, True], ids=["no references", "references"])
def test_a_datum_that_is_one_named_constant_gives_the_gradient(references):
    stokes, misfit, alpha = _box(references)
    adjoint = float(stokes.gradient(misfit, parameters=[alpha])["parameters"][alpha])
    a0 = float(alpha.sym)
    h = 1e-4 * a0
    values = []
    for a in (a0 + h, a0 - h):
        alpha.sym = a
        stokes.solve(zero_init_guess=True)
        values.append(float(stokes.gradient(misfit, parameters=[])["J"]))
    alpha.sym = a0
    fd = (values[0] - values[1]) / (2 * h)
    assert abs(fd) > 1e-8
    assert adjoint == pytest.approx(fd, rel=1e-5)
