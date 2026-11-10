r"""Adjoint gradients in a model with reference quantities (#763, #1048).

The adjoint state, the residual templates and the duals live in the frame PETSc
assembles in. Two sites read a number back through the dimensional gateway
instead, and so came back scaled by reference quantities:

* ``J``, the volume and facet sensitivities and the explicit term read
  ``Integral.evaluate()``, which re-dimensionalises. A dimensional viscosity
  :math:`\alpha\eta` gave :math:`\partial J/\partial\alpha` 1e21 times too large,
  the viscosity reference (#763).
* The Dirichlet-datum term read the datum's derivative through
  ``uw.function.evaluate``, re-dimensionalised to m²: 9e8 times too large on a
  30 km box (#1048).

Every assertion is absolute: against central differences of the solve, which
sees no adjoint machinery, and ``J`` against the same non-dimensional problem
solved with no reference quantities. A factor common to ``J`` and its gradient
cannot pass either. The length reference is 30 km, not 1, and the viscosity
reference (1e21 Pa s) differs from the viscosity (1e22 Pa s), so no scale cancels
by accident.
"""
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

# Central differences with a relative step of 1e-4 on a solve converged to 1e-11.
_FD = 1e-6


def _box(case, references):
    """A lid-driven or buoyant Stokes box in which ``alpha`` enters as ``case``
    says. Returns the solver, its misfit and ``alpha``."""
    if references:
        uw.get_default_model().set_reference_quantities(
            length=uw.quantity(30, "km"), viscosity=uw.quantity(1e21, "Pa*s"),
            time=uw.quantity(1, "Myr"))
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1 / 8)
    x, y = mesh.X
    alpha = uw.expression(r"\alpha", 2.0, "dimensionless knob")
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    if case == "dimensional viscosity":
        eta = uw.expression(r"\eta_0", uw.quantity(1e22, "Pa*s"), "viscosity")
        stokes.constitutive_model.Parameters.shear_viscosity_0 = alpha * eta
    else:
        stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.bodyforce = sympy.Matrix([0, -sympy.sin(sympy.pi * x)])
    lid = alpha * x * (1 - x) if case == "datum" else 0.0
    stokes.add_dirichlet_bc((lid, None if case == "traction" else 0.0), "Top")
    for wall in ("Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    if case == "traction":
        stokes.add_natural_bc((0.0, alpha * sympy.sin(sympy.pi * x)), "Top")
    stokes.tolerance = 1.0e-11
    stokes.solve()
    u = stokes.Unknowns.u
    misfit = u.sym[1] * sympy.sin(sympy.pi * x) + u.sym[0] * y
    return stokes, misfit, alpha


def _central_difference(stokes, misfit, alpha):
    a0 = float(alpha.sym)
    h = 1e-4 * a0
    values = []
    for a in (a0 + h, a0 - h):
        alpha.sym = a
        stokes.solve(zero_init_guess=True)
        values.append(float(stokes.gradient(misfit, parameters=[])["J"]))
    alpha.sym = a0
    return (values[0] - values[1]) / (2 * h)


@pytest.mark.parametrize("case", ["dimensional viscosity", "datum", "traction"])
def test_the_gradient_matches_central_differences(case):
    stokes, misfit, alpha = _box(case, references=True)

    g = stokes.gradient(misfit, parameters=[alpha])

    assert g["parameters"][alpha] == pytest.approx(
        _central_difference(stokes, misfit, alpha), rel=_FD)


@pytest.mark.parametrize("case", ["datum", "traction"])
def test_J_is_the_non_dimensional_problem_s(case):
    stokes, misfit, alpha = _box(case, references=False)
    plain = stokes.gradient(misfit, parameters=[alpha])
    uw.reset_default_model()

    stokes, misfit, alpha = _box(case, references=True)
    scaled = stokes.gradient(misfit, parameters=[alpha])

    assert scaled["J"] == pytest.approx(plain["J"], rel=1e-10)
    assert scaled["parameters"][alpha] == pytest.approx(plain["parameters"][alpha], rel=1e-10)
