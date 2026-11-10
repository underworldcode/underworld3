r"""Adjoint sensitivities are non-dimensional in a units model (#763, #765).

Linear Stokes is an exact oracle. With the viscosity :math:`\alpha\eta` constant and
the body force independent of it, the discrete velocity scales as
:math:`1/(\alpha\eta)`, so for any misfit :math:`J` linear in :math:`u`

.. math::

    \frac{\partial J}{\partial \alpha} = -\frac{J}{\alpha}.

The reference quantities are chosen so that no scale can cancel by accident: the
length reference is 30 km, not 1, and the viscosity reference (1e21 Pa s) differs
from the viscosity (1e22 Pa s). Before the fix the sensitivity came back 1e21 times
too large, the viscosity reference (#763). A dimensional value anywhere in the
residual, the viscosity here or the default ``yield_stress_min`` of ``0 Pa``,
stopped every gradient with ``SympifyError`` first (#765).
"""
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _sheared_box(viscosity, flow_model=uw.constitutive_models.ViscousFlowModel, **parameters):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1 / 8)
    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = flow_model
    stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity
    for name, value in parameters.items():
        setattr(stokes.constitutive_model.Parameters, name, value)
    stokes.bodyforce = sympy.Matrix([0, -sympy.sin(sympy.pi * x)])
    for wall in ("Bottom", "Top", "Left", "Right"):
        stokes.add_dirichlet_bc((0.0, 0.0), wall)
    stokes.tolerance = 1.0e-10
    stokes.solve()
    misfit = stokes.Unknowns.u.sym[1] * sympy.sin(sympy.pi * x)
    return stokes, misfit


def test_sensitivity_to_a_dimensionless_knob_on_a_dimensional_viscosity():
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(30, "km"), viscosity=uw.quantity(1e21, "Pa*s"),
        time=uw.quantity(1, "Myr"))
    alpha = uw.expression(r"\alpha", 2.0, "dimensionless knob")
    eta = uw.expression(r"\eta_0", uw.quantity(1e22, "Pa*s"), "viscosity")
    stokes, misfit = _sheared_box(alpha * eta)

    g = stokes.gradient(misfit, parameters=[alpha])

    assert g["parameters"][alpha] == pytest.approx(-g["J"] / 2.0, rel=1e-6)


def test_stock_viscoplastic_model_without_units_has_a_gradient():
    # yield_stress_min defaults to 0 Pa, a dimensional value, in a model with no
    # reference quantities. The yield stress is far above the flow's stress, so the
    # viscosity is alpha and the closed form holds.
    alpha = uw.expression(r"\alpha", 2.0, "viscosity")
    stokes, misfit = _sheared_box(alpha, uw.constitutive_models.ViscoPlasticFlowModel,
                                  yield_stress=1.0e6)

    g = stokes.gradient(misfit, parameters=[alpha])

    assert g["parameters"][alpha] == pytest.approx(-g["J"] / 2.0, rel=1e-6)
