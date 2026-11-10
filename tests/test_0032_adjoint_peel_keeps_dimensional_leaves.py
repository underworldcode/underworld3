r"""A dimensional value in the residual does not stop an adjoint gradient (#765),
and a misfit written as a named expression is expanded.

The peels that expand named expressions for the chain rule kept a named
expression as a leaf only when its value was a bare number. A dimensional value
is a Pint quantity, not a SymPy object, so substituting it raised
``SympifyError``. That took out every gradient of a residual holding one, whatever
the parameter, including a stock ``ViscoPlasticFlowModel``, whose
``yield_stress_min`` defaults to ``0 Pa`` even in a model with no reference
quantities.

Linear Stokes gives the oracle. With the viscosity constant and the body force
independent of it, the discrete velocity scales as :math:`1/\eta`, so for a misfit
:math:`J` linear in :math:`u`, :math:`\partial J/\partial\alpha = -J/\alpha` when
:math:`\eta = \alpha\,c`.
"""
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _gradient(viscosity, alpha, flow_model=uw.constitutive_models.ViscousFlowModel,
              misfit=None, **parameters):
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
    density = stokes.Unknowns.u.sym[1] * sympy.sin(sympy.pi * x)
    return stokes.gradient(misfit(density) if misfit else density, parameters=[alpha])


def test_stock_viscoplastic_model_without_units_has_a_gradient():
    # The yield stress is far above the flow's stress, so the viscosity is alpha.
    alpha = uw.expression(r"\alpha", 2.0, "viscosity")
    g = _gradient(alpha, alpha, uw.constitutive_models.ViscoPlasticFlowModel,
                  yield_stress=1.0e6)

    assert g["parameters"][alpha] == pytest.approx(-g["J"] / 2.0, rel=1e-6)


def test_a_dimensional_factor_beside_the_parameter_stays_a_leaf():
    # No reference quantities: the dimensional factor is a constant that the chain
    # rule must carry, not substitute.
    alpha = uw.expression(r"\alpha", 2.0, "dimensionless knob")
    c = uw.expression(r"c_0", uw.quantity(1.0, "Pa*s"), "dimensional constant")
    g = _gradient(alpha * c, alpha)

    assert g["parameters"][alpha] == pytest.approx(-g["J"] / 2.0, rel=1e-6)


@pytest.mark.parametrize("named", ["bare", "product"])
def test_a_misfit_written_as_a_named_expression_has_its_gradient(named):
    # The peel's search for named expressions missed a bare one, and a product
    # made only of named ones, so the misfit kept no field and dJ/dalpha was 0.
    def misfit(density):
        m = uw.expression(r"m_J", density, "misfit density")
        return m if named == "bare" else m * uw.expression(r"k_J", 1.0, "weight")

    alpha = uw.expression(r"\alpha", 2.0, "viscosity")
    g = _gradient(alpha, alpha, misfit=misfit)

    assert g["parameters"][alpha] == pytest.approx(-g["J"] / 2.0, rel=1e-6)
