r"""Non-dimensionalising an expression (UNFILED 41).

``uw.scaling.non_dimensionalise`` took a ``uw.quantity`` but not a
``uw.expression`` holding one: it looked for ``.unitless``, then multiplied by a
pint Unit, and raised DimensionalityError (km to dimensionless). So
``Annulus(radiusOuter=uw.expression("r_o", uw.quantity(6370, "km"), ...))``
failed at once (tutorial 15). ``uw.non_dimensionalise`` refused a sympy
expression or Matrix built from fields ("Cannot non-dimensionalise object of
type MutableDenseMatrix", tutorial 12's ``temperature / velocity[0]``).

A named expression holding a quantity takes that quantity's path; an
expression or Matrix gives its model-unit form, as the solver reads it
(``uw.unwrap(expr, keep_constants=False)``).
"""
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture
def units_model():
    model = uw.get_default_model()
    model.set_reference_quantities(
        length=uw.quantity(1000, "km"), viscosity=uw.quantity(1e21, "Pa*s"),
        velocity=uw.quantity(5, "cm/yr"), temperature=uw.quantity(1500, "K"))
    yield model
    uw.reset_default_model()


@pytest.mark.parametrize("entry", ["uw", "uw.scaling"])
def test_an_expression_holding_a_quantity_is_its_quantity(units_model, entry):
    nd = {"uw": uw.non_dimensionalise, "uw.scaling": uw.scaling.non_dimensionalise}[entry]
    r_o = uw.expression(r"r_o", uw.quantity(6370, "km"), "outer radius")
    assert float(nd(r_o)) == pytest.approx(float(nd(uw.quantity(6370, "km"))))
    assert float(nd(r_o)) == pytest.approx(6.37)


def test_an_annulus_takes_its_radii_as_expressions(units_model):
    r_o = uw.expression(r"r_o", uw.quantity(6370, "km"), "outer radius")
    r_i = uw.expression(r"r_i", uw.quantity(3480, "km"), "inner radius")
    by_expression = uw.meshing.Annulus(radiusOuter=r_o, radiusInner=r_i, cellSize=uw.quantity(1000, "km"))
    by_quantity = uw.meshing.Annulus(radiusOuter=uw.quantity(6370, "km"),
                                     radiusInner=uw.quantity(3480, "km"),
                                     cellSize=uw.quantity(1000, "km"))
    import numpy as np
    r = lambda m: np.linalg.norm(np.asarray(m._coords), axis=1)
    assert r(by_expression).max() == pytest.approx(r(by_quantity).max())
    assert r(by_expression).min() == pytest.approx(r(by_quantity).min())


def test_an_expression_of_fields_is_its_model_unit_form(units_model):
    """Tutorial 12: temperature / velocity[0]."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    T = uw.discretisation.MeshVariable("T_n", mesh, 1, degree=1, units="K")
    v = uw.discretisation.MeshVariable("v_n", mesh, 2, degree=2, units="cm/yr")
    expression = T / v[0]
    nd = uw.non_dimensionalise(expression)
    assert sympy.Matrix(nd) == sympy.Matrix(uw.unwrap(expression, keep_constants=False))
    assert uw.scaling.non_dimensionalise(expression) == nd


def test_a_symbolic_named_expression_is_its_model_unit_form(units_model):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)
    T = uw.discretisation.MeshVariable("T_s", mesh, 1, degree=1, units="K")
    dT = uw.expression(r"\Delta T", uw.quantity(3000, "K"), "contrast")
    e = uw.expression("theta", T.sym[0] / dT, "normalised")
    assert sympy.simplify(uw.non_dimensionalise(e) - T.sym[0] / 2) == 0


def test_dimensionalise_round_trips_an_expression_value(units_model):
    r_o = uw.expression(r"r_o", uw.quantity(6370, "km"), "outer radius")
    back = uw.dimensionalise(uw.non_dimensionalise(r_o), uw.quantity(1, "km").dimensionality)
    assert float(back.to("km").magnitude) == pytest.approx(6370)
