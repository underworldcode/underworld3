r"""A dimension the model has no reference for, beside one it has (UNFILED item 17).

The model's scale for a dimensionality was built from its fundamental scales,
and a fundamental dimension without one (amount of substance, or a temperature
with no temperature reference) raised. Every caller that caught the error fell
back to a raw number:

- ``UWQuantity.data`` returned the value in the quantity's own units, so an
  activation energy of 3e5 J/mol gave 3e5 and a gas constant 8.314: their
  ratio, 36084 K, never met the temperature reference. With 1000 K, E/(R T)
  was 1000 times too large, and exp(E/(R T) - E/(R T0)) overflowed to NaN.
- The same quantity written as 300 kJ/mol gave 300.

A fundamental dimension with no reference now has the SI base unit as its
scale (one mole, one kelvin), the scale a field of that dimension already runs
at. Mole cancels in every ratio that has a meaning, and a temperature with no
reference stays in kelvin.

Every value is checked against the hand computation, through evaluate (the
lambdified path) and an Integral (the JIT's constants).
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

_E, _R, _T, _T0 = 3.0e5, 8.314, 1400.0, 1600.0
_ARRHENIUS = np.exp(_E / _R * (1 / _T - 1 / _T0))          # 25.0722...


@pytest.fixture
def thermal_model():
    model = uw.get_default_model()
    model.set_reference_quantities(
        length=uw.quantity(30, "km"), viscosity=uw.quantity(1e24, "Pa*s"),
        velocity=uw.quantity(2.5, "mm/year"), temperature=uw.quantity(1000, "K"))
    yield model
    uw.reset_default_model()


def _constants():
    E = uw.expression("E_a", uw.quantity(_E, "J/mol"), "activation energy")
    R = uw.expression("R_g", uw.quantity(_R, "J/(mol*K)"), "gas constant")
    T0 = uw.expression("T_r", uw.quantity(_T0, "K"), "reference temperature")
    return E, R, T0


def _temperature(mesh):
    T = uw.discretisation.MeshVariable("T_u", mesh, 1, degree=1, units="K")
    T.array[...] = uw.quantity(_T, "K")
    return T.sym[0]


def _box():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)


def test_an_activation_over_the_gas_constant_meets_the_temperature_reference(thermal_model):
    """E and R each carry a mole; their model values divide to E/R in units of
    the temperature reference (1000 K), not in kelvin."""
    E, R, _T0 = _constants()
    assert float(E.data) / float(R.data) == pytest.approx(_E / _R / 1000.0)
    assert float(uw.quantity(_E / _R, "K").data) == pytest.approx(_E / _R / 1000.0)


def test_the_same_energy_in_other_units_has_the_same_model_value(thermal_model):
    a = uw.quantity(_E, "J/mol")
    b = uw.quantity(_E / 1000.0, "kJ/mol")
    assert float(b.data) == pytest.approx(float(a.data))
    assert float(uw.non_dimensionalise(b)) == pytest.approx(float(uw.non_dimensionalise(a)))


@pytest.mark.parametrize("form", ["sum", "product", "named ratio"])
def test_the_normalised_arrhenius_factor_evaluates_to_its_value(thermal_model, form):
    mesh = _box()
    T = _temperature(mesh)
    E, R, T0 = _constants()
    A = uw.expression("A_n", E / (R * T0), "E/(R T0)")
    fn = {"sum": sympy.exp(E / (R * T) - E / (R * T0)),
          "product": sympy.exp(E * (1 / T - 1 / T0) / R),
          "named ratio": sympy.exp(A * T0 / T) * sympy.exp(-A)}[form]
    r = uw.function.evaluate(fn, np.array([[0.5, 0.5]]))
    assert float(np.asarray(r).ravel()[0]) == pytest.approx(_ARRHENIUS, rel=1e-6)


def test_the_arrhenius_factor_integrates_to_its_value(thermal_model):
    """The JIT reads the constants as model values: one unit area of model."""
    mesh = _box()
    T = _temperature(mesh)
    E, R, T0 = _constants()
    value = uw.maths.Integral(mesh, sympy.exp(E * (1 / T - 1 / T0) / R)).evaluate()
    assert float(np.asarray(getattr(value, "magnitude", value))) == pytest.approx(_ARRHENIUS, rel=1e-6)


def test_without_a_temperature_reference_kelvin_is_the_scale():
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(30, "km"), viscosity=uw.quantity(1e24, "Pa*s"),
        velocity=uw.quantity(2.5, "mm/year"))
    try:
        assert float(uw.non_dimensionalise(uw.quantity(_T0, "K"))) == pytest.approx(_T0)
        _E_, _R_, T0 = _constants()
        assert float(T0.data) == pytest.approx(_T0)
    finally:
        uw.reset_default_model()


def test_an_energy_per_mole_round_trips(thermal_model):
    """dimensionalise used the same scale, and refused where it raised."""
    q = uw.quantity(_E, "J/mol")
    back = uw.dimensionalise(uw.non_dimensionalise(q), q.dimensionality)
    assert float(back.to("J/mol").magnitude) == pytest.approx(_E)
