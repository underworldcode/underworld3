r"""A raw pint quantity is taken as a uw.quantity is (UNFILED 35).

``1e-6 * uw.units.m**2 / uw.units.s`` is a pint Quantity, the form the units
tutorial wrote. A constitutive parameter took ``uw.quantity(1e-6, "m**2/s")``
but sent the raw form to ``sympy.sympify(float(q))``, which raised a pint
DimensionalityError naming neither fix. CLAUDE.md's rule is to accept the
forms a user writes and store pint objects, so each setter that takes a
quantity converts a raw one as it does a uw.quantity. Each case is checked
against the uw.quantity form.
"""
import numpy as np
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


def _box():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.5)


def _raw(value, units):
    return value * uw.units(units)


def test_a_parameter_takes_a_raw_quantity(units_model):
    mesh = _box()
    T = uw.discretisation.MeshVariable("T_p", mesh, 1, degree=1, units="K")
    a = uw.systems.Poisson(mesh, u_Field=T)
    a.constitutive_model = uw.constitutive_models.DiffusionModel
    a.constitutive_model.Parameters.diffusivity = _raw(1e-6, "m**2/s")
    b = uw.systems.Poisson(mesh, u_Field=T)
    b.constitutive_model = uw.constitutive_models.DiffusionModel
    b.constitutive_model.Parameters.diffusivity = uw.quantity(1e-6, "m**2/s")
    as_raw = uw.unwrap(a.constitutive_model.Parameters.diffusivity, keep_constants=False)
    as_quantity = uw.unwrap(b.constitutive_model.Parameters.diffusivity, keep_constants=False)
    assert float(as_raw) == pytest.approx(float(as_quantity))


def test_the_body_force_takes_raw_quantities(units_model):
    mesh = _box()
    a = uw.systems.Stokes(mesh)
    a.bodyforce = (0 * uw.units("Pa/m"), _raw(-3e4, "Pa/m"))
    b = uw.systems.Stokes(mesh)
    b.bodyforce = (uw.quantity(0, "Pa/m"), uw.quantity(-3e4, "Pa/m"))
    as_raw = [float(e) for e in sympy.Matrix(uw.unwrap(a.bodyforce, keep_constants=False))]
    as_quantity = [float(e) for e in sympy.Matrix(uw.unwrap(b.bodyforce, keep_constants=False))]
    assert as_raw == pytest.approx(as_quantity)
    assert as_quantity[1] != 0.0


def test_an_expression_takes_a_raw_quantity(units_model):
    a = uw.expression("k_r", _raw(3.0, "W/(m*K)"), "conductivity")
    b = uw.expression("k_q", uw.quantity(3.0, "W/(m*K)"), "conductivity")
    assert float(a.data) == pytest.approx(float(b.data))
    assert (1 * a.units).dimensionality == (1 * b.units).dimensionality


def test_non_dimensionalise_takes_a_raw_quantity(units_model):
    assert float(uw.non_dimensionalise(_raw(6370, "km"))) == pytest.approx(
        float(uw.non_dimensionalise(uw.quantity(6370, "km"))))


def test_a_field_takes_a_raw_quantity_array(units_model):
    mesh = _box()
    a = uw.discretisation.MeshVariable("T_a", mesh, 1, degree=1, units="K")
    b = uw.discretisation.MeshVariable("T_b", mesh, 1, degree=1, units="K")
    n = a.array.shape[0]
    a.array[...] = np.full((n, 1, 1), 1300.0) * uw.units("K")
    b.array[...] = uw.quantity(1300.0, "K")
    assert np.allclose(np.asarray(a.data), np.asarray(b.data))
