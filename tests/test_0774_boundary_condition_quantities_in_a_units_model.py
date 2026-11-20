r"""A boundary condition given as a quantity, in a model with reference quantities (#1238).

``add_condition`` replaced a ``uw.quantity`` or pint Quantity by its SI magnitude
and applied that number in model units. With a 10 mm/yr velocity reference a wall
at 5 mm/yr was given 1.58e-10 where 0.5 is right, a factor of 3e9, with no
warning. The same value as a ``uw.expression`` was right. Essential and natural
conditions share the code, in a tuple and as a single value.

An older path divides a plain number by the field's ``scaling_coefficient``.
For a velocity on a field the solver made (which carries units) that happened
to give the model value, but not when the fields were passed in without units
(the issue's case: scale 1), nor for a scalar field without units, nor for a
traction (divided by the velocity scale). A quantity is now non-dimensionalised
directly and not divided again.

The recorded value is checked against the model value worked out by hand from
the references, and two solves check the wall moves at it. Without reference
quantities the model works in SI, and the SI magnitude is still the value.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

# 10 km, 1e21 Pa s, 10 mm/yr: the stress scale is eta*V/L.
_YEAR = 365.25 * 24 * 3600
_STRESS = 1e21 * (10e-3 / _YEAR) / 10e3


@pytest.fixture
def units_model():
    model = uw.get_default_model()
    model.set_reference_quantities(
        length=uw.quantity(10, "km"), viscosity=uw.quantity(1e21, "Pa*s"),
        velocity=uw.quantity(10, "mm/yr"))
    yield model
    uw.reset_default_model()


def _stokes(fields_without_units=False):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    if fields_without_units:
        v = uw.discretisation.MeshVariable("v_q", mesh, 2, degree=2)
        p = uw.discretisation.MeshVariable("p_q", mesh, 1, degree=1)
        st = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    else:
        st = uw.systems.Stokes(mesh)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = uw.expression(
        r"\eta_b", uw.quantity(1e21, "Pa*s"), "viscosity")
    return st


def _recorded(bc):
    fn = bc.fn if hasattr(bc, "fn") else bc.fn_f
    return [float(v) for v in fn if v != sympy.oo]


@pytest.mark.parametrize("make", [lambda: uw.quantity(5, "mm/yr"),
                                  lambda: 5 * uw.units("mm/yr")],
                         ids=["uw.quantity", "pint"])
def test_an_essential_quantity_is_recorded_in_model_units(units_model, make):
    st = _stokes()
    st.add_essential_bc((make(), None), "Left")
    assert _recorded(st.essential_bcs[-1]) == pytest.approx([0.5])


@pytest.mark.parametrize("fields_without_units", [False, True],
                         ids=["solver's fields", "fields passed without units"])
def test_the_wall_moves_at_the_quantity(units_model, fields_without_units):
    st = _stokes(fields_without_units)
    st.add_essential_bc((uw.quantity(5, "mm/yr"), 0.0), "Left")
    st.add_essential_bc((0.0, 0.0), "Right")
    st.solve(zero_init_guess=True)
    v = st.Unknowns.u
    left = np.isclose(np.asarray(v.coords_nd)[:, 0], 0.0)
    assert v.data[left, 0] == pytest.approx(0.5)


def test_a_natural_quantity_is_recorded_in_model_units(units_model):
    st = _stokes()
    st.add_natural_bc((0.0, uw.quantity(1, "MPa")), "Top")
    assert _recorded(st.natural_bcs[-1]) == pytest.approx([0.0, 1e6 / _STRESS])


def test_a_single_quantity_is_recorded_in_model_units(units_model):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    poisson = uw.systems.Poisson(mesh)
    poisson.add_dirichlet_bc(uw.quantity(5, "km"), "Left")
    assert _recorded(poisson.essential_bcs[-1]) == pytest.approx([0.5])


def test_without_reference_quantities_the_si_magnitude_is_the_value():
    st = _stokes()
    st.add_essential_bc((uw.quantity(5, "mm/yr"), None), "Left")
    assert _recorded(st.essential_bcs[-1]) == pytest.approx([5e-3 / _YEAR])


def test_a_dimension_without_a_reference_is_in_si():
    """No temperature reference: a kelvin field runs at scale 1, in SI, and a
    boundary temperature is its SI magnitude (test_0812's Poisson problem)."""
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(10, "km"), viscosity=uw.quantity(1e21, "Pa*s"),
        velocity=uw.quantity(10, "mm/yr"))
    try:
        mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                                 cellSize=0.25)
        T = uw.discretisation.MeshVariable("T_q", mesh, 1, degree=1, units="K")
        poisson = uw.systems.Poisson(mesh, u_Field=T)
        poisson.add_dirichlet_bc(uw.quantity(300, "K"), "Left")
        assert _recorded(poisson.essential_bcs[-1]) == pytest.approx([300.0])
    finally:
        uw.reset_default_model()


def test_a_quantity_of_the_wrong_dimension_is_refused_on_a_field_with_units(units_model):
    """1 MPa on a velocity wall: refused, naming both dimensions, rather than
    converted to a believable number in the wrong units."""
    st = _stokes()
    with pytest.raises(ValueError, match=r"\[mass\].*\[length\] / \[time\]"):
        st.add_essential_bc((uw.quantity(1, "MPa"), None), "Left")
    st.add_essential_bc((uw.quantity(1, "m/s"), None), "Left")      # the field's dimension


def test_a_pure_number_on_a_field_without_units_is_accepted(units_model):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    poisson = uw.systems.Poisson(mesh)
    poisson.add_dirichlet_bc(0.5, "Left")
    assert _recorded(poisson.essential_bcs[-1]) == pytest.approx([0.5])


@pytest.mark.parametrize("value", [uw.quantity(1300, "K"), uw.quantity(1026.85, "degC"),
                                   uw.quantity(0, "degC")], ids=["K", "degC", "0 degC"])
def test_a_temperature_on_a_celsius_field_is_accepted(units_model, value):
    """An offset unit: the field's dimension is read from the unit itself."""
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    T = uw.discretisation.MeshVariable("T_c", mesh, 1, degree=1, units="degC")
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    poisson.add_dirichlet_bc(value, "Left")


def test_a_length_on_a_celsius_field_is_refused(units_model):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    T = uw.discretisation.MeshVariable("T_c2", mesh, 1, degree=1, units="degC")
    poisson = uw.systems.Poisson(mesh, u_Field=T)
    with pytest.raises(ValueError, match=r"essential \(Dirichlet\) condition on 'Left'"):
        poisson.add_dirichlet_bc(uw.quantity(5, "km"), "Left")
