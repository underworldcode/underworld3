"""The units contract of ``evaluate`` (#771, #904, #881, #786, #844).

Louis's ruling, 2026-11-13:

- ``evaluate`` is the dimensional output gateway: in a model with reference
  quantities it returns dimensional values. It REFUSES an expression that mixes a
  field without units with a leaf that has units, because the field's stored
  number is non-dimensional with no known dimension and the product is
  incoherent (#771, #904). A field without units on its own is a pure number
  and is returned as one.
- Solver fields carry units: ``Stokes(mesh)`` in a units model creates its
  velocity in m/s and its pressure in Pa, so the default path never builds the
  mix. A field passed in is kept, whichever of the two it is (#844).
- With no reference quantities, parameter defaults carry no units (#881).
- A point outside the domain is refused (#786). "Outside" is measured with a
  scale taken from the mesh, so a query on the boundary or a roundoff past it is
  answered as before.

Reference quantities: 30 km, 1e24 Pa s and 2.5 mm/yr, so V/L = 2.64e-15 1/s.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

_PTS = np.array([[0.5, 0.5]])


@pytest.fixture
def units_model():
    model = uw.get_default_model()
    model.set_reference_quantities(
        length=uw.quantity(30, "km"),
        viscosity=uw.quantity(1e24, "Pa*s"),
        velocity=uw.quantity(2.5, "mm/year"),
    )
    yield model
    uw.reset_default_model()


def _box(cell=0.25):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=cell
    )


def _shear(v):
    """An ND unit shear, dvx/dy = 1 in model units."""
    v.data[:, 0] = np.asarray(v.coords_nd)[:, 1]
    v.data[:, 1] = 0.0


def _E01(mesh, v):
    x, y = mesh.X
    return 0.5 * (v.sym[0].diff(y) + v.sym[1].diff(x))


# --- the gateway and the refusal (#771, #904) -------------------------------------


def test_a_unitless_field_times_a_dimensional_constant_is_refused(units_model):
    mesh = _box()
    v = uw.discretisation.MeshVariable("v_nu_760", mesh, 2, degree=2)  # no units
    _shear(v)
    eta = uw.expression(r"\eta_{760}", uw.quantity(1e24, "Pa*s"), "viscosity")
    with pytest.raises(ValueError, match="units") as raised:
        uw.function.evaluate(2 * eta * _E01(mesh, v), _PTS)
    # The message names the unitless field and the dimensional leaf.
    assert "v_nu_760" in str(raised.value)
    assert r"\eta_{760}" in str(raised.value)


def test_the_refusal_also_holds_in_global_evaluate(units_model):
    mesh = _box()
    T = uw.discretisation.MeshVariable("T_nu_760", mesh, 1, degree=1)  # no units
    T.data[:, 0] = 7.0
    eta = uw.expression(r"\eta_{760g}", uw.quantity(1e24, "Pa*s"), "viscosity")
    with pytest.raises(ValueError, match="T_nu_760"):
        uw.function.global_evaluate(T.sym[0] * eta, _PTS)


def test_a_unitless_field_alone_is_a_pure_number(units_model):
    mesh = _box()
    v = uw.discretisation.MeshVariable("v_alone_760", mesh, 2, degree=2)
    _shear(v)
    r = uw.function.evaluate(_E01(mesh, v), _PTS)
    assert getattr(r, "units", None) is None
    assert np.asarray(r).ravel()[0] == pytest.approx(0.5)


def test_with_units_on_every_leaf_the_gateway_is_dimensional(units_model):
    mesh = _box()
    v = uw.discretisation.MeshVariable("v_u_760", mesh, 2, degree=2, units="m/s")
    _shear(v)
    eta = uw.expression(r"\eta_{760u}", uw.quantity(1e24, "Pa*s"), "viscosity")
    scale = uw.quantity(2.5, "mm/year").to("m/s").magnitude / 30e3
    E01 = uw.function.evaluate(_E01(mesh, v), _PTS)
    tau = uw.function.evaluate(2 * eta * _E01(mesh, v), _PTS)
    assert float(np.asarray(E01.to("1/s")).ravel()[0]) == pytest.approx(0.5 * scale, rel=1e-6)
    assert float(np.asarray(tau.to("Pa")).ravel()[0]) == pytest.approx(1e24 * scale, rel=1e-6)


def test_two_dimensions_each_come_back_in_their_own_units(units_model):
    """A single uniform factor cannot pass this: viscosity and density have
    different reference scales."""
    eta = uw.expression(r"\eta_{760d}", uw.quantity(1e23, "Pa*s"), "viscosity")
    rho = uw.expression(r"\rho_{760d}", uw.quantity(2700, "kg/m**3"), "density")
    _box()
    assert float(np.asarray(uw.function.evaluate(eta, _PTS).to("Pa*s")).ravel()[0]) == pytest.approx(1e23)
    assert float(np.asarray(uw.function.evaluate(rho, _PTS).to("kg/m**3")).ravel()[0]) == pytest.approx(2700)


# --- solver fields carry units, and a field passed in is kept (#844) --------------


def test_stokes_creates_its_fields_with_units_in_a_units_model(units_model):
    mesh = _box()
    st = uw.systems.Stokes(mesh)
    assert st.Unknowns.u.units is not None and st.Unknowns.p.units is not None
    assert uw.quantity(1, st.Unknowns.u.units).to("m/s").magnitude > 0
    assert uw.quantity(1, st.Unknowns.p.units).to("Pa").magnitude > 0


def test_stokes_default_stress_evaluates_in_pascal(units_model):
    mesh = _box()
    st = uw.systems.Stokes(mesh)
    st.constitutive_model = uw.constitutive_models.ViscousFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e24, "Pa*s")
    _shear(st.Unknowns.u)
    tau = uw.function.evaluate(st.stress[0, 1], _PTS)
    scale = uw.quantity(2.5, "mm/year").to("m/s").magnitude / 30e3
    assert float(np.asarray(tau.to("Pa")).ravel()[0]) == pytest.approx(1e24 * scale, rel=1e-6)


def test_stokes_without_units_creates_fields_without_units():
    mesh = _box()
    st = uw.systems.Stokes(mesh)
    assert st.Unknowns.u.units is None and st.Unknowns.p.units is None


def test_stokes_keeps_a_velocity_passed_without_a_pressure():
    mesh = _box()
    v = uw.discretisation.MeshVariable("v_844", mesh, 2, degree=2)
    st = uw.systems.Stokes(mesh, velocityField=v)
    assert st.Unknowns.u is v
    assert st.Unknowns.p.degree == 1


def test_stokes_keeps_a_pressure_passed_without_a_velocity():
    mesh = _box()
    p = uw.discretisation.MeshVariable("p_844", mesh, 1, degree=1)
    st = uw.systems.Stokes(mesh, pressureField=p)
    assert st.Unknowns.p is p
    assert st.Unknowns.u.degree == 2


# --- no reference quantities: no units on parameter defaults (#881) ---------------


def test_viscoplastic_stress_is_plain_without_reference_quantities():
    mesh = _box()
    st = uw.systems.Stokes(mesh)
    st.constitutive_model = uw.constitutive_models.ViscoPlasticFlowModel
    st.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    st.constitutive_model.Parameters.yield_stress = 1
    s = uw.function.evaluate(st.stress, _PTS)
    assert getattr(s, "units", None) is None
    assert uw.get_units(st.stress) is None


# --- points outside the domain (#786) ----------------------------------------------


def test_a_point_far_outside_the_mesh_is_refused():
    mesh = _box()
    T = uw.discretisation.MeshVariable("T_786", mesh, 1, degree=1)
    T.data[:, 0] = np.asarray(T.coords)[:, 0]
    with pytest.raises(ValueError, match="outside"):
        uw.function.evaluate(T.sym[0], np.array([[6.0e4, -3.0e4], [0.5, 0.5]]))


def test_a_point_on_or_just_past_the_boundary_is_answered():
    mesh = _box()
    T = uw.discretisation.MeshVariable("T_786b", mesh, 1, degree=1)
    T.data[:, 0] = np.asarray(T.coords)[:, 0]
    pts = np.array([[1.0, 0.5], [1.0 + 1e-9, 0.5], [0.5, -1e-9]])
    vals = np.asarray(uw.function.evaluate(T.sym[0], pts)).ravel()
    assert vals == pytest.approx([1.0, 1.0, 0.5], abs=1e-6)


def test_a_caller_that_asks_for_the_extrapolation_mask_gets_it_instead():
    mesh = _box()
    T = uw.discretisation.MeshVariable("T_786c", mesh, 1, degree=1)
    T.data[:, 0] = 1.0
    _, mask = uw.function.evaluate(
        T.sym[0], np.array([[6.0e4, -3.0e4], [0.5, 0.5]]), check_extrapolated=True
    )
    assert list(np.asarray(mask).ravel()) == [True, False]
