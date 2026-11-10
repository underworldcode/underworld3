"""An expression's units come from its own atoms, not from other variables (#843).

In a model with reference quantities, a velocity without units is sheared so that
``dvx/dy = 1``. Its derived quantities have no units whatever else exists on the
mesh. Before the fix, units inference accepted any unit-carrying variable whose
function had the same Python type, so ``(dvx/dy)**2`` took the units of an
unrelated pressure field (Pa**2), and its value was rescaled by that field's
reference scale (6.97e18 rather than 1).
"""
import numpy as np
import pytest

# A derivative is evaluated through a projection with smoothing 1e-6 (evaluate's
# default), which leaves a relative error of about 5e-7 on this exact P2 field.
_PROJECTION = 1e-5
import underworld3 as uw
from underworld3.units import get_units

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


@pytest.fixture
def sheared_velocity_beside_a_pressure():
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(30, "km"), viscosity=uw.quantity(1e24, "Pa*s"),
        velocity=uw.quantity(2.5, "mm/year"))
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1 / 8)
    v = uw.discretisation.MeshVariable("v_843", mesh, 2, degree=2)
    v.data[:, 0] = np.asarray(v.coords_nd)[:, 1]
    v.data[:, 1] = 0.0
    uw.discretisation.MeshVariable("p_843", mesh, 1, degree=1, units="Pa")
    return mesh, v


def test_a_square_of_a_unitless_gradient_has_no_units(sheared_velocity_beside_a_pressure):
    mesh, v = sheared_velocity_beside_a_pressure
    g = v.sym[0].diff(mesh.X[1])

    assert get_units(g**2) is None
    value = uw.function.evaluate(g**2, np.array([[0.5, 0.5], [0.25, 0.75]]))
    assert getattr(value, "units", None) is None
    assert np.asarray(value) == pytest.approx(1.0, rel=_PROJECTION)


def test_a_strain_rate_of_a_unitless_velocity_has_no_units(sheared_velocity_beside_a_pressure):
    mesh, v = sheared_velocity_beside_a_pressure
    x, y = mesh.X
    e01 = (v.sym[0].diff(y) + v.sym[1].diff(x)) / 2

    assert get_units(e01) is None
    value = uw.function.evaluate(e01, np.array([[0.5, 0.5]]))
    assert np.asarray(value) == pytest.approx(0.5, rel=_PROJECTION)
