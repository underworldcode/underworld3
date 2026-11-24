"""``normal=`` of a rotated free-slip wall refuses a field, with the reason.

``normal=`` takes an expression of the mesh coordinates (or a constant), which
the rotation evaluates at each wall node. A mesh variable's ``.sym`` passed the
check for stray symbols, since a field is a function of the coordinates, and
then failed inside ``lambdify`` with "SyntaxError: invalid syntax" (H6, UNFILED
37). It is now refused where the condition is added, with a message saying what
``normal=`` takes.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _stokes():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)
    v = uw.discretisation.MeshVariable("Vn", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("Pn", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    return mesh, s


def test_a_field_given_as_the_normal_is_refused_where_it_is_added():
    mesh, s = _stokes()
    n_field = uw.discretisation.MeshVariable("Nfield", mesh, 2, degree=2)
    with pytest.raises(ValueError, match="expression of the mesh coordinates"):
        s.add_rotated_freeslip_bc(0, "Top", normal=n_field.sym)


def test_an_expression_of_the_coordinates_is_still_taken():
    mesh, s = _stokes()
    x, y = mesh.X
    s.add_rotated_freeslip_bc(0, "Top", normal=sympy.Matrix([[0 * x, 1 + 0 * y]]))
