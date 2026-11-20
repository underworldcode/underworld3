r"""Arithmetic on a variable's derivative (UNFILED 40).

``MeshVariable.diff()`` returns a ``UnitAwareDerivativeMatrix``, which indexes
into unit-aware elements but supported no arithmetic, with or without a units
model: ``s.diff(x) + s.diff(y)`` raised TypeError, which stops tutorial 3
(Symbolic_Forms). ``s.sym.diff(x) + s.sym.diff(y)`` worked.

The result is the plain sympy Matrix of the same arithmetic on the underlying
derivatives; a scalar applies to every entry. Its units are what get_units
reads from the derivatives.
"""
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _mesh_and_field(units=None):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(-1, -1), maxCoords=(1, 1), cellSize=0.25)
    s = uw.discretisation.MeshVariable("s_d", mesh, 1, degree=2, units=units)
    return mesh, s


def test_derivatives_add_subtract_and_scale_as_their_sympy_form():
    mesh, s = _mesh_and_field()
    x, y = mesh.X
    dx, dy = s.sym.diff(x), s.sym.diff(y)
    assert sympy.Matrix(s.diff(x) + s.diff(y)) == dx + dy
    assert sympy.Matrix(s.diff(x) - s.diff(y)) == dx - dy
    assert sympy.Matrix(2 * s.diff(x)) == 2 * dx
    assert sympy.Matrix(s.diff(x) * 2) == 2 * dx
    assert sympy.Matrix(s.diff(x) / 2) == dx / 2
    assert sympy.Matrix(-s.diff(x)) == -dx
    assert sympy.Matrix(s.diff(x) + dy) == dx + dy          # with a plain Matrix
    assert sympy.Matrix(dy + s.diff(x)) == dx + dy


def test_a_scalar_applies_to_every_entry():
    mesh, s = _mesh_and_field()
    x, _y = mesh.X
    dx = s.sym.diff(x)
    assert sympy.Matrix(s.diff(x) + 1) == dx.applyfunc(lambda e: e + 1)
    assert sympy.Matrix(1 - s.diff(x)) == dx.applyfunc(lambda e: 1 - e)


def test_a_sum_of_two_derivatives_carries_their_units():
    uw.get_default_model().set_reference_quantities(
        length=uw.quantity(10, "km"), viscosity=uw.quantity(1e21, "Pa*s"),
        velocity=uw.quantity(1, "cm/yr"), temperature=uw.quantity(1000, "K"))
    try:
        mesh, s = _mesh_and_field(units="K")
        x, y = mesh.X
        total = s.diff(x) + s.diff(y)
        units = uw.get_units(total)
        assert units is not None
        assert (1 * units).dimensionality == (1 * uw.units("K/m")).dimensionality
    finally:
        uw.reset_default_model()
