"""The zero entries of the spherical and geographic frames are exactly zero.

The east (azimuthal) direction of a spherical or geographic frame has no z
component. ``coordinates.py`` used to write that zero as ``independent_of_N``,
the expression ``vanishing * x * y * z`` with ``vanishing`` = 1e-18, so that a
matrix mixing constant and coordinate-dependent entries would lambdify to one
array. The frames are also used to build solver forms, so every form built from
the east direction (a rotated frame, a tangential body force) carried
1e-18 * x * y * z in model units: nothing in a model whose values are of order
one, an error of order one in a model whose model-unit values are of order 1e-18.

The baseline is exact: each zero entry is ``sympy.Integer(0)``, and neither the
frames nor a Stokes form built from the east direction contain the term.
"""
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _shell():
    return uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5, cellSize=0.5)


def _geographic():
    return uw.meshing.RegionalGeographicBox(numElements=(2, 2, 2))


def _frames(mesh):
    cs = mesh.CoordinateSystem
    frames = {"rRotN": cs._rRotN, "unit_e_2": cs.unit_e_2}
    if getattr(cs, "_rRotN_sym", None) is not None:
        frames["rRotN_sym"] = cs._rRotN_sym
    for name in ("_geoRotN", "_unit_WE", "_unit_SN"):
        if getattr(cs, name, None) is not None:
            frames[name] = getattr(cs, name)
    return cs, frames


@pytest.mark.parametrize("build", [_shell, _geographic], ids=["shell", "geographic"])
def test_the_frames_carry_no_vanishing_term(build):
    cs, frames = _frames(build())
    for name, frame in frames.items():
        assert not frame.has(cs.independent_of_N), name
        assert not frame.has(uw.maths.functions.vanishing), name


@pytest.mark.parametrize("build", [_shell, _geographic], ids=["shell", "geographic"])
def test_the_east_direction_has_an_exact_zero_z_component(build):
    cs, _frames_ = _frames(build())
    assert cs._rRotN[2, 2] == sympy.Integer(0)
    assert cs.unit_e_2[0, 2] == sympy.Integer(0)


def test_a_stokes_form_built_from_the_east_direction_carries_no_vanishing_term():
    mesh = _shell()
    cs = mesh.CoordinateSystem
    v = uw.discretisation.MeshVariable("V1296", mesh, 3, degree=2)
    p = uw.discretisation.MeshVariable("P1296", mesh, 1, degree=1)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.bodyforce = cs.unit_e_2.T
    form = uw.function.expressions.unwrap(stokes.F0.sym)
    assert not form.has(cs.independent_of_N)
    assert not form.has(uw.maths.functions.vanishing)
