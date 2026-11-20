"""A wall-normal datum with a net flux into an enclosed domain is refused (UNFILED 36).

With the pressure null space declared (``petsc_use_pressure_nullspace``), the
user says the domain is enclosed: no open or traction wall, so the net normal
flux through the boundary must vanish. A datum with a net flux then has no
solution. The lifted right-hand side carries that flux on the pressure rows
(its sum is -0.2/pi for u.n = 0.1 sin(pi x) on the top of a unit box), and the
solve converged in the pressure-gauge quotient space and only warned afterwards.
It is now refused at the lift, before any iteration, naming the net flux.

Without the declared null space nothing is refused, and a domain with an open
wall may carry a net flux.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _stokes(mesh):
    v = uw.discretisation.MeshVariable("V1295", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("P1295", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[0, 0]])
    s.tolerance = 1.0e-8
    return s


def _box():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=3)


def test_a_net_flux_into_an_enclosed_box_is_refused():
    mesh = _box()
    x, _y = mesh.X
    s = _stokes(mesh)
    s.petsc_use_pressure_nullspace = True
    s.add_rotated_freeslip_bc(0.1 * sympy.sin(sympy.pi * x), "Top")
    s.add_rotated_freeslip_bc(0, "Left")
    s.add_rotated_freeslip_bc(0, "Right")
    s.add_dirichlet_bc((0.0, 0.0), "Bottom")
    with pytest.raises(RuntimeError, match="net flux"):
        s.solve()


def test_a_flux_free_datum_on_the_annulus_is_accepted():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
    _r, th = mesh.CoordinateSystem.xR
    s = _stokes(mesh)
    s.petsc_use_pressure_nullspace = True
    s.add_rotated_freeslip_bc(0.1 * sympy.cos(2 * th), "Upper")
    s.add_rotated_freeslip_bc(0, "Lower")
    s.solve()
    assert s._rotated_freeslip_info["converged"]


def test_the_same_box_with_an_open_wall_is_accepted():
    mesh = _box()
    x, _y = mesh.X
    s = _stokes(mesh)
    s.add_rotated_freeslip_bc(0.1 * sympy.sin(sympy.pi * x), "Top")
    s.add_rotated_freeslip_bc(0, "Left")
    s.add_dirichlet_bc((0.0, 0.0), "Bottom")
    # Right is natural (traction free): open, so the flux leaves there
    s.solve()
    assert s._rotated_freeslip_info["converged"]
