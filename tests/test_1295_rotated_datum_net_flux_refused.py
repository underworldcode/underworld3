"""A wall-normal datum's net flux through a boundary declared enclosed (UNFILED 36).

With the pressure null space declared (``petsc_use_pressure_nullspace``) the user
says the domain is enclosed, so the datum g = u.n must carry no net flux. The
test is the datum's own surface integral against its scale, F/S with
F = ∫ g ds and S = ∫ |g| ds over the walls that carry it:

* above 2e-2 the solve is refused before any iteration (a closed box with
  u.n = 0.1 sin πx on its top: F/S = 1; an annulus whose outer wall was moved
  off the circle and slid along it: 4.9e-2);
* from 1e-3 to 2e-2 it is warned about as a real mass-conservation violation
  (the same moved annulus, half as far: 7e-3);
* below 1e-3 a large projected-out gauge component is attributed to the wall
  discretisation (nodes slid along an unchanged circle: 5e-4, falling as h²),
  where the solve used to warn of a conservation violation;
* the annulus as built (cos 2θ on a circle) is silent.

Without the declared null space nothing is refused, and a domain with an open
wall may carry a net flux.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _stokes(mesh, tag):
    v = uw.discretisation.MeshVariable(f"V1295{tag}", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable(f"P1295{tag}", mesh, 1, degree=1)
    s = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    s.bodyforce = sympy.Matrix([[0, 0]])
    s.tolerance = 1.0e-6
    return s


def _box():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, qdegree=3)


def _annulus_solver(tag, off=0.0, slide=0.0):
    """u.n = 0.1 cos 2θ on Upper, declared enclosed; the mesh moved by ``off``
    (a radial wobble cos 3θ, largest on Upper) and ``slide`` (θ + slide sin 2θ)."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2, qdegree=3)
    if off or slide:
        X = np.array(mesh.X.coords)
        r, th = np.hypot(X[:, 0], X[:, 1]), np.arctan2(X[:, 1], X[:, 0])
        r = r * (1.0 + off * np.cos(3 * th) * (r - 0.5) / 0.5)
        th = th + slide * np.sin(2 * th)
        mesh.deform(np.column_stack([r * np.cos(th), r * np.sin(th)]))
    x, y = mesh.X
    s = _stokes(mesh, tag)
    s.petsc_use_pressure_nullspace = True
    s.add_rotated_freeslip_bc(0.1 * sympy.cos(2 * sympy.atan2(y, x)), "Upper")
    s.add_rotated_freeslip_bc(0, "Lower")
    return s


def test_a_net_flux_into_an_enclosed_box_is_refused():
    mesh = _box()
    x, _y = mesh.X
    s = _stokes(mesh, "box")
    s.petsc_use_pressure_nullspace = True
    s.add_rotated_freeslip_bc(0.1 * sympy.sin(sympy.pi * x), "Top")
    s.add_rotated_freeslip_bc(0, "Left")
    s.add_rotated_freeslip_bc(0, "Right")
    s.add_dirichlet_bc((0.0, 0.0), "Bottom")
    with pytest.raises(RuntimeError, match="net flux"):
        s.solve()


def test_a_wall_moved_far_enough_to_carry_a_large_flux_is_refused():
    s = _annulus_solver("far", off=0.2, slide=0.2)          # F/S = 4.9e-2
    with pytest.raises(RuntimeError, match="net flux"):
        s.solve()


def test_a_moved_wall_with_a_small_real_flux_is_warned_about(capsys):
    s = _annulus_solver("near", off=0.1, slide=0.1)         # F/S = 7e-3
    capsys.readouterr()
    s.solve()
    out = capsys.readouterr().out
    assert "violates mass conservation by that amount" in out, out
    assert "wall discretisation" not in out, out


def test_nodes_slid_along_the_circle_are_put_down_to_discretisation(capsys):
    s = _annulus_solver("slid", slide=0.1)                  # F/S = 5e-4
    capsys.readouterr()
    s.solve()
    out = capsys.readouterr().out
    assert s._rotated_freeslip_info["converged"]
    assert "wall discretisation" in out, out
    assert "violates mass conservation" not in out, out


def test_the_annulus_as_built_is_silent(capsys):
    s = _annulus_solver("built")
    capsys.readouterr()
    s.solve()
    out = capsys.readouterr().out
    assert s._rotated_freeslip_info["converged"]
    assert "net flux" not in out and "gauge component" not in out, out


def test_the_same_box_with_an_open_wall_is_accepted():
    mesh = _box()
    x, _y = mesh.X
    s = _stokes(mesh, "open")
    s.add_rotated_freeslip_bc(0.1 * sympy.sin(sympy.pi * x), "Top")
    s.add_rotated_freeslip_bc(0, "Left")
    s.add_dirichlet_bc((0.0, 0.0), "Bottom")
    # Right is natural (traction free): open, so the flux leaves there
    s.solve()
    assert s._rotated_freeslip_info["converged"]
