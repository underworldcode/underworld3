"""Stokes-family solvers refuse a P1 velocity on simplices (ruling 2026-11-15).

Louis, 2026-11-15: "The degree=1 is almost an incomplete specification", so a
Stokes-family solver on a simplex mesh with a velocity of degree 1 is refused,
on every rank, with the advice to use degree=2. A P1 velocity on simplices does
not pair stably with any pressure the solver offers (P1/P0 diverged in
docs/examples' SolNL, #944). Q1 velocities on quads, Darcy velocities and
advection fields are not refused.

Before this, only the one-field path checked degrees (#844), and only for a
mismatch: Stokes(mesh, degree=1) and a P1/P0 pair passed together were
accepted.
"""
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def _simplex():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)


def test_degree_one_with_no_fields_is_refused_on_simplices():
    with pytest.raises(ValueError, match="2026-11-15"):
        uw.systems.Stokes(_simplex(), degree=1)


def test_a_p1_velocity_passed_alone_is_refused_on_simplices():
    mesh = _simplex()
    v = uw.discretisation.MeshVariable("v1291a", mesh, 2, degree=1)
    with pytest.raises(ValueError, match="degree=2"):
        uw.systems.Stokes(mesh, velocityField=v, degree=1)


@pytest.mark.parametrize("p_degree", [0, 1])
def test_a_p1_velocity_passed_with_its_pressure_is_refused_on_simplices(p_degree):
    mesh = _simplex()
    v = uw.discretisation.MeshVariable(f"v1291b{p_degree}", mesh, 2, degree=1)
    p = uw.discretisation.MeshVariable(f"p1291b{p_degree}", mesh, 1, degree=p_degree)
    with pytest.raises(ValueError, match="2026-11-15"):
        uw.systems.Stokes(mesh, velocityField=v, pressureField=p)


def test_navier_stokes_refuses_it_too():
    mesh = _simplex()
    v = uw.discretisation.MeshVariable("v1291ns", mesh, 2, degree=1)
    p = uw.discretisation.MeshVariable("p1291ns", mesh, 1, degree=0)
    with pytest.raises(ValueError, match="2026-11-15"):
        uw.systems.NavierStokes(mesh, v, p, rho=1.0, order=1)


def test_q1_on_quads_and_p2_on_simplices_are_accepted():
    quads = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
    v = uw.discretisation.MeshVariable("v1291q", quads, 2, degree=1)
    p = uw.discretisation.MeshVariable("p1291q", quads, 1, degree=0)
    uw.systems.Stokes(quads, velocityField=v, pressureField=p)
    uw.systems.Stokes(_simplex(), degree=2)
