"""Stokes-family solvers refuse a degree-1 velocity on a simplex mesh.

P1/P0 locks and P1/P1 is unstable, so on triangles and tetrahedra a velocity of
degree 1 is "almost an incomplete specification" (Louis, approving this as a
change to solver behaviour). It is refused in the Stokes base class, whether it
comes from ``degree=1`` or from a passed P1 velocity, so Stokes, NavierStokes,
VE_Stokes and Stokes_Constrained are all covered; the message points to
degree=2 (P2/P1). Quadrilaterals keep Q1/P0, the classic Underworld element, and
P2/P0 on simplices is allowed. Decided from the mesh's agreed ``isSimplex``.
"""
import pytest

import underworld3 as uw


def _tri():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.25)


def _quad():
    return uw.meshing.StructuredQuadBox(elementRes=(4, 4))


@pytest.mark.level_1
@pytest.mark.tier_b
@pytest.mark.parametrize("solver", ["Stokes", "NavierStokes", "VE_Stokes", "Stokes_Constrained"])
def test_degree_one_on_simplices_is_refused(solver):
    """Every solver of the family, given a P1 velocity (NavierStokes needs both
    fields passed)."""
    cls = getattr(uw.systems, solver)
    mesh = _tri()
    v = uw.discretisation.MeshVariable(f"v1_{solver}", mesh, 2, degree=1)
    p = uw.discretisation.MeshVariable(f"p0_{solver}", mesh, 1, degree=0, continuous=False)
    with pytest.raises(ValueError, match="degree=2"):
        cls(mesh, velocityField=v, pressureField=p, degree=1)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_degree_one_alone_on_simplices_is_refused():
    with pytest.raises(ValueError, match="degree=2"):
        uw.systems.Stokes(_tri(), degree=1)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_a_passed_p1_velocity_on_simplices_is_refused():
    mesh = _tri()
    v = uw.discretisation.MeshVariable("v1_p1s", mesh, 2, degree=1)
    p = uw.discretisation.MeshVariable("p0_p1s", mesh, 1, degree=0, continuous=False)
    with pytest.raises(ValueError, match="degree=2"):
        uw.systems.Stokes(mesh, velocityField=v, pressureField=p)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_q1_on_quadrilaterals_is_allowed():
    mesh = _quad()
    v = uw.discretisation.MeshVariable("v_q1", mesh, 2, degree=1)
    p = uw.discretisation.MeshVariable("p_q1", mesh, 1, degree=0, continuous=False)
    uw.systems.Stokes(mesh, velocityField=v, pressureField=p)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_p2_p0_on_simplices_is_allowed():
    mesh = _tri()
    v = uw.discretisation.MeshVariable("v_p2p0", mesh, 2, degree=2)
    p = uw.discretisation.MeshVariable("p_p2p0", mesh, 1, degree=0, continuous=False)
    uw.systems.Stokes(mesh, velocityField=v, pressureField=p)


@pytest.mark.level_1
@pytest.mark.tier_b
def test_darcy_is_not_affected():
    mesh = _tri()
    h = uw.discretisation.MeshVariable("h_darcy_p1s", mesh, 1, degree=1)
    v = uw.discretisation.MeshVariable("v_darcy_p1s", mesh, 2, degree=1)
    uw.systems.SteadyStateDarcy(mesh, h_Field=h, v_Field=v)
