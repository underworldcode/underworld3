"""The circular-wall methods take each wall's circle from the mesh's declaration.

``add_cylindrical_wall_transfer``, ``add_cylindrical_zero_slip_transfer`` and
``cylindrical_pressure_lifting`` read the circle from
``mesh.analytic_surface(label)`` and accept no radius from the caller. They
refuse a boundary the mesh does not declare as an origin-centred circle:
undeclared, released (after ``deform``), or declared as a plane. After a move
that keeps the walls declared (a mover that slides them and restores them onto
their circles), the lagged solve builds the wall geometry again.

The baselines are exact: the refusal's reason, the declared radius, the
labelled facets' current coordinates, and the quiescent solution u = 0.
"""
import numpy as np
import pytest
import sympy as sp
from mpi4py import MPI

import underworld3 as uw
from underworld3.meshing.smoothing.graph import _pinned_mask

pytestmark = [pytest.mark.level_1]


def _stokes(mesh, tag):
    # The solver set-up of test_1182's quiescent case: a zero load has a zero
    # residual, which a line search reports as divergence.
    v = uw.discretisation.MeshVariable(f"DeclV{tag}", mesh, 2, degree=2, continuous=True)
    p = uw.discretisation.MeshVariable(f"DeclP{tag}", mesh, 1, degree=1, continuous=True)
    stokes = uw.systems.Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = sp.Integer(1)
    stokes.saddle_preconditioner = sp.Integer(1)
    stokes.petsc_use_pressure_nullspace = True
    stokes.tolerance = 1e-9
    stokes.petsc_options["snes_type"] = "ksponly"
    stokes.petsc_options["ksp_atol"] = 1e-12
    return stokes


def _annulus():
    return uw.meshing.Annulus(radiusInner=1.22, radiusOuter=2.22, cellSize=1 / 8,
                              qdegree=3, degree=1)


@pytest.mark.tier_b
def test_the_wall_takes_the_declared_radius():
    mesh = _annulus()
    stokes = _stokes(mesh, "R")
    wall = stokes.add_cylindrical_zero_slip_transfer(boundary="Upper", sign=1, quadrature=6)
    assert wall.radius == 2.22


@pytest.mark.tier_b
def test_a_radius_from_the_caller_is_not_accepted():
    mesh = _annulus()
    stokes = _stokes(mesh, "C")
    with pytest.raises(TypeError):
        stokes.add_cylindrical_zero_slip_transfer(boundary="Upper", radius=2.22, sign=1,
                                                  quadrature=6)


@pytest.mark.tier_b
def test_an_undeclared_wall_is_refused():
    mesh = _annulus()
    # A mesh whose wall has a label and no declared circle (a mesh read from a
    # file declares nothing): removed from the registry here.
    del mesh.bounding_surfaces["Lower"]
    stokes = _stokes(mesh, "U")
    with pytest.raises(ValueError, match="no analytic surface is registered for boundary 'Lower'"):
        stokes.add_cylindrical_zero_slip_transfer(boundary="Lower", sign=-1, quadrature=6)
    with pytest.raises(ValueError, match="no analytic surface is registered for boundary 'Lower'"):
        stokes.add_cylindrical_wall_transfer(boundary="Lower", sign=-1, viscosity=1,
                                             pressure_jump=sp.S.Zero, quadrature=6)


@pytest.mark.tier_b
def test_a_wall_released_by_deform_is_refused():
    mesh = _annulus()
    mesh.deform(1.001 * np.array(mesh.X.coords))
    stokes = _stokes(mesh, "D")
    with pytest.raises(ValueError, match="radial surface of boundary 'Upper' was released"):
        stokes.add_cylindrical_wall_transfer(boundary="Upper", sign=1, viscosity=1,
                                             pressure_jump=sp.S.Zero, quadrature=6)


@pytest.mark.tier_b
def test_a_plane_is_refused():
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=1 / 4, qdegree=3)
    stokes = _stokes(mesh, "B")
    with pytest.raises(ValueError, match="boundary 'Top' is declared as a plane, not a circle"):
        stokes.add_cylindrical_zero_slip_transfer(boundary="Top", sign=1, quadrature=6)


@pytest.mark.tier_b
def test_the_lift_refuses_a_mesh_that_declares_no_interface():
    mesh = _annulus()
    stokes = _stokes(mesh, "L")
    with pytest.raises(ValueError, match="no analytic surface is registered for boundary 'Internal'"):
        stokes.cylindrical_pressure_lifting(interface="Internal", outer="Upper", wavenumber=2,
                                            amplitude=1.0, quadrature_points=4)


@pytest.mark.tier_b
def test_the_lift_takes_both_declared_radii(monkeypatch):
    from underworld3.utilities import _cylindrical_pressure
    mesh = uw.meshing.AnnulusInternalBoundary(
        radiusInner=1.22, radiusInternal=2.0, radiusOuter=2.22,
        cellSize_Inner=1 / 8, cellSize_Internal=1 / 8, cellSize_Outer=1 / 8, qdegree=3)
    stokes = _stokes(mesh, "T")
    received = {}

    def record(mesh, interface_radius, outer_radius, *args):
        received.update(interface=interface_radius, outer=outer_radius)

    monkeypatch.setattr(_cylindrical_pressure, "_create_circular_pressure_lift", record)
    stokes.cylindrical_pressure_lifting(interface="Internal", outer="Upper", wavenumber=2,
                                        amplitude=1.0, quadrature_points=4)
    assert received == dict(interface=2.0, outer=2.22)


def _labelled_vertex_coordinates(mesh, label):
    on_label = _pinned_mask(mesh.dm, (label,))
    return np.asarray(mesh.X.coords, dtype=float)[on_label]


def _radial_metric(mesh):
    field = uw.discretisation.MeshVariable("DeclMetric", mesh, 1, degree=1)
    r = np.linalg.norm(np.asarray(field.coords), axis=1)
    field.array[:, 0, 0] = 1.0 + 4.0 * np.exp(-((r - 1.72) / 0.15) ** 2)
    return field.sym[0]


@pytest.mark.tier_b
def test_a_slide_that_keeps_the_walls_declared_rebuilds_the_wall_geometry():
    """The mover slides both walls' vertices along their circles and restores
    them, so the declarations hold; the lagged solve must then use the moved
    chords, and the quiescent solution stays exactly zero."""
    mesh = _annulus()
    stokes = _stokes(mesh, "S")
    stokes.bodyforce = sp.Matrix([0, 0])
    walls = {name: stokes.add_cylindrical_zero_slip_transfer(boundary=name, sign=sign,
                                                            quadrature=6)
             for name, sign in (("Lower", -1), ("Upper", 1))}
    before = {name: wall.edges.copy() for name, wall in walls.items()}

    uw.meshing.smooth_mesh_interior(
        mesh, metric=_radial_metric(mesh), method="mmpde", slip_surfaces=True,
        method_kwargs=dict(n_outer=6, step_frac=0.2, tol=5.0e-3))

    for name, wall in walls.items():
        assert mesh.analytic_surface(name).radius == wall.radius

    history = stokes.solve_cylindrical_zero_slip_updates(walls=walls, updates=1)

    for name, wall in walls.items():
        # The wall's chords are the labelled facets' current vertices.
        current = _labelled_vertex_coordinates(mesh, name)
        local = {tuple(row) for row in current}
        gathered = set().union(*uw.mpi.comm.allgather(local))
        assert {tuple(point) for point in wall.edges.reshape(-1, 2)} == gathered
        assert np.max(np.abs(wall.edges - before[name])) > 1.0e-4, f"{name} did not slide"
    for step in history:
        assert step["converged"]
    assert uw.mpi.comm.allreduce(float(np.max(np.abs(stokes.u.array), initial=0.0)),
                                 op=MPI.MAX) < 1.0e-12


@pytest.mark.tier_b
def test_a_solve_after_deform_is_refused():
    mesh = _annulus()
    stokes = _stokes(mesh, "F")
    walls = {name: stokes.add_cylindrical_zero_slip_transfer(boundary=name, sign=sign,
                                                            quadrature=6)
             for name, sign in (("Lower", -1), ("Upper", 1))}
    mesh.deform(1.001 * np.array(mesh.X.coords))
    with pytest.raises(ValueError, match="was released"):
        stokes.solve_cylindrical_zero_slip_updates(walls=walls, updates=1)


@pytest.mark.tier_b
def test_a_release_that_moves_no_coordinate_refuses_the_solve():
    """The cache follows the declaration, not only the coordinates: a wall whose
    surface is released without any vertex moving is refused like a moved one,
    on every rank."""
    mesh = _annulus()
    stokes = _stokes(mesh, "N")
    walls = {name: stokes.add_cylindrical_zero_slip_transfer(boundary=name, sign=sign,
                                                            quadrature=6)
             for name, sign in (("Lower", -1), ("Upper", 1))}
    version = mesh._geometry_version
    mesh.bounding_surfaces["Lower"].release("released for the test, with no move")
    assert mesh._geometry_version == version
    with pytest.raises(ValueError, match="radial surface of boundary 'Lower' was released"):
        stokes.solve_cylindrical_zero_slip_updates(walls=walls, updates=1)


def _driven_stokes(mesh, tag):
    # A cos(2 theta) radial load drives a flow, so the lagged datum is not zero
    # when the mesh moves.
    stokes = _stokes(mesh, tag)
    x, y = mesh.X
    r2 = x**2 + y**2
    stokes.bodyforce = sp.Matrix([x, y]) * (x**2 - y**2) / r2**sp.Rational(3, 2)
    walls = {name: stokes.add_cylindrical_zero_slip_transfer(boundary=name, sign=sign,
                                                            quadrature=6)
             for name, sign in (("Lower", -1), ("Upper", 1))}
    return stokes, walls


@pytest.mark.tier_b
def test_a_slide_under_a_driven_flow_restarts_the_lagged_datum():
    """After a slide that keeps the walls declared, the lagged solve must give
    what a fresh solver with fresh walls gives on the moved mesh: the datum
    lagged from the old mesh is not carried over.

    Both solves assemble the same operator and the same data from a zero datum,
    so they agree to round-off. Measured 2026-11-17 on this fixture (largest
    speed 6.8e-3): relative difference 1.1e-31 serial and at np 2, 1.3e-16 at
    np 3, 3.2e-16 at np 4. With the restart removed, so that the datum lagged
    from the old mesh is carried over the rebuild, it is 1.1e-5 (serial and
    np 2). The bound of 1e-10 relative sits five orders of magnitude from each.
    """
    mesh = _annulus()
    stokes, walls = _driven_stokes(mesh, "G")
    stokes.solve_cylindrical_zero_slip_updates(walls=walls, updates=2)
    assert all(np.any(wall.values != 0.0) for wall in walls.values()), \
        "the driven flow left a zero lagged datum, so this test cannot see a carried one"

    uw.meshing.smooth_mesh_interior(
        mesh, metric=_radial_metric(mesh), method="mmpde", slip_surfaces=True,
        method_kwargs=dict(n_outer=6, step_frac=0.2, tol=5.0e-3))
    stokes.solve_cylindrical_zero_slip_updates(walls=walls, updates=2)

    fresh, fresh_walls = _driven_stokes(mesh, "H")
    fresh.solve_cylindrical_zero_slip_updates(walls=fresh_walls, updates=2)

    speed = uw.mpi.comm.allreduce(float(np.max(np.abs(fresh.u.array), initial=0.0)),
                                  op=MPI.MAX)
    difference = uw.mpi.comm.allreduce(
        float(np.max(np.abs(np.asarray(stokes.u.array) - np.asarray(fresh.u.array)),
                     initial=0.0)), op=MPI.MAX)
    assert speed > 1.0e-6, "the load drove no flow"
    assert difference <= 1.0e-10 * speed, (
        f"the lagged solve after the slide differs from a fresh solve by "
        f"{difference:.3e} (largest speed {speed:.3e}): a datum lagged from the old mesh "
        f"was carried over the rebuild")


@pytest.mark.tier_b
def test_a_wall_off_its_declared_circle_is_refused_with_the_distance():
    """A labelled wall whose vertices left the declared circle (here moved by
    1e-9 of the radius, ten times the match tolerance, without releasing the
    declaration) is refused, and the refusal says by how much and against what
    tolerance."""
    mesh = _annulus()
    on_upper = _pinned_mask(mesh.dm, ("Upper",))
    coords = np.array(mesh.X.coords)
    coords[on_upper] *= 1.0 + 1.0e-9
    mesh._deform_mesh(coords)  # the low-level move: it keeps the declaration
    stokes = _stokes(mesh, "O")
    with pytest.raises(ValueError, match=r"labelled vertices of boundary 'Upper' are off its "
                       r"declared circle of radius 2\.22: the largest distance is 1\.0\d*e-09 "
                       r"of the radius, beyond the match tolerance of 1\.0e-10"):
        stokes.add_cylindrical_zero_slip_transfer(boundary="Upper", sign=1, quadrature=6)
