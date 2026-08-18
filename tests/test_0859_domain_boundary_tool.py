"""The domain-boundary tool (:mod:`place_surface`): the mesh's own boundary
rebuilt as OCC geometry, so an assembly can be clipped against the domain a
mesh ACTUALLY has rather than against an analytic box.

The mesh boundary IS the domain — there is no circle an annulus is failing
to be — so the tool must reproduce the discrete boundary exactly: the loop
(shell, in 3-D) of the mesh's own boundary facets, collinear (coplanar) runs
compressed to their corners so a box rebuilds the four-sided (six-faced)
tool the analytic clip used. Exactness is asserted two ways in each
dimension: the tool's OCC mass equals the mesh's own summed cell measure to
round-off, and — the negative control — it DIFFERS from the smooth-geometry
measure by far more than that tolerance, which proves the tool is the
polygon and not the circle it approximates.
"""
import gmsh
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.place_surface import (
    _compress_collinear_loop, _domain_boundary_facets, _domain_loops_2d,
    _occ_domain_2d, _occ_domain_3d, _snap_to_boundary_2d,
    _snap_to_boundary_3d, cell_areas, _owned_cell_volume)

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b,
              pytest.mark.skipif(uw.mpi.size > 1,
                                 reason="serial suite; the parallel form "
                                        "joins ptest_0855")]


def _shoelace(P):
    return 0.5 * float(P[:, 0] @ np.roll(P[:, 1], -1)
                       - P[:, 1] @ np.roll(P[:, 0], -1))


def _occ_mass_2d(loops):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("uw_domain_tool_test")
        occ = gmsh.model.occ
        tool = _occ_domain_2d(occ, [_compress_collinear_loop(L)
                                    for L in loops])
        occ.synchronize()
        return float(occ.getMass(2, tool))
    finally:
        gmsh.finalize()


def _occ_mass_3d(verts, tris):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.model.add("uw_domain_tool_test")
        occ = gmsh.model.occ
        vol, planes = _occ_domain_3d(occ, verts, tris)
        occ.synchronize()
        return float(occ.getMass(3, vol)), planes
    finally:
        gmsh.finalize()


# ------------------------------------------------------------------------ 2-D

def test_a_box_boundary_compresses_to_its_four_corners():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.11,
        regular=False, qdegree=2)
    loops = _domain_loops_2d(mesh.dm)
    assert len(loops) == 1
    comp = _compress_collinear_loop(loops[0])
    assert len(comp) == 4
    assert (sorted(map(tuple, comp))
            == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
    assert _occ_mass_2d(loops) == pytest.approx(1.0, rel=1e-12)


def test_the_annulus_tool_is_the_polygon_not_the_circle():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=0.2, qdegree=2)
    loops = _domain_loops_2d(mesh.dm)
    assert len(loops) == 2

    mesh_area = float(cell_areas(mesh.dm).sum())
    loop_area = sum(abs(_shoelace(np.asarray(L))) for L in loops)
    outer, inner = sorted((abs(_shoelace(np.asarray(L))) for L in loops),
                          reverse=True)
    assert outer - inner == pytest.approx(mesh_area, rel=1e-12)

    mass = _occ_mass_2d(loops)
    assert mass == pytest.approx(mesh_area, rel=1e-9)
    # Negative control: the polygon is NOT the smooth annulus. If the tool
    # ever silently becomes the circle, this fires before anything subtle.
    smooth = np.pi * (1.0 ** 2 - 0.5 ** 2)
    assert abs(mass - smooth) > 1e3 * abs(mass - mesh_area)
    assert loop_area > 0.0


def test_the_2d_snap_restores_wall_values_exactly():
    square = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]
    xy = np.array([[0.3, 1e-10],          # near the bottom wall
                   [1.0 - 1e-10, 0.7],    # near the right wall
                   [1e-10, 1e-10],        # near a corner
                   [0.5, 2e-9],           # OUTSIDE the snap tolerance
                   [0.4, 0.6]])           # interior
    out = _snap_to_boundary_2d(xy.copy(), square)
    assert out[0, 1] == 0.0 and out[0, 0] == 0.3
    assert out[1, 0] == 1.0 and out[1, 1] == 0.7
    assert out[2, 0] == 0.0 and out[2, 1] == 0.0
    assert out[3, 1] == 2e-9              # untouched: the control
    assert np.array_equal(out[4], xy[4])


# ------------------------------------------------------------------------ 3-D

def test_a_cube_boundary_merges_to_its_six_faces():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.35, regular=False, qdegree=2)
    verts, tris = _domain_boundary_facets(mesh.dm)
    mass, planes = _occ_mass_3d(verts, tris)
    assert len(planes) == 6
    assert mass == pytest.approx(1.0, rel=1e-12)


def test_the_spherical_shell_tool_is_the_polyhedron_not_the_sphere():
    mesh = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5,
                                     cellSize=0.4, qdegree=2)
    verts, tris = _domain_boundary_facets(mesh.dm)
    mesh_volume = _owned_cell_volume(mesh.dm)
    mass, _planes = _occ_mass_3d(verts, tris)
    assert mass == pytest.approx(mesh_volume, rel=1e-9)
    # Negative control, as in 2-D: the discrete shell, not the smooth one.
    smooth = 4.0 / 3.0 * np.pi * (1.0 ** 3 - 0.5 ** 3)
    assert abs(mass - smooth) > 1e3 * abs(mass - mesh_volume)


def test_the_3d_snap_restores_wall_values_exactly():
    verts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                      [0.0, 1.0, 0.0]])
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    planes = [(verts[0], np.array([0.0, 0.0, 1.0]))]
    xyz = np.array([[0.3, 0.4, 1e-10],
                    [0.6, 0.2, -1e-10],
                    [0.5, 0.5, 2e-9],
                    [0.5, 0.5, 0.5]])
    out = _snap_to_boundary_3d(xyz.copy(), verts, tris, planes)
    assert out[0, 2] == 0.0 and out[0, 0] == 0.3 and out[0, 1] == 0.4
    assert out[1, 2] == 0.0
    assert out[2, 2] == 2e-9              # untouched: the control
    assert np.array_equal(out[3], xyz[3])
