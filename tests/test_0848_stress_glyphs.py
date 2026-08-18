"""Principal-stress glyph and trajectory geometry.

These are figure primitives, so what can be asserted is the geometry
they BUILD, not what it looks like: that a known tensor produces bars
of the right length, direction, and sign class, and that the mod-180
trajectory integrator follows a direction field whose eigenvector
sign flips underfoot — the case an ordinary streamline integrator
gets wrong by reversing mid-line.
"""
import numpy as np
import pytest

import underworld3.visualisation as vis

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

import pyvista

pyvista.OFF_SCREEN = True


@pytest.fixture(autouse=True)
def _close_every_plotter():
    yield
    pyvista.close_all()


def test_uniaxial_compression_cross():
    # sigma = diag(-2, 0): one compressive bar of half-length 2*scale
    # along x, one zero-length tensile bar along y.
    coords = np.array([[0.0, 0.0]])
    stress = np.array([[[-2.0, 0.0], [0.0, 0.0]]])
    glyphs = vis.principal_stress_glyphs(coords, stress, scale=0.25)

    assert glyphs.n_cells == 2
    tensile = glyphs.cell_data["tensile"]
    assert sorted(tensile) == [0.0, 1.0]

    # Bar k's endpoints are consecutive points 2k, 2k+1.
    spans = np.ptp(np.asarray(glyphs.points).reshape(-1, 2, 3), axis=1)
    lengths = np.linalg.norm(spans, axis=1)
    compressive = int(np.argmin(tensile))
    assert np.isclose(lengths[compressive], 2 * 0.25 * 2.0)
    assert np.isclose(lengths[int(np.argmax(tensile))], 0.0)
    assert np.allclose(spans[compressive][1:], 0.0)


def test_shear_gives_45_degree_cross():
    # Pure shear sigma_xy = 1: principal axes at 45 degrees, one
    # tensile and one compressive bar of equal length.
    coords = np.array([[0.0, 0.0]])
    stress = np.array([[[0.0, 1.0], [1.0, 0.0]]])
    glyphs = vis.principal_stress_glyphs(coords, stress, scale=1.0)

    spans = np.ptp(np.asarray(glyphs.points).reshape(-1, 2, 3), axis=1)
    assert np.allclose(spans[:, 0], spans[:, 1])
    assert sorted(glyphs.cell_data["tensile"]) == [0.0, 1.0]


def test_three_bars_in_3d():
    coords = np.array([[0.0, 0.0, 0.0]])
    stress = np.array([np.diag([-3.0, 1.0, 2.0])])
    glyphs = vis.principal_stress_glyphs(coords, stress, scale=1.0)

    assert glyphs.n_cells == 3
    assert sorted(glyphs.cell_data["tensile"]) == [0.0, 1.0, 1.0]


def test_trajectory_survives_eigenvector_sign_flip():
    # A uniform horizontal direction field whose reported sign
    # alternates with x — legitimate for eigenvectors (mod 180). The
    # integrator must keep heading one way and cross the whole box.
    def direction_at(p):
        sign = 1.0 if np.sin(20.0 * p[0]) >= 0 else -1.0
        return sign * np.array([1.0, 0.0])

    def inside(p):
        return 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0

    seeds = np.array([[0.5, 0.5]])
    lines = vis.direction_trajectories(
        direction_at, seeds, inside, step=0.01, separation=0.05
    )
    assert len(lines) == 1
    line = lines[0]
    assert np.ptp(line[:, 0]) > 0.9  # spans the box
    assert np.ptp(line[:, 1]) < 1.0e-12  # never turns


def test_annulus_default_seeds_avoid_the_hole():
    # The default seed grid spans the bounding box; on an annulus the
    # box centre is not in the mesh, and evaluating there would fail.
    import sympy
    import underworld3 as uw

    mesh = uw.meshing.Annulus(radiusInner=0.5, radiusOuter=1.0,
                              cellSize=0.2)
    x, y = mesh.X
    stress = sympy.Matrix([[x, y], [y, -x]])

    pl = vis.plot_stress_glyphs(mesh, stress, num_seeds=12)
    pl.close()

    # Rebuild the same default seeding to inspect it directly.
    import pyvista as pv

    pvmesh = vis.mesh_to_pv_mesh(mesh)
    bounds = np.asarray(pvmesh.bounds).reshape(3, 2)
    spacing = (bounds[:2, 1] - bounds[:2, 0]).max() / 12
    axes = [
        np.arange(bounds[k, 0] + 0.5 * spacing, bounds[k, 1], spacing)
        for k in range(2)
    ]
    gx, gy = np.meshgrid(*axes, indexing="ij")
    seeds = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
    _cells, closest = pvmesh.find_closest_cell(
        seeds, return_closest_point=True
    )
    inside = np.linalg.norm(closest - seeds, axis=1) < 1.0e-6
    radii = np.linalg.norm(seeds[inside, :2], axis=1)
    assert inside.sum() > 0
    assert radii.min() > 0.45  # nothing seeded in the hole
    assert radii.max() < 1.0 + 1.0e-6


def test_trajectories_respect_separation():
    def direction_at(p):
        return np.array([1.0, 0.0])

    def inside(p):
        return 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0

    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 9), np.linspace(0.1, 0.9, 9))
    seeds = np.column_stack([gx.ravel(), gy.ravel()])
    lines = vis.direction_trajectories(
        direction_at, seeds, inside, step=0.01, separation=0.1
    )
    assert len(lines) > 1
    heights = sorted(line[0, 1] for line in lines)
    gaps = np.diff(heights)
    assert gaps.min() > 0.05  # no two lines share a corridor
