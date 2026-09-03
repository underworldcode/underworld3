"""Conservative level set under rigid rotation, on both transport solvers.

A circle carried once round the box centre must come back: the enclosed
volume is held by the mass correction throughout, the 0.5 contour returns to
its initial position, and the reinitialisation keeps the profile at its
thickness rather than letting it smear.

Run: pixi run python -m pytest tests/test_1100_levelset_rotation.py -v
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.systems import level_set

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

RADIUS = 0.15
CENTRE = (0.5, 0.75)


def _setup(tag, advection):
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 32, qdegree=3)
    x, y = mesh.X
    psi = uw.discretisation.MeshVariable(f"psi_{tag}", mesh, 1, degree=2)
    eps = level_set.interface_thickness(mesh, psi, scale=0.35)
    distance = RADIUS - np.sqrt((psi.coords[:, 0] - CENTRE[0]) ** 2 + (psi.coords[:, 1] - CENTRE[1]) ** 2)
    level_set.initialise_psi(psi, eps, signed_distance=distance)
    psi0 = np.array(psi.array)
    # rigid rotation about the box centre, one revolution in 2 pi
    velocity = sympy.Matrix([[-(y - 0.5), x - 0.5]])
    # the rotating flow crosses the walls: impose the far-field value there
    solver = uw.systems.LevelSetSolver(psi, velocity=velocity, epsilon=eps, advection=advection,
                                       far_field=0.0, reini_steps=1, reini_frequency=5)
    return mesh, psi, psi0, solver


@pytest.mark.parametrize("advection", ["supg", "slcn"])
def test_circle_returns_after_one_revolution(advection):
    mesh, psi, psi0, solver = _setup(advection, advection)
    area0 = solver.interface_volume()
    assert area0 == pytest.approx(np.pi * RADIUS ** 2, rel=0.02)

    n_steps = 200
    dt = 2.0 * np.pi / n_steps
    for _ in range(n_steps):
        solver.solve(dt)
        assert solver.interface_volume() == pytest.approx(area0, rel=1e-6)

    data = np.asarray(psi.array).reshape(-1)
    assert data.min() >= 0.0 and data.max() <= 1.0
    # profile stays sharp: the transition band holds a small share of nodes
    in_band = np.mean((data > 0.05) & (data < 0.95))
    assert in_band < 0.12, in_band
    # the 0.5 contour is back: nodal mismatch of the indicator is small
    mismatch = np.mean(np.abs((data > 0.5).astype(float) - (psi0.reshape(-1) > 0.5).astype(float)))
    assert mismatch < 0.01, mismatch


def test_initialise_from_polygon_needs_shapely_or_a_distance():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=2)
    psi = uw.discretisation.MeshVariable("psi_poly", mesh, 1, degree=1)
    eps = level_set.interface_thickness(mesh, psi)
    pytest.importorskip("shapely")
    angles = np.linspace(0.0, 2.0 * np.pi, 33)
    circle = np.column_stack((CENTRE[0] + RADIUS * np.cos(angles), CENTRE[1] + RADIUS * np.sin(angles)))
    level_set.initialise_psi(psi, eps, interface_geometry="polygon", interface_coordinates=circle)
    data = np.asarray(psi.array).reshape(-1)
    assert 0.0 <= data.min() and data.max() <= 1.0 and 0.0 < data.mean() < 0.2


def test_material_property_field_blends():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 8, qdegree=2)
    psi = uw.discretisation.MeshVariable("psi_mat", mesh, 1, degree=1)
    field = level_set.material_property_field(psi.sym[0], [1.0, 100.0], "arithmetic")
    assert field.subs(psi.sym[0], 1) == 100.0 and field.subs(psi.sym[0], 0) == 1.0
    with pytest.raises(ValueError, match="interface must be one of"):
        level_set.material_property_field(psi.sym[0], [1.0, 100.0], "cubic")
