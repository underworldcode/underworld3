"""The remap of a smooth field across a mesh move is accurate, with no limiter by default.

The monotone clamp bounds every remapped value by its nearest DOFs on one rank, which adds
error on a smooth field. A target outside the old mesh takes the location fallback in
global_evaluate, which reads the variable's kd-tree, so the tree has to follow the nodes.
A P2 field can still leave its nodal range inside a cell at a sharp front; nothing here
bounds that.
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]


def test_a_wall_slide_keeps_a_smooth_field_accurate():
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    field = (r - 0.5) * 2 + 0.5 * sympy.cos(4 * sympy.atan2(y, x)) * r
    T = uw.discretisation.MeshVariable("T0768", mesh, 1, degree=2)
    T.array[:, 0, 0] = uw.function.evaluate(field, T.coords).reshape(-1)

    coords = np.array(mesh.X.coords)
    radius = np.linalg.norm(coords, axis=1)
    wall = (np.abs(radius - 1.0) < 1.0e-9) | (np.abs(radius - 0.5) < 1.0e-9)
    angle = 0.2 * (1 / 8) / 0.5     # a fifth of a cell on the inner wall
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    coords[wall] = coords[wall] @ rotation
    mesh.deform(coords)

    dof = np.asarray(T.coords).reshape(-1, 2)
    error = np.abs(np.asarray(T.array).reshape(-1) - uw.function.evaluate(field, dof).reshape(-1))
    dof_radius = np.linalg.norm(dof, axis=1)
    interior = (dof_radius < 0.99) & (dof_radius > 0.51)
    assert error[interior].max() < 2.0e-3, error[interior].max()    # 9.8e-3 under the clamp, 8.3e-4 without
    assert error[~interior].max() < 4.0e-2, error[~interior].max()  # the fallback's order: 2.7e-2


def test_the_variable_kdtree_follows_a_mover():
    """Off by 0.2 on this case when the tree stayed at the pre-move nodes."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 8, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    T = uw.discretisation.MeshVariable("T0768k", mesh, 1, degree=1)
    T.array[:, 0, 0] = np.asarray(T.coords)[:, 0]
    start = np.array(T.coords)
    T.rbf_interpolate(start)                    # the tree exists before the move

    density = 1 + 24 * sympy.exp(-(((r - 0.9) / 0.06) ** 2))
    uw.meshing.node_redistribution(mesh, density, slip_surfaces=True, skip_threshold=None,
                                   method_kwargs=dict(accel="none", n_outer=20))
    nodes = np.array(T.coords)
    assert np.abs(nodes - start).max() > 1.0e-3       # the mover did move them
    read_back = np.asarray(T.rbf_interpolate(nodes)).reshape(-1)
    assert np.abs(read_back - np.asarray(T.array).reshape(-1)).max() < 1.0e-9


def test_a_p1_field_stays_in_range_at_an_unresolved_front():
    """P1 interpolation is a convex combination, so a viscosity-bearing field kept to P1
    needs no limiter. P2 at this front leaves its range by 9 %."""
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=1 / 16, qdegree=3)
    x, y = mesh.X
    r = sympy.sqrt(x ** 2 + y ** 2)
    T = uw.discretisation.MeshVariable("T0768p", mesh, 1, degree=1)
    T.array[:, 0, 0] = uw.function.evaluate(
        0.5 * (1 + sympy.tanh((r - 0.75) / 0.005)), T.coords).reshape(-1)
    start = np.array(T.array)

    density = 1 + 24 * sympy.exp(-(((r - 0.75) / 0.06) ** 2))
    uw.meshing.node_redistribution(mesh, density, slip_surfaces=True, skip_threshold=None,
                                   method_kwargs=dict(accel="none", n_outer=20))
    remapped = np.asarray(T.array)
    assert np.abs(remapped - start).max() > 1.0e-3      # the remap did something
    assert remapped.min() >= start.min() - 1.0e-12
    assert remapped.max() <= start.max() + 1.0e-12
