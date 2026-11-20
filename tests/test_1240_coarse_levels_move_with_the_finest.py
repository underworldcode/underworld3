"""The coarse levels of a refined hierarchy move with the finest level.

Louis Moresi, 2026-11-19: "They should move, really, even if, in the FMG, we
don't know their coordinates. If we extract a coarse level as a submesh, I
think it should be an injection of the fine-level coordinates."

A move of the finest mesh used to leave the coarse levels where they were
made. A quadrilateral or hexahedral level's transfers are built from where it
stands, so a coarse level left behind took the transfers from an unmoved
coarse level to a moved fine one: on a box whose top wall rises by 2.5 % of
its depth a step, the velocity solve took 3 to 11 iterations over 20 steps
(measured serially, development 246d6cef). Each coarse vertex now takes the
displacement of the finest vertex it is under the refinement
(``_plex_topology.inherited_vertices``), at the next solve. That held the same
box at 3 to 5.

The checks are topological: a coarse vertex and its finest vertex are paired
by the refinement's own map, never by position.
"""

import numpy as np
import pytest
import sympy

import underworld3 as uw
from underworld3.discretisation.discretisation_mesh import _vertex_coordinates
from underworld3.utilities import custom_mg
from underworld3.utilities._plex_topology import inherited_vertices

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _boxes():
    return {
        "triangles": uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25, qdegree=2, refinement=2),
        "quadrilaterals": uw.meshing.StructuredQuadBox(
            elementRes=(4, 4), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), qdegree=2, refinement=2),
    }


def _to_finest(mesh, k):
    """Level ``k``'s vertices as rows of the finest level's, by the refinement."""
    h = mesh.dm_hierarchy
    rows = None
    for j in range(len(h) - 2, k - 1, -1):
        one = inherited_vertices(h[j]) - h[j + 1].getDepthStratum(0)[0]
        rows = one if rows is None else rows[one]
    return rows


def _raise_the_top(mesh, rise=0.1):
    c = np.asarray(mesh.X.coords).copy()
    c[:, -1] *= 1 + rise * np.sin(np.pi * c[:, 0])
    mesh.deform(c)


@pytest.mark.parametrize("cells", ["triangles", "quadrilaterals"])
def test_the_refinement_map_takes_a_coarse_vertex_to_its_own_place(cells):
    # Checks the map itself against the coordinates: at rest, with no
    # refinement callback, a coarse vertex and its finest vertex coincide bit
    # for bit.
    mesh = _boxes()[cells]
    finest = _vertex_coordinates(mesh.dm)
    for k in range(len(mesh.dm_hierarchy) - 1):
        assert np.array_equal(_vertex_coordinates(mesh.dm_hierarchy[k]),
                              finest[_to_finest(mesh, k)])


@pytest.mark.parametrize("cells", ["triangles", "quadrilaterals"])
def test_a_coarse_vertex_follows_its_finest_vertex(cells):
    mesh = _boxes()[cells]
    wraps = list(mesh.multigrid_levels[:-1])
    built = [_vertex_coordinates(w.dm).copy() for w in wraps]
    _raise_the_top(mesh)
    mesh.clone_dm_hierarchy()                       # where a solver reads the levels
    finest = _vertex_coordinates(mesh.dm)
    for k, wrap in enumerate(wraps):
        expected = finest[_to_finest(mesh, k)]
        assert np.array_equal(_vertex_coordinates(mesh.dm_hierarchy[k]), expected)
        assert np.array_equal(_vertex_coordinates(wrap.dm), expected)
        # where the level was made is kept, for a transfer read as built
        assert np.array_equal(wrap._transfer_vertices, built[k])


def test_a_move_put_back_puts_the_coarse_levels_back():
    mesh = _boxes()["quadrilaterals"]
    wraps = list(mesh.multigrid_levels[:-1])
    built = [_vertex_coordinates(w.dm).copy() for w in wraps]
    at_rest = np.asarray(mesh.X.coords).copy()
    _raise_the_top(mesh)
    mesh.clone_dm_hierarchy()
    mesh.deform(at_rest)
    mesh.clone_dm_hierarchy()
    for k, wrap in enumerate(wraps):
        assert np.array_equal(_vertex_coordinates(wrap.dm), built[k])
        assert np.array_equal(_vertex_coordinates(mesh.dm_hierarchy[k]), built[k])


def test_a_level_snapped_by_a_refinement_callback_stays_where_it_was_made():
    # The annulus's refinement snaps each level onto the circles, so a coarse
    # vertex is not bit for bit its finest vertex. The displacement, zero at
    # rest, leaves it where it was made.
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.25, refinement=1)
    wraps = list(mesh.multigrid_levels[:-1])
    built = [_vertex_coordinates(w.dm).copy() for w in wraps]
    mesh.clone_dm_hierarchy()
    for k, wrap in enumerate(wraps):
        assert np.array_equal(_vertex_coordinates(wrap.dm), built[k])


def _velocity_iterations_after(steps, cells="quadrilaterals"):
    mesh = _boxes()[cells]
    coarse = list(mesh.multigrid_levels[:-1])
    x, y = mesh.X
    s = uw.systems.Stokes(mesh)
    s.constitutive_model = uw.constitutive_models.ViscousFlowModel
    s.constitutive_model.Parameters.shear_viscosity_0 = 1
    for wall in ("Bottom", "Left", "Right"):
        s.add_dirichlet_bc((0.0, 0.0), wall)
    s.bodyforce = sympy.Matrix([[0, sympy.sin(sympy.pi * x)]])
    s.tolerance = 1.0e-6
    custom_mg.set_custom_fmg(s, coarse, field_id=0)
    for _ in range(steps):
        _raise_the_top(mesh, rise=0.025)
    s.solve()
    return s.snes.getKSP().getPC().getFieldSplitSubKSP()[0].getIterationNumber()


def test_quadrilateral_multigrid_holds_on_a_rising_wall():
    # Measured: 4 velocity iterations after 10 rises with the coarse levels
    # moved, serially, and 5 at np 2 and 3 (the smoother differs with the
    # partition); 8 serially on development 246d6cef, with them left where
    # they were made.
    expected = 4 if uw.mpi.size == 1 else 5
    iterations = _velocity_iterations_after(10)
    if uw.mpi.size <= 3:
        assert iterations == expected
    else:                                   # not measured above np 3
        assert iterations <= expected
