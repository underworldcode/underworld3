"""A ladder rung a hair inside the wall goes to the outcrop cap (#1090).

The ladder outcrop builder keeps the structured band on the rows inside the
domain and closes it with an unstructured cap from the last inside rung to the
wall, clipped against the domain in CAD. A row whose points sat just inside the
wall left a cap sliver thinner than the geometry kernel's tolerance, which the
clip merged away. Measured on development:

* an annulus (cell size 0.5, refinement 4), a trace dipping 35 degrees: with a
  rail point about 3e-8 inside the chord polygon the cap meshed short of its CAD
  area, at about 3e-7 it lost a rung vertex (#1090's two refusals; a rotation
  sweep in the issue found 100 such placements in 2001);
* a unit square, the rung 1e-7 or 1e-8 below the top wall: "the ladder's outcrop
  cap meets no part of the domain".

A row now counts as inside only when its points are at least a quarter of a cell
from the boundary; a closer rung goes to the cap, which then starts one rung
back. This file uses the square, which every platform meshes the same.
"""
import numpy as np
import pytest

from underworld3.utilities import place_surface

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

SIZE = 0.03
WIDTH = 2 * SIZE
SQUARE = [np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])]


@pytest.mark.parametrize("gap", [1e-3, 1e-5, 1e-7, 1e-8])
def test_a_rung_just_under_the_wall_still_closes_with_a_cap(gap):
    # a vertical spine at x = 1/2 whose top inside rung sits `gap` under y = 1,
    # with one row past the wall
    y = (1.0 - gap) - SIZE * np.arange(12)[::-1]
    S = np.vstack([np.column_stack([np.full(12, 0.5), y]), [0.5, 1.0 - gap + SIZE]])
    R = np.tile([1.0, 0.0], (len(S), 1))
    pts, tris, cad_area = place_surface._ladder_outcrop_assembly_2d(S, R, WIDTH, SIZE, SQUARE)
    Q = pts[tris]
    twice = ((Q[:, 1, 0] - Q[:, 0, 0]) * (Q[:, 2, 1] - Q[:, 0, 1])
             - (Q[:, 1, 1] - Q[:, 0, 1]) * (Q[:, 2, 0] - Q[:, 0, 0]))
    assert (twice > 0).all() or (twice < 0).all(), "a cap triangle is inverted"
    mesh_area = float(np.abs(twice).sum() / 2.0)
    # the band from the bottom rung to the wall, the width of the ribbon
    expected = WIDTH * (1.0 - y[0])
    assert abs(mesh_area - cad_area) <= 1e-12 * cad_area, (mesh_area, cad_area)
    assert abs(cad_area - expected) <= 1e-12 * expected, (cad_area, expected)
