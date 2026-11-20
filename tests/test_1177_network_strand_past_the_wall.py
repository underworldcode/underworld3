"""mesher="network" refuses a strand that reaches the domain wall (#1177).

The network mesher embeds each strand's spine, the polyline itself, as points
and edges of the ribbon face, and the face is clipped to the domain. A strand
drawn past a wall leaves spine points outside the clipped face, and gmsh,
handed embedded points outside their face, either meshes nothing or does not
return. Measured on development (one rank, gmsh 4.15.2):

* two strands dipping 20 degrees from one rung under the top wall, 3 band
  widths apart (#1177's reproduction): the second crosses the right wall, and
  the placement did not return in 55 minutes;
* that second strand alone: no return in 120 s;
* the same strand 0.22 lower: "the ribbon assembly meshed to no triangles"
  after 6.6 s.

* the same strand ended exactly on the right wall: "meshed to no triangles".

The network path does not build an outcrop. The assembly now refuses, before
gmsh meshes, a ribbon the domain clips (the domain removed part of its area),
and names the meshers that build an outcrop: the default and "ladder".
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import place_surface

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(120, method="thread")]

H = 0.03                                  # rung and base cell size
W = 2.0 * H                               # band width
GAP = 3.0                                 # band widths between the strands' tops


def _straight(top, n=12, dip_degrees=20.0):
    dip = np.radians(dip_degrees)
    return np.asarray(top, float) + (np.arange(n) * H)[:, None] * np.array(
        [np.cos(dip), -np.sin(dip)])


SECOND_X = 0.2 + GAP * W / np.sin(np.radians(20.0))      # 0.726; it ends at x = 1.036
CASES = {
    "two strands, the issue's reproduction": [("f0", _straight((0.2, 1.0 - H))),
                                              ("f1", _straight((SECOND_X, 1.0 - H)))],
    "one strand past the right wall": [("f1", _straight((SECOND_X, 0.75)))],
}


def _ended_on_the_right_wall():
    P = _straight((SECOND_X, 0.75))
    cut = int(np.argmax(P[:, 0] > 1.0))
    a, b = P[cut - 1], P[cut]
    return np.vstack([P[:cut], a + (1.0 - a[0]) / (b[0] - a[0]) * (b - a)])


CASES["one strand ended on the right wall"] = [("f1", _ended_on_the_right_wall())]


@pytest.mark.parametrize("case", list(CASES))
def test_a_strand_reaching_the_wall_is_refused_before_meshing(case):
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=H, qdegree=2)
    with pytest.raises(RuntimeError, match="does not build an outcrop") as refusal:
        place_surface.place_fault_ribbon_2d(base, CASES[case], W, margin_rings=1,
                                            clearance=0.7, split=False, mesher="network")
    assert "mesher='ladder'" in str(refusal.value) and "mesher=None" in str(refusal.value)


def test_a_strand_inside_the_domain_still_builds():
    # the first strand alone stops 0.06 of a rung under the top wall
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=H, qdegree=2)
    mesh = place_surface.place_fault_ribbon_2d(
        base, CASES["two strands, the issue's reproduction"][:1], W, margin_rings=1,
        clearance=0.7, split=False, mesher="network")[0]
    assert mesh.dm.getHeightStratum(0)[1] > base.dm.getHeightStratum(0)[1]
