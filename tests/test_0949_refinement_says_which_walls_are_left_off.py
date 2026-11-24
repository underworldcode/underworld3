"""A refinement that leaves wall vertices off their wall says which walls, and
what to do (#949, item 1).

CubedSphere(simplex=True) holds wall vertices back at refinement: the guard
keeps the cells valid by leaving vertices where refinement put them, on a crease
between two wall patches. How many depends on the platform's gmsh mesh:
numElements 3 at refinement 2 held 21 vertices on macOS (12-core Apple M4) and
47 on the Linux CI runner (3 at refinement 1, the issue's count); numElements 4
held none on both. So the test asserts a positive count, not one. The guard
warned only that "N vertices ... stay where refinement placed them". The warning
now names the walls they are left off and, for CubedSphere, the remedy. The
cross-patch flip that would place them is still open on #949.
"""
import ast
import re
import warnings

import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _messages(numElements):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        uw.meshing.CubedSphere(radiusOuter=1.0, radiusInner=0.547, numElements=numElements,
                               simplex=True, refinement=2)
    return [str(w.message) for w in record if "stay where refinement placed them" in str(w.message)]


def test_a_crease_that_holds_vertices_back_names_the_walls_and_the_remedy():
    messages = _messages(3)
    with uw.selective_ranks(0) as selected:
        if selected:
            assert len(messages) == 1, messages
            # the count, and which walls hold the crease vertices, are the
            # platform's gmsh mesh's: on macOS all on Lower, on a Linux build
            # not (H5, 2026-11-22); so any non-empty list of the shell's walls
            held = re.search(r"(\d+) vertices it moved", messages[0])
            assert held and int(held.group(1)) > 0, messages[0]
            walls = re.search(r"off the walls (\[[^\]]*\])", messages[0])
            assert walls, messages[0]
            named = ast.literal_eval(walls.group(1))
            assert named and set(named) <= {"Lower", "Upper"}, messages[0]
            assert "numElements=4" in messages[0] and "simplex=False" in messages[0]


def test_a_mesh_whose_refinement_holds_nothing_back_says_nothing():
    assert _messages(4) == []
