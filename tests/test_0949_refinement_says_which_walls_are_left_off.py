"""A refinement that leaves wall vertices off their wall says which walls, and
what to do (#949, item 1).

CubedSphere(simplex=True) holds wall vertices back at refinement: the guard
keeps the cells valid by leaving vertices where refinement put them, on a crease
between two wall patches. Measured on development: numElements 3 held 1 vertex
at refinement 1 and 21 at refinement 2; numElements 4 held none. The guard
warned only that "N vertices ... stay where refinement placed them". The warning
now names the walls they are left off and, for CubedSphere, the remedy. The
cross-patch flip that would place them is still open on #949.
"""
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
            # 21 vertices, all on the inner wall (measured)
            assert "21 vertices" in messages[0] and "off the walls ['Lower']" in messages[0]
            assert "numElements=4" in messages[0] and "simplex=False" in messages[0]


def test_a_mesh_whose_refinement_holds_nothing_back_says_nothing():
    assert _messages(4) == []
