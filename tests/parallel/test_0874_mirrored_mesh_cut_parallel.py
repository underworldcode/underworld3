"""A mesh wound clockwise throughout is cut as the mirror image of its mirror (#874).

The inversion count after a cut reads each cell's sign under the mesh's own
winding, voted over every rank. A box with x mirrored is wound clockwise
throughout; read with the planar sign alone, every cell counted as inverted
and ``cut_along_lines`` refused it at any ``snap_frac``, even 0. Here the
mirrored box, cut along the mirrored line, must give exactly the mirror image
of the box's cut: the same vertices with x negated, bit for bit. This file
carries no ``mpi(min_size=2)`` mark: it runs serially and in the parallel
batches.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.line_cut import cut_along_lines

pytestmark = [pytest.mark.level_1, pytest.mark.timeout(300, method="thread")]


def _box_dm(mirror):
    box = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                            cellSize=0.08)
    dm = box.dm.clone()
    coords = dm.getCoordinatesLocal().copy()
    if mirror:
        X = coords.array.reshape(-1, 2)
        X[:, 0] = -X[:, 0]
        coords.array[...] = X.ravel()
    dm.setCoordinatesLocal(coords)
    return dm


def _vertices(dm):
    """Every rank's vertex coordinates, gathered and sorted (shared vertices
    appear once)."""
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    X = np.concatenate(uw.mpi.comm.allgather(X))
    X = np.unique(X, axis=0)
    return X[np.lexsort((X[:, 1], X[:, 0]))]


@pytest.mark.tier_b
@pytest.mark.parametrize("snap_frac", [0.0, 0.3])
def test_a_mirrored_box_is_cut_as_the_mirror_image(snap_frac):
    line = np.array([[0.0, 0.37], [1.0, 0.61]])
    plain, _ = cut_along_lines(_box_dm(False), [line], snap_frac=snap_frac)
    mirrored, _ = cut_along_lines(_box_dm(True), [line * np.array([-1.0, 1.0])],
                                  snap_frac=snap_frac)
    expected = _vertices(plain) * np.array([-1.0, 1.0])
    expected = expected[np.lexsort((expected[:, 1], expected[:, 0]))]
    np.testing.assert_array_equal(_vertices(mirrored), expected)
