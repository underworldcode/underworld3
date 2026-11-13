"""A mesh wound clockwise throughout is cut like its mirror image (#874).

In two dimensions the inversion count after a cut read every cell's planar
sign with no convention, so a mesh wound clockwise throughout (a box with x
mirrored) read every cell as inverted, and ``cut_along_lines`` refused it at
any ``snap_frac``, even 0, with "snapping inverted N cell(s); snap_frac=0.0 is
too large for this mesh" (W1). The count now reads the cells under the
mesh's own winding, voted over every rank. The mirrored box must be cut, with
the same number of cut edges as the box itself. Runs serially and in parallel.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.line_cut import cut_along_lines

pytestmark = [pytest.mark.level_1, pytest.mark.timeout(300)]


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


@pytest.mark.tier_b
@pytest.mark.parametrize("snap_frac", [0.0, 0.3])
def test_a_mirrored_box_is_cut_like_the_box(snap_frac):
    line = np.array([[0.0, 0.37], [1.0, 0.61]])
    mirrored_line = line * np.array([-1.0, 1.0])
    _, plain = cut_along_lines(_box_dm(False), [line], snap_frac=snap_frac)
    _, mirrored = cut_along_lines(_box_dm(True), [mirrored_line], snap_frac=snap_frac)
    assert mirrored["n_cut_edges"] == plain["n_cut_edges"]
