"""Two lines cut within an edge of each other each keep a complete chain (#494).

``cut_along_lines`` skipped every edge with an endpoint on ANY line, so where
two lines ran within an edge of each other, line B's crossing of an edge that
touched line A was never split, and B's chain of cut edges had gaps, silently:
measured on a cellSize-0.05 box, 3 gaps in each chain at 0.7 h apart, 1 at
0.9 h, in one call; two sequential cuts also left gaps. A complete open chain on
V collinear vertices has V - 1 edges between them.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.line_cut import cut_along_lines

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]

H = 0.05
A = np.array([[0.0, 0.41], [1.0, 0.47]])


def _gaps(dm, line):
    vS, vE = dm.getDepthStratum(0)
    X = dm.getCoordinatesLocal().array.reshape(-1, 2)
    d = np.cross(line[1] - line[0], X - line[0])
    on = set((np.flatnonzero(np.abs(d) / np.linalg.norm(line[1] - line[0]) < 1e-9) + vS).tolist())
    eS, eE = dm.getDepthStratum(1)
    edges = sum(1 for e in range(eS, eE) if set(int(c) for c in dm.getCone(e)) <= on)
    return len(on) - 1 - edges


@pytest.fixture(scope="module")
def box():
    return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=H)


@pytest.mark.parametrize("apart", [0.7, 0.9])
def test_one_call_keeps_both_chains_complete(box, apart):
    B = A + [0.0, apart * H]
    dm, _ = cut_along_lines(box.dm.clone(), [A, B], snap_frac=0.30)
    assert (_gaps(dm, A), _gaps(dm, B)) == (0, 0)


@pytest.mark.parametrize("apart", [0.7, 0.9])
def test_sequential_cuts_keep_both_chains_complete(box, apart):
    B = A + [0.0, apart * H]
    first, _ = cut_along_lines(box.dm.clone(), [A], label="LineA", snap_frac=0.30)
    second, _ = cut_along_lines(first, [B], label="LineB", snap_frac=0.30)
    assert (_gaps(second, A), _gaps(second, B)) == (0, 0)
