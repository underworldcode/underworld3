"""A cut does not snap a vertex of a labelled internal boundary (#595).

``cut_along_lines`` snaps the vertex nearest a crossing onto the line. Cutting a
fault along the centre of a placed thin band, a vertex of the band's skin (its
boundary with the host) could be the one snapped, dragging the band shut at that
point with nothing refused: 2 to 19 of the 38 skin vertices of a 0.02-wide band
moved, depending on where the cut fell. Such a vertex now keeps its place, its
crossing is split instead, and the cut warns how many were spared. ``add_fault``
passes its 2-D snapping options to the cut, and a call without them makes the
same cut as before.
"""
import warnings

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.line_cut import cut_along_lines
from underworld3.utilities.place_surface import place_thin_volume

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b, pytest.mark.timeout(300)]


def _skin_coords(dm):
    name = [dm.getLabelName(i) for i in range(dm.getNumLabels())
            if dm.getLabelName(i).endswith("_skin")][0]
    lab = dm.getLabel(name)
    vS, vE = dm.getDepthStratum(0)
    X = dm.getCoordinatesLocal().array.reshape(-1, 2)
    out = set()
    for v in lab.getValueIS().getIndices():
        for p in lab.getStratumIS(int(v)).getIndices():
            out.update(q for q in dm.getTransitiveClosure(int(p))[0] if vS <= q < vE)
    return X[np.array(sorted(out)) - vS]


@pytest.mark.parametrize("offset", [0.0, -0.004])
def test_the_band_skin_keeps_its_shape(offset):
    mesh = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.04, regular=False)
    band, _ = place_thin_volume(mesh.dm, [np.array([[0.3, 0.5], [0.7, 0.53]])], width=0.02)
    skin = _skin_coords(band)
    line = np.array([[0.0, 0.4955 + offset], [1.0, 0.5705 + offset]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cut, _ = cut_along_lines(band.clone(), [line], label="Cut", snap_frac=0.30)
    X = cut.getCoordinatesLocal().array.reshape(-1, 2)
    moved = [p for p in skin if np.min(np.linalg.norm(X - p, axis=1)) > 1e-12]
    assert not moved, f"{len(moved)} skin vertices moved"
    assert any("#595" in str(w.message) for w in caught)


def _vertices(mesh):
    vS, vE = mesh.dm.getDepthStratum(0)
    return uw.mpi.comm.allreduce(vE - vS)


def test_add_fault_without_snap_options_makes_the_same_cut():
    def faulted(**snap):
        box = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                                cellSize=0.08, regular=False)
        return box.add_fault([("F595", np.array([[0.25, 0.3], [0.75, 0.62]]))], **snap)

    plain = _vertices(faulted())
    assert plain == _vertices(faulted(snap_frac=0.10, snap_quality=0.15, snap_dist=0.0))
    assert plain != _vertices(faulted(snap_frac=0.0))      # the option reaches the cut
