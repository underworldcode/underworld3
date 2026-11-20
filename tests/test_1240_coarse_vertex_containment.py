"""The containment rule for a coarse vertex read in the level above it.

Stage 2 of the coarse levels moving with the fine (add-on and split levels): a
coarse vertex takes the displacement of the level above it, interpolated where
the vertex was made, in a candidate cell from the recorded parents. A vertex
outside every candidate (H5 measured coarse cut vertices up to 0.027 outside,
barycentric weights up to 2.0 at the tips) is clamped into the candidate that
holds it best, and the clamp is reported: counted with its worst weight before
clipping, warned below -1e-2 and refused below -0.5.

The fixture is two triangles, (0,0) (1,0) (0,1) and (1,0) (1,1) (0,1), and a
linear field, which a point inside either is read exactly.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import fault_mg
from underworld3.utilities.custom_mg import CollectiveBuildError

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

VXY = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
CV = np.array([[0, 1, 2], [1, 3, 2]])
FIELD = np.column_stack([1.0 + 2.0 * VXY[:, 0] - VXY[:, 1], 3.0 * VXY[:, 1]])


def _linear(P):
    return np.column_stack([1.0 + 2.0 * P[:, 0] - P[:, 1], 3.0 * P[:, 1]])


def test_a_point_inside_a_candidate_is_read_exactly_in_the_best_one():
    P = np.array([[0.25, 0.25], [0.75, 0.6], [0.5, 0.5]])
    # every point offered both cells; (0.5, 0.5) lies on their shared edge
    owner = np.array([0, 0, 1, 1, 2, 2])
    cells = np.array([0, 1, 0, 1, 0, 1])
    out, n, worst = fault_mg._interpolate_in_candidates(P, owner, cells, VXY, CV, FIELD)
    assert np.abs(out - _linear(P)).max() < 1e-14
    assert (n, worst) == (0, 0.0)


@pytest.mark.parametrize("below, outcome", [(0.005, "noted"), (0.05, "warned"),
                                            (0.7, "refused")])
def test_a_point_outside_every_candidate_is_clamped_and_reported(below, outcome, recwarn):
    # under the bottom edge of the first cell, its only candidate
    P = np.array([[0.4, -below]])
    out, n, worst = fault_mg._interpolate_in_candidates(P, [0], [0], VXY, CV, FIELD)
    # clamped onto the edge: the weight -below clipped to 0, the rest renormalised
    lam = np.array([1.0 - 0.4 + below, 0.4, -below])
    w = np.clip(lam, 0.0, 1.0) / np.clip(lam, 0.0, 1.0).sum()
    assert np.abs(out[0] - w @ FIELD[CV[0]]).max() < 1e-14
    assert n == 1 and worst == pytest.approx(-below, rel=1e-12)
    if outcome == "refused":
        with pytest.raises(CollectiveBuildError, match="half a cell or more outside"):
            fault_mg._containment_verdict(n, worst, uw.mpi.comm)
        return
    notes = fault_mg._containment_verdict(n, worst, uw.mpi.comm)
    assert notes == {"clamped": 1, "worst_weight": pytest.approx(-below, rel=1e-12)}
    warned = [w for w in recwarn if "were clamped into the nearest" in str(w.message)]
    assert len(warned) == (1 if outcome == "warned" else 0)


def test_a_point_with_no_candidate_is_left_unread():
    out, n, worst = fault_mg._interpolate_in_candidates(
        np.array([[0.2, 0.2], [0.9, 0.9]]), [1], [1], VXY, CV, FIELD)
    assert np.isnan(out[0]).all() and not np.isnan(out[1]).any()
    assert n == 0
