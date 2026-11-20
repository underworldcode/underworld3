"""The fault replay on the coarse add-on levels records shared control points
without warning (#559).

``pull_vertex_onto`` never moves a vertex an earlier target of the call holds:
a later target with no free corner goes without, is counted, and on the
user's own mesh is named in a warning. The replay of a fault on the coarse
levels of its multigrid tail (``fault_mg.split_tail_levels``) sees the fine
trace sampled finer than each level, where sharing is expected; a warning
there would fire on every faulted multigrid build. It passes
``on_shared="note"``, and the count stays on the level.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import fault_split

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])


def test_the_coarse_replay_passes_note_and_the_finest_cut_warns(monkeypatch):
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, refinement=1, qdegree=2)
    pts = np.column_stack([np.linspace(*TRACE[:, 0], 9), np.linspace(*TRACE[:, 1], 9),
                           np.zeros(9)])
    guide = uw.meshing.Surface("guide_559", base, pts)
    guide.discretize()
    child = base.adapt(guide.refinement_metric_function(
        h_near=0.05 / 4, h_far=0.05, width=0.08, profile="linear"), max_levels=2)

    calls = []
    original = fault_split.add_fault

    def spy(mesh, faults, *a, **k):
        calls.append((k.get("partition", True), k.get("on_shared", "warn")))
        return original(mesh, faults, *a, **k)

    monkeypatch.setattr(fault_split, "add_fault", spy)
    mesh = child.add_fault([("Flt", TRACE)])
    replays = [on_shared for partition, on_shared in calls if partition is False]
    assert replays and set(replays) == {"note"}, calls
    # the mesh the cut returns carries its count of shared control points
    assert mesh._fault_shared_control_points == 0


@pytest.mark.parametrize("direction", ["forward", "reversed"])
def test_a_sphere_fault_cuts_in_either_direction_and_counts_its_shared_points(direction):
    """The prescribed-normal sheet's fault (test_1092): nine control points over
    0.6 radians on a sphere of cell size 0.15, two pairs of which share a vertex.
    The later of each pair keeps it, so which point goes without follows the
    trace's direction (measured: phi -0.15 and 0.075 forward, their mirror images
    reversed). The cut and the split succeed either way, and the two are counted
    on the mesh returned."""
    from underworld3.utilities import line_cut

    # the sphere of test_1092
    sphere = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.15, degree=1, qdegree=5)
    phi = np.linspace(-0.3, 0.3, 9)
    trace = np.column_stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
    if direction == "reversed":
        trace = trace[::-1]
    with pytest.warns(UserWarning, match="left without a vertex"):
        split = sphere.add_fault(("Flt", trace))
    assert line_cut.last_shared_targets["count"] == 2
    assert split._fault_shared_control_points == 2
