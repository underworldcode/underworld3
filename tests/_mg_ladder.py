"""Shared check that an adapt child's multigrid levels really do coarsen.

Imported by ``test_0836_nvb_graded_adapt`` and ``test_0840_nvb_3d_serial_adapt``,
which previously carried a verbatim copy each.

**The estimator here is deliberately NOT the one the implementation uses.**
``_subsample_mg_levels`` chooses levels by ``percentile(cell_diameters, 5)``.
Measuring the result with that same statistic asks the implementation whether it
did what it decided to do, which it always did: on the 2-D band case it reported
a step of 1.817 against a 1.800 threshold — a 0.9 % margin — while an independent
measure of the same step gave 1.401. The number being asserted was the
selector's own opinion, and a 0.9 % margin on a gmsh mesh is a CI flake waiting
for a version bump.

The estimator below is the MEAN EDGE LENGTH inside the refined region: a mean
rather than a percentile, edges rather than cell diameters, and the region the
metric was actually asked about rather than the whole mesh. It agrees with the
selector about the thing that matters — whether a level is a near-duplicate of
its neighbour — and disagrees about the exact ratio, which is why it is worth
having.
"""

import numpy as np


def refined_resolution(dm, inside):
    """Mean length of the edges whose midpoint lies in the refined region.

    ``inside`` takes an ``(n, dim)`` array of midpoints and returns a boolean
    mask. Whole-mesh statistics will not do for adapt-on-top: the mesh only grows
    where the feature is, so a genuine halving of `h` there shows up as a global
    cell-count ratio near 1 and a flat whole-mesh median.
    """
    vS, _vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    cdim = dm.getCoordinateDim()
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, cdim)
    ends = np.array([dm.getCone(e) for e in range(eS, eE)], dtype=np.int64) - vS
    A, B = X[ends[:, 0]], X[ends[:, 1]]
    sel = inside(0.5 * (A + B))
    assert sel.any(), "no edge lies in the refined region — check `inside`"
    return float(np.linalg.norm(A - B, axis=1)[sel].mean())


def assert_coarsening_ladder(child, inside, ratio=2.0, slack=0.9, floor=1.25):
    """No level may be a near-duplicate, and the adapted span must match `ratio`.

    Two separate claims, because they fail differently:

    * **no near-duplicate step.** This is the defect ``mg_coarsening_ratio``
      exists to remove — hierarchies whose top levels differed by under 1 % in
      `h`, each costing a full Galerkin RAP and smoother sweep for no correction,
      measured 2.3-7.3x slower for the same iteration count.
    * **the adapted SPAN matches the request.** Asserted cumulatively rather than
      per step. An engine lands near a target, not on it, and the individual
      steps of a graded refinement are legitimately uneven (1.40 then 3.02 for a
      requested 2.0); what the knob promises is one level per ``ratio`` in `h`
      across the adapted range, and that is what is checked.

    The base tail is excluded — it is a uniform hierarchy with its own spacing —
    but its finest level is the rung the first adapted step is measured from.
    """
    h = [refined_resolution(m.dm, inside)
         for m in child._custom_mg_coarse_meshes] + \
        [refined_resolution(child.dm, inside)]

    n_base = len(child.parent.dm_hierarchy)
    adapted = h[n_base - 1:]
    steps = [adapted[i] / adapted[i + 1] for i in range(len(adapted) - 1)]
    assert steps, "no adapted level was recorded"

    for i, r in enumerate(steps):
        assert r >= floor, (
            f"adapted step {i} coarsens by only {r:.2f} (levels "
            f"{[f'{x:.5f}' for x in adapted]}): a near-duplicate level, which is "
            f"the defect mg_coarsening_ratio exists to remove")

    span = adapted[0] / adapted[-1]
    assert span >= (ratio ** len(steps)) * slack, (
        f"{len(steps)} adapted level(s) span only {span:.2f}x in h, short of the "
        f"{ratio}x per level requested (levels {[f'{x:.5f}' for x in adapted]})")
