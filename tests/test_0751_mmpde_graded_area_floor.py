"""Regression for #419: the MMPDE mover must actually move on a GRADED mesh.

The line-search collapse guard used to be one absolute floor,
``area_floor_frac x median cell volume``. On a graded mesh the finest cells
start orders of magnitude below that floor, so every trial step was
rejected, the line search backtracked to ``scale=0``, and the mover
returned having moved nothing -- silently. The floor is now per-cell
relative (no cell may shrink below a fraction of its OWN starting volume).
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

CENTRE, H_NEAR, H_FAR, WIDTH = 0.5, 0.01, 0.25, 0.08


def _metric(centroids):
    d = np.abs(np.asarray(centroids)[:, 1] - CENTRE)
    slope = np.sqrt(H_FAR**2 - H_NEAR**2) / WIDTH
    return 1.0 / np.minimum(np.sqrt(H_NEAR**2 + (slope * d) ** 2), H_FAR) ** 2


def _density(mesh):
    x, y = mesh.X
    slope = np.sqrt(H_FAR**2 - H_NEAR**2) / WIDTH
    d = sympy.Abs(y - CENTRE)
    return 1.0 / sympy.Min(
        sympy.sqrt(H_NEAR**2 + (slope * d) ** 2), H_FAR) ** 2


def _graded_child(max_levels):
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2,
        refinement=1, qdegree=2)
    return base.adapt(_metric, max_levels=max_levels)


def _coords(mesh):
    return mesh.dm.getCoordinatesLocal().array.reshape(-1, mesh.cdim).copy()


def _volume_spread(mesh):
    """max/min cell volume -- how graded the mesh actually is."""
    dm = mesh.dm
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    p = _coords(mesh)
    t = np.asarray([[q - vS for q in dm.getTransitiveClosure(c)[0]
                     if vS <= q < vE] for c in range(cS, cE)])
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    ar = np.abs(0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                       - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])))
    return float(ar.max() / ar.min())


def test_mover_moves_on_a_graded_mesh():
    """The bug: on a strongly graded mesh the mover moved nothing at all."""
    child = _graded_child(max_levels=4)
    # Guard the guard: if this mesh were not strongly graded the test could
    # pass without exercising the floor at all. The spread saturates once
    # the metric is satisfied, so it is capped by H_FAR/H_NEAR (~200 here),
    # not by max_levels -- do not raise this without widening the metric.
    assert _volume_spread(child) > 150.0

    before = _coords(child)
    child.redistribute_nodes(_density(child),
                             method_kwargs=dict(n_outer=8))
    moved = float(np.abs(_coords(child) - before).max())

    h_fine = H_NEAR
    assert moved > 1.0e-3 * h_fine, (
        f"mover did not move a graded mesh (max|dx|={moved:.3e}); the "
        f"collapse floor is rejecting every trial step (#419)")


def test_collapse_guard_still_rejects_folding():
    """The floor must still do its job: no cell may collapse or invert."""
    child = _graded_child(max_levels=3)

    dm = child.dm
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    tris = np.asarray([[q - vS for q in dm.getTransitiveClosure(c)[0]
                        if vS <= q < vE] for c in range(cS, cE)])

    def signed(p):
        a, b, c = p[tris[:, 0]], p[tris[:, 1]], p[tris[:, 2]]
        return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                      - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))

    s0 = signed(_coords(child))
    child.redistribute_nodes(_density(child),
                             method_kwargs=dict(n_outer=25))
    s1 = signed(_coords(child))

    sign = np.sign(np.median(s0))
    assert np.all(sign * s1 > 0.0), "a cell folded during redistribution"
    # and none collapsed below the documented fraction of its own start
    assert float((s1 / s0).min()) > 0.01
