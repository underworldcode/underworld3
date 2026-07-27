"""``mesh.relax()`` and ``adapt(..., relax=...)`` — shape relaxation.

Relaxation improves element SHAPE without changing topology or the size
distribution the refinement installed. The invariants that matter:

* topology is untouched (cell and vertex counts, DOF layout);
* nothing folds;
* the graded spacing survives (this is what separates relaxation from
  re-adaptation);
* the max-angle tail actually improves — the property relaxation is for.
"""
import numpy as np
import pytest
import sympy
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]

CENTRE, H_NEAR, H_FAR, WIDTH = 0.5, 0.012, 0.30, 0.10
_SLOPE = np.sqrt(H_FAR**2 - H_NEAR**2) / WIDTH


def _metric(centroids):
    d = np.abs(np.asarray(centroids)[:, 1] - CENTRE)
    return 1.0 / np.minimum(
        np.sqrt(H_NEAR**2 + (_SLOPE * d) ** 2), H_FAR) ** 2


def _density(mesh):
    x, y = mesh.X
    d = sympy.Abs(y - CENTRE)
    return 1.0 / sympy.Min(
        sympy.sqrt(H_NEAR**2 + (_SLOPE * d) ** 2), H_FAR) ** 2


def _child(max_levels=3, **kw):
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.25,
        refinement=1, qdegree=2)
    return base.adapt(_metric, max_levels=max_levels, **kw)


def _tris(mesh):
    dm = mesh.dm
    vS, vE = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    return np.asarray([[q - vS for q in dm.getTransitiveClosure(c)[0]
                        if vS <= q < vE] for c in range(cS, cE)])


def _coords(mesh):
    return mesh.dm.getCoordinatesLocal().array.reshape(-1, 2).copy()


def _signed_area(p, t):
    a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                  - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))


def _max_angles(p, t):
    P = p[t]
    out = []
    for k in range(3):
        u, v = P[:, (k + 1) % 3] - P[:, k], P[:, (k + 2) % 3] - P[:, k]
        out.append(np.degrees(np.arccos(np.clip(
            np.sum(u * v, axis=1) / (np.linalg.norm(u, axis=1)
                                     * np.linalg.norm(v, axis=1) + 1e-300),
            -1.0, 1.0))))
    return np.stack(out, axis=1).max(axis=1)


def _h_on_feature(p, t):
    ar = np.abs(_signed_area(p, t))
    on = np.abs(p[t].mean(axis=1)[:, 1] - CENTRE) < 3 * H_NEAR
    return float(np.median(np.sqrt(2 * ar)[on]))


def test_relax_preserves_topology_and_does_not_fold():
    child = _child()
    t = _tris(child)
    n_cells, n_verts = len(t), len(_coords(child))
    s0 = _signed_area(_coords(child), t)

    child.relax(method_kwargs=dict(n_outer=20))

    assert len(_tris(child)) == n_cells
    assert len(_coords(child)) == n_verts
    s1 = _signed_area(_coords(child), t)
    assert np.all(np.sign(np.median(s0)) * s1 > 0.0), "a cell folded"


def test_relax_preserves_the_grading():
    """The point of the 'ideal' frame: size is held, only shape moves."""
    child = _child()
    t = _tris(child)
    h0 = _h_on_feature(_coords(child), t)

    child.relax(method_kwargs=dict(n_outer=20))

    h1 = _h_on_feature(_coords(child), t)
    assert h1 == pytest.approx(h0, rel=0.25), (
        f"relaxation changed on-feature spacing {h0:.4f} -> {h1:.4f}; it "
        f"should preserve the size distribution, not re-derive it")


def test_pure_shape_relax_improves_the_max_angle_tail():
    """The property relaxation exists to fix.

    NOTE the metric-free call. Passing a metric puts the mover in the
    ideal-METRIC frame, where it re-grades as well as reshapes and will
    trade element shape away to chase the size field: measured on this
    mesh, 117.9 -> 113.8 degrees without a metric but 117.9 -> 127.4 WITH
    one. Relaxing per generation does best of all (117.9 -> 101.2). So
    "relax" and "relax to a metric" are different operations and only the
    first is a shape guarantee.
    """
    child = _child(max_levels=4)
    t = _tris(child)
    before = np.percentile(_max_angles(_coords(child), t), 99)

    child.relax(method_kwargs=dict(n_outer=40))

    after = np.percentile(_max_angles(_coords(child), t), 99)
    assert after <= before, (
        f"99th-percentile max angle got worse: {before:.1f} -> {after:.1f}")


def test_adapt_relax_placements_run_and_at_end_keeps_the_cell_count():
    plain = len(_tris(_child()))
    at_end = _child(relax=True, relax_kwargs=dict(
        method_kwargs=dict(n_outer=10)))
    per_gen = _child(relax="per-generation", relax_kwargs=dict(
        method_kwargs=dict(n_outer=10)))

    # relaxing after the last generation cannot change the topology
    assert len(_tris(at_end)) == plain
    # relaxing inside the loop changes what gets marked, so the count may
    # differ -- it must still be a sane mesh
    assert len(_tris(per_gen)) > 0
    s = _signed_area(_coords(per_gen), _tris(per_gen))
    assert np.all(np.sign(np.median(s)) * s > 0.0)


def test_adapt_relax_placements_run_in_3d():
    """Relaxation composes with adapt in 3D.

    NB this does NOT cover the cell-list refinement engine: 3D prefers the
    native uwnvb transform too when it is built (the cell-list engine is the
    np=1 fallback), so both this and the 2D case exercise the native path.
    The cell-list relax hook remains uncovered.
    """
    def metric3d(c):
        d = np.sqrt((c[:, 0] - 0.5)**2 + (c[:, 1] - 0.5)**2)
        return 1.0/np.minimum(
            np.sqrt(0.05**2 + (2.0*np.abs(d - 0.3))**2), 0.3)**2

    def build(relax):
        base = uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
            cellSize=0.5, refinement=1, qdegree=2)
        return base.adapt(metric3d, max_levels=1, relax=relax,
                          relax_kwargs=dict(method_kwargs=dict(n_outer=8)))

    def ncells(m):
        cs, ce = m.dm.getHeightStratum(0)
        return ce - cs

    plain, at_end = build(False), build(True)
    assert ncells(at_end) == ncells(plain)   # end-relax cannot retopologise
    per_gen = build("per-generation")
    assert ncells(per_gen) > 0
    for m in (at_end, per_gen):
        p = m.dm.getCoordinatesLocal().array.reshape(-1, 3)
        assert np.isfinite(p).all()


def test_relax_rejects_a_contradictory_reference_frame():
    child = _child(max_levels=2)
    with pytest.raises(ValueError, match="ideal-reference-frame"):
        child.relax(method_kwargs=dict(reference="mesh"))


def test_adapt_rejects_an_unknown_relax_placement():
    with pytest.raises(ValueError, match="relax="):
        _child(max_levels=2, relax="sideways")
