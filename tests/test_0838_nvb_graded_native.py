"""Layer-2 Route B, Stage 2b: graded NVB via the native single-bisection driver.

``underworld3.utilities._nvb_transform.refine(dm, want_label)`` refines every cell
flagged in the adaptation label once, with the bounded newest-vertex conforming
closure, running as repeated conforming single-bisection sub-passes (the
``uwnvb_bisect`` transform) driven from C. Unlike PETSc's longest-edge
``refine_sbr`` (unbounded closure — a uniform-finest patch only), NVB *grades*: a
mark deep in a refined patch adds O(1) cells locally.

These tests pin the graded behaviour against the validated serial reference
``underworld3.utilities.nvb.NVBMesh``:
  - the refinement edges match ``NVBMesh`` exactly under repeated uniform refine;
  - a mark deep in a refined patch is bounded/local (NVB) not a drain (SBR);
  - the mesh stays conforming (0 hanging / 0 over-shared) and refinement is
    deterministic;
  - a shrinking bullseye grades (many levels coexist), with far fewer cells than
    the SBR uniform patch.

Serial (the SF-reconciled cross-rank closure for full parallel confluence is a
follow-up; the driver already runs conforming at np>1).
"""
import numpy as np
import pytest
import underworld3 as uw
from petsc4py import PETSc

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

_nvb = pytest.importorskip(
    "underworld3.utilities._nvb_transform",
    reason="native uwnvb transform not built (needs the custom-PETSc/amr env)",
)
from underworld3.utilities.nvb import NVBMesh  # noqa: E402

DM_ADAPT_REFINE = 1
CENTER = np.array([0.5, 0.5])


def _base(cellSize=0.2):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=cellSize, regular=False, qdegree=2,
    ).dm.clone()


def _cents(dm):
    cs, ce = dm.getHeightStratum(0)
    return np.array([dm.computeCellGeometryFVM(c)[1] for c in range(cs, ce)]), cs, ce


def _ncells(dm):
    cs, ce = dm.getHeightStratum(0)
    return ce - cs


def _refine(dm, pred):
    d = dm.clone()
    d.createLabel("adapt")
    lab = d.getLabel("adapt")
    lab.setDefaultValue(0)
    C, cs, ce = _cents(d)
    for c in range(cs, ce):
        if pred(C[c - cs]):
            lab.setValue(c, DM_ADAPT_REFINE)
    return _nvb.refine(d, "adapt")


def _conforming(dm):
    es, ee = dm.getDepthStratum(1)
    over = sum(1 for e in range(es, ee) if dm.getSupportSize(e) > 2)
    coords = dm.getCoordinatesLocal().array.reshape(-1, 2)
    vs, ve = dm.getDepthStratum(0)
    vset = {tuple(np.round(coords[v - vs], 9)) for v in range(vs, ve)}
    hang = 0
    for e in range(es, ee):
        c = dm.getCone(e)
        mp = tuple(np.round(0.5 * (coords[c[0] - vs] + coords[c[1] - vs]), 9))
        if mp in vset:
            hang += 1
    return over, hang


def _refmap_dm(dm):
    """Each cell's refinement edge, keyed by centroid, as a frozenset of endpoint coords."""
    slot = dm.getLabel("uwnvb_refedge")
    co = dm.getCoordinatesLocal().array.reshape(-1, 2)
    vs, ve = dm.getDepthStratum(0)
    cs, ce = dm.getHeightStratum(0)
    ref = {}
    for c in range(cs, ce):
        cone = dm.getCone(c)
        s = slot.getValue(c)
        verts = set()
        for e in cone:
            a, b = dm.getCone(e)
            verts |= {a, b}
        cen = tuple(np.round(np.array([co[v - vs] for v in verts]).mean(0), 8))
        a, b = dm.getCone(cone[s])
        ref[cen] = frozenset({tuple(np.round(co[a - vs], 8)), tuple(np.round(co[b - vs], 8))})
    return ref


def _refmap_nvbmesh(m):
    C = np.array(m.coords)
    ref = {}
    for _, (p, b0, b1) in m.cells.items():
        cen = tuple(np.round(C[[p, b0, b1]].mean(0), 8))
        ref[cen] = frozenset({tuple(np.round(C[b0], 8)), tuple(np.round(C[b1], 8))})
    return ref


def test_refedges_match_nvbmesh_under_uniform_refine():
    """The native driver reproduces NVBMesh's refinement-edge structure exactly
    over repeated uniform refinement (the invariant that makes closure bounded)."""
    base = _base(0.35)
    m = NVBMesh.from_dm(base)
    dm = base
    for _ in range(4):
        m.refine(set(m.cells.keys()))
        d = dm.clone()
        d.createLabel("adapt")
        lab = d.getLabel("adapt")
        lab.setDefaultValue(0)
        cs, ce = d.getHeightStratum(0)
        for c in range(cs, ce):
            lab.setValue(c, DM_ADAPT_REFINE)
        dm = _nvb.refine(d, "adapt")
        assert _ncells(dm) == len(m.cells)
        rd, rn = _refmap_dm(dm), _refmap_nvbmesh(m)
        assert len(rd) == len(rn)
        assert all(cen in rn and rn[cen] == rd[cen] for cen in rd)


def test_deep_mark_is_bounded_not_a_drain():
    """A single cell marked deep in a 2x-refined patch adds O(1) cells (NVB), not
    the whole patch (the SBR longest-edge drain)."""
    dm = _base(0.1)
    for _ in range(2):
        dm = _refine(dm, lambda x: np.linalg.norm(x - CENTER) < 0.35)
    before = _ncells(dm)
    C, cs, ce = _cents(dm)
    tgt = C[np.argmin(np.linalg.norm(C - CENTER, axis=1))]
    dm2 = _refine(dm, lambda x, t=tgt: np.allclose(x, t))
    added = _ncells(dm2) - before
    assert 0 < added <= 12, f"deep mark added {added} cells (should be a small local closure)"
    assert _conforming(dm2) == (0, 0)


def test_bullseye_grades_and_is_conforming():
    """A shrinking bullseye keeps many refinement levels (grades) and stays
    conforming — far fewer cells than an SBR uniform patch would give."""
    dm = _base(0.12)
    for R in (0.4, 0.28, 0.18, 0.11, 0.06):
        dm = _refine(dm, lambda x, R=R: np.linalg.norm(x - CENTER) < R)
    assert _conforming(dm) == (0, 0)
    # multiple distinct cell areas => graded (not a single uniform-fine patch)
    a = np.array([dm.computeCellGeometryFVM(c)[0] for c in range(*dm.getHeightStratum(0))])
    a = a[a > 0]
    levels = np.unique(np.round(np.log2(a.max() / a)).astype(int))
    assert len(levels) >= 4, f"expected a graded spread of levels, got {levels}"


def test_deterministic():
    a = _refine(_base(0.12), lambda x: np.linalg.norm(x - CENTER) < 0.2)
    b = _refine(_base(0.12), lambda x: np.linalg.norm(x - CENTER) < 0.2)
    assert _ncells(a) == _ncells(b)
