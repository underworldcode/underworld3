"""Layer-2 Route B, Stage 2a: the native ``uwnvb`` DMPlexTransform.

``underworld3.utilities._nvb_transform`` registers a self-contained C
``DMPlexTransform`` named ``uwnvb`` into PETSc on import (no PETSc rebuild). This
stage is the *SBR-equivalent* base of the native newest-vertex transform: it must
reproduce PETSc's ``refine_sbr`` (longest-edge bisection) byte-for-byte, which
exercises the whole native stack end-to-end — registration, the ``SetUp``
closure, ``DMLabelPropagate`` cross-rank propagation, the cell transform, subcell
orientation, and coordinate mapping — on the pinned PETSc.

The graded newest-vertex edge choice is layered on this base in a later stage;
here we only pin the equivalence so the transform infrastructure can't regress.

Skips cleanly where the extension isn't built (e.g. a non-custom-PETSc env).
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

DM_ADAPT_REFINE = 1


def _base_dm(cellSize=0.2):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=cellSize, regular=False,
        qdegree=2,
    ).dm.clone()


def _refine(dm, ttype, marked):
    """Refine ``dm`` with transform ``ttype`` on the given marked cell ids, via
    the same ``adaptLabel`` path production code uses."""
    opts = PETSc.Options()
    had = opts.hasName("dm_plex_transform_type")
    prev = opts.getString("dm_plex_transform_type") if had else None
    opts.setValue("dm_plex_transform_type", ttype)
    try:
        d = dm.clone()
        d.createLabel("adapt")
        lab = d.getLabel("adapt")
        lab.setDefaultValue(0)
        for c in marked:
            lab.setValue(int(c), DM_ADAPT_REFINE)
        return d.adaptLabel("adapt")
    finally:
        if had:
            opts.setValue("dm_plex_transform_type", prev)
        else:
            opts.delValue("dm_plex_transform_type")


def _triangulation(dm):
    """The set of triangles as sorted rounded vertex-coordinate triples — a
    geometry-level fingerprint independent of point numbering."""
    coords = dm.getCoordinatesLocal().array.reshape(-1, 2)
    vs, ve = dm.getDepthStratum(0)
    cs, ce = dm.getHeightStratum(0)
    tris = []
    for c in range(cs, ce):
        cl = dm.getTransitiveClosure(c)[0]
        verts = [p for p in cl if vs <= p < ve]
        tris.append(tuple(sorted(tuple(np.round(coords[p - vs], 10)) for p in verts)))
    return sorted(tris)


def test_registered():
    assert _nvb.registered is True


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_uwnvb_matches_refine_sbr(seed):
    """Native ``uwnvb`` produces an identical triangulation to ``refine_sbr`` for
    a variety of marked sets, including full uniform refinement."""
    rng = np.random.default_rng(seed)
    base = _base_dm()
    ncell = base.getHeightStratum(0)[1]
    if seed == 0:
        marked = [0]
    elif seed == 1:
        marked = sorted(rng.choice(ncell, size=min(5, ncell), replace=False).tolist())
    else:
        marked = list(range(ncell))  # full uniform refine

    tri_sbr = _triangulation(_refine(_base_dm(), "refine_sbr", marked))
    tri_nvb = _triangulation(_refine(_base_dm(), "uwnvb", marked))
    assert len(tri_nvb) == len(tri_sbr)
    assert tri_nvb == tri_sbr
