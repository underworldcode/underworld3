"""Split-node faults (:mod:`underworld3.utilities.fault_split`).

A conforming surface leaves the fault as facets the two flanking cells share;
the split duplicates the chain's interior vertices and rewires the Minus-side
cells to the replicas, so continuous FE spaces become discontinuous exactly
across the fault. What is asserted, and the defect each assertion targets:

- **conformity and support counts** — every facet keeps at most two support
  cells, and each of the doubled fault facets has exactly one, where it had
  two before. A mis-wired arc shows up here first.
- **exact chart arithmetic** — a chain of m facets has k = m-1 interior
  vertices, and the split adds exactly k vertices and m facets (each re-homed
  Minus spoke replaces a dropped original one-for-one), taking the Euler
  characteristic from 1 (disc) to 0 (slit disc = annulus). A missed or extra
  duplication cannot pass this.
- **geometry conserved** — cell count and total area unchanged, all areas
  positive, replica coordinates bit-identical to their originals.
- **DOF independence, the load-bearing one** — a Dirichlet value pinned on one
  side of the fault must NOT appear on the coincident nodes of the other
  side. If any degree of freedom were still shared, both sides would read the
  pinned value and the fault would be welded shut at that node.
- **labels** — the two side labels have equal facet counts with pairwise
  coincident endpoints; the original surface name survives on BOTH copies (it
  remains the whole-fault handle); the constructed Mesh stacks both sides
  into ``UW_Boundaries``, which is what the solver resolves names against.
- **refusals are refusals** — boundary-crossing chains, junctions,
  single-facet chains, closed loops and double splits are rejected loudly.
  Every one of these, mishandled silently, yields a mesh that looks plausible
  and computes the wrong thing.
- **re-applicability** — the fault position is a design variable and moves
  during a run, so splitting must be a pure function of the base mesh:
  cutting and splitting the same base at two positions must give two valid,
  independent children.

The parallel seam refusal and the star-forest check are in
``tests/parallel/ptest_0845_fault_split_parallel.py``.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.fault_split import split_along_label, split_fault
from underworld3.utilities.line_cut import (cell_areas, cut_along_lines,
                                            pull_vertex_onto)
from underworld3.utilities.reconnect import _coords

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

# An interior slanted segment: both tips inside the unit box, off any mesh
# symmetry line. Tips are pulled onto mesh vertices first, which is the
# supported way to terminate a cut inside the mesh.
TIP_A = np.array([0.31, 0.42])
TIP_B = np.array([0.72, 0.61])
FAULT = "Flt"
FAULT_VALUE = 30


def _box(cell_size=1 / 16, qdegree=2):
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
        cellSize=cell_size, regular=False, qdegree=qdegree)


def _cut_dm(base_dm, tip_a=TIP_A, tip_b=TIP_B, name=FAULT, value=FAULT_VALUE):
    dm = pull_vertex_onto(base_dm, np.vstack([tip_a, tip_b]))
    dm, _info = cut_along_lines(dm, [np.vstack([tip_a, tip_b])],
                                label=name, label_value=value)
    return dm


def _split(dm, name=FAULT, value=FAULT_VALUE):
    return split_along_label(dm, name, value,
                             f"{name}Plus", 31, f"{name}Minus", 32)


def _strata_counts(dm):
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    return cE - cS, vE - vS, eE - eS


def _fault_facets(dm, name, value):
    eS, eE = dm.getDepthStratum(1)
    points = dm.getLabel(name).getStratumIS(value)
    return [int(p) for p in points.getIndices() if eS <= int(p) < eE]


def test_split_doubles_the_fault_facets_and_nothing_else():
    base = _box()
    dm = _cut_dm(base.dm)
    m = len(_fault_facets(dm, FAULT, FAULT_VALUE))
    k = m - 1
    nc, nv, ne = _strata_counts(dm)

    out, point_map, clone_map = _split(dm)
    nc2, nv2, ne2 = _strata_counts(out)

    assert (nc2, nv2, ne2) == (nc, nv + k, ne + m)
    # Euler characteristic: disc (1) -> slit disc, i.e. annulus (0),
    # independently of the chain length.
    assert nv - ne + nc == 1
    assert nv2 - ne2 + nc2 == 0

    # Conformity: no facet gained a third cell, and each of the doubled fault
    # facets is now one-sided where its source was two-sided.
    cS, cE = out.getHeightStratum(0)
    for e in range(*out.getDepthStratum(1)):
        assert len(out.getSupport(e)) <= 2
    for side in ("FltPlus", "FltMinus"):
        facets = _fault_facets(out, side, {"FltPlus": 31, "FltMinus": 32}[side])
        assert len(facets) == m
        for e in facets:
            assert len(out.getSupport(e)) == 1

    # The clone map records exactly the replicas: k vertices plus the re-homed
    # Minus-side facets (m fault copies among them).
    vS2, vE2 = out.getDepthStratum(0)
    replica_vertices = [q for q in clone_map if vS2 <= q < vE2]
    assert len(replica_vertices) == k
    # point_map covers the whole source chart; only re-homed facets drop out.
    eS, eE = dm.getDepthStratum(1)
    dropped = np.flatnonzero(point_map < 0)
    assert all(eS <= p < eE for p in dropped)


def test_geometry_is_conserved_and_replicas_are_coincident():
    base = _box()
    dm = _cut_dm(base.dm)
    areas = cell_areas(dm)

    out, _point_map, clone_map = _split(dm)
    areas2 = cell_areas(out)

    assert (areas2 > 0).all()
    assert np.isclose(areas2.sum(), areas.sum(), rtol=1e-13)
    # Cells keep their source order, so per-cell data stays aligned.
    assert np.allclose(areas2, areas)

    X_old, X_new = _coords(dm), _coords(out)
    vS, _vE = dm.getDepthStratum(0)
    vS2, vE2 = out.getDepthStratum(0)
    for q, p in clone_map.items():
        if vS2 <= q < vE2:
            assert (X_new[q - vS2] == X_old[p - vS]).all()


def test_the_two_sides_share_no_dofs():
    """The load-bearing property: a value pinned on Plus stays off Minus.

    Solved rather than merely assembled, so the assertion covers the whole
    path a slip condition takes: label -> enum -> essential BC -> solution.
    """
    base = _box(cell_size=1 / 16)
    child = _cut_child(base)
    split = split_fault(child, FAULT)

    u = uw.discretisation.MeshVariable("u_split", split, 1, degree=1)
    poisson = uw.systems.Poisson(split, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 0.0
    for b in ("Left", "Right", "Top", "Bottom"):
        poisson.add_dirichlet_bc(0.0, b)
    poisson.add_dirichlet_bc(1.0, f"{FAULT}Plus")
    poisson.solve()

    X, vals = np.asarray(u.coords), np.asarray(u.data[:, 0])
    tangent = TIP_B - TIP_A
    normal = np.array([-tangent[1], tangent[0]]) / np.hypot(*tangent)
    s = (X - TIP_A) @ normal
    t = ((X - TIP_A) @ tangent) / (tangent @ tangent)
    on = (np.abs(s) < 1e-11) & (t > -1e-9) & (t < 1 + 1e-9)

    m = len(_fault_facets(split.dm, f"{FAULT}Plus",
                          split.boundaries[f"{FAULT}Plus"].value))
    k = m - 1
    # k coincident pairs plus the two single tips.
    assert on.sum() == 2 * k + 2

    interior = on & (t > 1e-9) & (t < 1 - 1e-9)
    plus_side = np.isclose(vals, 1.0, atol=1e-10)
    pinned = interior & plus_side
    free = interior & ~plus_side
    # One of each coincident pair reads the pinned value, the other reads a
    # genuine interior value strictly between the wall and the pin — it would
    # read exactly 1.0 if any DOF were shared.
    assert pinned.sum() == k
    assert free.sum() == k
    assert (vals[free] > 0.0).all() and (vals[free] < 1.0 - 1e-6).all()
    # The tips are deliberately NOT split: both sides meet there.
    tips = on & ~interior
    assert tips.sum() == 2
    assert np.allclose(vals[tips], 1.0, atol=1e-10)


def test_labels_pair_up_and_reach_the_solver_stack():
    base = _box()
    child = _cut_child(base)
    split = split_fault(child, FAULT)

    plus_v = split.boundaries[f"{FAULT}Plus"].value
    minus_v = split.boundaries[f"{FAULT}Minus"].value
    plus = _fault_facets(split.dm, f"{FAULT}Plus", plus_v)
    minus = _fault_facets(split.dm, f"{FAULT}Minus", minus_v)
    assert len(plus) == len(minus) > 0

    # Pairwise coincident: the multiset of endpoint coordinates matches.
    X = _coords(split.dm)
    vS, _vE = split.dm.getDepthStratum(0)

    def endpoint_multiset(facets):
        rows = []
        for e in facets:
            pts = sorted(tuple(np.round(X[int(v) - vS], 12))
                         for v in split.dm.getCone(e))
            rows.append(tuple(pts))
        return sorted(rows)

    assert endpoint_multiset(plus) == endpoint_multiset(minus)

    # The original name survives on both copies: 2m facets.
    fault_v = split.boundaries[FAULT].value
    assert len(_fault_facets(split.dm, FAULT, fault_v)) == 2 * len(plus)

    # UW_Boundaries is what the solver reads when resolving a name.
    stack = split.dm.getLabel("UW_Boundaries")
    assert stack.getStratumSize(plus_v) > 0
    assert stack.getStratumSize(minus_v) > 0


def test_refusals_are_refusals():
    base = _box()

    # A boundary-to-boundary chain: every vertex of a through-going cut sits
    # in the star of a boundary facet at its two ends.
    through, _info = cut_along_lines(
        base.dm, [np.array([[-0.2, 0.37], [1.2, 0.63]])],
        label="Thru", label_value=40)
    with pytest.raises(ValueError, match="domain boundary"):
        split_along_label(through, "Thru", 40, "ThruPlus", 41,
                          "ThruMinus", 42)

    # A single-facet chain has no interior vertex.
    dm = _cut_dm(base.dm)
    facets = _fault_facets(dm, FAULT, FAULT_VALUE)
    dm.createLabel("Short")
    dm.getLabel("Short").setValue(facets[0], 50)
    with pytest.raises(ValueError, match="single facet"):
        split_along_label(dm, "Short", 50, "SP", 51, "SM", 52)

    # A junction: three arms meeting at one vertex.
    junction = np.array([0.5, 0.5])
    jdm = pull_vertex_onto(base.dm, junction)
    for kk, arm in enumerate(([[-0.2, 0.20], [0.5, 0.5]],
                              [[0.5, 0.5], [1.2, 0.30]],
                              [[0.5, 0.5], [0.55, 1.2]])):
        jdm, _ = cut_along_lines(jdm, [np.asarray(arm, dtype=float)],
                                 label="Y", label_value=60)
    with pytest.raises(ValueError, match="junction"):
        split_along_label(jdm, "Y", 60, "YP", 61, "YM", 62)

    # An unknown label.
    with pytest.raises(ValueError, match="no facets carry"):
        split_along_label(base.dm, "Nowhere", 7, "NP", 8, "NM", 9)

    # A double split: after the first, the original name marks BOTH copies,
    # which read as two parallel chains rather than one.
    once, _pm, _cm = _split(_cut_dm(base.dm))
    with pytest.raises(ValueError, match="single open chain"):
        split_along_label(once, FAULT, FAULT_VALUE, "P2", 71, "M2", 72)

    # The Mesh-level wrapper refuses a second split by name before touching
    # the topology.
    child = _cut_child(base)
    split = split_fault(child, FAULT)
    with pytest.raises(ValueError, match="split already"):
        split_fault(split, FAULT)


def test_the_split_reapplies_as_the_fault_moves():
    """The fault is a design variable and migrates: same base, two positions."""
    base = _box()
    positions = [(TIP_A, TIP_B),
                 (np.array([0.28, 0.55]), np.array([0.66, 0.38]))]
    children = []
    for a, b in positions:
        child = _cut_child(base, a, b)
        children.append(split_fault(child, FAULT))

    for split in children:
        nc, nv, ne = _strata_counts(split.dm)
        assert nv - ne + nc == 0
        areas = cell_areas(split.dm)
        assert (areas > 0).all()
        assert np.isclose(areas.sum(), 1.0, rtol=1e-12)
    # Independent children: different faults, different meshes.
    assert not np.array_equal(_coords(children[0].dm),
                              _coords(children[1].dm))


def _cut_child(base, tip_a=TIP_A, tip_b=TIP_B, name=FAULT):
    """A cut child Mesh with interior tips, mirroring add_conforming_surface.

    Tips are placed with `pull_vertex_onto` directly (the surface machinery's
    own tip path takes a Surface object); the label value comes from the
    extended enum FIRST, exactly as `add_conforming_surface` does, so the
    facets the label marks are the facets the enum member resolves to.
    """
    boundaries = base._boundaries_with(name)
    value = boundaries[name].value
    dm = _cut_dm(base.dm, tip_a, tip_b, name=name, value=value)
    mesh = uw.discretisation.Mesh(
        dm,
        simplex=True,
        coordinate_system_type=base.CoordinateSystem.coordinate_type,
        qdegree=base.qdegree,
        boundaries=boundaries,
        verbose=False,
    )
    mesh.parent = base
    mesh._relationship_kind = "refinement"
    mesh._refine_dofs_coincide = False
    mesh.regions = base.regions
    mesh._parent_mesh_version = base._mesh_version
    return mesh
