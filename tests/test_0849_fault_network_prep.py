"""prepare_fault_network: crossings/abutments become splittable offset
junctions with the promised Euclidean clearance.

The split-node pipeline refuses shared vertices; imported trace sets
cross and abut. The preparer's contract: every junction kind (X, T,
near-miss) is converted, the prepared traces keep at least the ligament
of Euclidean clearance (pull-backs are angle-corrected — an oblique
junction pulls back by ligament / sin theta), and the result splits in
one ``add_fault`` call.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.meshing.surfaces import prepare_fault_network

pytestmark = [pytest.mark.level_1, pytest.mark.tier_b]

H, LIG_F = 0.02, 1.5
LIG = LIG_F * H

RAW = [
    ("A", np.array([[0.2, 0.5], [0.8, 0.5]])),      # through-going
    ("B", np.array([[0.5, 0.2], [0.5, 0.8]])),      # X crossing, 90 deg
    ("C", np.array([[0.65, 0.5], [0.85, 0.75]])),   # oblique T abutment
    ("E", np.array([[0.35, 0.51], [0.30, 0.80]])),  # near-miss abutment
]


def _poly_clearance(P, Q):
    best = np.inf
    for a, b in zip(Q[:-1], Q[1:]):
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        best = min(best, np.linalg.norm(
            P - (a + t[:, None] * ab), axis=1).min())
    return best


def test_junctions_become_offset_form_and_split():
    prepared, report = prepare_fault_network(
        [(n, p.copy()) for n, p in RAW], spacing=H, ligament=LIG_F,
        verbose=False)

    kinds = " ".join(report)
    assert "X crossing" in kinds
    assert "T abutment" in kinds
    assert "near-miss" in kinds
    names = [n for n, _ in prepared]
    # the X crossing cuts A; the T abutment does NOT (the through-going
    # trace stays continuous, only the abutter is trimmed)
    assert sum(n.startswith("A") for n in names) == 2
    assert sum(n.startswith("B") for n in names) == 2

    # the promised EUCLIDEAN clearance, everywhere (this is the check
    # that caught the along-trace pull-back being too short at oblique
    # junctions — the negative control for the angle correction)
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            d = min(_poly_clearance(prepared[i][1], prepared[j][1]),
                    _poly_clearance(prepared[j][1], prepared[i][1]))
            assert d >= 0.95 * LIG, (
                f"{names[i]} vs {names[j]}: clearance {d:.4f} < {LIG}")

    # and the whole prepared set is splittable in one call
    child = uw.meshing.UnstructuredSimplexBox(cellSize=H).add_fault(
        prepared)
    assert sorted(child._fault_point_pairs) == sorted(names)


def test_through_master_stays_continuous():
    """A declared master is never cut at an X crossing — the other
    trace yields on both sides; two crossing masters is a hard error."""
    A, B = RAW[0], RAW[1]
    prepared, _rep = prepare_fault_network(
        [(n, p.copy()) for n, p in (A, B)], spacing=H, ligament=LIG_F,
        through=["A"], verbose=False)
    names = [n for n, _ in prepared]
    assert names.count("A") == 1                 # continuous
    assert sum(n.startswith("B") for n in names) == 2
    d = min(_poly_clearance(p, dict(prepared)["A"])
            for n, p in prepared if n.startswith("B"))
    assert d >= 0.95 * LIG

    with pytest.raises(ValueError, match="through-going"):
        prepare_fault_network([(n, p.copy()) for n, p in (A, B)],
                              spacing=H, ligament=LIG_F,
                              through=["A", "B"], verbose=False)


def test_disjoint_traces_pass_through_unchanged():
    faults = [("P", np.array([[0.2, 0.3], [0.6, 0.3]])),
              ("Q", np.array([[0.2, 0.7], [0.6, 0.7]]))]
    prepared, report = prepare_fault_network(
        faults, spacing=H, ligament=LIG_F, verbose=False)
    assert report == []
    assert [n for n, _ in prepared] == ["P", "Q"]
    for (n0, p0), (_n1, p1) in zip(faults, prepared):
        assert np.array_equal(p0, p1)


def test_touching_junction_refused_with_the_real_reason():
    """Three chains terminating at ONE shared vertex: the second split
    lands its tip on the first fault's slit. The refusal must name the
    touching junction (and its physics), not 'the domain boundary'."""
    J = np.array([0.50, 0.50])
    faults = [("West", np.array([[0.15, 0.45], J])),
              ("East", np.array([J, [0.85, 0.55]])),
              ("Splay", np.array([J, [0.80, 0.76]]))]
    with pytest.raises(ValueError, match="touching junction"):
        uw.meshing.UnstructuredSimplexBox(cellSize=0.04).add_fault(faults)
