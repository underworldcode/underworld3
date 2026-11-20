"""Every coarse fault copy has fine images in the split-level transfer (#1120).

A fault cut into an adapt child is replayed on the add-on levels, and the
transfer between two split levels (``fault_mg.cell_located_prolongation``)
locates each fine cell's centroid in the coarse cells. Next to the fault the
coarse cut leaves thin pieces, and a coarse fault-pair node whose copy on one
side sits on such a piece got no fine image on that side: its column of the
transfer was empty. After the pair rotation the copy's mean and jump columns are
then proportional, and the Galerkin coarse level is singular (the issue measured
3 such pairs at rest, 3 null directions).

The fix takes, for the rows that reach such a copy, the recorded parent's row:
``fault_split.add_fault`` records each split cell's host cell through the cut and
the split (a cell label the DMPlex transform and the split carry), the two hosts
nest, and a fine node takes the best-holding same-side cut piece of its star's
nested parents. Only those rows change (``fault_mg._WEAK_COPY``).

Measured on the fixture of the issue (serial, a 96-thread Linux workstation),
the node transfer between the finest split level and the split level below it:

==========================================  ============  ==========
case                                        development   this fix
==========================================  ============  ==========
at rest: empty fault-copy columns            4             0
every add-on level moved by 0.015 across     12 (+1 weak)  0
==========================================  ============  ==========
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities import fault_mg

pytestmark = [pytest.mark.level_2, pytest.mark.tier_b, pytest.mark.timeout(600)]

TRACE = np.array([[0.3, 0.45], [0.7, 0.55]])
DEGREE = 2


def _child():
    base = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                             cellSize=0.1, refinement=1, qdegree=2)
    pts = np.column_stack([np.linspace(*TRACE[:, 0], 9), np.linspace(*TRACE[:, 1], 9),
                           np.zeros(9)])
    guide = uw.meshing.Surface("guide_1120", base, pts)
    guide.discretize()
    return base.adapt(guide.refinement_metric_function(
        h_near=0.05 / 4, h_far=0.05, width=0.08, profile="linear"), max_levels=2)


def _move_every_addon_level(child, amp):
    """Move the child and every add-on level under it by the same field, across
    the fault (the levels nest, so a shared vertex moves the same)."""
    tail = child._custom_mg_coarse_meshes
    for mesh in [child] + [tail[i] for i in fault_mg.addon_level_indices(tail)]:
        X = np.asarray(mesh.X.coords).copy()
        X[:, 1] += amp * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
        mesh.deform(X)


def _copy_column_norms(mesh):
    """The finest split pair's node transfer, and the column norms of the coarse
    level's fault copies (both copies of every split point)."""
    coarse = mesh._custom_mg_coarse_meshes[-1]
    assert fault_mg.is_split_level(coarse), "the level below the finest is not split"
    P = fault_mg.cell_located_prolongation(coarse, mesh, DEGREE, True)
    copies = _copies_by_point(coarse)
    norms = np.asarray(abs(P).sum(axis=0)).ravel()[copies]
    return norms


def _copies_by_point(mesh):
    """The node rows of every split point's two copies, read from the section of
    the paired points (written here, not imported, so the test does not use the
    code it tests). Collective: the basis DM is built on every rank."""
    pairs = getattr(mesh, "_fault_point_pairs", None) or {}
    points = sorted({int(q) for pp in pairs.values() for kv in pp.items() for q in kv})
    dmb = mesh._basis_coordinate_dm(DEGREE, True)
    sec = dmb.getLocalSection()
    rows = [r for q in points if sec.getDof(q)
            for r in range(sec.getOffset(q) // mesh.cdim,
                           (sec.getOffset(q) + sec.getDof(q)) // mesh.cdim)]
    dmb.destroy()
    return np.unique(rows)


@pytest.mark.skipif(uw.mpi.size > 1, reason="serial: the node transfer is the serial one")
@pytest.mark.parametrize("amp", [0.0, 0.002, 0.015],
                         ids=["at-rest", "moved-0.002", "moved-0.015"])
def test_every_coarse_fault_copy_has_fine_images(amp):
    child = _child()
    if amp:
        _move_every_addon_level(child, amp)
    mesh = child.add_fault([("Flt", TRACE)])
    norms = _copy_column_norms(mesh)
    assert norms.size > 0
    weak = norms < 0.1 * np.median(norms)
    assert not weak.any(), (
        f"{int((norms == 0).sum())} fault-copy column(s) empty and "
        f"{int(weak.sum())} below a tenth of the median copy column "
        f"({np.sort(norms)[:6]})")


@pytest.mark.skipif(uw.mpi.size > 1, reason="serial")
def test_the_split_records_each_cell_s_host_cell():
    child = _child()
    mesh = child.add_fault([("Flt", TRACE)])
    record = mesh._cut_parent
    cS, cE = child.dm.getHeightStratum(0)
    assert record.shape == (mesh.dm.getHeightStratum(0)[1] - mesh.dm.getHeightStratum(0)[0],)
    # every cell has a host, every host cell is the host of some cell
    assert record.min() >= 0
    assert np.array_equal(np.unique(record), np.arange(cE - cS))
    # the label is not left on any mesh
    assert not mesh.dm.hasLabel("UW_HostCell") and not child.dm.hasLabel("UW_HostCell")


@pytest.mark.skipif(uw.mpi.size > 1, reason="serial")
@pytest.mark.parametrize("amp, flagged_max, kept_min", [
    (0.0, 0.0, 0.255), (0.002, 0.0, 0.221), (0.015, 0.069, 0.379)],
    ids=["at-rest", "moved-0.002", "moved-0.015"])
def test_the_weak_copy_threshold_sits_in_the_measured_gap(amp, flagged_max, kept_min,
                                                          monkeypatch):
    """``fault_mg._WEAK_COPY`` must lie between the weakest copy the located rule
    needs repaired and the weakest it does not, in every case: a value moved
    past either edge of the gap fails here. The edges are measured live from the
    located rule alone (the repair switched off), and checked against the values
    recorded when the threshold was set."""
    child = _child()
    if amp:
        _move_every_addon_level(child, amp)
    mesh = child.add_fault([("Flt", TRACE)])
    threshold = fault_mg._WEAK_COPY
    monkeypatch.setattr(fault_mg, "_WEAK_COPY", 0.0)      # the located rule alone
    norms = _copy_column_norms(mesh)
    ratios = np.sort(norms / np.median(norms))
    needs = ratios[ratios < 0.5 * kept_min]              # the copies to repair
    sound = ratios[ratios >= 0.5 * kept_min]
    assert needs.size and needs.max() == pytest.approx(flagged_max, abs=0.005)
    assert sound.min() == pytest.approx(kept_min, abs=0.005)
    assert needs.max() < threshold < sound.min(), (
        f"_WEAK_COPY = {threshold} is outside the gap ({needs.max():.3f}, "
        f"{sound.min():.3f}) measured at amp={amp}")


@pytest.mark.skipif(uw.mpi.size > 1, reason="serial")
def test_copies_left_weak_are_reported_not_passed_in_silence(monkeypatch):
    """Only the child moves (0.015 along the fault), the add-on levels stay
    where they were built, and no recorded-parent row reaches the weak copies.
    The transfer must say so: the record counts them and a warning names the
    solve-report entry that carries them."""
    child = _child()
    X = np.asarray(child.X.coords).copy()
    X[:, 0] += 0.015 * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    child.deform(X)
    mesh = child.add_fault([("Flt", TRACE)])
    coarse = mesh._custom_mg_coarse_meshes[-1]
    monkeypatch.setattr(fault_mg, "_warned_unrepaired", False)
    with pytest.warns(UserWarning, match="unrepaired_fault_copies"):
        fault_mg.cell_located_prolongation(coarse, mesh, DEGREE, True)
    record = fault_mg.last_parent_rows
    assert record["weak_copies"] > 0 and record["remaining"] > 0, record
    # no cell of the pair touches both copies of the fault (H5's guard, defensive)
    assert record["side_conflicts"] == 0, record


def test_a_cell_touching_both_copies_of_a_fault_has_no_side():
    """A defensive guard: a cell whose star holds both the Minus and the Plus copy
    of one fault's points has no side (0), so the side filter excludes nothing
    for it, and it is counted. It used to take whichever copy was visited last.
    No real mesh has shown one; the pairing here makes one by pairing a vertex
    with itself, so every cell around it touches both copies."""
    import types

    box = uw.meshing.UnstructuredSimplexBox(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
                                            cellSize=0.25, qdegree=2)
    dm = box.dm
    v0, _v1 = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    star = np.asarray(dm.getTransitiveClosure(v0, useCone=False)[0])
    around = np.sort(star[(star >= cS) & (star < cE)] - cS)
    paired = types.SimpleNamespace(dm=dm, _fault_point_pairs={"F": {v0: v0}})
    side, names, conflicts = fault_mg._cell_fault_sides(paired)
    assert names == ["F"]
    assert conflicts == around.size > 0
    assert not side[around, 0].any()
