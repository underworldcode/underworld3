"""Split-node faults in parallel: redistribute first, split rank-interior.

The contract: the LOW-LEVEL splitter (``split_along_label``) is legal only
when the fault's cell fans stay off the partition seam — the replicas are
then rank-local, the star-forest leaf set is unchanged, and the renumbering
broadcast of ``reconnect._rebuild_point_sf`` carries it over. A
seam-touching fault at that level is REFUSED, with the same error on every
rank. The USER-facing path (``Mesh.add_fault`` / ``split_fault``) never hits
the refusal: it REDISTRIBUTES the fault's cell star (plus one growth layer)
onto the rank that already owns most of it first, so the split always runs
with serial topology — the same strategy the 3-D splitter ships.

What is asserted:

* **outcomes agree across ranks**, whatever they are — the collectivity
  contract itself;
* a fault interior to one rank splits cleanly, and the star-forest still
  resolves every leaf to its own coordinates (`_sf_coordinate_drift`, the
  check a mis-renumbered forest cannot pass while conformity, Euler and area
  all stay rank-locally perfect);
* globally, the split adds exactly its replicas: the owned-point Euler
  characteristic drops from 1 (disc) to 0 (slit disc) and the total area is
  unchanged;
* a seam-straddling fault raises the seam refusal — not a chain-fragment
  symptom — on EVERY rank at the low level, and ``add_fault`` splits the
  SAME geometry cleanly via the redistribution;
* a seam-crossing fault SPLITS AND SLIPS through ``add_fault`` at every
  rank count — no skips, no tolerated refusals.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/ptest_0845_fault_split_parallel.py
    mpirun -n 3 python -m pytest --with-mpi tests/parallel/ptest_0845_fault_split_parallel.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities.fault_split import split_along_label
from underworld3.utilities.line_cut import cut_along_lines, pull_vertex_onto

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(600)]

# A short fault tucked into a corner: at modest rank counts the partitioner
# leaves its one-ring inside a single rank, which is the success path. If a
# partition does land the seam on it, the split refuses collectively and the
# success path is reported as skipped rather than silently untested.
CORNER = (np.array([0.08, 0.10]), np.array([0.24, 0.17]))
# Near-corner to near-corner: crosses any contiguous partition of the box.
SPANNING = (np.array([0.06, 0.08]), np.array([0.94, 0.92]))

FAULT, VALUE, PLUS, MINUS = "Flt", 30, 31, 32


def _cut(tips):
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 16,
        regular=False, qdegree=2)
    dm = pull_vertex_onto(base.dm, np.vstack(tips))
    dm, _info = cut_along_lines(dm, [np.vstack(tips)],
                                label=FAULT, label_value=VALUE)
    return dm


def _attempt(dm):
    """The split's outcome on this rank: ("ok", result) or ("seam", None).

    Only the seam refusal is caught; any other error is a genuine failure and
    propagates.
    """
    try:
        result = split_along_label(dm, FAULT, VALUE,
                                   f"{FAULT}Plus", PLUS,
                                   f"{FAULT}Minus", MINUS)
        return "ok", result
    except RuntimeError as err:
        if "seam" not in str(err):
            raise
        return "seam", None


def _global(x, op=MPI.SUM):
    return uw.mpi.comm.allreduce(x, op=op)


def _owned(dm, points):
    try:
        _nroots, ilocal, _iremote = dm.getPointSF().getGraph()
    except (ValueError, TypeError):
        ilocal = None
    leaves = set() if ilocal is None else {int(p) for p in ilocal}
    return [p for p in points if p not in leaves]


def _owned_euler(dm):
    cS, cE = dm.getHeightStratum(0)
    vS, vE = dm.getDepthStratum(0)
    eS, eE = dm.getDepthStratum(1)
    nc = len(_owned(dm, range(cS, cE)))
    nv = len(_owned(dm, range(vS, vE)))
    ne = len(_owned(dm, range(eS, eE)))
    return _global(nv) - _global(ne) + _global(nc)


def _owned_area(dm):
    cS, cE = dm.getHeightStratum(0)
    return _global(sum(abs(dm.computeCellGeometryFVM(c)[0])
                       for c in _owned(dm, range(cS, cE))))


def _sf_coordinate_drift(dm):
    """Broadcast every vertex's coordinates root-to-leaf; leaves must agree.

    Lifted from ``ptest_0844_reconnect_parallel``: the one check a
    mis-renumbered star-forest cannot pass while every rank-local invariant
    stays perfect.
    """
    pStart, pEnd = dm.getChart()
    vS, vE = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    sf = dm.getPointSF()
    worst = 0.0
    for comp in range(2):
        root = np.zeros(pEnd - pStart, dtype=np.float64)
        root[vS - pStart: vE - pStart] = X[: vE - vS, comp]
        leaf = np.full(pEnd - pStart, np.nan, dtype=np.float64)
        sf.bcastBegin(MPI.DOUBLE, root, leaf, MPI.REPLACE)
        sf.bcastEnd(MPI.DOUBLE, root, leaf, MPI.REPLACE)
        seen = np.isfinite(leaf[vS - pStart: vE - pStart])
        if seen.any():
            worst = max(worst, float(np.abs(
                leaf[vS - pStart: vE - pStart][seen]
                - X[: vE - vS, comp][seen]).max()))
    return _global(worst, op=MPI.MAX)


def test_a_rank_interior_fault_splits_cleanly():
    dm = _cut(CORNER)
    assert _owned_euler(dm) == 1
    area = _owned_area(dm)

    outcome, result = _attempt(dm)
    outcomes = uw.mpi.comm.allgather(outcome)
    assert len(set(outcomes)) == 1, (
        f"rank {uw.mpi.rank}: outcomes {outcomes} disagree — the refusal is "
        "not collective")
    if outcomes[0] == "seam":
        pytest.skip(f"the corner fault touches the seam at np={uw.mpi.size}; "
                    "the refusal contract is asserted separately")

    out, _point_map, _clone_map = result
    assert _sf_coordinate_drift(out) == 0.0, (
        "a leaf no longer resolves to its own coordinates; the grown chart "
        "was not propagated to the star-forest")
    assert _owned_euler(out) == 0, "the slit did not open (or opened twice)"
    assert _owned_area(out) == pytest.approx(area, rel=1e-12)

    fS, fE = out.getHeightStratum(1)
    assert _global(sum(1 for f in range(fS, fE)
                       if len(out.getSupport(f)) > 2)) == 0

    # The two sides carry the same number of facets, all one-sided.
    counts = {}
    for name, value in ((f"{FAULT}Plus", PLUS), (f"{FAULT}Minus", MINUS)):
        # An empty stratum hands back a null IS that segfaults in
        # getIndices(); a rank owning no part of the fault is the normal case.
        facets = []
        if out.hasLabel(name) and out.getLabel(name).getStratumSize(value) > 0:
            facets = [int(p) for p in
                      out.getLabel(name).getStratumIS(value).getIndices()]
        assert all(len(out.getSupport(f)) == 1 for f in facets)
        counts[name] = _global(len(_owned(out, facets)))
    assert counts[f"{FAULT}Plus"] == counts[f"{FAULT}Minus"] > 0

    uw.pprint(f"[ptest_0845] np={uw.mpi.size}: split "
              f"{counts[f'{FAULT}Plus']} fault facets cleanly")


def test_a_fault_may_run_close_to_the_seam():
    """The refined seam rule: only a CHAIN VERTEX on the seam refuses.

    A fault whose cell fans merely touch the seam is rank-local in every
    point the split moves (an unshared vertex owns its whole fan), so it
    must split cleanly — the original blanket star-based refusal rejected
    these. The fault below reaches toward the box centre, where the np=2/3
    partitions put their seams; whichever outcome the partition produces
    must be collective, and a success must be a valid slit.
    """
    dm = _cut((np.array([0.10, 0.12]), np.array([0.46, 0.22])))
    outcome, result = _attempt(dm)
    outcomes = uw.mpi.comm.allgather(outcome)
    assert len(set(outcomes)) == 1, f"outcomes {outcomes} disagree"
    if outcomes[0] == "seam":
        pytest.skip(f"a chain vertex landed on the np={uw.mpi.size} seam; "
                    "the low-level splitter refuses (add_fault would "
                    "redistribute — asserted in the crossing test)")
    out, _pm, _cm = result
    assert _sf_coordinate_drift(out) == 0.0
    assert _owned_euler(out) == 0


def test_an_organic_cut_across_the_seam_is_refused_then_redistributed():
    """A raw cut straddling the seam is the configuration the LOW-LEVEL
    splitter refuses (collectively, with the seam verdict — never a
    chain-fragment symptom); the SAME geometry through Mesh.add_fault
    splits cleanly, because add_fault redistributes the fault's cell star
    onto one rank before splitting."""
    dm = _cut(SPANNING)
    outcome, _result = _attempt(dm)
    outcomes = uw.mpi.comm.allgather(outcome)
    assert len(set(outcomes)) == 1, (
        f"rank {uw.mpi.rank}: outcomes {outcomes} disagree — a rank that "
        "raises alone leaves its peers blocked in the next collective")
    assert outcomes[0] == "seam", (
        "a fault spanning the box did not touch the seam; the refusal "
        "contract went untested")

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 16,
        regular=False, qdegree=2)
    split = base.add_fault(("Flt", np.vstack(SPANNING)))
    assert _sf_coordinate_drift(split.dm) == 0.0
    assert _owned_euler(split.dm) == 0, (
        "add_fault did not open the slit on the redistributed mesh")


def test_a_seam_crossing_fault_redistributes_and_slips():
    """The redistribute-first strategy end to end, on the geometry that
    used to exercise the (now removed) crossing machinery: add_fault moves
    the fault's cell star onto one rank, the split runs with serial
    topology, and a frictionless solve slips with machine-zero opening —
    asserted as SUCCESS at every rank count, no tolerated refusals."""
    from underworld3.utilities.fault_contact import fault_slip

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 16,
        regular=False, qdegree=2)
    pts = np.array([[0.15, 0.35], [0.85, 0.65]])
    split = base.add_fault(("Flt", pts))

    assert _sf_coordinate_drift(split.dm) == 0.0, (
        "a leaf does not resolve to its own coordinates — the redistributed "
        "split's star-forest is wrong")
    assert _owned_euler(split.dm) == 0, "the slit did not open globally"
    fS, fE = split.dm.getHeightStratum(1)
    assert _global(sum(1 for f in range(fS, fE)
                       if len(split.dm.getSupport(f)) > 2)) == 0

    x, y = split.X
    v = uw.discretisation.MeshVariable("vX", split, 2, degree=2)
    p = uw.discretisation.MeshVariable("pX", split, 1, degree=0,
                                       continuous=False)
    stokes = uw.systems.Stokes(split, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
    stokes.tolerance = 1e-6
    stokes.petsc_use_pressure_nullspace = True
    for side in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), side)
    stokes.add_fault_bc(0, "Flt")
    stokes.solve()
    info = stokes._rotated_freeslip_info
    assert info["converged"]

    _s, V, leak = fault_slip(stokes, "Flt", info)
    leak_max = _global(float(np.abs(leak).max()) if len(leak) else 0.0,
                       op=MPI.MAX)
    peak = _global(float(np.abs(V).max()) if len(V) else 0.0, op=MPI.MAX)
    assert leak_max < 1e-12, "a coincident pair opened"
    half = 0.5 * float(np.linalg.norm(pts[1] - pts[0]))
    assert peak > 0.15 * half, "the redistributed fault barely slips"
    uw.pprint(f"[ptest_0845] np={uw.mpi.size}: redistributed fault slipped, "
              f"peak {peak:.4f}, leak {leak_max:.1e}")
