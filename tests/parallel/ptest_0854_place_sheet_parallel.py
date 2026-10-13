"""Parallel placement of a 3-D sheet (:func:`place_surface.place_sheet`).

The parallel mechanism is gather-first: the sheet's region is redistributed
onto one rank, the serial carve-and-fill runs there, and every rank rebuilds
collectively through uninterpolate + ``DMPlexInterpolate``. Because the region
is rank-interior, nothing the surgery deletes or adds is shared — the leaf set
only renumbers — and the fill sees the identical cavity whatever the incoming
partition, so the topology is partition-independent by construction.

What is asserted is chosen to be DETERMINISTIC across gmsh/PETSc versions:
the labelled facet count equals the input triangle count exactly (the fill is
forbidden to remesh the sheet), the domain volume is conserved to round-off,
the global Euler number is a ball's, every wall keeps its owned facet count,
and every placed point is a vertex on the surgery rank. Absolute cell counts
are NOT pinned — they are version-dependent (the test_0842 lesson).

Every refusal path in the module is collective; the marking is SF-reconciled
so no rank's verdict depends on the partition. Run:

    mpirun -np 2 python -m pytest tests/parallel/ptest_0854_place_sheet_parallel.py --with-mpi
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities.place_surface import place_sheet

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(600)]


def _sheet(centre, tilt=0.25, half=0.2, n=5):
    u = np.array([1.0, 0.0, tilt])
    u = u / np.linalg.norm(u)
    v = np.array([0.0, 1.0, 0.0])
    s = np.linspace(-half, half, n)
    pts = np.array([np.asarray(centre) + a * u + b * v for a in s for b in s])
    tris = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [(a, b, d), (a, d, c)]
    return pts, np.array(tris, dtype=np.int64)


def _owned_label_count(dm, name, value):
    """Owned points under (name, value): a shared point counts once."""
    pStart, _pEnd = dm.getChart()
    leaves = np.zeros(dm.getChart()[1] - pStart, dtype=bool)
    try:
        _n, ilocal, _ir = dm.getPointSF().getGraph()
        if ilocal is not None and len(ilocal):
            leaves[np.asarray(ilocal, dtype=np.int64) - pStart] = True
    except (ValueError, TypeError):
        pass
    n = 0
    if dm.hasLabel(name) and dm.getLabel(name).getStratumSize(value) > 0:
        for p in dm.getLabel(name).getStratumIS(value).getIndices():
            if not leaves[int(p) - pStart]:
                n += 1
    return int(uw.mpi.comm.allreduce(n, op=MPI.SUM))


def test_parallel_placement_matches_the_serial_contract():
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.11, regular=False, qdegree=2)
    pts, tris = _sheet((0.5, 0.5, 0.5))

    walls_before = {b.name: _owned_label_count(mesh.dm, b.name, b.value)
                    for b in mesh.boundaries}

    dm, info = place_sheet(mesh.dm, pts, tris, label="Fault", label_value=44)

    # The deterministic identities. place_sheet itself gates volume, Euler
    # and the moved-node checks collectively; asserting the info here proves
    # the gates ran and agreed on every rank.
    assert info["n_surface_facets"] == len(tris)
    assert info["n_on_surface"] == 0
    assert info["n_removed"] > 0 and info["n_placed"] >= len(pts)
    assert _owned_label_count(dm, "Fault", 44) == len(tris)

    for name, before in walls_before.items():
        after = _owned_label_count(dm, name, mesh.boundaries[name].value)
        assert after == before, f"wall {name}: {before} -> {after}"

    # The sheet's points are vertices, all on the surgery rank (the gather
    # leaves the fault region resident on one rank by design).
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 3)
    vS, vE = dm.getDepthStratum(0)
    X = X[: vE - vS]
    found_local = 0
    if len(X):
        from scipy.spatial import cKDTree
        d, _ = cKDTree(X).query(pts)
        found_local = int((d == 0.0).sum())
    per_rank = comm.allgather(found_local)
    assert max(per_rank) == len(pts), (
        f"placed points found per rank {per_rank}; all must live on the "
        f"surgery rank")

    # The info dict is identical on every rank — the collective-verdict
    # discipline, asserted rather than assumed.
    gathered = comm.allgather(info)
    assert all(g == gathered[0] for g in gathered)


def test_refusals_are_collective():
    """A sheet through the wall must refuse IDENTICALLY on every rank.

    The 2-D cut's hardest-won lesson: a rank-local raise is a hang at np>=3,
    and the happy path cannot see this defect class at all.
    """
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.15, regular=False, qdegree=2)
    # Centre near the wall: the cavity must reach it and be refused.
    pts, tris = _sheet((0.5, 0.5, 0.08), tilt=0.0)

    message = None
    try:
        place_sheet(mesh.dm, pts, tris, label="Bad", label_value=9)
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
    messages = comm.allgather(message)
    assert all(m is not None for m in messages), (
        f"some rank did NOT raise: {[m is None for m in messages]}")
    assert len(set(messages)) == 1, "ranks raised different errors"


def test_an_outcropping_sheet_traces_partition_independently():
    """The general-boundary outcrop in parallel: the trace chain's edge
    count is a pure function of the sheet and the gathered boundary
    complex, never of the partition. The clip and chain run identically
    on every rank BEFORE the gather; the labelled count is read off the
    result and must equal them at every rank count (the #589 discipline,
    for the sheet's chain instead of the zone's band)."""
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.12, regular=False, qdegree=2)
    strike = np.array([1.0, 0.0, 0.0])
    dip = np.array([0.0, 0.15, -1.0])
    dip /= np.linalg.norm(dip)
    top = np.array([0.5, 0.5, 1.1])
    s = np.linspace(-0.22, 0.22, 5)
    d = np.linspace(0.0, 0.5, 5)
    pts = np.array([top + a * strike + b * dip for b in d for a in s])
    tris, n = [], 5
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, e = (i + 1) * n + j, (i + 1) * n + j + 1
            tris += [(a, b, e), (a, e, c)]

    dm, info = place_sheet(mesh.dm, pts, np.array(tris, dtype=np.int64),
                           label="FltA", label_value=41)
    assert info["n_trace_edges"] > 0, "the sheet left no trace"
    gathered = comm.allgather(info)
    assert all(g == gathered[0] for g in gathered)

    # Owned labelled trace EDGES across all ranks == the chain's count.
    pStart, _pEnd = dm.getChart()
    eS, eE = dm.getDepthStratum(1)
    leaves = np.zeros(dm.getChart()[1] - pStart, dtype=bool)
    try:
        _n, ilocal, _ir = dm.getPointSF().getGraph()
        if ilocal is not None and len(ilocal):
            leaves[np.asarray(ilocal, dtype=np.int64) - pStart] = True
    except (ValueError, TypeError):
        pass
    n_local = 0
    trace = dm.getLabel("FltA_trace")
    if dm.hasLabel("FltA_trace") and trace.getStratumSize(41) > 0:
        for p in trace.getStratumIS(41).getIndices():
            if eS <= int(p) < eE and not leaves[int(p) - pStart]:
                n_local += 1
    n_edges = int(comm.allreduce(n_local, op=MPI.SUM))
    assert n_edges == info["n_trace_edges"], (
        f"{n_edges} owned trace edges for {info['n_trace_edges']} in the "
        "chain; the trace depends on the partition")
