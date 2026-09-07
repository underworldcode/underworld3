"""Parallel embedding of a thin volume (:func:`place_surface.place_thin_volume`).

The mechanism is place_sheet's, inherited whole: the assembly is meshed once
(rank 0) and broadcast so every rank marks against the identical skin; the
zone's region is gathered onto one rank; the carve and the annular hole fill
run there; every rank rebuilds collectively. The gathered region is
rank-interior, so nothing the surgery touches is shared and the topology is
partition-independent by construction.

Deterministic assertions only (the test_0842 lesson): the info dict identical
on every rank, labels of the advertised kinds and sizes, volume conserved by
the routine's own gate, and refusals collective — a rank-local raise is a
hang at np>=3 and the happy path cannot see that defect class at all. Run:

    mpirun -np 2 python -m pytest tests/parallel/ptest_0855_place_thin_volume_parallel.py --with-mpi
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities.place_surface import place_thin_volume

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(600)]

CROSS = [np.array([[0.3, 0.3, 0.5], [0.7, 0.3, 0.5],
                   [0.7, 0.7, 0.5], [0.3, 0.7, 0.5]]),
         np.array([[0.3, 0.5, 0.3], [0.7, 0.5, 0.3],
                   [0.7, 0.5, 0.7], [0.3, 0.5, 0.7]])]


def _owned_label_count(dm, name, value):
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


def test_parallel_embedding_matches_the_serial_contract():
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.09, regular=False, qdegree=2)
    new, info = place_thin_volume(mesh.dm, CROSS, width=0.045,
                                  label="Zone", label_value=5)

    # The routine gates volume, Euler and the constraint surfaces itself,
    # collectively; asserting the info here proves the gates ran and agreed.
    assert info["n_zone_cells"] > 0
    assert _owned_label_count(new, "Zone", 5) == info["n_zone_cells"]
    assert _owned_label_count(new, "Zone_skin", 5) == info["n_skin_faces"]

    gathered = comm.allgather(info)
    assert all(g == gathered[0] for g in gathered)


def test_refusals_are_collective():
    """A zone through the wall must refuse IDENTICALLY on every rank."""
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.15, regular=False, qdegree=2)
    low = np.array([[0.3, 0.3, 0.08], [0.7, 0.3, 0.08],
                    [0.7, 0.7, 0.08], [0.3, 0.7, 0.08]])
    message = None
    try:
        place_thin_volume(mesh.dm, [low], width=0.05, label="Bad")
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
    messages = comm.allgather(message)
    assert all(m is not None for m in messages), (
        f"some rank did NOT raise: {[m is None for m in messages]}")
    assert len(set(messages)) == 1, "ranks raised different errors"


def test_a_2d_outcropping_ribbon_embeds_in_parallel():
    """The 2-D outcrop through the same gather-first mechanism, at np>=2.

    The box, the band split and the chain decomposition are collective;
    the splice and relabel run on the surgery rank. Asserted: the info
    dict identical on every rank, the labels of the advertised sizes, a
    band present on the top wall, and no top-wall edge left without Top.
    """
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.07,
        regular=False, qdegree=2)
    line = np.array([[0.35, 0.40], [0.60, 1.10]])   # past the top wall
    new, info = place_thin_volume(mesh.dm, [line], width=0.04,
                                  label="Zone", label_value=5)
    assert info["n_zone_cells"] > 0
    assert _owned_label_count(new, "Zone", 5) == info["n_zone_cells"]
    assert _owned_label_count(new, "Zone_skin", 5) == info["n_skin_faces"]
    gathered = comm.allgather(info)
    assert all(g == gathered[0] for g in gathered)

    pStart, pEnd = new.getChart()
    leaves = np.zeros(pEnd - pStart, dtype=bool)
    try:
        _n, ilocal, _ir = new.getPointSF().getGraph()
        if ilocal is not None and len(ilocal):
            leaves[np.asarray(ilocal, dtype=np.int64) - pStart] = True
    except (ValueError, TypeError):
        pass
    fS, fE = new.getHeightStratum(1)
    vS, vE = new.getDepthStratum(0)
    Xn = np.asarray(new.getCoordinatesLocal().array).reshape(-1, 2)[: vE - vS]
    top = new.getLabel("Top")
    skin = new.getLabel("Zone_skin")
    n_band = n_bare = 0
    for f in range(fS, fE):
        if leaves[f - pStart] or len(new.getSupport(f)) != 1:
            continue
        verts = [int(q) - vS for q in new.getTransitiveClosure(f)[0]
                 if vS <= int(q) < vE]
        if all(Xn[v][1] == 1.0 for v in verts):
            if top.getValue(f) < 0:
                n_bare += 1
            if skin.getValue(f) == 5:
                n_band += 1
    n_band = int(comm.allreduce(n_band, op=MPI.SUM))
    n_bare = int(comm.allreduce(n_bare, op=MPI.SUM))
    assert n_band > 0, "the ribbon left no band on the surface"
    assert n_bare == 0, "the relabel left top-wall edges without Top"


def test_a_2d_outcrop_on_the_annulus_embeds_in_parallel():
    """The general-boundary outcrop at np>=2, on the annulus.

    The domain loops are gathered from UNSHARED support-1 edges — a
    partition seam also has local support 1, and taking it would eat the
    mesh at the seam differently at every rank count — so the clip, the
    trace and the splice are functions of the mesh, not of the partition.
    The gathered complex is sorted into a canonical order, so the info
    dict (trace count included) is identical at any rank count; the
    np=1 value is asserted equal by running the serial suite.
    """
    comm = uw.mpi.comm
    mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                              cellSize=0.06, qdegree=2)
    theta = 0.4
    ray = np.array([np.cos(theta), np.sin(theta)])
    line = np.array([0.62 * ray, 1.30 * ray])
    new, info = place_thin_volume(mesh.dm, [line], width=0.03,
                                  label="Zone", label_value=5)
    assert info["n_zone_cells"] > 0
    assert info["n_trace_facets"] > 0, "the ribbon left no trace"
    assert _owned_label_count(new, "Zone", 5) == info["n_zone_cells"]
    assert _owned_label_count(new, "Zone_skin", 5) == info["n_skin_faces"]
    gathered = comm.allgather(info)
    assert all(g == gathered[0] for g in gathered)

    # Every true boundary edge still carries a wall label, and the trace
    # edges among them number exactly what the info advertises. A domain
    # boundary edge is support 1 AND unshared — a partition-seam edge also
    # has local support 1, and counting it would report phantom bare edges
    # (the _true_wall_vertex_mask distinction, met here by the probe).
    from underworld3.utilities.place_surface import _shared_point_flags
    pStart, _pEnd = new.getChart()
    shared = _shared_point_flags(new).astype(bool)
    fS, fE = new.getHeightStratum(1)
    lower = new.getLabel("Lower")
    upper = new.getLabel("Upper")
    trace = new.getLabel("Zone_trace")
    n_trace = n_bare = 0
    for f in range(fS, fE):
        if shared[f - pStart] or len(new.getSupport(f)) != 1:
            continue
        if lower.getValue(f) < 0 and upper.getValue(f) < 0:
            n_bare += 1
        if trace.getValue(f) == 5:
            n_trace += 1
    n_trace = int(comm.allreduce(n_trace, op=MPI.SUM))
    n_bare = int(comm.allreduce(n_bare, op=MPI.SUM))
    assert n_trace == info["n_trace_facets"]
    assert n_bare == 0, "the relabel left boundary edges without a label"


def test_a_3d_outcrop_on_a_rotated_box_embeds_in_parallel():
    """The general 3-D cap at np>=2: no wall axis-aligned.

    The same mesh, rotation and patch as the serial
    test_0860_outcrop_general_boundary_3d, so the trace count is a
    partition-independence check against the serial value by suite. The
    boundary complex is gathered from UNSHARED support-1 faces and sorted
    canonically, so the frame masks, the collar and the info dict are
    functions of the mesh, not of the partition.
    """
    comm = uw.mpi.comm
    a, b = np.deg2rad(20.0), np.deg2rad(15.0)
    Rz = np.array([[np.cos(a), -np.sin(a), 0.0],
                   [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)],
                   [0.0, np.sin(b), np.cos(b)]])
    R = Rz @ Rx
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.12, regular=False, qdegree=2)
    vec = mesh.dm.getCoordinatesLocal()
    vec.array[:] = (vec.array.reshape(-1, 3) @ R.T).reshape(-1)
    mesh.dm.setCoordinatesLocal(vec)
    patch = np.array([[0.3, 0.45, 1.1], [0.7, 0.45, 1.1],
                      [0.72, 0.55, 0.55], [0.32, 0.55, 0.55]]) @ R.T
    new, info = place_thin_volume(mesh.dm, [patch], width=0.05,
                                  label="Zone", label_value=5)
    assert info["n_zone_cells"] > 0
    assert info["n_trace_facets"] > 0, "the zone left no trace"
    assert _owned_label_count(new, "Zone", 5) == info["n_zone_cells"]
    assert _owned_label_count(new, "Zone_skin", 5) == info["n_skin_faces"]
    assert _owned_label_count(new, "Zone_trace", 5) > 0
    gathered = comm.allgather(info)
    assert all(g == gathered[0] for g in gathered)

    # Every true boundary face still carries a wall label — the relabel
    # restored the rotated wall's labels per removed face.
    from underworld3.utilities.place_surface import _shared_point_flags
    pStart, _pEnd = new.getChart()
    shared = _shared_point_flags(new).astype(bool)
    fS, fE = new.getHeightStratum(1)
    walls = [new.getLabel(n)
             for n in ("Top", "Bottom", "Left", "Right", "Front", "Back")]
    n_bare = 0
    for f in range(fS, fE):
        if shared[f - pStart] or len(new.getSupport(f)) != 1:
            continue
        if all(w.getValue(f) < 0 for w in walls):
            n_bare += 1
    n_bare = int(comm.allreduce(n_bare, op=MPI.SUM))
    assert n_bare == 0, "the relabel left boundary faces without a label"


def test_2d_refusals_are_collective():
    """A ribbon stopping short of the wall refuses IDENTICALLY everywhere."""
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.07,
        regular=False, qdegree=2)
    short = np.array([[0.35, 0.40], [0.60, 0.97]])
    message = None
    try:
        place_thin_volume(mesh.dm, [short], width=0.04, label="Bad")
    except (RuntimeError, ValueError) as exc:
        message = str(exc)
    messages = comm.allgather(message)
    assert all(m is not None for m in messages), (
        f"some rank did NOT raise: {[m is None for m in messages]}")
    assert len(set(messages)) == 1, "ranks raised different errors"


def test_the_gather_moves_only_the_shell_around_the_zone():
    """The gather exists for the seam rule: the cells the carve drops,
    their vertex star, and one more layer so the ring's points are
    unshared. That is a shell about three cells thick around the zone,
    and nothing else may move (#670: a distance blanket two cells wide
    ahead of that growth moved 91% of a fixture whose cavity was 6%,
    onto one rank, for good). Measured on this box, np=2 and np=4 alike:
    the shell is 5046 cells against 7269 within three median cell
    diameters, and the old mask moved 9831 (this test fails on it). A
    box eight times the unit fixture, so a shell fits inside it."""
    from underworld3.utilities.edge_split import cell_diameters

    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-0.5, -0.5, -0.5), maxCoords=(1.5, 1.5, 1.5),
        cellSize=0.24, refinement=1, regular=False, qdegree=2)
    width = 0.045
    cells = np.asarray(mesh._cell_node_indices(1, True))
    X = np.asarray(mesh.X.coords)[:, :3]
    centroid = X[cells].mean(axis=1)
    diameters = np.concatenate(comm.allgather(
        np.asarray(cell_diameters(mesh.dm), dtype=float)))
    h_med = float(np.median(diameters))

    def slab_distance(P):
        lo, hi = P.min(axis=0) - 0.5 * width, P.max(axis=0) + 0.5 * width
        return np.linalg.norm(np.maximum(np.maximum(lo - centroid, 0.0),
                                         centroid - hi), axis=1)

    d = np.min([slab_distance(P) for P in CROSS], axis=0)
    within_three = int(comm.allreduce(int((d < 3.0 * h_med).sum()),
                                      op=MPI.SUM))

    new, info = place_thin_volume(mesh.dm, CROSS, width=width,
                                  label="Zone", label_value=5)
    assert info["n_zone_cells"] > 0
    assert all(g == info for g in comm.allgather(info))
    assert info["n_gathered"] <= within_three, (
        f"the gather moved {info['n_gathered']} cells; the shell rule "
        f"allows at most the {within_three} within three cells of the zone")


def test_two_zones_apart_are_two_regions():
    """Two patches a domain apart are two connected components of the
    assembly, so two regions of the gather, each to its own rank
    (#670): the surgeries run concurrently, and the sewn mesh is the
    same one the single-region gather produced (zone, skin and removed
    counts). Measured here at np=2: 304 cells moved where one region
    for the pair moved 8451, and the two owners are different ranks."""
    comm = uw.mpi.comm
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-0.5, -0.5, -0.5), maxCoords=(1.5, 1.5, 1.5),
        cellSize=0.24, refinement=1, regular=False, qdegree=2)
    apart = [np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0],
                       [0.2, 0.0, 0.4], [-0.2, 0.0, 0.4]]),
             np.array([[0.8, 1.0, 0.6], [1.2, 1.0, 0.6],
                       [1.2, 1.0, 1.0], [0.8, 1.0, 1.0]])]
    new, info = place_thin_volume(mesh.dm, apart, width=0.045,
                                  label="Zone", label_value=5)
    assert all(g == info for g in comm.allgather(info))
    assert info["n_regions"] == 2, info
    assert info["n_zone_cells"] > 0
    assert _owned_label_count(new, "Zone", 5) == info["n_zone_cells"]
    assert _owned_label_count(new, "Zone_skin", 5) == info["n_skin_faces"]
    # the two regions do not share cells: what moved is at most one of
    # them, never both (both to one rank would be the old behaviour)
    assert info["n_moved"] < info["n_gathered"]
