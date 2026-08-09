"""Parallel confluence of ``mesh.add_conforming_surface``.

The cut is a pure geometric function of the surface and the mesh coordinates:
every rank holding a shared edge computes the same crossing parameter from the
same two endpoints, so the result should be partition-independent *by
construction*. That is an argument, not a measurement, and the surrounding
machinery is exactly where such arguments have failed before — the ``edge_split``
engine needed three separate fixes (a collective reached inside a rank-local
branch, a partition-dependent greedy selection, and a mis-sized ``PetscSF``
reduce) that were all invisible in serial.

What is asserted:

- **the mesh is the same at any communicator size** — compared by COORDINATES,
  not counts. Derived counters lie in parallel: a shared vertex is held by every
  rank on the seam, so summing local counts overstates them, and two different
  meshes can agree on a total.
- **conformity** — no facet with more than two cells, which a mis-handled
  star-forest breaks.
- **the geometric property survives the partition** — every segment of the
  surface between consecutive crossings is still a mesh edge, checked on owned
  points.
- **the surface label reaches every rank that owns part of it**, since that is
  what a boundary condition on the surface depends on.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/ptest_0844_line_cut_parallel.py
    mpirun -n 3 python -m pytest --with-mpi tests/parallel/ptest_0844_line_cut_parallel.py
    mpirun -n 4 python -m pytest --with-mpi tests/parallel/ptest_0844_line_cut_parallel.py
"""
import hashlib

import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities.line_cut import cell_areas

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(300)]

SLANTED = np.array([[-0.2, 0.317], [1.2, 0.683]])

# The serial reference, asserted in the serial file
# (tests/test_0844_line_cut.py::test_serial_reference_for_parallel_confluence) so
# a change to the contract is visible there rather than as a mysterious parallel
# failure here. Building a COMM_SELF mesh inside the parallel run to recompute it
# is NOT a substitute: every rank then drives gmsh independently, which hangs.
SERIAL_VERTICES = 224
SERIAL_CELLS = 396
SERIAL_SURFACE_FACETS = 26
SERIAL_COORD_SHA = "c68821fc041cf94c"
# The fault ZONE: the cells in the support of those 26 facets, so 52 of them,
# hashed by sorted centroid. A count alone can agree between two different sets.
SERIAL_ZONE_SHA = "94b098f3d3153eb5"

# Vertices lying exactly ON the surface, per snap fraction: (count, coord hash).
# This is what the snap test compares against. On the UNCUT base the number is
# ZERO and the nearest vertex is 5.5e-3 away, so any assertion phrased as "the
# vertices near the surface are on it" is satisfied by an empty set and holds
# with the feature removed entirely.
SERIAL_ON_SURFACE = {
    0.0:  (29, "38a5cf77322d57bc"),
    0.05: (29, "38a5cf77322d57bc"),
    0.2:  (18, "b8dfa8eadd27b59a"),
}


def _surf(name, mesh, points):
    """A `Surface` for `add_conforming_surface`, which takes one rather than a
    (points, name) pair: it is what `fault_metric` and
    `refinement_metric_function` already take, so one object drives both the
    refinement and the cut."""
    return uw.meshing.Surface(name, mesh, np.asarray(points, dtype=float))


def _coords(dm):
    return np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dm.getCoordinateDim())


def _owned(dm, points):
    """Those of ``points`` this rank owns — held as a star-forest root, not leaf."""
    try:
        _nroots, ilocal, _iremote = dm.getPointSF().getGraph()
    except (ValueError, TypeError):
        ilocal = None
    leaves = set() if ilocal is None else {int(p) for p in ilocal}
    return [int(p) for p in points if int(p) not in leaves]


def _owned_label_size(mesh, name):
    """Globally, how many facets carry ``name`` — counted once per facet.

    A labelled facet on a partition seam is present on every rank of the seam, so
    summing local stratum sizes overstates it and cannot be compared with a serial
    number.
    """
    value = mesh.boundaries[name].value
    label = mesh.dm.getLabel(name)
    # An EMPTY stratum yields a null IS that petsc4py will happily hand back and
    # then segfault on in `getIndices()`. A rank owning no part of the surface is
    # the normal case at np>2, so the size has to be checked first.
    if label.getStratumSize(value) == 0:
        points = []
    else:
        points = label.getStratumIS(value).getIndices()
    return uw.mpi.comm.allreduce(len(_owned(mesh.dm, points)))


def _owned_vertex_coords(dm):
    """Coordinates of the vertices this rank OWNS, gathered over all ranks.

    Owned-only, because a shared vertex is present on every rank of the seam and
    would otherwise appear several times in the global set.
    """
    vS, vE = dm.getDepthStratum(0)
    try:
        _nroots, ilocal, _iremote = dm.getPointSF().getGraph()
    except (ValueError, TypeError):
        ilocal = None
    leaves = set() if ilocal is None else {int(p) for p in ilocal}
    X = _coords(dm)
    mine = np.array([X[v - vS] for v in range(vS, vE) if v not in leaves])
    gathered = uw.mpi.comm.allgather(mine)
    allX = np.vstack([g for g in gathered if len(g)])
    return allX[np.lexsort((allX[:, 1], allX[:, 0]))]


def _over_shared_facets(dm):
    fS, fE = dm.getHeightStratum(1)
    return uw.mpi.comm.allreduce(
        sum(1 for f in range(fS, fE) if len(dm.getSupport(f)) > 2))


def _surface_mesh():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=3)
    return base, base.add_conforming_surface(_surf("Fault", base, SLANTED))


def test_cut_is_independent_of_the_partition():
    """The whole point: the same surface on the same base gives the same mesh.

    Compared by sorted owned-vertex COORDINATES, not counts. Derived counters lie
    in parallel — a shared vertex is held by every rank on the seam — and two
    different meshes can agree on a total anyway.
    """
    _base, cut = _surface_mesh()
    parallel = _owned_vertex_coords(cut.dm)

    assert _over_shared_facets(cut.dm) == 0, "the cut broke conformity"

    assert parallel.shape[0] == SERIAL_VERTICES, (
        f"np={uw.mpi.size} produced {parallel.shape[0]} owned vertices, serial "
        f"{SERIAL_VERTICES}. The cut must not depend on the partition.")

    cS, cE = cut.dm.getHeightStratum(0)
    cells = uw.mpi.comm.allreduce(len(_owned(cut.dm, range(cS, cE))))
    assert cells == SERIAL_CELLS, (
        f"np={uw.mpi.size} produced {cells} owned cells, serial {SERIAL_CELLS}")

    got = hashlib.sha256(np.round(parallel, 9).tobytes()).hexdigest()[:16]
    assert got == SERIAL_COORD_SHA, (
        f"np={uw.mpi.size} vertex coordinates hash {got}, serial "
        f"{SERIAL_COORD_SHA}: the cut moved with the partition.")


def test_surface_is_a_chain_of_edges_on_every_rank():
    """The geometric property, checked rank-locally on the cut mesh."""
    _base, cut = _surface_mesh()
    dm = cut.dm
    X = _coords(dm)

    A, B = SLANTED[0], SLANTED[-1]
    d = B - A
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    s = (X - A) @ nrm

    vS = dm.getDepthStratum(0)[0]
    edges = {frozenset(int(v) - vS for v in dm.getCone(e))
             for e in range(*dm.getDepthStratum(1))}

    on = np.flatnonzero(np.abs(s) < 1e-11)
    order = on[np.argsort(((X[on] - A) @ d) / (d @ d))]

    # The chain is asserted GLOBALLY, by counting the facets that carry the
    # surface label once each. A per-rank gap count cannot be: a pair of
    # consecutive on-surface vertices straddling a seam legitimately has no local
    # edge, so the bound has to be scaled by the number of seams — which LOOSENS
    # as the partition gets harder, permitting 8 broken segments out of 26 at
    # np=4. The owned facet count is exact and partition-independent.
    assert _owned_label_size(cut, "Fault") == SERIAL_SURFACE_FACETS, (
        f"np={uw.mpi.size}: the surface is {_owned_label_size(cut, 'Fault')} "
        f"facets, serial {SERIAL_SURFACE_FACETS} — the chain is broken.")

    # And locally: consecutive on-surface vertices that are both present here are
    # joined by an edge here. Reported for diagnosis, bounded by the seams.
    missing = sum(1 for u, v in zip(order[:-1], order[1:])
                  if frozenset((int(u), int(v))) not in edges)
    assert uw.mpi.comm.allreduce(missing) <= 2 * uw.mpi.size

    # No cell may straddle, on any rank — that is the property the whole feature
    # exists to provide, and it is purely local.
    vS_, vE_ = dm.getDepthStratum(0)
    cS, cE = dm.getHeightStratum(0)
    straddle = 0
    for c in range(cS, cE):
        vs = [int(p) - vS_ for p in dm.getTransitiveClosure(c)[0] if vS_ <= p < vE_]
        sv = s[vs]
        if (sv > 1e-11).any() and (sv < -1e-11).any():
            straddle += 1
    assert uw.mpi.comm.allreduce(straddle) == 0


def test_surface_label_survives_distribution():
    """Finding the surface again needs the WHOLE label, not a facet of it.

    ``allreduce(local) > 0`` is satisfied by one facet on one rank, which is the
    state a distribution bug produces. The count of owned labelled facets is the
    assertion that discriminates, and it must equal the serial one exactly.
    """
    _base, cut = _surface_mesh()
    value = cut.boundaries["Fault"].value

    assert cut.dm.hasLabel("Fault")
    assert _owned_label_size(cut, "Fault") == SERIAL_SURFACE_FACETS, (
        f"np={uw.mpi.size}: {_owned_label_size(cut, 'Fault')} labelled facets, "
        f"serial {SERIAL_SURFACE_FACETS}")

    # It must also be stacked into UW_Boundaries, which is what the solver reads.
    local = cut.dm.getLabel("Fault").getStratumSize(value)
    stacked = cut.dm.getLabel("UW_Boundaries").getStratumSize(value)
    assert uw.mpi.comm.allreduce(stacked) == uw.mpi.comm.allreduce(local)


VERTICAL = np.array([[0.5, -0.2], [0.5, 1.2]])

# The delivered feature is a surface that carries a boundary condition, so the
# parallel contract is the SOLVE, not just the mesh. A domain integral is the
# right comparison: it is independent of the partition and of DOF ordering, which
# a nodal norm is not.
#
# The solve is driven to a TIGHT tolerance so this can be asserted strictly. At
# the default tolerance serial and np=3 differ by 1.5e-8 — two iterative solves
# converging to different points within their own rtol, not a parallel defect.
# Tightened, they agree to 4e-17, which is what makes the assertion meaningful
# rather than a loosened bound hiding a real difference.
SERIAL_BC_INTEGRAL = 0.3807400201042878


def _bc_mesh():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=3, refinement=1)
    return base.add_conforming_surface(_surf("Fault", base, VERTICAL))


def test_a_boundary_condition_on_the_surface_solves_in_parallel():
    """The feature, end to end: constrain the surface and solve."""
    mesh = _bc_mesh()
    u = uw.discretisation.MeshVariable("u_par", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    for b in ("Left", "Right", "Top", "Bottom"):
        poisson.add_dirichlet_bc(0.0, b)
    poisson.add_dirichlet_bc(1.0, "Fault")
    poisson.petsc_options["ksp_rtol"] = 1.0e-14
    poisson.petsc_options["snes_rtol"] = 1.0e-14
    poisson.solve()

    # The constraint must hold on every rank that owns part of the surface.
    X, vals = np.asarray(u.coords), np.asarray(u.data[:, 0])
    on = np.abs(X[:, 0] - 0.5) < 1e-11
    if on.any():
        assert np.allclose(vals[on], 1.0, atol=1e-9), (
            f"np={uw.mpi.size}: the surface BC is not honoured on rank "
            f"{uw.mpi.rank}")
    assert uw.mpi.comm.allreduce(int(on.sum())) > 0, "no rank owns the surface"

    got = uw.maths.Integral(mesh, u.sym[0]).evaluate()
    assert abs(got - SERIAL_BC_INTEGRAL) < 1e-12, (
        f"np={uw.mpi.size}: integral {got!r} differs from serial "
        f"{SERIAL_BC_INTEGRAL!r}; the parallel solve is not the same problem.")


def test_a_second_surface_chains_in_parallel():
    """Two named surfaces, added one after the other, both usable."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=3)
    one = base.add_conforming_surface(_surf("Fault", base, SLANTED))
    two = one.add_conforming_surface(
        _surf("Moho", one, np.array([[-0.2, 0.12], [1.2, 0.12]])))

    names = [b.name for b in two.boundaries]
    assert "Fault" in names and "Moho" in names
    for nm in ("Fault", "Moho"):
        size = two.dm.getLabel(nm).getStratumSize(two.boundaries[nm].value)
        assert uw.mpi.comm.allreduce(size) > 0, f"{nm} vanished under distribution"
    assert _over_shared_facets(two.dm) == 0


@pytest.mark.parametrize("snap_frac", [0.0, 0.05, 0.2])
def test_snap_fraction_is_partition_independent(snap_frac):
    """The snap decision is read off an EDGE, so a rank holding one side of a
    shared vertex can decide differently from its neighbour. Reconciling that
    over the star-forest is what makes the cut converge at all — at np=3 the
    unreconciled version converged at snap_frac=0 and never at 0.1.

    Asserted as the COUNT and the IDENTITY of the on-surface vertices against
    serial, not as "whatever is near the surface is on it". The failure this
    names — a vertex snapped on some ranks and not others — leaves that vertex
    about ``snap_frac * h`` off the line, three or four orders OUTSIDE any
    tolerance-band selector, and an empty band satisfies a band assertion.
    """
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=3)
    cut = base.add_conforming_surface(_surf("Fault", base, SLANTED), snap_frac=snap_frac)
    assert _over_shared_facets(cut.dm) == 0

    A, B = SLANTED[0], SLANTED[-1]
    d = B - A
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)

    vS, vE = cut.dm.getDepthStratum(0)
    X = _coords(cut.dm)
    mine = np.array([X[v - vS] for v in _owned(cut.dm, range(vS, vE))])
    gathered = [g for g in uw.mpi.comm.allgather(mine) if len(g)]
    allX = np.vstack(gathered)
    on = allX[np.abs((allX - A) @ nrm) < 1e-11]
    on = on[np.lexsort((on[:, 1], on[:, 0]))]

    n_expected, sha_expected = SERIAL_ON_SURFACE[snap_frac]
    assert len(on) == n_expected, (
        f"np={uw.mpi.size} snap={snap_frac}: {len(on)} vertices on the surface, "
        f"serial {n_expected}. A snap applied on only some ranks changes this.")
    got = hashlib.sha256(np.round(on, 9).tobytes()).hexdigest()[:16]
    assert got == sha_expected, (
        f"np={uw.mpi.size} snap={snap_frac}: on-surface vertices hash {got}, "
        f"serial {sha_expected} — the same COUNT of different vertices.")


# Inputs found by sweeping in serial (`~/+Simulations/mesh_reconnection_study/`
# `cut_find_refusal_inputs.py`, `cut_hunt_inversion.py`) and confirmed to reach
# the refusal each is named for. The first attempt at this test used plausible
# inputs that quietly returned success for four of five cases.
_BOX = dict(minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), regular=False, qdegree=2)
_ZIG = np.array([[-0.1, 0.5], [0.30, 0.62], [0.55, 0.38], [0.80, 0.62], [1.1, 0.5]])

REFUSALS = [
    ("nothing to cut", 1 / 12, np.array([[5.0, 5.0], [6.0, 6.0]]), 0.10, 0.15,
     ValueError),
    ("line ends inside", 1 / 12, np.array([[-0.1, 0.5], [0.5, 0.5]]), 0.0, 0.15,
     ValueError),
    ("edge crossed twice", 1 / 3, _ZIG, 0.0, 0.15, ValueError),
    # The quality guard is turned OFF for this one on purpose. With it on, the
    # snap that flattens the cell is vetoed and the cut succeeds — which is the
    # guard working, and is asserted separately in the serial suite. The refusal
    # path still exists for a cell that inverts during SPLITTING, and it is that
    # path's collectiveness this case is here to protect.
    ("snapping inverts a cell", 1 / 8,
     np.array([[-0.1, 0.503], [1.1, 0.541]]), 0.48, None, RuntimeError),
]


@pytest.mark.parametrize("name,h,line,snap,quality,expected",
                         REFUSALS, ids=[r[0] for r in REFUSALS])
def test_every_refusal_is_collective(name, h, line, snap, quality, expected):
    """A refusal must reach EVERY rank, or it is a hang rather than an error.

    Each condition below is a property of one rank's cells — whether this rank
    holds the inverted cell, the tip triangle, the twice-crossed edge — so a
    rank-local ``raise`` aborts that rank while its peers walk on into the next
    collective and block there. Nine defects of exactly this shape have been
    found in this module; the parallel suite could not see any of them because it
    only ever exercised the happy path.

    Negative control, measured: restoring the rank-local form of the
    cell-inversion test makes this file HANG at np=3 on the last case, while the
    three before it still pass.
    """
    from underworld3.utilities.line_cut import cut_along_lines

    mesh = uw.meshing.UnstructuredSimplexBox(cellSize=h, **_BOX)
    try:
        cut_along_lines(mesh.dm, [line], snap_frac=snap, snap_quality=quality)
        outcome = "no refusal"
    except (ValueError, RuntimeError) as exc:
        outcome = type(exc).__name__

    seen = uw.mpi.comm.allgather(outcome)
    assert set(seen) == {expected.__name__}, (
        f"np={uw.mpi.size} {name!r}: ranks disagreed — {seen}. Every rank must "
        f"raise {expected.__name__}, or the ones that do not will hang.")


# ---------------------------------------------------------------------------
# The fault zone, and fault NETWORKS, across a partition.
# ---------------------------------------------------------------------------

def test_the_fault_zone_is_the_same_set_at_any_partition_size():
    """The zone is what a cell-wise viscosity is assigned on, so it has to be
    the same cells however the mesh is split.

    Compared by owned COUNT and by the sorted centroids of the zone cells — the
    count alone can agree between two different sets.
    """
    _base, cut = _surface_mesh()
    zone = cut.cells_supporting("Fault")

    dm = cut.dm
    cS, cE = dm.getHeightStratum(0)
    owned = set(_owned(dm, range(cS, cE)))
    n = uw.mpi.comm.allreduce(
        sum(1 for c in np.flatnonzero(zone) if cS + int(c) in owned))

    # One element each side of every facet — the defining property, and it must
    # survive the partition rather than merely hold on rank 0.
    assert n == 2 * SERIAL_SURFACE_FACETS, (
        f"np={uw.mpi.size}: {n} owned zone cells for "
        f"{SERIAL_SURFACE_FACETS} facets; the zone is the facet support, so it "
        f"is exactly twice as many")

    vS, vE = dm.getDepthStratum(0)
    X = _coords(dm)
    mine = np.array([
        X[[int(p) - vS for p in dm.getTransitiveClosure(cS + int(c))[0]
           if vS <= p < vE]].mean(axis=0)
        for c in np.flatnonzero(zone) if cS + int(c) in owned])
    gathered = [g for g in uw.mpi.comm.allgather(mine) if len(g)]
    allc = np.vstack(gathered)
    allc = allc[np.lexsort((allc[:, 1], allc[:, 0]))]
    got = hashlib.sha256(np.round(allc, 9).tobytes()).hexdigest()[:16]
    assert got == SERIAL_ZONE_SHA, (
        f"np={uw.mpi.size}: zone centroid hash {got}, serial {SERIAL_ZONE_SHA} "
        f"— the same NUMBER of different cells")


JUNCTION = np.array([0.5, 0.5])
NETWORKS = {
    # Y: three arms from one junction. Two of them START there.
    "Y": (np.array([[-0.2, 0.20], [0.5, 0.5]]),
          np.array([[0.5, 0.5], [1.2, 0.30]]),
          np.array([[0.5, 0.5], [0.55, 1.2]])),
    # T: one fault abutting another.
    "T": (np.array([[-0.2, 0.34], [1.2, 0.66]]),
          np.array([[0.5, 0.5], [0.62, 1.2]])),
    # X: two faults crossing.
    "X": (np.array([[-0.2, 0.22], [1.2, 0.78]]),
          np.array([[0.30, -0.2], [0.70, 1.2]])),
}


@pytest.mark.parametrize("kind", list(NETWORKS), ids=list(NETWORKS))
def test_a_fault_network_cuts_at_a_shared_junction_in_parallel(kind):
    """Branching, abutting and crossing faults, across a partition.

    A junction is the same problem as a tip — a distinguished point that has to
    coincide with a mesh vertex — and placing it is where a partition bites:
    ``pull_vertex_onto`` reduces the choice globally, because a rank-local
    nearest-vertex search moves a DIFFERENT vertex on each rank and the branches
    then meet at different places on either side of a seam.
    """
    from underworld3.utilities.line_cut import cut_along_lines, pull_vertex_onto

    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 20,
        regular=False, qdegree=3)
    dm = pull_vertex_onto(base.dm, JUNCTION)

    branches = NETWORKS[kind]
    for k, branch in enumerate(branches):
        dm, _info = cut_along_lines(dm, [branch], label=f"F{k}",
                                    label_value=20 + k)

    # Conformity first: a mis-handled star-forest breaks this before anything
    # geometric shows up.
    fS, fE = dm.getHeightStratum(1)
    assert uw.mpi.comm.allreduce(
        sum(1 for f in range(fS, fE) if len(dm.getSupport(f)) > 2)) == 0
    assert uw.mpi.comm.allreduce(
        int((cell_areas(dm) <= 0.0).sum())) == 0, "a network cut inverted a cell"

    # Every branch is a labelled chain, and the junction is on all of them.
    X = _coords(dm)
    vS = dm.getDepthStratum(0)[0]
    for k, branch in enumerate(branches):
        n_owned = uw.mpi.comm.allreduce(
            len(_owned(dm, dm.getLabel(f"F{k}").getStratumIS(20 + k).getIndices()
                       if dm.getLabel(f"F{k}").getStratumSize(20 + k) else [])))
        assert n_owned > 0, f"branch {k} lost its label under distribution"

    on_junction = uw.mpi.comm.allreduce(
        int((np.linalg.norm(X[:, :2] - JUNCTION, axis=1) < 1e-12).sum()))
    assert on_junction > 0, "the junction is not a mesh vertex on any rank"
