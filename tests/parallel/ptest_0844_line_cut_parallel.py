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


def _coords(dm):
    return np.asarray(dm.getCoordinatesLocal().array).reshape(-1, dm.getCoordinateDim())


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
    return base, base.add_conforming_surface(SLANTED, name="Fault")


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

    # Consecutive on-surface vertices that are BOTH local must be joined by a
    # local edge. A pair straddling a partition seam legitimately is not.
    missing = 0
    for u, v in zip(order[:-1], order[1:]):
        if frozenset((int(u), int(v))) not in edges:
            missing += 1
    assert uw.mpi.comm.allreduce(missing) <= 2 * uw.mpi.size, (
        "more gaps in the surface chain than partition seams can explain")

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
    """A boundary condition on the surface needs the label on every owning rank."""
    _base, cut = _surface_mesh()
    value = cut.boundaries["Fault"].value

    assert cut.dm.hasLabel("Fault")
    local = cut.dm.getLabel("Fault").getStratumSize(value)
    assert uw.mpi.comm.allreduce(local) > 0, "the surface label vanished"

    # It must also be stacked into UW_Boundaries, which is what the solver reads.
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
    return base.add_conforming_surface(VERTICAL, name="Fault")


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
    one = base.add_conforming_surface(SLANTED, name="Fault")
    two = one.add_conforming_surface(np.array([[-0.2, 0.12], [1.2, 0.12]]),
                                     name="Moho")

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
    unreconciled version converged at snap_frac=0 and never at 0.1."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1 / 12,
        regular=False, qdegree=3)
    cut = base.add_conforming_surface(SLANTED, name="Fault", snap_frac=snap_frac)
    assert _over_shared_facets(cut.dm) == 0
    # Every vertex NEAR the surface must lie exactly ON it: a snap that only some
    # ranks applied leaves its vertex a hair off, which is how the disagreement
    # shows up geometrically.
    X = _coords(cut.dm)
    A, B = SLANTED[0], SLANTED[-1]
    d = B - A
    nrm = np.array([-d[1], d[0]]) / np.hypot(*d)
    distance = np.abs((X - A) @ nrm)
    near = distance < 1e-6
    worst = float(distance[near].max()) if near.any() else 0.0
    assert uw.mpi.comm.allreduce(worst, op=max) < 1e-12, (
        f"np={uw.mpi.size}: a surface vertex sits {worst:.2e} off the line")
