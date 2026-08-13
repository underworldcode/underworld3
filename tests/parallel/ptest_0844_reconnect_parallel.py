"""Reconnection repair under a frozen partition seam.

The parallel contract here is deliberately *not* partition independence. Repair
gives that up by construction: which cavities may be flipped depends on where the
partitioner drew the seam, so the flip set — and therefore the mesh — differs with
rank count. What must hold at every rank count is everything else, and that is
what this file asserts:

* **no shared point's cone changes.** This is the freeze rule stated as a
  postcondition, and it is the load-bearing one. It is what allows the rebuilt DM
  to reuse the point star-forest verbatim instead of reconstructing it by matching
  seam coordinates — a spatial query standing in for an identity lookup, which is
  the failure mode ``nvb._exact_vertex_map`` exists to refuse;
* the **chart, cell count and total area** are invariant, globally;
* the mesh still carries a solve, which is the only real proof the labels and the
  star-forest came through usable rather than merely present.

Run with:
    mpirun -n 2 python -m pytest --with-mpi tests/parallel/ptest_0844_reconnect_parallel.py
    mpirun -n 3 python -m pytest --with-mpi tests/parallel/ptest_0844_reconnect_parallel.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from underworld3.utilities import edge_split, reconnect

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2,
              pytest.mark.tier_b, pytest.mark.timeout(600)]

CENTRE = np.array([0.4, 0.55])


def _refined_dm():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2,
        regular=False, qdegree=2)
    dm = base.dm
    for _ in range(20):
        cS, cE = dm.getHeightStratum(0)
        if cE > cS:
            cen = np.array([dm.computeCellGeometryFVM(c)[1]
                            for c in range(cS, cE)])
            d = np.linalg.norm(cen - CENTRE, axis=1)
            target = np.where(d < 0.25, 0.05, 0.4)
            sel = np.flatnonzero(edge_split.cell_diameters(dm) > target) + cS
        else:
            sel = np.empty(0, dtype=np.int64)
        dm, n = edge_split.bisect_longest_edges(dm, sel)
        if n == 0:
            break
    return dm


def _owned(dm, points):
    try:
        _nroots, ilocal, _iremote = dm.getPointSF().getGraph()
    except (ValueError, TypeError):
        ilocal = None
    leaves = set() if ilocal is None else {int(p) for p in ilocal}
    return [p for p in points if p not in leaves]


def _global(x, op=MPI.SUM):
    return uw.mpi.comm.allreduce(x, op=op)


def _owned_cells_and_area(dm):
    cS, cE = dm.getHeightStratum(0)
    owned = _owned(dm, range(cS, cE))
    area = sum(abs(dm.computeCellGeometryFVM(c)[0]) for c in owned)
    return len(owned), area


def test_shared_points_are_untouched():
    """The freeze rule as a postcondition — the invariant the design rests on."""
    dm = _refined_dm()
    shared = reconnect._shared_points(dm)
    pStart, _pEnd = dm.getChart()
    idx = np.flatnonzero(shared) + pStart
    assert _global(len(idx)) > 0, (
        "no point is shared, so this run cannot exercise the freeze rule")
    before = {int(p): tuple(int(x) for x in dm.getCone(p)) for p in idx}

    out, nflips = reconnect.flip_to_reduce_max_angle(dm)

    assert _global(nflips, op=MPI.MAX) > 0, "nothing flipped anywhere"
    for p, cone in before.items():
        assert tuple(int(x) for x in out.getCone(p)) == cone, (
            f"rank {uw.mpi.rank}: shared point {p} was rewired; the point "
            f"star-forest can no longer be reused verbatim")


def test_geometry_and_conformity_survive():
    dm = _refined_dm()
    chart = dm.getChart()
    ncells, area = _owned_cells_and_area(dm)

    out, _ = reconnect.flip_to_reduce_max_angle(dm)

    assert out.getChart() == chart
    ncells_after, area_after = _owned_cells_and_area(out)
    assert _global(ncells_after) == _global(ncells)
    assert _global(area_after) == pytest.approx(_global(area), rel=1e-12)

    fS, fE = out.getHeightStratum(1)
    assert _global(sum(1 for f in range(fS, fE)
                       if len(out.getSupport(f)) > 2)) == 0


def test_repaired_mesh_still_solves():
    """Labels and the star-forest present is not the same as usable."""
    dm = _refined_dm()
    out, _ = reconnect.flip_to_reduce_max_angle(dm)

    mesh = uw.discretisation.Mesh(out, qdegree=2)
    u = uw.discretisation.MeshVariable("u_par", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "All_Boundaries")
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0

    # Integrating 1 exercises the assembled section over the rebuilt topology on
    # every rank at once, which a rank-local area sum does not.
    one = uw.discretisation.MeshVariable("one_par", mesh, 1, degree=1)
    one.array[:, 0, 0] = 1.0
    assert uw.maths.Integral(mesh, one.sym[0]).evaluate() == pytest.approx(
        1.0, rel=1e-10)


def test_adapt_with_repair_runs_in_parallel():
    """The full ``mesh.adapt(..., repair=True)`` path."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2,
        regular=False, refinement=1, qdegree=3)

    def metric(centroids):
        d = np.linalg.norm(np.asarray(centroids) - CENTRE, axis=1)
        return 1.0 / np.where(d < 0.2, 0.04, 0.15) ** 2

    child = base.adapt(metric, max_levels=2, engine="edge_split", repair=True)

    fS, fE = child.dm.getHeightStratum(1)
    assert _global(sum(1 for f in range(fS, fE)
                       if len(child.dm.getSupport(f)) > 2)) == 0
    # Flips move no vertex, so the exact vertex prolongation must survive.
    assert child._adapt_prolongation and all(
        P is not None for P in child._adapt_prolongation)

    # The cell-parent map must NOT survive a repair, because a flipped cell can
    # straddle two coarse cells and using it would transfer from the wrong
    # parent. Checked at mg_coarsening_ratio=1.0, which is the only setting where
    # the claim is observable: at the default 2.0 a level spans several
    # generations, a cell has no single parent whatever the repair did, and the
    # map is None in BOTH arms — so asserting it there says nothing about repair.
    arms = {}
    for repair in (False, True):
        arm = base.adapt(metric, max_levels=2, engine="edge_split",
                         repair=repair, mg_coarsening_ratio=1.0)
        arms[repair] = arm._adapt_parent_cells

    assert any(pc is not None for pc in arms[False]), (
        "no parent-cell map survived WITHOUT repair, so the assertion below "
        "cannot distinguish repair from anything else")
    assert all(pc is None for pc in arms[True]), (
        "a parent-cell map survived a repair pass; a flipped cell spans two "
        "coarse cells and the transfer would read the wrong parent")

    uw.pprint(f"[ptest_0844] np={uw.mpi.size}: repaired child "
              f"{_global(_owned_cells_and_area(child.dm)[0])} cells")


# ------------------------------------------------------- the removal primitive

def _sf_coordinate_drift(dm):
    """Broadcast every vertex's coordinates root-to-leaf; leaves must agree.

    Deletion compacts the point chart, so the star-forest cannot be reused
    verbatim the way a flip's can: every point after a deleted one shifts, and a
    leaf's *remote* index is a number only its owner holds.
    ``rebuild_cavities`` renumbers locally and broadcasts the new
    numbering once to close that gap.

    This is the check a mis-renumbering cannot pass and nothing else catches.
    Conformity, Euler and area are all rank-local: they stay perfect while the
    forest points at the wrong points, and only a solve — much later — disagrees.
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


def test_removal_renumbers_the_star_forest():
    dm = _refined_dm()
    vS, vE = dm.getDepthStratum(0)
    ncells, area = _owned_cells_and_area(dm)
    assert _global(len(_owned(dm, range(*dm.getChart()))) ) > 0

    out, n = reconnect.remove_vertices(dm, np.arange(vS, vE))

    assert _global(n, op=MPI.MAX) > 0, (
        "nothing was deleted on any rank, so the renumbering is untested")
    assert _sf_coordinate_drift(out) == 0.0, (
        "a leaf no longer resolves to its own coordinates; the compacted chart "
        "was not propagated correctly")

    ncells_after, area_after = _owned_cells_and_area(out)
    assert _global(ncells_after) < _global(ncells), "no cell was removed"
    assert _global(area_after) == pytest.approx(_global(area), rel=1e-12)
    fS, fE = out.getHeightStratum(1)
    assert _global(sum(1 for f in range(fS, fE)
                       if len(out.getSupport(f)) > 2)) == 0


def test_removal_leaves_the_seam_alone():
    """No shared point may be deleted, which is what keeps the leaf set intact.

    ``rebuild_cavities`` renumbers the star-forest but does not rebuild
    it, so a deleted shared point would leave a leaf pointing at nothing. The
    pass freezes any cavity touching the seam; this is that rule as a
    postcondition, checked by coordinates because the numbering has moved.
    """
    dm = _refined_dm()
    shared = reconnect._shared_points(dm)
    pStart, _pEnd = dm.getChart()
    vS, vE = dm.getDepthStratum(0)
    X = np.asarray(dm.getCoordinatesLocal().array).reshape(-1, 2)
    seam = {tuple(X[v - vS]) for v in np.flatnonzero(shared) + pStart
            if vS <= v < vE}
    assert _global(len(seam)) > 0, "no vertex is shared; the rule is untested"

    out, n = reconnect.remove_vertices(dm, np.arange(vS, vE))
    assert _global(n, op=MPI.MAX) > 0

    oS, oE = out.getDepthStratum(0)
    Y = np.asarray(out.getCoordinatesLocal().array).reshape(-1, 2)[: oE - oS]
    survived = {tuple(row) for row in Y}
    assert all(p in survived for p in seam), (
        f"rank {uw.mpi.rank}: a shared vertex was deleted")


def test_reduced_mesh_still_solves():
    """The only real proof the rebuilt forest and labels came through usable."""
    dm = _refined_dm()
    out, n = reconnect.remove_vertices(dm, np.arange(*dm.getDepthStratum(0)))
    assert _global(n, op=MPI.MAX) > 0

    mesh = uw.discretisation.Mesh(out, qdegree=2)
    u = uw.discretisation.MeshVariable("u_del", mesh, 1, degree=1)
    poisson = uw.systems.Poisson(mesh, u_Field=u)
    poisson.constitutive_model = uw.constitutive_models.DiffusionModel
    poisson.constitutive_model.Parameters.diffusivity = 1.0
    poisson.f = 1.0
    poisson.add_dirichlet_bc(0.0, "All_Boundaries")
    poisson.solve()
    assert poisson.snes.getConvergedReason() > 0

    one = uw.discretisation.MeshVariable("one_del", mesh, 1, degree=1)
    one.array[:, 0, 0] = 1.0
    assert uw.maths.Integral(mesh, one.sym[0]).evaluate() == pytest.approx(
        1.0, rel=1e-10)
