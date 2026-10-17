"""The add-on is partitioned once, before any surgery, as one hierarchy.

``partition_addon`` computes one assignment on the finest add-on level — the
fault regions glued and sent to their majority rank, the load balanced by
diffusion over the seams — and applies it to every add-on level through the
parent-cell relation, with provenance riding the move. The contracts asserted
here are the ones the fault machinery and the multigrid transfers build on
(design note ``docs/developer/design/parallel-addon-distribution-2026-10.md``):

* every fault region (the trace's cell star plus two growth layers) lies on
  one rank, so the split's pairs and fans are rank-local by construction;
* every add-on pair is co-partitioned exactly: each fine cell's parent cell on
  the level below is local;
* every cell's recorded parent, fetched by its current address, contains the
  cell's centroid (the relation is exact, on the base boundary too);
* the finest cells per rank are within the balance tolerance of the mean, or
  as close as the seams allow (the diffusion stops when nothing can move).

Then ``add_fault`` on the partitioned add-on: no level is moved by the split,
the split-level transfers report no cross-rank miss, and the solve takes about
as many velocity iterations as in serial.
"""
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import addon_partition as ap
from underworld3.utilities.fault_mg import (_barycentric, _cell_vertex_table,
                                            addon_level_indices)
from underworld3.utilities.place_surface import _region_cells

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900)]

comm = MPI.COMM_WORLD
TIP_A, TIP_B = np.array([0.30, 0.45]), np.array([0.70, 0.55])
TRACE = np.vstack([TIP_A, TIP_B])
H_BASE = 0.05
K = 2
TOLERANCE = 0.1


def _adapted():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, refinement=1,
        qdegree=2)
    pts = np.column_stack([np.linspace(TIP_A[0], TIP_B[0], 9),
                           np.linspace(TIP_A[1], TIP_B[1], 9), np.zeros(9)])
    guide = uw.meshing.Surface("guide", base, pts)
    guide.discretize()
    return base.adapt(guide.refinement_metric_function(
        h_near=H_BASE / 2**K, h_far=H_BASE, width=0.08, profile="linear"),
        max_levels=K)


def _cells(mesh):
    cS, cE = mesh.dm.getHeightStratum(0)
    return cE - cS


def _levels(mesh):
    """``(static, addon_levels_including_mesh)`` of a mesh-owned tail."""
    tail = mesh._custom_mg_coarse_meshes
    idx = addon_level_indices(tail)
    static = [tail[i] for i in range(len(tail)) if i not in idx]
    return static, [tail[i] for i in idx] + [mesh]


def _parents_contain_centroids(below, lev):
    """Fetch each cell's parent (by its CURRENT address) and test containment.
    Returns the global count of cells NOT inside their parent."""
    par = lev._addon_parent
    vxy_b, cv_b = _cell_vertex_table(below)
    nb = cv_b.shape[0]
    verts = np.ascontiguousarray(vxy_b[cv_b].reshape(nb, -1))
    sf = PETSc.SF().create(comm=PETSc.COMM_WORLD)
    sf.setGraph(nb, None, np.ascontiguousarray(par, dtype=np.int32))
    sf.setUp()
    got = np.zeros((par.shape[0], verts.shape[1]))
    unit = MPI.DOUBLE.Create_contiguous(verts.shape[1])
    unit.Commit()
    sf.bcastBegin(unit, verts, got, MPI.REPLACE)
    sf.bcastEnd(unit, verts, got, MPI.REPLACE)
    unit.Free()
    sf.destroy()
    fcent, _v, _c = ap._cell_centroids(lev)
    lam = _barycentric(fcent, got.reshape(par.shape[0], cv_b.shape[1], -1))
    return comm.allreduce(int((lam.min(axis=1) < -1e-9).sum()))


def test_partition_invariants_on_the_adapt_child():
    child = _adapted()
    marks = ap.mark_polylines([TRACE])
    out = ap.partition_addon(child, marks, layers=1, tolerance=TOLERANCE)
    assert out is not child
    record = out._addon_partition
    assert record["regions"] == 1

    # the fault region is rank-interior on the new finest AND on every add-on
    # level (the replay splits each level on its own star)
    static, levels = _levels(out)
    for lev in levels:
        cell_region, _canon, regions = _region_cells(lev.dm, marks(lev.dm), layers=1)
        counts = np.asarray(comm.allgather(
            [int((cell_region == r).sum()) for r in regions]))
        assert all(np.count_nonzero(counts[:, j]) == 1 for j in range(len(regions)))

    # add-on pairs are co-partitioned: parents local, provenance exact
    assert len(levels) == len(_levels(child)[1])
    for l in range(1, len(levels)):
        ap._parent_cells_local(levels[l - 1], levels[l])     # raises if not
        assert _parents_contain_centroids(levels[l - 1], levels[l]) == 0
    assert _parents_contain_centroids(static[-1], levels[0]) == 0
    # a cell of the coarsest add-on level has a remote parent exactly when
    # its unit moved
    remote = comm.allreduce(int((levels[0]._addon_parent[:, 0] != comm.rank).sum()))
    assert remote == record["units_moved"]

    # the finest cells are shared out: within tolerance of the mean, or down
    # to the one indivisible fault region
    load = np.asarray(comm.allgather(_cells(out)), dtype=float)
    mean = load.mean()
    region = max(record["region_weight"].values())
    assert load.max() <= max(mean * (1.0 + TOLERANCE), region) + TOLERANCE * mean, \
        (load, record)
    # the total is conserved on every level
    for a, b in zip(_levels(child)[1], levels):
        assert comm.allreduce(_cells(a)) == comm.allreduce(_cells(b))


def test_partition_runs_once():
    child = _adapted()
    out = ap.partition_addon(child, ap.mark_polylines([TRACE]), layers=1)
    with pytest.raises(NotImplementedError):
        ap.partition_addon(out, ap.mark_polylines([TRACE]))


def _solve(mesh):
    x, y = mesh.X
    stokes = uw.systems.Stokes(mesh)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = 1
    stokes.tolerance = 1.0e-6
    stokes.petsc_use_pressure_nullspace = True
    for side in ("Top", "Bottom", "Left", "Right"):
        stokes.add_dirichlet_bc((y - 0.5, 0.0), side)
    stokes.add_fault_bc(0, "Flt")
    stokes.solve()
    return stokes


def test_add_fault_moves_nothing_after_the_partition_and_converges():
    """``add_fault`` on the adapt child partitions once and then cuts and
    splits in place: the cut child's parent is the partitioned mesh, every
    split level's pairs are whole, the split-level transfers report no
    cross-rank miss, and the solve converges in about the serial count."""
    from underworld3.utilities.fault_split import _fault_labels_touch_seam
    child = _adapted()
    out = child.add_fault([("Flt", TRACE)])
    partitioned = out.parent.parent                 # split <- cut <- partitioned
    assert partitioned._addon_partition is not None
    static, levels = _levels(out)
    for lev in levels:
        assert getattr(lev, "_fault_point_pairs", None), "a level was not split"
        labels = [(f"Flt{side}", int(lev.boundaries[f"Flt{side}"].value))
                  for side in ("Plus", "Minus")]
        assert not _fault_labels_touch_seam(lev.dm, labels)
    stokes = _solve(out)
    info = stokes._rotated_freeslip_info
    assert info["velocity_pc"] == "custom-FMG" and info["converged"], info
    assert info["vel_its_last"][-1] <= 9, info["vel_its_last"]
    stats = stokes._custom_mg["hierarchy"].level_transfer_stats
    # pairs between add-on levels (and into the finest) are co-partitioned:
    # the cell-located transfer finds every node locally
    n_addon = len(levels)
    for st in stats[-(n_addon - 1):] if n_addon > 1 else []:
        assert st is not None and st["misses"] == 0, st
