"""The add-on is partitioned once, before any surgery, as one hierarchy.

``partition_addon`` computes one assignment on the finest add-on level — the
fault regions glued and sent to their majority rank, the load balanced by
diffusion over the seams — and applies it to every add-on level through the
parent-cell relation, with provenance riding the move. The contracts asserted
here are the ones the fault machinery and the multigrid transfers build on
(design note ``docs/developer/design/parallel-addon-distribution-2026-10.md``):

* every fault region (the trace's cell star plus a growth layer) lies on one
  rank, on EVERY add-on level, so the split's pairs and fans are rank-local by
  construction wherever the faults are replayed;
* every add-on pair is co-partitioned exactly: each fine cell's parent cell on
  the level below is local;
* every cell's recorded parent, fetched by its current address, contains the
  cell's centroid (the relation is exact, on the base boundary too);
* the finest cells per rank are within the balance tolerance of the mean, or
  as close as the fault region allows: the region is one indivisible unit.

Then ``add_fault`` on the partitioned add-on: the cut, the split and the level
replay move nothing (each level's lineage runs back to the partitioned tail),
the split-level transfers report no cross-rank miss, the base boundary is
fetched by provenance, and the solve converges in about the serial count.

Measured 2026-10-17 on this fixture (k = 2): velocity iterations 6 at np 2 and
4 (serial 6; test_1087); finest cells per rank 742 / 618 at np 2.
"""
import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw
from fault_fixtures import TRACE, adapted_2d, shear_fault_solve
from underworld3.utilities import addon_partition as ap
from underworld3.utilities.fault_mg import (_barycentric, _cell_vertex_table,
                                            addon_level_indices)
from underworld3.utilities.place_surface import _region_cells

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900)]

comm = MPI.COMM_WORLD
K = 2
TOLERANCE = 0.1


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
    child = adapted_2d(K)
    marks = ap.mark_polylines([TRACE])
    out = ap.partition_addon(child, marks, layers=1, tolerance=TOLERANCE)
    record = out._addon_partition
    assert record["regions"] == 1
    assert record["units_moved"] > 0 and out is not child

    # the fault region is rank-interior on the new finest AND on every add-on
    # level (the replay splits each level on its own star)
    static, levels = _levels(out)
    for lev in levels:
        cell_region, _canon, regions = _region_cells(lev.dm, marks(lev.dm), layers=1)
        counts = np.asarray(comm.allgather(
            [int((cell_region == r).sum()) for r in regions]))
        assert all(np.count_nonzero(counts[:, j]) == 1 for j in range(len(regions)))

    # add-on pairs are co-partitioned and the recorded parents are exact: an
    # independent fetch of each parent by its current address contains the
    # cell's centroid, on the add-on pairs and on the base boundary
    assert len(levels) == len(_levels(child)[1])
    for l in range(1, len(levels)):
        assert _parents_contain_centroids(levels[l - 1], levels[l]) == 0
        assert comm.allreduce(int((levels[l]._addon_parent[:, 0] != comm.rank).sum())) == 0
    assert _parents_contain_centroids(static[-1], levels[0]) == 0
    # a cell of the coarsest add-on level has a remote parent exactly when
    # its unit moved
    remote = comm.allreduce(int((levels[0]._addon_parent[:, 0] != comm.rank).sum()))
    assert remote == record["units_moved"]

    # the finest cells are shared out: within tolerance of the mean, or down
    # to the one indivisible fault region. The slack beyond the region is the
    # diffusion's own stop rule: a rank stops offering when every lighter
    # neighbour is within tolerance of it, so a load up to tolerance x mean
    # above the region can remain on the owner.
    load = np.asarray(comm.allgather(_cells(out)), dtype=float)
    mean = load.mean()
    region = max(record["region_weight"].values())
    assert load.max() <= max(mean * (1.0 + TOLERANCE), region) + TOLERANCE * mean, \
        (load, record)
    # the total is conserved on every level
    for a, b in zip(_levels(child)[1], levels):
        assert comm.allreduce(_cells(a)) == comm.allreduce(_cells(b))


def test_a_second_partition_keeps_the_first_fault_whole():
    """A split add-on partitioned again for a second fault: the first fault's
    pairs survive the move whole (an interior labelled surface is glued like
    a marked one), both faults' stars are rank-interior, and the base
    boundary's relation is read from the first partition's record rather
    than searched (the pair is no longer co-partitioned)."""
    from underworld3.utilities.fault_split import _fault_labels_touch_seam
    # a third of the box from the first trace: the two regions' shells (the
    # stars plus a layer at the base spacing, about 0.15 each) do not touch,
    # so they stay two regions
    second = np.array([[0.30, 0.12], [0.70, 0.12]])
    split = adapted_2d(K).add_fault([("Flt", TRACE)])
    n_pairs = comm.allreduce(len(split._fault_point_pairs["Flt"]))
    out = ap.partition_addon(split, ap.mark_polylines([second]), layers=1,
                             tolerance=TOLERANCE)
    assert out._addon_partition["regions"] == 2          # the new and the glued
    static, levels = _levels(out)
    assert comm.allreduce(len(out._fault_point_pairs["Flt"])) == n_pairs
    for lev in levels:
        labels = [(f"Flt{side}", int(lev.boundaries[f"Flt{side}"].value))
                  for side in ("Plus", "Minus")]
        assert not _fault_labels_touch_seam(lev.dm, labels)
        cell_region, _c, regions = _region_cells(
            lev.dm, ap.mark_polylines([second])(lev.dm), layers=1)
        counts = np.asarray(comm.allgather(
            [int((cell_region == r).sum()) for r in regions]))
        assert all(np.count_nonzero(counts[:, j]) == 1 for j in range(len(regions)))
    assert _parents_contain_centroids(static[-1], levels[0]) == 0


def test_add_fault_moves_nothing_after_the_partition_and_converges():
    from underworld3.utilities.fault_split import _fault_labels_touch_seam
    child = adapted_2d(K)
    out = child.add_fault([("Flt", TRACE)])
    # split <- cut <- partitioned <- child, for the finest and for every
    # replayed add-on level: nothing was moved after the partition
    partitioned = out.parent.parent
    assert partitioned.parent is child
    assert partitioned._addon_partition["units_moved"] > 0
    static, levels = _levels(out)
    p_static, p_levels = _levels(partitioned)
    assert [s is t for s, t in zip(static, p_static)] == [True] * len(static)
    for lev, pre in zip(levels[:-1], p_levels[:-1]):
        assert lev.parent.parent is pre
    for lev in levels:
        assert getattr(lev, "_fault_point_pairs", None), "a level was not split"
        labels = [(f"Flt{side}", int(lev.boundaries[f"Flt{side}"].value))
                  for side in ("Plus", "Minus")]
        assert not _fault_labels_touch_seam(lev.dm, labels)
    stokes, info, _slip = shear_fault_solve(out)
    assert info["velocity_pc"] == "custom-FMG" and info["converged"], info
    assert info["vel_its_last"][-1] <= 9, info["vel_its_last"]
    stats = stokes._custom_mg["hierarchy"].level_transfer_stats
    # the base boundary (static finest -> first add-on level) is the pair
    # the partition moved apart: fetched by provenance, remote parents
    # exactly where units moved
    boundary = stats[len(static)]
    assert "fetched_remote" in boundary, boundary
    assert boundary["fetched_remote"] > 0
    # every add-on pair is co-partitioned: the cell-located transfer finds
    # every node locally (the old co-location left 14 of 2992 remote)
    for st in stats[len(static) + 1:]:
        assert st is not None and st["misses"] == 0 and "fetched_remote" not in st, st
