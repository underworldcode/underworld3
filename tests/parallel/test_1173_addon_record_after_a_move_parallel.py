"""A fault is partitioned, placed and solved on a base mesh that has moved (#1091).

In parallel, the fault routes (``Mesh.add_fault``, ``place_fault_ribbon_2d``)
start with ``addon_partition.partition_addon``, which records the parent of each
finest cell on the level below and moves the fault's cells onto one rank. A move
of the mesh (``mesh.deform``, the MMPDE mover, the Laplacian smoother) changes
the finest level's coordinates and leaves the level below where it was. The
partition located each cell's parent at its moved centroid: it refused wherever
that put the parent on another rank ("N fine cell(s) have no local parent cell
on the level below"), and recorded a wrong parent where it did not. Serial runs
skip the partition.

The mesh is moved three ways. ``mesh.deform`` moves it by one fine cell in the
interior, 0.7 of a cell of the level below. ``node_redistribution`` and
``smooth_mesh_interior`` are the movers a campaign drives, and at more than one
rank both write the coordinate vector of the DM, for their halo exchange,
before they hand the move to the mesh: a record of the built grid taken at that
hand-over was the moved grid. The fixture is a unit box refined twice; the
smoother takes the irregular one, as it leaves a regular grid where it is. The
serial half of the record is in tests/test_1173_addon_record_after_a_move.py.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1173_addon_record_after_a_move_parallel.py
"""
import numpy as np
import pytest
import sympy

import underworld3 as uw
from fault_fixtures import (TRACE, children_per_parent,
                            parents_contain_centroids, shear_fault_solve)
from underworld3.utilities import addon_partition as ap
from underworld3.utilities.fault_mg import _cell_vertex_table, reference_vertices
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900)]

comm = uw.mpi.comm
H_FINE = 1 / 32
N_RUNGS = 80
# What a degree-2 node of a regularly refined cell weighs on the nodes of its
# parent: the parent's basis at barycentric coordinates in quarters.
NESTED_P2 = np.array([-0.125, 0.0, 0.25, 0.375, 0.5, 0.75, 1.0])


def _shift(X, size=H_FINE):
    """``size`` in the interior, nothing on the walls."""
    w = size * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    return np.column_stack([w, w])


def _deform(mesh):
    X0 = np.asarray(mesh.X.coords).copy()
    mesh.deform(X0 + _shift(X0))


def _redistribute(mesh):
    x, y = mesh.X
    uw.meshing.node_redistribution(
        mesh, 1 + 6 * sympy.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.15 ** 2))


def _smooth(mesh):
    uw.meshing.smooth_mesh_interior(mesh, n_iters=20, method="laplacian")


# The mover, whether the box is regular, the least it moves a vertex, and the
# pressure iterations and peak slip of the solve on the fault placed after it.
# Measured in serial: a move of 0.0313 (one fine cell), 0.0467 and 0.0161, the
# bound half of each for the two movers, whose grid at another rank count has
# not been measured; and 19, 19 and 15 pressure iterations at 3 velocity
# iterations an application. After mesh.deform the count is 19 in serial and
# at np 2 and 20 at np 3, 4 and 5 (hyperion), and the slip the same at every
# rank count.
MOVERS = [pytest.param(_deform, True, 0.03, 19, 0.159472, id="deform"),
          pytest.param(_redistribute, True, 0.023, 19, 0.159522,
                       id="node_redistribution"),
          pytest.param(_smooth, False, 0.008, 15, 0.159507,
                       id="smooth_mesh_interior")]


def _moved_base(mover, regular):
    """The box, moved, and the vertex coordinates it was built with."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=regular,
        qdegree=2, refinement=2)
    built = _cell_vertex_table(base)[0].copy()
    mover(base)
    return base, built


def _largest(values):
    return max(comm.allgather(float(np.abs(values).max()) if values.size else 0.0))


@pytest.mark.parametrize("mover, regular, moved, pres_its, peak_slip", MOVERS)
def test_a_moved_base_is_partitioned_on_the_parents_it_was_refined_from(
        mover, regular, moved, pres_its, peak_slip):
    base, built = _moved_base(mover, regular)
    # the record is the grid the box was built with, on every rank, whoever
    # wrote the coordinates first
    assert _largest(_cell_vertex_table(base)[0] - built) > moved
    assert _largest(reference_vertices(base, ap._levels_of(base)[0]) - built) == 0.0
    out = ap.partition_addon(base, ap.mark_polylines([TRACE]), layers=1,
                             tolerance=0.1)
    assert out._addon_partition["units_moved"] > 0 and out is not base
    below = out._addon_below
    # the parents are those of the refinement: four children each
    count = children_per_parent(below, out)
    assert comm.allreduce(int((count != 4).sum())) == 0
    # the fixture moves far enough to tell: where the cells are now, some are
    # outside their parent (920 of the 2,048 after the redistribution and 51
    # of the 2,592 after the smoother, in serial)
    assert parents_contain_centroids(below, out) > 0
    held = out._addon_reference
    assert held is not None and held[0] is below
    reference = held[1]
    vxy, cv = _cell_vertex_table(out)
    if mover is _deform:
        # the geometry the base was made in rode the move vertex by vertex:
        # every vertex of the moved copy is its reference position, shifted
        assert _largest(vxy - (reference + _shift(reference))) < 1e-12
    # each child lies inside its parent where it was made
    made_at = reference[cv].mean(axis=1)
    assert parents_contain_centroids(below, out, centroids=made_at) == 0


@pytest.mark.parametrize("mover, regular, moved, pres_its, peak_slip", MOVERS)
def test_a_fault_placed_on_a_moved_base_solves_with_the_nested_transfer(
        mover, regular, moved, pres_its, peak_slip):
    length = float(np.linalg.norm(TRACE[1] - TRACE[0]))
    spacing = length / N_RUNGS
    trace = TRACE[0] + (np.arange(N_RUNGS + 1) * spacing)[:, None] * (
        TRACE[1] - TRACE[0]) / length
    base, _built = _moved_base(mover, regular)
    mesh = place_fault_ribbon_2d(base, [("Flt", trace)], 2 * spacing,
                                 margin_rings=1, clearance=0.7, split=True,
                                 mesher="ladder")[0]
    stokes, info, slip = shear_fault_solve(mesh)
    assert info["converged"] and info["velocity_pc"] == "custom-FMG"
    # The counts guard the convergence of the solve and the slip its answer;
    # what the move does to the transfer is asserted on the weights below. The
    # pressure bound leaves five iterations for the rank count and another
    # host's PETSc build.
    assert info["pres_its_last"][-1] <= pres_its + 5
    assert info["vel_its_last"][-1] <= 4
    assert abs(slip - peak_slip) < 5e-5
    # the pair the partition moved apart: the base's finest level, moved, over
    # the level it was refined from. Its transfer is built from the record, and
    # it is the transfer of the refinement, whatever the move did to the nodes.
    hierarchy = stokes._custom_mg["hierarchy"]
    levels = hierarchy.level_meshes
    pairs = [l for l in range(1, len(levels))
             if levels[l]._addon_reference is not None
             and getattr(levels[l], "_addon_below", None) is levels[l - 1]
             and getattr(levels[l], "_mg_static", False)]
    assert len(pairs) == 1, pairs
    pair = pairs[0]
    assert hierarchy.level_transfer_stats[pair] is not None
    _indptr, _indices, weights = hierarchy.transfers[pair - 1].getValuesCSR()
    assert comm.allreduce(int(weights.size)) > 0
    assert _largest(np.abs(weights[:, None] - NESTED_P2[None, :]).min(axis=1)) < 1e-9


def test_a_second_fault_on_a_moved_adapt_child_reads_the_split_level_where_it_was_made():
    """``Mesh.add_fault`` replaces each add-on level under an adapt child by
    its split copy. The child's reference was taken against the level
    replaced and has to serve against the copy: without it the partition for
    the next fault looks for each cell's parent under the cell as it stands,
    refuses where that is on another rank (96 cells at np 2 on this fixture)
    and records a wrong parent where it is not. The serial half is
    test_a_split_level_takes_over_the_reference_of_the_level_it_replaces."""
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 6, regular=True,
        refinement=1, qdegree=2)
    metric = uw.discretisation.MeshVariable("M", base, 1, degree=1)
    near = np.abs(np.asarray(metric.coords)[:, 1] - 0.5) < 0.2
    metric.array[:, 0, 0] = np.where(near, 48.0, 12.0) ** 2
    # two add-on levels, the finest cells 1/48 across beside the faults
    host = base.adapt(metric, max_levels=2, engine="nvb")
    X = np.asarray(host.X.coords).copy()
    host.deform(X + _shift(X, 1 / 48))
    first = host.add_fault([("Flt", np.array([[0.30, 0.45], [0.70, 0.55]]))])
    held = first._addon_reference
    assert held is not None
    assert any(held[0] is level for level in first._custom_mg_coarse_meshes)
    second = first.add_fault([("F2", np.array([[0.30, 0.64], [0.70, 0.60]]))])
    held = second._addon_reference
    assert held is not None
    assert any(held[0] is level for level in second._custom_mg_coarse_meshes)
    # every level that records its parents: each cell lies inside the parent
    # recorded for it, in the geometry the level was made in
    recorded = [level for level in [*second._custom_mg_coarse_meshes, second]
                if getattr(level, "_addon_parent", None) is not None]
    assert recorded
    astray = 0
    for level in recorded:
        # The parents are addressed on the level as the partition left it
        # (_addon_below). The reference of the finest names the split copy
        # that took that level's place, and holds the same geometry.
        below = level._addon_below
        held = level._addon_reference
        reference = (reference_vertices(level, below) if held is None
                     else held[1])
        vxy, cv = _cell_vertex_table(level)
        made_at = (vxy if reference is None else reference)[cv].mean(axis=1)
        assert parents_contain_centroids(below, level, centroids=made_at) == 0
        astray += parents_contain_centroids(below, level)
    # the fixture moves far enough to tell: where the cells stand, some are
    # outside the parent recorded for them
    assert astray > 0
