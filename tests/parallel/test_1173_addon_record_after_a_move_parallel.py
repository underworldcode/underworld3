"""A fault is partitioned, placed and solved on a base mesh that has moved (#1091).

In parallel, the fault routes (``Mesh.add_fault``, ``place_fault_ribbon_2d``)
start with ``addon_partition.partition_addon``, which records the parent of each
finest cell on the level below and moves the fault's cells onto one rank. A move
of the mesh (``mesh.deform``, the MMPDE mover) changes the finest level's
coordinates and leaves the level below where it was. The partition located each
cell's parent at its moved centroid: it refused wherever that put the parent on
another rank ("N fine cell(s) have no local parent cell on the level below"),
and recorded a wrong parent where it did not. Serial runs skip the partition.

The fixture is a unit box refined twice and moved by one fine cell in the
interior, 0.7 of a cell of the level below. The serial half of the record is in
tests/test_1173_addon_record_after_a_move.py.

    mpirun -n 2 python -m pytest --with-mpi tests/parallel/test_1173_addon_record_after_a_move_parallel.py
"""
import numpy as np
import pytest
from mpi4py import MPI

import underworld3 as uw
from fault_fixtures import (TRACE, children_per_parent, parents_contain_centroids,
                            shear_fault_solve)
from underworld3.utilities import addon_partition as ap
from underworld3.utilities.fault_mg import _cell_vertex_table
from underworld3.utilities.place_surface import place_fault_ribbon_2d

pytestmark = [pytest.mark.mpi(min_size=2), pytest.mark.level_2, pytest.mark.tier_b,
              pytest.mark.timeout(900)]

comm = MPI.COMM_WORLD
H_FINE = 1 / 32
N_RUNGS = 80
# What a degree-2 node of a regularly refined cell weighs on the nodes of its
# parent: the parent's basis at barycentric coordinates in quarters.
NESTED_P2 = np.array([-0.125, 0.0, 0.25, 0.375, 0.5, 0.75, 1.0])


def _shift(X):
    """One fine cell in the interior, nothing on the walls."""
    w = H_FINE * np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1])
    return np.column_stack([w, w])


def _moved_base():
    base = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0, 0), maxCoords=(1, 1), cellSize=1 / 8, regular=True,
        qdegree=2, refinement=2)
    X0 = np.asarray(base.X.coords).copy()
    base.deform(X0 + _shift(X0))
    return base


def _largest(values):
    return comm.allreduce(float(np.abs(values).max()) if values.size else 0.0,
                          op=MPI.MAX)


def test_a_moved_base_is_partitioned_on_the_parents_it_was_refined_from():
    base = _moved_base()
    out = ap.partition_addon(base, ap.mark_polylines([TRACE]), layers=1,
                             tolerance=0.1)
    assert out._addon_partition["units_moved"] > 0 and out is not base
    below = out._addon_below
    # the parents are those of the refinement: four children each
    count = children_per_parent(below, out)
    assert comm.allreduce(int((count != 4).sum())) == 0
    # the fixture moves far enough to tell: where the cells are now, some are
    # outside their parent
    assert parents_contain_centroids(below, out) > 0
    # the geometry the base was made in rode the move vertex by vertex: every
    # vertex of the moved copy is its reference position, shifted
    held = out._addon_reference
    assert held is not None and held[0] is below
    reference = held[1]
    vxy, cv = _cell_vertex_table(out)
    assert _largest(vxy - (reference + _shift(reference))) < 1e-12
    # each child lies inside its parent where it was made
    made_at = reference[cv].mean(axis=1)
    assert parents_contain_centroids(below, out, centroids=made_at) == 0


def test_a_fault_placed_on_a_moved_base_solves_with_the_nested_transfer():
    length = float(np.linalg.norm(TRACE[1] - TRACE[0]))
    spacing = length / N_RUNGS
    trace = TRACE[0] + (np.arange(N_RUNGS + 1) * spacing)[:, None] * (
        TRACE[1] - TRACE[0]) / length
    mesh = place_fault_ribbon_2d(_moved_base(), [("Flt", trace)], 2 * spacing,
                                 margin_rings=1, clearance=0.7, split=True,
                                 mesher="ladder")[0]
    stokes, info, slip = shear_fault_solve(mesh)
    assert info["converged"] and info["velocity_pc"] == "custom-FMG"
    # The counts guard the convergence of the solve and the slip its answer;
    # what the move does to the transfer is asserted on the weights below.
    # Serial, moved or not: 19 pressure iterations at 3 velocity iterations an
    # application, and a peak slip of 0.15947 on the moved base.
    assert info["pres_its_last"][-1] <= 22 and info["vel_its_last"][-1] <= 5
    assert abs(slip - 0.15947) < 5e-4
    # the pair the partition moved apart: the base's finest level, moved, over
    # the level it was refined from. Its transfer is built from the record, and
    # it is the transfer of the refinement, whatever the move did to the nodes.
    hierarchy = stokes._custom_mg["hierarchy"]
    levels = hierarchy.level_meshes
    moved = [l for l in range(1, len(levels))
             if levels[l]._addon_reference is not None
             and getattr(levels[l], "_addon_below", None) is levels[l - 1]
             and getattr(levels[l], "_mg_static", False)]
    assert len(moved) == 1, moved
    pair = moved[0]
    assert hierarchy.level_transfer_stats[pair] is not None
    _indptr, _indices, weights = hierarchy.transfers[pair - 1].getValuesCSR()
    assert comm.allreduce(int(weights.size)) > 0
    assert _largest(np.abs(weights[:, None] - NESTED_P2[None, :]).min(axis=1)) < 1e-9
