"""Boundary-slip classification is the serial one on every partition (#1086).

``Mesh.boundary_slip`` reads which walls a vertex lies on from the labelled
facets in its closure. A wall vertex on a partition seam is held by ranks that
do not all hold its wall facet, so each label's mask is combined over the ranks
holding the vertex, with OR: the vertex is on the wall if any of them sees it
there. AND would class such a vertex as interior, and it would leave its wall;
no reduction at all would class a corner whose two walls are on different
ranks as a slip vertex.

On unit boxes (with their registered plane surfaces) the classification is
known from coordinates (natural partitions rarely hold a wall vertex on a
rank without either of its wall facets, so the cells are also dealt to the
ranks at random, where they often do): corners are pinned, other wall vertices slip, and the
rest are free. Checked directly: every vertex is offset, the projection is
applied, and every owned wall vertex must be back on its wall (interior ones
untouched), with exactly the corners pinned. This file carries no
``mpi(min_size=2)`` mark: it runs serially and in the parallel batches.
"""
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a, pytest.mark.timeout(120)]

TOL = 1e-12


def _box(dim, cell_size, scatter):
    """A unit box; with ``scatter``, its cells dealt to the ranks at random by
    a shell partition installed before the Mesh constructor distributes the
    loaded DM (which holds every cell on rank 0)."""
    from petsc4py import PETSc

    dmod = uw.discretisation.discretisation_mesh
    original = dmod._from_plexh5

    def load(*args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, tuple):
            return result
        sf0, dm = result
        cs, ce = dm.getHeightStratum(0)
        owner = np.random.default_rng(scatter).integers(0, uw.mpi.size, ce - cs)
        partitioner = dm.getPartitioner()
        partitioner.setType(PETSc.Partitioner.Type.SHELL)
        partitioner.setShellPartition(
            uw.mpi.size, sizes=np.bincount(owner, minlength=uw.mpi.size).astype(np.int32),
            points=np.argsort(owner, kind="stable").astype(np.int32))
        return sf0, dm

    if scatter is not None and uw.mpi.size > 1:
        dmod._from_plexh5 = load
    try:
        return uw.meshing.UnstructuredSimplexBox(minCoords=(0.0,) * dim,
                                                 maxCoords=(1.0,) * dim, cellSize=cell_size)
    finally:
        dmod._from_plexh5 = original


@pytest.mark.parametrize("scatter", [None, 1086, 1087])
@pytest.mark.parametrize("dim,cell_size", [(2, 0.1), (3, 0.3)])
def test_wall_vertices_slip_and_only_corners_pin(dim, cell_size, scatter):
    from underworld3.meshing.smoothing import _owned_vertex_mask

    mesh = _box(dim, cell_size, scatter)
    ref = np.asarray(mesh.X.coords, dtype=float).copy()
    on_wall = (np.abs(ref) < TOL) | (np.abs(ref - 1.0) < TOL)        # per axis
    walls = on_wall.sum(axis=1)
    owned = np.asarray(_owned_vertex_mask(mesh.dm), dtype=bool)

    # the case under test: wall vertices this rank holds but does not own
    if uw.mpi.size > 1:
        assert uw.mpi.comm.allreduce(int(((walls > 0) & ~owned).sum())) > 0

    is_pinned, project = mesh.boundary_slip(True, reference_coords=ref)
    is_pinned = np.asarray(is_pinned, dtype=bool)
    # pinned exactly where a vertex is on two walls or more (corners, and the
    # edges of a 3-D box)
    wrong_pin = int((is_pinned != (walls >= 2)).sum())
    assert uw.mpi.comm.allreduce(wrong_pin) == 0, wrong_pin

    offset = np.array([0.003, 0.002, 0.0025][:dim])
    moved = project(ref + offset)
    one_wall = owned & (walls == 1)
    back_on_wall = np.abs(np.where(on_wall, moved - np.round(ref), 0.0)).max(axis=1) < TOL
    off_wall = int((one_wall & ~back_on_wall).sum())
    interior = owned & (walls == 0)
    untouched = int((interior & ~np.isclose(moved, ref + offset, rtol=0, atol=TOL).all(axis=1)).sum())
    assert uw.mpi.comm.allreduce(off_wall) == 0, off_wall
    assert uw.mpi.comm.allreduce(untouched) == 0, untouched
