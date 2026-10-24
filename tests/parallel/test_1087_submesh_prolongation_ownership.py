"""Injection must reach parent owners that have no local extracted-region node.

Check P1/P2 vector fields, nonzero untouched parent values and both transfer
operations. Global coordinate gathering belongs only to this small independent
test oracle; the production transfer communicates through the parent PETSc DM.
"""

import numpy as np
from mpi4py import MPI
import pytest
from scipy.spatial import cKDTree
import underworld3 as uw


pytestmark = [pytest.mark.level_2, pytest.mark.tier_b]


def _thin_slab_parent(partition: str) -> uw.discretisation.Mesh:
    parent = uw.meshing.BoxInternalBoundary(
        elementRes=(12, 12),
        zelementRes=(10, 2),
        zintCoord=0.85,
        minCoords=(0.0, 0.0),
        maxCoords=(1.0, 1.0),
        simplex=True,
        qdegree=2,
    )
    if partition == "default" or uw.mpi.size == 1:
        return parent

    # Concentrating the outer slab on the last rank makes the empty-child
    # case independent of Gmsh triangulation and the default partitioner.
    dm = parent.dm.clone()
    dm.setCoordinatesLocal(parent.dm.getCoordinatesLocal())
    start, end = dm.getHeightStratum(0)
    lower, upper = [], []
    for cell in range(start, end):
        _, centre, _ = dm.computeCellGeometryFVM(cell)
        (lower if centre[1] < 0.5 else upper).append(cell - start)
    sizes = [0] * uw.mpi.size
    sizes[0], sizes[-1] = len(lower), len(upper)
    partitioner = dm.getPartitioner()
    partitioner.setType("shell")
    partitioner.setShellPartition(uw.mpi.size, sizes, lower + upper)
    migration = dm.distribute()
    concentrated = uw.discretisation.Mesh(dm, boundaries=parent.boundaries, qdegree=2)
    concentrated.regions = parent.regions
    if migration is not None:
        migration.destroy()
    return concentrated


@pytest.mark.parametrize("partition", ("default", "single_rank_region"))
@pytest.mark.parametrize("degree", (1, 2))
@pytest.mark.parametrize("mode", ("replace", "add"))
def test_submesh_injection_preserves_all_parent_copies(degree, mode, partition):
    parent = _thin_slab_parent(partition)
    child = parent.extract_region("Outer")
    source = uw.discretisation.MeshVariable("Child", child, 2, degree=degree)
    target = uw.discretisation.MeshVariable("Parent", parent, 2, degree=degree)
    x = np.asarray(source.coords)
    source.array[:, 0, :] = np.column_stack((1 + x[:, 0] + 2 * x[:, 1], 3 - x[:, 0]))
    target.array[:, 0, :] = 7.0
    child.prolongate(source, target, mode=mode)

    global_coords = np.concatenate(uw.mpi.comm.allgather(x))
    distance, _ = cKDTree(global_coords).query(np.asarray(target.coords))
    selected = distance < 1e-10
    expected = np.full(target.array[:, 0, :].shape, 7.0)
    points = np.asarray(target.coords)[selected]
    values = np.column_stack((1 + points[:, 0] + 2 * points[:, 1], 3 - points[:, 0]))
    expected[selected] = values + (7.0 if mode == "add" else 0.0)
    local = float(np.max(np.abs(target.array[:, 0, :] - expected), initial=0.0))
    assert uw.mpi.comm.allreduce(local, op=MPI.MAX) < 1e-12
    if partition == "single_rank_region" and uw.mpi.size > 1:
        start, end = child.dm.getHeightStratum(0)
        counts = uw.mpi.comm.allgather(end - start)
        assert counts[-1] > 0 and all(count == 0 for count in counts[:-1])


def test_region_reextraction_keeps_empty_ranks_and_cell_sizes():
    parent = _thin_slab_parent("single_rank_region")
    child = parent.extract_region("Outer")
    start, end = child.dm.getHeightStratum(0)
    counts = uw.mpi.comm.allgather(end - start)
    child._re_extract_from_parent()
    start, end = child.dm.getHeightStratum(0)
    assert uw.mpi.comm.allgather(end - start) == counts
    child.cell_size()
    sizes = np.asarray(child._cell_size_variable.array[:, 0, 0])
    assert sizes.size == end - start
    assert np.all(sizes > 0.0)


def test_surface_extraction_keeps_empty_ranks():
    parent = _thin_slab_parent("single_rank_region")
    surface = parent.extract_surface("Top")
    start, end = surface.dm.getHeightStratum(0)
    counts = uw.mpi.comm.allgather(end - start)
    assert counts[-1] > 0 and all(count == 0 for count in counts[:-1])
    assert surface.dim == 1 and surface.cdim == 2
    with pytest.raises(ValueError):
        parent.extract_surface("Top", label_value=99999)
