"""Injection must reach parent owners that have no local extracted-region node.

Check P1/P2 vector fields, nonzero untouched parent values and both transfer
operations, on a partition where such owners provably exist (the region on
rank 0, its neighbour on the last rank), plus agreeing and disagreeing shared
copies and the rejection tolerance. Global coordinate gathering belongs only to this small independent
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

    # The outer slab (the extracted region, y > 0.85) on rank 0 and the inner
    # slab on the last rank: the interface at y = 0.85 is then shared, and
    # some of its parent DOFs are owned by the inner slab's rank, which has no
    # child cell, so no local copy to write. That is the case the reduction
    # exists for; a split elsewhere can leave it with no such DOF.
    dm = parent.dm.clone()
    dm.setCoordinatesLocal(parent.dm.getCoordinatesLocal())
    start, end = dm.getHeightStratum(0)
    inner, outer = [], []
    for cell in range(start, end):
        _, centre, _ = dm.computeCellGeometryFVM(cell)
        (inner if centre[1] < 0.85 else outer).append(cell - start)
    sizes = [0] * uw.mpi.size
    sizes[0], sizes[-1] = len(outer), len(inner)
    partitioner = dm.getPartitioner()
    partitioner.setType("shell")
    partitioner.setShellPartition(uw.mpi.size, sizes, outer + inner)
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
        assert counts[0] > 0 and all(count == 0 for count in counts[1:])
        # The case must actually arise: parent DOFs owned here, covered by the
        # child, with no child copy on this rank.
        assert uw.mpi.comm.allreduce(_orphans(child, source, target, selected)) > 0


def _orphans(child, source, target, covered):
    """Parent DOFs this rank owns that the child covers somewhere but not here."""
    _, local_parent_rows = child._build_dof_map(target, source)
    iset, field_dm = target.mesh.dm.createSubDM(target.field_id)
    local_section, global_section = field_dm.getLocalSection(), field_dm.getGlobalSection()
    ncomp = target.num_components
    owned = np.zeros(len(covered), dtype=bool)
    for point in range(*local_section.getChart()):
        dof = local_section.getDof(point)
        if dof and global_section.getOffset(point) >= 0:
            offset = local_section.getOffset(point)
            owned[offset // ncomp:(offset + dof) // ncomp] = True
    iset.destroy()
    field_dm.destroy()
    here = np.zeros(len(covered), dtype=bool)
    here[local_parent_rows] = True
    return int((owned & covered & ~here).sum())


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
    assert counts[0] > 0 and all(count == 0 for count in counts[1:])
    assert surface.dim == 1 and surface.cdim == 2
    with pytest.raises(ValueError):
        parent.extract_surface("Top", label_value=99999)


def test_surface_extraction_of_a_missing_label_raises():
    # The parent has no such label at all: a ValueError on every rank, not
    # the PETSc abort a null label handle used to give.
    parent = _thin_slab_parent("default")
    with pytest.raises(ValueError):
        parent.extract_surface("NoSuchLabel", label_value=5)


def _whole_domain_child(name):
    """A child that is the whole parent: in parallel its partition-seam nodes
    are held, with values, by more than one rank."""
    parent = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2, qdegree=2,
    )
    parent.dm.createLabel("WholeDomain")
    start, end = parent.dm.getHeightStratum(0)
    for cell in range(start, end):
        parent.dm.setLabelValue("WholeDomain", cell, 1)
    child = parent.extract_region("WholeDomain", label_value=1)
    source = uw.discretisation.MeshVariable(f"{name}Child", child, 2, degree=1)
    target = uw.discretisation.MeshVariable(f"{name}Parent", parent, 2, degree=1)
    coordinates = np.concatenate(uw.mpi.comm.allgather(np.asarray(source.coords)))
    assert len(np.unique(coordinates.round(12), axis=0)) < len(coordinates)
    return child, source, target


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("mode", ("replace", "add"))
def test_agreeing_shared_child_values_are_injected(mode):
    # Shared copies that agree are averaged, which must give the value itself.
    child, source, target = _whole_domain_child("Agreeing")
    x = np.asarray(source.coords)
    source.array[:, 0, :] = np.column_stack((1 + x[:, 0] + 2 * x[:, 1], 3 - x[:, 0]))
    target.array[:, 0, :] = 7.0
    child.prolongate(source, target, mode=mode)
    y = np.asarray(target.coords)
    expected = np.column_stack((1 + y[:, 0] + 2 * y[:, 1], 3 - y[:, 0]))
    expected = expected + (7.0 if mode == "add" else 0.0)
    local = float(np.max(np.abs(target.array[:, 0, :] - expected), initial=0.0))
    assert uw.mpi.comm.allreduce(local, op=MPI.MAX) < 1e-12


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("mode", ("replace", "add"))
@pytest.mark.parametrize("component", (0, 1))
def test_inconsistent_shared_child_values_are_rejected_collectively(mode, component):
    child, source, target = _whole_domain_child("Inconsistent")
    source.array[:, 0, :] = 2.0
    target.array[:, 0, :] = 7.0
    # Deliberately corrupt local copies without the public array writer's sync.
    source.vec.array.reshape(-1, 2)[:, component] += uw.mpi.rank
    with pytest.raises(ValueError, match="inconsistent.*shared"):
        child.prolongate(source, target, mode=mode)
    np.testing.assert_array_equal(target.array[:, 0, :], 7.0)


@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("perturbation, accepted", [
    (5e-13, True),            # inside rtol = 1e-12
    (5e-11, False),           # outside it
    (np.nan, False),          # not finite
])
def test_the_rejection_tolerance(perturbation, accepted):
    child, source, target = _whole_domain_child(f"Tolerance{str(perturbation)[:5]}")
    source.array[:, 0, :] = 1.0
    target.array[:, 0, :] = 7.0
    copies = source.vec.array.reshape(-1, 2)
    if uw.mpi.rank == uw.mpi.size - 1:
        copies[:, 0] = 1.0 + perturbation if np.isfinite(perturbation) else np.nan
    if accepted:
        child.prolongate(source, target, mode="replace")
    else:
        with pytest.raises(ValueError, match="inconsistent or nonfinite"):
            child.prolongate(source, target, mode="replace")
        np.testing.assert_array_equal(target.array[:, 0, :], 7.0)
