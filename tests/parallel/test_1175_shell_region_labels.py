"""Topological shell labels where true-sphere centroid classification is wrong."""

import numpy as np
import pytest
from mpi4py import MPI
from petsc4py import PETSc

import underworld3 as uw

from underworld3.utilities._fitted_regions import _label_fitted_shell_regions

pytestmark = pytest.mark.level_1


def _chordal_shell():
    """A chordal interface and a subdivided outer tetrahedron.

    Before subdivision, the outer tetrahedron has all four vertices on r=1,
    so its centroid lies
    strictly inside the sphere. Wall seeds and the tagged shared facet define
    its side independently of that misleading radius.
    """
    coordinates = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.],
                            [.2, .2, .2], np.ones(3)/np.sqrt(3)])
    coordinates = np.vstack((coordinates, coordinates[[0, 1, 2, 4]].mean(axis=0)))
    cells = np.array([[0, 1, 2, 3], [0, 2, 1, 5], [0, 1, 4, 5],
                      [0, 4, 2, 5], [1, 2, 4, 5]], dtype=PETSc.IntType)
    if PETSc.COMM_WORLD.rank:
        cells = np.empty((0, 4), dtype=PETSc.IntType)
        coordinates = np.empty((0, 3))
    dm = PETSc.DMPlex().createFromCellList(3, cells, coordinates,
                                         interpolate=True, comm=PETSc.COMM_WORLD)
    for name in ("Lower", "Upper", "Internal"):
        dm.createLabel(name)
    section, xyz = dm.getCoordinateSection(), dm.getCoordinatesLocal().array
    vs, ve = dm.getDepthStratum(0)
    fs, fe = dm.getHeightStratum(1)
    for face in range(fs, fe):
        vertices = [int(v) for v in dm.getTransitiveClosure(face)[0] if vs <= v < ve]
        points = np.array([xyz[section.getOffset(v):section.getOffset(v)+3] for v in vertices])
        if np.allclose(points.sum(axis=1), 1):
            dm.getLabel("Internal").setValue(face, 12)
        elif len(dm.getSupport(face)) != 1:
            continue
        elif np.any(np.all(np.isclose(points, [.2, .2, .2]), axis=1)):
            dm.getLabel("Lower").setValue(face, 11)
        else:
            dm.getLabel("Upper").setValue(face, 13)
    dm.markBoundaryFaces("All_Boundaries", 1001)
    sf = dm.distribute()
    if sf:
        sf.destroy()
    return dm


def _classify(dm):
    return _label_fitted_shell_regions(dm, ("Lower", 11), ("Upper", 13),
                                      ("Internal", 12), ("Inner", 101), ("Outer", 102))


@pytest.mark.tier_b
@pytest.mark.parametrize("overlap", [False, True])
def test_spherical_chordal_side_is_topological_not_centroid_radius(overlap):
    dm = _chordal_shell()
    if overlap and PETSc.COMM_WORLD.size > 1:
        sf = dm.distributeOverlap(1)
        if sf:
            sf.destroy()
    try:
        counts = _classify(dm)
        assert counts == dict(inner_cells=1, outer_cells=4)
        cs, ce = dm.getHeightStratum(0)
        for cell in range(cs, ce):
            centroid = dm.computeCellGeometryFVM(cell)[1]
            inner = dm.getLabel("Inner").getValue(cell) == 101
            outer = dm.getLabel("Outer").getValue(cell) == 102
            assert inner != outer
            # All cells have r_centroid<1, including every Outer cell.
            assert np.linalg.norm(centroid) < 1
        fs, fe = dm.getHeightStratum(1)
        for face in range(fs, fe):
            cells = dm.getSupport(face)
            if len(cells) == 2:
                sides = [dm.getLabel("Outer").getValue(int(c)) == 102 for c in cells]
                assert (sides[0] != sides[1]) == (dm.getLabel("Internal").getValue(face) == 12)
    finally:
        dm.destroy()


@pytest.mark.tier_b
@pytest.mark.parametrize("defect", ["missing_interface", "missing_outer_seed", "exterior_interface"])
def test_invalid_fitted_shell_partition_is_rejected_collectively(defect):
    dm = _chordal_shell()
    try:
        fs, fe = dm.getHeightStratum(1)
        if defect == "missing_interface":
            dm.getLabel("Internal").clearStratum(12)
        elif defect == "missing_outer_seed":
            dm.getLabel("Upper").clearStratum(13)
        else:
            for face in range(fs, fe):
                if dm.getLabel("Lower").getValue(face) == 11:
                    dm.getLabel("Internal").setValue(face, 12)
        with pytest.raises(ValueError):
            _classify(dm)
    finally:
        dm.destroy()


def _internal_facets_separating(mesh):
    """(Internal facets with both cells on this rank, those with one Inner and one
    Outer cell), summed over ranks."""
    dm = mesh.dm
    fs, fe = dm.getHeightStratum(1)
    internal = dm.getLabel(mesh.boundaries.Internal.name)
    inner, outer = (dm.getLabel(region.name) for region in (mesh.regions.Inner, mesh.regions.Outer))
    facets = separating = 0
    for face in range(fs, fe):
        cells = dm.getSupport(face)
        if internal.getValue(face) != mesh.boundaries.Internal.value or len(cells) != 2:
            continue
        facets += 1
        sides = sorted((inner.getValue(int(c)) >= 0, outer.getValue(int(c)) >= 0) for c in cells)
        separating += sides == [(False, True), (True, False)]
    return uw.mpi.comm.allreduce(facets, op=MPI.SUM), uw.mpi.comm.allreduce(separating, op=MPI.SUM)


def _owned_inner_cells(mesh):
    dm = mesh.dm
    cs, ce = dm.getHeightStratum(0)
    owned = np.asarray(dm.getCellNumbering().getIndices()) >= 0
    inner = dm.getLabel(mesh.regions.Inner.name)
    local = sum(1 for c in range(cs, ce) if owned[c - cs] and inner.getValue(c) >= 0)
    return uw.mpi.comm.allreduce(local, op=MPI.SUM)


@pytest.mark.tier_b
def test_refined_shell_keeps_each_cell_on_its_side_of_the_internal_surface():
    """Refinement leaves new Internal vertices on the chords, inside the sphere.

    A cell just outside the refined surface then has its centroid below
    ``radiusInternal``, and a radius test labels it Inner: 1363 cells on this
    mesh at two refinements. Two exact properties hold for correct labels.
    Every Internal facet has an Inner cell on one side and an Outer cell on
    the other, and each refinement splits a tetrahedron into eight on the
    same side, so the Inner count is 64 times the unrefined one.
    """
    shell = dict(radiusOuter=1.0, radiusInternal=0.8, radiusInner=0.547, cellSize=0.3, qdegree=2)
    coarse = uw.meshing.SphericalShellInternalBoundary(**shell)
    refined = uw.meshing.SphericalShellInternalBoundary(refinement=2, **shell)

    facets, separating = _internal_facets_separating(refined)
    assert facets > 0
    assert separating == facets
    assert _owned_inner_cells(refined) == 64 * _owned_inner_cells(coarse)
