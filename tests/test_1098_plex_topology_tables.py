"""The compiled DMPlex topology tables answer exactly what DMPlex answers.

``utilities/_plex_topology`` builds, in one C loop per stratum, the tables the
fault, placement and add-on code used to assemble one petsc4py call at a time:
each point's closure in one depth, each point's support, a per-cell section and
its offsets, each cell's dof rows. The baseline is DMPlex itself, queried point
by point: the tables must agree entry for entry, in closure order, on 2-D and
3-D simplex meshes, on whatever partition the run has (the module is rank-local;
run under ``mpirun`` it is checked on every rank's chart).
"""
import numpy as np
import pytest
from petsc4py import PETSc

import underworld3 as uw
from underworld3.utilities import _plex_topology as pt

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


@pytest.fixture(scope="module", params=[2, 3], ids=["2d", "3d"])
def mesh(request):
    if request.param == 2:
        return uw.meshing.UnstructuredSimplexBox(
            minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.1, qdegree=2)
    return uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.3,
        regular=False, qdegree=2)


def _closure_in_depth(dm, p, depth):
    dS, dE = dm.getDepthStratum(depth)
    return [int(q) for q in dm.getTransitiveClosure(p)[0] if dS <= q < dE]


def test_closure_points_match_the_plex(mesh):
    dm = mesh.dm
    cS, cE = dm.getHeightStratum(0)
    fS, fE = dm.getHeightStratum(1)
    for points in (np.arange(cS, cE), np.arange(fS, fE)):
        for depth in range(dm.getDimension()):
            table = pt.closure_points(dm, points, depth)
            for row, p in zip(table, points):
                ref = _closure_in_depth(dm, int(p), depth)
                assert row[row >= 0].tolist() == ref
                assert (row[len(ref):] == -1).all()


def test_supports_match_the_plex(mesh):
    dm = mesh.dm
    for stratum in (dm.getHeightStratum(1), dm.getDepthStratum(0)):
        points = np.arange(*stratum)
        table = pt.supports(dm, points)
        for row, p in zip(table, points):
            assert row[row >= 0].tolist() == [int(q) for q in dm.getSupport(int(p))]


def test_cell_section_and_offsets_match_the_plex(mesh):
    dm = mesh.dm
    cS, cE = dm.getHeightStratum(0)
    cells = np.arange(cS, cE)
    section = PETSc.Section().create(comm=dm.getComm())
    pt.fill_section(section, dm, cells, 3)
    assert section.getChart() == dm.getChart()
    assert section.getStorageSize() == 3 * len(cells)
    offsets = pt.section_offsets(section, cells)
    assert offsets.tolist() == [section.getOffset(int(c)) for c in cells]
    assert sorted(offsets.tolist()) == list(range(0, 3 * len(cells), 3))


def test_closure_dof_rows_match_the_plex(mesh):
    dm = mesh.dm.getCoordinateDM()
    section = mesh.dm.getCoordinateSection()
    cdim = mesh.dm.getCoordinateDim()
    cS, cE = mesh.dm.getHeightStratum(0)
    table = pt.closure_dof_rows(dm, section, np.arange(cS, cE), cdim)
    for row, c in zip(table, range(cS, cE)):
        ref = []
        for q in dm.getTransitiveClosure(c)[0]:
            n = section.getDof(int(q)) // cdim
            if n:
                o = section.getOffset(int(q)) // cdim
                ref.extend(range(o, o + n))
        assert row[row >= 0].tolist() == ref
