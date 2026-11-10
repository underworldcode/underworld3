"""Exact FE source snapshots remain local beside fitted spherical interfaces."""

import os
from enum import Enum

import numpy as np
import pytest
import underworld3 as uw

from underworld3.utilities._fitted_regions import _label_fitted_shell_regions
from underworld3.utilities._spherical_stokes_gap import _prepare_source_snapshot

pytestmark = pytest.mark.level_2
SPACES = [(0, False), (1, False), (2, False), (1, True), (2, True)]


def _polynomial(points, degree):
    x, y, z = np.moveaxis(points, -1, 0)
    value = np.broadcast_to([1., -2., 3.], points.shape).copy()
    if degree >= 1:
        value += np.stack((2*x-y, 3*y+z, -x+4*z), axis=-1)
    if degree == 2:
        value += np.stack((x*y+z*z, y*y-x*z, x*x+y*z), axis=-1)
    return value


@pytest.fixture(scope="module")
def source_fixture():
    class Boundaries(Enum):
        Centre = 1
        Lower = 11
        Internal = 12
        Upper = 13

    filename = os.environ.get("UW_SPHERICAL_SOURCE_TEST_MESH")
    if filename:
        mesh = uw.discretisation.Mesh(
            filename, degree=1, qdegree=3,
            coordinate_system_type=uw.coordinates.CoordinateSystemType.SPHERICAL,
            boundaries=Boundaries, useMultipleTags=True, useRegions=True,
            markVertices=True, boundary_normals=None,
        )
    else:
        mesh = uw.meshing.SphericalShellInternalBoundary(
            radiusInner=.55, radiusInternal=.775, radiusOuter=1,
            cellSize=1/8, degree=1, qdegree=3,
        )
    _label_fitted_shell_regions(
        mesh.dm, ("Lower", 11), ("Upper", 13), ("Internal", 12),
        ("SourceTestInner", 201), ("SourceTestOuter", 202),
    )
    fields = [uw.discretisation.MeshVariable(
        f"SnapshotSource{index}", mesh, 3, degree=degree, continuous=continuous,
    ) for index, (degree, continuous) in enumerate(SPACES)]
    dm = mesh.dm
    cs, ce = dm.getHeightStratum(0)
    vs, ve = dm.getDepthStratum(0)
    section = dm.getCoordinateSection()
    native = dm.getCoordinatesLocal().array
    vertex_ids = [[int(v) for v in dm.getTransitiveClosure(cell)[0] if vs <= v < ve]
                  for cell in range(cs, ce)]
    vertices = np.array([[native[section.getOffset(v):section.getOffset(v)+3]
                          for v in ids] for ids in vertex_ids])
    centres = vertices.mean(axis=1)
    outer = mesh.cells_labelled("SourceTestOuter", 202)
    offset = centres/4 + (2*outer[:, None]-1)*[4., 5., -6.]
    owned = np.flatnonzero(mesh._get_owned_cells_mask())
    snapshots = []
    for field, (degree, continuous) in zip(fields, SPACES):
        coordinates = np.asarray(field.coords)
        if continuous:
            field.array[:, 0, :] = _polynomial(coordinates, degree)
        else:
            rows = mesh._cell_node_indices(degree, False)
            field.array[rows, 0, :] = _polynomial(coordinates[rows], degree) + offset[:, None]
        snapshots.append(_prepare_source_snapshot(field.sym, vertices, mesh))
    # A snapshot freezes prescribed data; later field changes must not alter it.
    for field in fields:
        field.array[:, 0, :] = 0
    adjacent, probes = [], []
    epsilon = np.array([1e-2, 1e-5, 1e-8])
    for cell in owned:
        for face in dm.getCone(int(cell)+cs):
            if dm.getLabel("Internal").getValue(face) != 12:
                continue
            face_ids = {int(v) for v in dm.getTransitiveClosure(int(face))[0] if vs <= v < ve}
            opposite, = [i for i, v in enumerate(vertex_ids[cell]) if v not in face_ids]
            centre = np.delete(vertices[cell], opposite, axis=0).mean(axis=0)
            adjacent.append(cell)
            probes.append((1-epsilon[:, None])*centre + epsilon[:, None]*vertices[cell, opposite])
    return mesh, fields, snapshots, centres, owned, np.array(adjacent, dtype=int), np.array(probes).reshape(-1, 3, 3), offset


@pytest.mark.tier_b
@pytest.mark.parametrize("index", range(5))
def test_source_snapshot_exact_polynomials_in_unequal_and_empty_batches(source_fixture, monkeypatch, index):
    mesh, fields, snapshots, centres, owned, adjacent, probes, offset = source_fixture
    degree, continuous = SPACES[index]
    snapshot = snapshots[index]
    assert not snapshot.coefficients.flags.writeable

    def forbidden_read(*args, **kwargs):
        raise AssertionError("A prepared source batch must not read a variable or mesh")

    monkeypatch.setattr(type(fields[index]), "array", property(forbidden_read))
    monkeypatch.setattr(type(fields[index]), "coords", property(forbidden_read))
    monkeypatch.setattr(mesh, "_cell_node_indices", forbidden_read)
    empty = snapshot.evaluate(np.empty((0, 3, 3)), np.empty(0, dtype=int))
    assert empty.shape == (0, 3, 3)
    batch_size = 37 + mesh.dm.comm.rank
    for cells, points in ((owned, centres[owned, None]), (adjacent, probes)):
        for start in range(0, len(cells), batch_size):
            selected = cells[start:start+batch_size]
            sample = points[start:start+batch_size]
            expected = _polynomial(sample, degree) + (0 if continuous else offset[selected, None])
            actual = snapshot.evaluate(sample, selected)
            scale = max(1., float(np.max(np.abs(expected), initial=0)))
            np.testing.assert_allclose(actual, expected, rtol=0, atol=4096*np.finfo(float).eps*scale)
