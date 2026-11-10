"""Output and reload of a manifold mesh: triangles in three-space (#869, #870).

* #869: the XDMF writer picked the cell type from the embedding dimension and
  the corner count alone, so the triangles of a ``SphericalManifold`` (three
  corners in three-space) were written as Hexahedron. The cell's own
  dimension is on the cells dataset (``cell_dim``) and decides it now.
* #870: a manifold mesh rebuilt from its checkpoint lost its third
  coordinate (DMPlexCoordinatesLoad left cdim at the topological dimension),
  and ``read_checkpoint``'s default path on a mesh built from a DMPlex
  object, which has no topology SF, failed inside ``sectionLoad`` (a
  segfault when found). The first is fixed; the second refuses and names the
  routes that apply.
"""
import os

import h5py
import numpy as np
import pytest

import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _field(v):
    X = np.asarray(v.coords)
    return X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2]


@pytest.fixture
def written(tmp_path):
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3)
    v = uw.discretisation.MeshVariable("s869", mesh, 1, degree=2)
    v.array[:, 0, 0] = _field(v)
    out = str(tmp_path)
    mesh.write_timestep("sheet", index=0, outputPath=out, meshVars=[v], petsc_reload=True)
    return mesh, v, out


def test_a_manifold_mesh_is_written_to_xdmf_as_triangles_in_three_space(written):
    mesh, v, out = written
    xdmf = open(os.path.join(out, "sheet.mesh.00000.xdmf")).read()
    assert 'TopologyType="Triangle"' in xdmf, [l for l in xdmf.splitlines() if "TopologyType" in l]
    assert 'TopologyType="Hexahedron"' not in xdmf
    assert 'GeometryType="XYZ"' in xdmf
    with h5py.File(os.path.join(out, "sheet.mesh.00000.h5"), "r") as h5:
        geometry = "viz/geometry/vertices" if "viz/geometry" in h5 else "geometry/vertices"
        assert h5[geometry].shape[1] == 3


def test_a_manifold_mesh_rebuilt_from_its_checkpoint_keeps_its_embedding_and_reads_back(written):
    mesh, v, out = written
    rebuilt = uw.discretisation.Mesh(os.path.join(out, "sheet.mesh.00000.h5"))
    assert (rebuilt.dim, rebuilt.cdim) == (2, 3)
    r = np.linalg.norm(np.asarray(rebuilt.X.coords), axis=1)
    assert np.allclose(r, 1.0, atol=1e-10)
    w = uw.discretisation.MeshVariable("s869", rebuilt, 1, degree=2)
    w.read_checkpoint(os.path.join(out, "sheet.mesh.s869.00000.h5"))
    assert np.abs(np.asarray(w.array)[:, 0, 0] - _field(w)).max() < 1e-12


def test_read_checkpoint_on_a_mesh_built_from_a_dm_refuses_and_names_the_routes(written):
    mesh, v, out = written
    path = os.path.join(out, "sheet.mesh.s869.00000.h5")
    v.array[...] = 0.0
    with pytest.raises(RuntimeError, match="same_layout=True"):
        v.read_checkpoint(path)
    v.read_checkpoint(path, same_layout=True)        # the writer, in its own layout
    assert np.abs(np.asarray(v.array)[:, 0, 0] - _field(v)).max() < 1e-12
