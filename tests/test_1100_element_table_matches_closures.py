"""The mesh's element table describes the cells the mesh actually has.

``mesh.element`` says how many faces a cell has (``entities[1]``), how many
vertices (``entities[dim]``) and how many vertices a face has
(``face_entities[dim]``); the face-control-point builders, the geometry audit
and the visualisation slice cell closures by these counts. The table had no
1-D entry, so a 1-D surface mesh (an annulus boundary, ``extract_surface``)
fell through to the tetrahedron entry and read 4 faces and 4 vertices per line
cell; slicing the last 4 points of a line's 3-point closure took the line
itself as a vertex.

The baseline is DMPlex: every cell's cone size, the vertices in its closure,
and the vertices of each face's closure, on a 1-D surface, 2-D triangles and
quadrilaterals, 3-D tetrahedra and a 2-D surface in 3-D.
"""
import numpy as np
import pytest

import underworld3 as uw
from underworld3.utilities._plex_topology import closure_points

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _annulus_surface():
    annulus = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5, cellSize=0.2)
    return annulus.extract_surface("Upper")


def _shell_surface():
    shell = uw.meshing.SphericalShell(radiusOuter=1.0, radiusInner=0.5,
                                      cellSize=0.3)
    return shell.extract_surface("Upper")


MESHES = {
    "line (annulus surface)": _annulus_surface,
    "triangle": lambda: uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=0.2),
    "quadrilateral": lambda: uw.meshing.StructuredQuadBox(elementRes=(4, 3)),
    "tetrahedron": lambda: uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0), maxCoords=(1.0, 1.0, 1.0), cellSize=0.4),
    "triangle in 3-D (shell surface)": _shell_surface,
}


@pytest.mark.parametrize("name", list(MESHES))
def test_element_table_matches_the_cells(name):
    mesh = MESHES[name]()
    dm, el = mesh.dm, mesh.element
    cS, cE = dm.getHeightStratum(0)
    fS, fE = dm.getHeightStratum(1)
    assert cE > cS
    cells = np.arange(cS, cE)
    cone_sizes = {dm.getConeSize(int(c)) for c in cells}
    assert cone_sizes == {el.entities[1]}, (name, cone_sizes, el)
    verts = closure_points(dm, cells, 0)
    assert verts.shape[1] == el.entities[mesh.dim] and (verts >= 0).all(), \
        (name, verts.shape, el)
    face_verts = closure_points(dm, np.arange(fS, fE), 0)
    assert face_verts.shape[1] == el.face_entities[mesh.dim] \
        and (face_verts >= 0).all(), (name, face_verts.shape, el)
