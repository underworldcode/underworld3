"""Regression test for 3D mesh cell orientation in VTK/meshio output.

PETSc DMPlex uses opposite vertex winding to VTK for 3D cells.
mesh_to_pv_mesh() must reorder vertices so that cell Jacobians are
positive (right-handed orientation), otherwise face normals, shading,
clipping, and back-face culling break.

See: https://github.com/underworldcode/underworld3/issues/40
"""

import pytest
import numpy as np
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def _tet_jacobian_sign(verts):
    """Signed volume of tetrahedron. Positive = VTK right-handed."""
    e01 = verts[1] - verts[0]
    e02 = verts[2] - verts[0]
    e03 = verts[3] - verts[0]
    return np.dot(np.cross(e01, e02), e03)


def _hex_jacobian_sign(verts):
    """Trilinear Jacobian sign at vertex 0. Positive = VTK convention."""
    e01 = verts[1] - verts[0]
    e03 = verts[3] - verts[0]
    e04 = verts[4] - verts[0]
    return np.dot(np.cross(e01, e03), e04)


def _get_viz_cell_points(mesh):
    """Replicate the cell extraction and reordering from mesh_to_pv_mesh."""
    cStart, cEnd = mesh.dm.getHeightStratum(0)
    pStart, _ = mesh.dm.getDepthStratum(0)
    n = mesh.element.entities[mesh.dim]

    cell_points_list = []
    for cell_id in range(cStart, cEnd):
        pts = mesh.dm.getTransitiveClosure(cell_id)[0][-n:] - pStart
        cell_points_list.append(pts)

    if mesh.dim == 3:
        if mesh.dm.isSimplex():
            reorder = [0, 2, 1, 3]
        else:
            reorder = [0, 3, 2, 1, 4, 5, 6, 7]
        cell_points_list = [pts[reorder] for pts in cell_points_list]

    return cell_points_list


def test_hex_orientation_3d():
    mesh = uw.meshing.StructuredQuadBox(
        minCoords=(0.0, 0.0, 0.0),
        maxCoords=(1.0, 1.0, 1.0),
        elementRes=(2, 2, 2),
    )
    coords = np.asarray(mesh.X.coords, dtype=np.double)
    cell_points_list = _get_viz_cell_points(mesh)

    for i, pts in enumerate(cell_points_list):
        det = _hex_jacobian_sign(coords[pts])
        assert det > 0, f"Hex cell {i} has negative Jacobian ({det})"


def test_tet_orientation_3d():
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0, 0.0),
        maxCoords=(1.0, 1.0, 1.0),
        cellSize=0.5,
    )
    coords = np.asarray(mesh.X.coords, dtype=np.double)
    cell_points_list = _get_viz_cell_points(mesh)

    for i, pts in enumerate(cell_points_list):
        det = _tet_jacobian_sign(coords[pts])
        assert det > 0, f"Tet cell {i} has negative Jacobian ({det})"


def test_2d_unaffected():
    """Verify 2D meshes are not modified by the 3D reordering."""
    for mesh in [
        uw.meshing.StructuredQuadBox(elementRes=(3, 3)),
        uw.meshing.UnstructuredSimplexBox(),
    ]:
        coords = np.asarray(mesh.X.coords, dtype=np.double)
        cell_points_list = _get_viz_cell_points(mesh)

        for i, pts in enumerate(cell_points_list):
            verts = coords[pts]
            e01 = verts[1] - verts[0]
            e02 = verts[2] - verts[0]
            cross = e01[0] * e02[1] - e01[1] * e02[0]
            assert cross > 0, f"2D cell {i} has wrong winding ({cross})"
