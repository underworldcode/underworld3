"""Navigation kd-tree indexing must survive distribution and deformation.

Regression for issue #360: every moving-mesh workflow (mesh.deform / free
surface / r-adapt) crashed at np>1 with

    cell_point_coords = nav_coords[points - pStart]
    IndexError: index ... is out of bounds for axis 0 ...

in _build_kd_tree_index. Two defects, both fixed together:

1. On manifold meshes (dim != cdim) the nav DM is a 1-cell-overlap clone
   whose local coordinate vector is LARGER than the main DM's. The
   nuke_coords_and_rebuild refresh pushed the main DM's (smaller) local
   vector into the nav DM via setCoordinatesLocal, silently truncating the
   halo coordinates — the very first kd-tree build at np>1 then indexed
   past the end. The fix rebuilds the overlap clone from the updated DM.

2. Vertex plex-points were mapped to coordinate rows with the affine shift
   ``points - pStart``, which is only a layout coincidence. Navigation now
   maps through the coordinate PetscSection offsets
   (Mesh._vertex_coord_rows), valid for any section ordering.

Meaningful at any np; the failure mode is parallel-only (np>=2):

    pixi run mpirun -np 4 python -m pytest tests/test_0059_*.py
"""

import numpy as np
import pytest
import underworld3 as uw

pytestmark = [pytest.mark.level_1, pytest.mark.tier_a]


def test_volume_mesh_deform_and_locate():
    """Repeated deforms of a distributed volume mesh keep point location alive."""
    mesh = uw.meshing.UnstructuredSimplexBox(
        minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0), cellSize=1.0 / 12.0
    )
    uw.discretisation.MeshVariable("T_v360", mesh, 1, degree=2)

    for _ in range(2):
        coords = np.array(mesh.X.coords)
        new_coords = coords.copy()
        new_coords[:, 1] += (
            0.01 * np.sin(np.pi * coords[:, 0]) * np.sin(np.pi * coords[:, 1])
        )
        mesh.deform(new_coords)

    # coordinate-row map must cover the deformed nav coords exactly
    rows = mesh._vertex_coord_rows(mesh.dm)
    assert rows.min() >= 0
    assert rows.max() < mesh._nav_coords.shape[0]

    cells = mesh.get_closest_cells(np.array([[0.5, 0.5]]))
    assert cells[0] >= 0


def test_manifold_mesh_build_deform_and_locate():
    """Manifold (dim != cdim) meshes: construction alone crashed at np>1
    pre-fix (truncated nav-DM coordinates), and deform re-triggers the
    same rebuild path."""
    mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.4)
    uw.discretisation.MeshVariable("T_m360", mesh, 1, degree=1)

    nav_dm = mesh._nav_dm if mesh._nav_dm is not None else mesh.dm
    pStart, pEnd = nav_dm.getDepthStratum(0)
    assert mesh._nav_coords.shape[0] == pEnd - pStart, (
        "nav coords truncated: the nav DM sees "
        f"{pEnd - pStart} vertices but _nav_coords has "
        f"{mesh._nav_coords.shape[0]} rows (issue #360)"
    )

    # radial bulge, applied through the public deform path
    coords = np.array(mesh.X.coords)
    r = np.sqrt((coords**2).sum(axis=1))
    mesh.deform(coords * (1.0 + 0.05 * (coords[:, 2:3] / r[:, None]) ** 2))

    nav_dm = mesh._nav_dm if mesh._nav_dm is not None else mesh.dm
    pStart, pEnd = nav_dm.getDepthStratum(0)
    assert mesh._nav_coords.shape[0] == pEnd - pStart

    # the deformed north pole sits at r = 1.05; locate a nearby point
    cells = mesh.get_closest_cells(np.array([[0.0, 0.0, 1.05]]))
    assert cells[0] >= 0
