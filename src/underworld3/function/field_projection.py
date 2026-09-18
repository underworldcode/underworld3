"""Project or prolong MeshVariable data to a different polynomial degree.

Uses a scratch PETSc DM with ``createInterpolation`` — no MeshVariable
is created, the mesh DM is never modified, and all scratch objects are
destroyed before returning.

Typical use cases
-----------------
* Down-sample a P2 field to P1 vertex values for visualisation / XDMF output.
* Prolong a P1 field to P2 DOF values for initialisation.
* Obtain vertex values (degree-1) from any higher-order variable.
* Write vertex values directly to an HDF5 file via PETSc ViewerHDF5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from petsc4py import PETSc

if TYPE_CHECKING:
    from underworld3.discretisation import MeshVariable


def project_to_degree(
    mesh_var: "MeshVariable",
    target_degree: int = 1,
    continuous: bool = True,
    include_ghosts: bool = True,
) -> np.ndarray:
    """Project a MeshVariable to a different polynomial degree.

    Parameters
    ----------
    mesh_var
        Source MeshVariable (any degree, scalar/vector/tensor).
    target_degree
        Polynomial degree of the target space (default 1 = vertex values).
    continuous
        Whether the target space is continuous (default ``True``).
    include_ghosts
        If ``True`` (default), return the full local vector including ghost
        DOFs.  If ``False``, return only the owned partition (suitable for
        parallel HDF5 writing where each rank writes its own slice).

    Returns
    -------
    np.ndarray
        Projected values with shape ``(n_dofs, num_components)``.

    Notes
    -----
    This creates a transient scratch DM, builds the PETSc interpolation
    matrix, applies it, and destroys everything.  The mesh DM and all
    existing MeshVariables are completely untouched.

    For ``target_degree == mesh_var.degree`` the interpolation matrix is
    the identity and the result matches the source data exactly.
    """

    mesh = mesh_var.mesh
    nc = mesh_var.num_components

    # --- scratch DM with a single field at the target degree ---
    options = PETSc.Options()
    prefix = "_fieldproj_"
    options.setValue(f"{prefix}petscspace_degree", target_degree)
    options.setValue(f"{prefix}petscdualspace_lagrange_continuity", continuous)
    options.setValue(f"{prefix}petscdualspace_lagrange_node_endpoints", False)

    fe_target = PETSc.FE().createDefault(
        mesh.dim, nc, mesh.isSimplex, mesh.qdegree, prefix, PETSc.COMM_SELF,
    )

    dm_scratch = mesh.dm.clone()
    dm_scratch.addField(fe_target)
    dm_scratch.createDS()

    # --- source sub-DM for this variable's field ---
    iset, subdm_src = mesh.dm.createSubDM(mesh_var.field_id)

    # --- interpolation matrix ---
    # PETSc: source_subdm.createInterpolation(target_dm) returns a matrix
    # whose mult() maps source global vec → target global vec.
    interp, _scale = subdm_src.createInterpolation(dm_scratch)

    # --- apply ---
    g_src = subdm_src.createGlobalVec()
    subdm_src.localToGlobal(mesh_var._lvec, g_src)

    g_dst = dm_scratch.createGlobalVec()
    interp.mult(g_src, g_dst)

    if include_ghosts:
        l_dst = dm_scratch.createLocalVec()
        dm_scratch.globalToLocal(g_dst, l_dst)
        result = l_dst.array.reshape(-1, nc).copy()
        l_dst.destroy()
    else:
        result = g_dst.array.reshape(-1, nc).copy()

    # --- cleanup ---
    for obj in (interp, g_src, g_dst, dm_scratch, fe_target, subdm_src, iset):
        obj.destroy()
    if _scale is not None:
        _scale.destroy()

    return result


def project_to_vertices(mesh_var: "MeshVariable") -> np.ndarray:
    """Shorthand: project any MeshVariable to P1 (vertex) values.

    Parameters
    ----------
    mesh_var
        Source MeshVariable.

    Returns
    -------
    np.ndarray
        Values at mesh vertices, shape ``(n_vertices, num_components)``.
    """
    return project_to_degree(mesh_var, target_degree=1, continuous=True)


def _repack_tensor_to_paraview(data, vtype, dim):
    """Repack UW3 tensor data to ParaView 9-component (3x3) format.

    The checkpoint path (``/fields/``) stores tensors in UW3's internal
    ``_data_layout`` ordering.  The visualisation path
    (``/vertex_fields/``) must repack to ParaView's expected row-major
    3x3 layout: ``[xx, xy, xz, yx, yy, yz, zx, zy, zz]``.

    UW3 ``_data_layout`` ordering:

    - TENSOR 2D:     ``[xx, xy, yx, yy]``  (row-major, 4 components)
    - SYM_TENSOR 2D: ``[xx, yy, xy]``       (3 unique components)
    - TENSOR 3D:     ``[xx, xy, xz, yx, yy, yz, zx, zy, zz]``
                     (row-major, already ParaView-compatible)
    - SYM_TENSOR 3D: ``[xx, yy, zz, xy, xz, yz]``  (6 unique components)

    .. note::

       The original PR #69 assumed 2D TENSOR stored ``[xx, yy, xy, yx]``.
       This corrected version uses the actual ``_data_layout`` row-major
       ordering ``[xx, xy, yx, yy]``.
    """
    import underworld3 as uw

    n = data.shape[0]
    ncomp = data.shape[1]
    out = np.zeros((n, 9), dtype=data.dtype)

    if vtype == uw.VarType.TENSOR:
        if dim == 2 and ncomp == 4:
            # [xx, xy, yx, yy] → [xx, xy, 0, yx, yy, 0, 0, 0, 0]
            out[:, 0] = data[:, 0]  # xx
            out[:, 1] = data[:, 1]  # xy
            out[:, 3] = data[:, 2]  # yx
            out[:, 4] = data[:, 3]  # yy
        elif dim == 3 and ncomp == 9:
            out[:, :] = data[:, :]
        else:
            return data  # unknown layout, pass through
    elif vtype == uw.VarType.SYM_TENSOR:
        if dim == 2 and ncomp == 3:
            # [xx, yy, xy] → [xx, xy, 0, xy, yy, 0, 0, 0, 0]
            out[:, 0] = data[:, 0]  # xx
            out[:, 1] = data[:, 2]  # xy
            out[:, 3] = data[:, 2]  # xy (symmetric)
            out[:, 4] = data[:, 1]  # yy
        elif dim == 3 and ncomp == 6:
            # [xx, yy, zz, xy, xz, yz] → full 3x3
            out[:, 0] = data[:, 0]  # xx
            out[:, 4] = data[:, 1]  # yy
            out[:, 8] = data[:, 2]  # zz
            out[:, 1] = data[:, 3]  # xy
            out[:, 3] = data[:, 3]  # yx = xy
            out[:, 2] = data[:, 4]  # xz
            out[:, 6] = data[:, 4]  # zx = xz
            out[:, 5] = data[:, 5]  # yz
            out[:, 7] = data[:, 5]  # zy = yz
        else:
            return data
    else:
        # MATRIX or unknown — pass through
        return data

    return out


def _write_vec_to_group(viewer, data_array, name, group, comm):
    """Write a numpy array as a standalone PETSc Vec under an HDF5 group.

    DM-associated Vecs ignore ``pushGroup`` because the DM's HDF5 writer
    pushes its own ``/fields/`` prefix.  This helper creates a plain Vec
    (no DM) so ``pushGroup`` is respected.

    When *data_array* is 2D ``(N, ncomp)``, the Vec block size is set to
    *ncomp* so the HDF5 dataset is written as ``(N, ncomp)`` rather than
    flat ``(N*ncomp,)``.

    Parameters
    ----------
    viewer
        An open ``PETSc.ViewerHDF5``.
    data_array : np.ndarray
        Local (owned) data to write — shape ``(n_local,)`` or
        ``(n_local, ncomp)``.
    name : str
        Dataset name in the HDF5 group.
    group : str
        HDF5 group path (e.g. ``/vertex_fields``).
    comm
        MPI communicator.
    """
    vec = PETSc.Vec().createWithArray(data_array.ravel(), comm=comm)
    if data_array.ndim == 2 and data_array.shape[1] > 1:
        vec.setBlockSize(data_array.shape[1])
    vec.setName(name)
    viewer.pushGroup(group)
    viewer(vec)
    viewer.popGroup()
    vec.destroy()


def _write_index_array_to_group(viewer, data_array, name, group, comm):
    """Write a distributed integer connectivity array to an HDF5 group."""
    indices = PETSc.IS().createGeneral(
        np.asarray(data_array, dtype=PETSc.IntType).reshape(-1), comm=comm
    )
    if data_array.ndim == 2:
        indices.setBlockSize(data_array.shape[1])
    indices.setName(name)
    viewer.pushGroup(group)
    viewer(indices)
    viewer.popGroup()
    indices.destroy()


def _physical_visualisation_enabled(units):
    """Return whether declared units require a physical XDMF copy."""
    import underworld3 as uw

    return (
        units is not None
        and uw.get_default_model().has_units_active()
        and uw.is_nondimensional_scaling_active()
    )


def _physical_visualisation_values(data, units):
    """Return output values and their declared unit label.

    Solver and checkpoint vectors use model magnitudes.  XDMF arrays are a
    user-facing boundary, so an active nondimensional model is converted to
    the units declared by the mesh or variable before those arrays are written.
    """
    import underworld3 as uw

    if units is None:
        return data, None

    target_units = uw.units(units).units if isinstance(units, str) else units
    unit_label = str(target_units)
    if not _physical_visualisation_enabled(units):
        return data, unit_label

    dimensionality = dict(target_units.dimensionality)
    if not dimensionality:
        return data, unit_label

    physical = uw.dimensionalise(
        np.asarray(data),
        target_dimensionality=dimensionality,
    ).to(target_units)
    return np.asarray(physical), unit_label


def write_vertices_to_viewer(
    mesh_var: "MeshVariable",
    viewer: "PETSc.ViewerHDF5",
    group: str = "/vertex_fields",
    name: str | None = None,
) -> None:
    """Project a MeshVariable to P1 vertex values and write via PETSc ViewerHDF5.

    For P1 continuous variables, the existing global vector data is
    written directly (DOFs already match mesh vertices).  For P2+
    continuous variables, a scratch DM and PETSc interpolation matrix
    project to degree 1.

    Tensor variables (TENSOR, SYM_TENSOR) are repacked from UW3's
    internal ``_data_layout`` ordering to ParaView's 9-component
    row-major 3x3 format.  The checkpoint data in ``/fields/`` is
    unchanged — only the visualisation copy is repacked.

    Data is written as a standalone Vec (no DM) so that ``pushGroup``
    is respected — DM-associated Vecs would be redirected to ``/fields/``
    by the DMPlex HDF5 writer.

    The vector is written under *group* (default ``/vertex_fields``)
    with the dataset name *name* (default ``<clean_name>_<clean_name>``
    to match the existing XDMF convention).

    Parameters
    ----------
    mesh_var
        Source MeshVariable (any degree, scalar/vector/tensor).
    viewer
        An open ``PETSc.ViewerHDF5`` in append or write mode.
    group
        HDF5 group path to write into.
    name
        Dataset name.  Defaults to ``<clean_name>_<clean_name>``.
    """

    if name is None:
        name = f"{mesh_var.clean_name}_{mesh_var.clean_name}"

    mesh = mesh_var.mesh
    nc = mesh_var.num_components
    is_p1 = mesh_var.continuous and mesh_var.degree == 1

    if is_p1:
        # DOFs = vertices — use gvec data directly (already owned-only)
        mesh_var._sync_lvec_to_gvec()
        data = mesh_var._gvec.array.reshape(-1, nc).copy()
    else:
        # include_ghosts=False → owned partition only, suitable for
        # parallel HDF5 writing where each rank writes its own slice.
        data = project_to_degree(
            mesh_var, target_degree=1, continuous=True, include_ghosts=False,
        )

    # Repack tensors to ParaView 9-component format for visualisation.
    # The projected data uses the same _data_layout as the source variable.
    import underworld3 as uw

    is_tensor = hasattr(mesh_var, "vtype") and mesh_var.vtype in (
        uw.VarType.TENSOR,
        uw.VarType.SYM_TENSOR,
    )
    if is_tensor:
        data = _repack_tensor_to_paraview(data, mesh_var.vtype, mesh.dim)

    data, _ = _physical_visualisation_values(data, mesh_var.units)

    _write_vec_to_group(viewer, data, name, group, PETSc.COMM_WORLD)


def write_field_to_viewer(
    mesh_var: "MeshVariable",
    viewer: "PETSc.ViewerHDF5",
    group: str,
    name: str,
) -> None:
    """Write a variable's owned values in physical units without projection."""
    mesh_var._sync_lvec_to_gvec()
    data = mesh_var._gvec.array.reshape(-1, mesh_var.num_components).copy()
    data, _ = _physical_visualisation_values(data, mesh_var.units)
    _write_vec_to_group(viewer, data, name, group, PETSc.COMM_WORLD)


def write_field_coordinates_to_viewer(
    mesh_var: "MeshVariable",
    viewer: "PETSc.ViewerHDF5",
    group: str,
    name: str = "coordinates",
) -> None:
    """Write owned coordinates for a variable's exact finite-element layout."""
    mesh = mesh_var.mesh
    coordinate_dm = mesh._basis_coordinate_dm(mesh_var.degree, mesh_var.continuous)
    local = coordinate_dm.getLocalVec()
    global_vector = coordinate_dm.getGlobalVec()
    local.array[...] = np.asarray(mesh_var.coords_nd).reshape(-1)
    coordinate_dm.localToGlobal(local, global_vector, addv=False)
    coordinates = global_vector.array.reshape(-1, mesh.cdim).copy()
    coordinates, _ = _physical_visualisation_values(coordinates, mesh.units)
    _write_vec_to_group(viewer, coordinates, name, group, PETSc.COMM_WORLD)
    coordinate_dm.restoreGlobalVec(global_vector)
    coordinate_dm.restoreLocalVec(local)
    coordinate_dm.destroy()


def write_projected_field_to_viewer(
    mesh_var: "MeshVariable",
    viewer: "PETSc.ViewerHDF5",
    target_degree: int,
    continuous: bool,
    group: str,
    name: str,
) -> None:
    """Write a compact visualization projection for an unsupported layout."""
    data = project_to_degree(
        mesh_var,
        target_degree=target_degree,
        continuous=continuous,
        include_ghosts=False,
    )
    data, _ = _physical_visualisation_values(data, mesh_var.units)
    _write_vec_to_group(viewer, data, name, group, PETSc.COMM_WORLD)


def write_p2_simplex_topology_to_viewer(mesh_var, viewer, group="/fields"):
    """Write VTK-ordered quadratic triangle or tetrahedron connectivity."""
    mesh = mesh_var.mesh
    if not (
        mesh_var.continuous
        and mesh_var.degree == 2
        and mesh.isSimplex
        and mesh.dim in (2, 3)
        and mesh.cdim == mesh.dim
    ):
        raise NotImplementedError(
            "direct P2 XDMF requires a full-dimensional triangle or tetrahedron mesh"
        )

    coordinate_dm = mesh._basis_coordinate_dm(2, True)
    local_section = coordinate_dm.getLocalSection()
    global_section = coordinate_dm.getGlobalSection()
    local_to_global = np.full(len(mesh_var.coords_nd), -1, dtype=PETSc.IntType)
    point_start, point_end = local_section.getChart()
    for point in range(point_start, point_end):
        node_count = local_section.getDof(point) // mesh.cdim
        if node_count == 0:
            continue
        local_offset = local_section.getOffset(point) // mesh.cdim
        global_offset = global_section.getOffset(point)
        if global_offset < 0:
            global_offset = -(global_offset + 1)
        global_offset //= mesh.cdim
        local_to_global[local_offset : local_offset + node_count] = np.arange(
            global_offset, global_offset + node_count, dtype=PETSc.IntType
        )
    coordinate_dm.destroy()

    cell_start, cell_end = mesh.dm.getHeightStratum(0)
    owned = np.ones(cell_end - cell_start, dtype=bool)
    if mesh.dm.comm.getSize() > 1:
        _, leaves, remote = mesh.dm.getPointSF().getGraph()
        if leaves is None:
            leaves = np.arange(len(remote))
        leaves = np.asarray(leaves)
        owned[leaves[(leaves >= cell_start) & (leaves < cell_end)] - cell_start] = False

    corner_count = mesh.dim + 1
    node_count = 6 if mesh.dim == 2 else 10
    p2_rows = mesh._cell_node_indices(2, True).reshape(-1, node_count)[owned]
    vertex_rows = mesh._cell_node_indices(1, True).reshape(-1, corner_count)[owned]
    p2_coordinates = mesh_var.coords_nd[p2_rows]
    corners = mesh._get_coords_for_basis(1, True)[vertex_rows]
    negative = np.linalg.det((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1)) < 0
    if mesh.dim == 2:
        corners[negative] = corners[negative][:, [0, 2, 1]]
        edge_pairs = ((0, 1), (1, 2), (2, 0))
        topology_name = "Triangle_6"
    else:
        corners[negative] = corners[negative][:, [0, 2, 1, 3]]
        edge_pairs = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
        topology_name = "Tetrahedron_10"

    connectivity = np.empty_like(p2_rows, dtype=PETSc.IntType)
    for cell_index, (row, nodes, vertices) in enumerate(
        zip(p2_rows, p2_coordinates, corners, strict=True)
    ):
        targets = np.vstack(
            (vertices, *(0.5 * (vertices[a] + vertices[b]) for a, b in edge_pairs))
        )
        order = [np.argmin(np.linalg.norm(nodes - target, axis=1)) for target in targets]
        if len(set(order)) != node_count:
            raise RuntimeError(f"could not map UW3 P2 nodes to {topology_name} ordering")
        connectivity[cell_index] = local_to_global[row[order]]
    if np.any(connectivity < 0):
        raise RuntimeError("P2 XDMF connectivity contains an unmapped global node")
    _write_index_array_to_group(
        viewer, connectivity, "cells", group, PETSc.COMM_WORLD
    )


def write_coordinates_to_viewer(
    mesh,
    viewer: "PETSc.ViewerHDF5",
    group: str = "/vertex_fields",
    name: str = "coordinates",
) -> None:
    """Write mesh vertex coordinates via PETSc ViewerHDF5.

    Parameters
    ----------
    mesh
        Source mesh.
    viewer
        An open ``PETSc.ViewerHDF5`` in append or write mode.
    group
        HDF5 group path to write into.
    name
        Dataset name (default ``coordinates``).
    """
    coord_gvec = mesh.dm.getCoordinates()
    coords = coord_gvec.array.reshape(-1, mesh.cdim).copy()
    coords, _ = _physical_visualisation_values(coords, mesh.units)
    _write_vec_to_group(viewer, coords, name, group, PETSc.COMM_WORLD)


def write_cell_field_to_viewer(
    mesh_var: "MeshVariable",
    viewer: "PETSc.ViewerHDF5",
    group: str = "/cell_fields",
    name: str | None = None,
) -> None:
    """Write a cell (discontinuous/DG-0) variable via PETSc ViewerHDF5.

    Data is written as a standalone Vec (no DM) so that ``pushGroup``
    is respected.

    Parameters
    ----------
    mesh_var
        Source MeshVariable (discontinuous or degree 0).
    viewer
        An open ``PETSc.ViewerHDF5`` in append or write mode.
    group
        HDF5 group path to write into.
    name
        Dataset name.  Defaults to ``<clean_name>_<clean_name>``.
    """

    if name is None:
        name = f"{mesh_var.clean_name}_{mesh_var.clean_name}"

    nc = mesh_var.num_components
    mesh_var._sync_lvec_to_gvec()
    data = mesh_var._gvec.array.reshape(-1, nc).copy()
    data, _ = _physical_visualisation_values(data, mesh_var.units)
    _write_vec_to_group(viewer, data, name, group, PETSc.COMM_WORLD)


def _write_dg1_to_viewer(
    mesh_var,
    viewer,
    group="/dg1",
    coordinate_name="vertices",
    value_name="values",
    repack_tensors=True,
):
    """Write owned simplex cells with independent vertices and DG1 traces.

    Coordinate-section cell maps preserve element ownership and node ordering;
    no point location, coordinate matching, or inter-element averaging is used.
    Interior DG interpolation nodes define an affine polynomial, evaluated at
    that same cell's vertices. Native checkpoint vectors are untouched.
    """
    mesh = mesh_var.mesh
    if (
        mesh_var.continuous or mesh_var.degree != 1 or not mesh.isSimplex
        or mesh.dim not in (2, 3) or mesh.cdim != mesh.dim
    ):
        raise NotImplementedError("DG1 XDMF requires a full-dimensional triangle/tetrahedron mesh")
    cstart, cend = mesh.dm.getHeightStratum(0)
    owned = np.ones(cend - cstart, dtype=bool)
    # Serial DMPlex may have an unset point SF; no cells are ghosts there.
    if mesh.dm.comm.getSize() > 1:
        _, leaves, remote = mesh.dm.getPointSF().getGraph()
        if leaves is None:
            leaves = np.arange(len(remote))
        leaves = np.asarray(leaves)
        owned[leaves[(leaves >= cstart) & (leaves < cend)] - cstart] = False
    rows = mesh._cell_node_indices(1, False).reshape(-1, mesh.dim + 1)[owned]
    vertex_rows = mesh._cell_node_indices(1, True).reshape(-1, mesh.dim + 1)[owned]
    corners = mesh._get_coords_for_basis(1, True)[vertex_rows]
    # Closure order is arbitrary; give VTK positively oriented simplices.
    negative = np.linalg.det((corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1)) < 0
    corners[negative] = corners[negative][:, [0, 2, 1] if mesh.dim == 2 else [0, 2, 1, 3]]
    nodes = mesh_var.coords_nd[rows]
    coefficients = mesh_var._lvec.array.reshape(-1, mesh_var.num_components)[rows]
    matrix = (nodes[:, 1:] - nodes[:, :1]).transpose(0, 2, 1)
    local = np.linalg.solve(matrix, (corners - nodes[:, :1]).transpose(0, 2, 1))
    weights = np.concatenate((1 - local.sum(axis=1, keepdims=True), local), axis=1)
    values = np.einsum("cij,cik->cjk", weights, coefficients).reshape(-1, mesh_var.num_components)
    if repack_tensors:
        values = _repack_tensor_to_paraview(values, mesh_var.vtype, mesh.dim)
    corners, _ = _physical_visualisation_values(corners, mesh.units)
    values, _ = _physical_visualisation_values(values, mesh_var.units)
    corner_rows = corners.reshape(-1, mesh.cdim)
    _write_vec_to_group(
        viewer, corner_rows, coordinate_name, group, PETSc.COMM_WORLD
    )
    _write_vec_to_group(viewer, values, value_name, group, PETSc.COMM_WORLD)
    local_count = len(corner_rows)
    offset = mesh.dm.comm.tompi4py().exscan(local_count)
    if offset is None:
        offset = 0
    cells = np.arange(offset, offset + local_count, dtype=PETSc.IntType).reshape(
        -1, mesh.dim + 1
    )
    _write_index_array_to_group(viewer, cells, "cells", group, PETSc.COMM_WORLD)
