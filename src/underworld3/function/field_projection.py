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

    Returns
    -------
    np.ndarray
        Projected values with shape ``(n_target_dofs, num_components)``.

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
    l_dst = dm_scratch.createLocalVec()

    interp.mult(g_src, g_dst)
    dm_scratch.globalToLocal(g_dst, l_dst)

    result = l_dst.array.reshape(-1, nc).copy()

    # --- cleanup ---
    for obj in (interp, g_src, g_dst, l_dst, dm_scratch, fe_target):
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


def _write_vec_to_group(viewer, data_array, name, group, comm):
    """Write a numpy array as a standalone PETSc Vec under an HDF5 group.

    DM-associated Vecs ignore ``pushGroup`` because the DM's HDF5 writer
    pushes its own ``/fields/`` prefix.  This helper creates a plain Vec
    (no DM) so ``pushGroup`` is respected.

    Parameters
    ----------
    viewer
        An open ``PETSc.ViewerHDF5``.
    data_array : np.ndarray
        Local (owned) data to write — shape ``(n_local, ...)``.
    name : str
        Dataset name in the HDF5 group.
    group : str
        HDF5 group path (e.g. ``/vertex_fields``).
    comm
        MPI communicator.
    """
    vec = PETSc.Vec().createWithArray(data_array.ravel(), comm=comm)
    vec.setName(name)
    viewer.pushGroup(group)
    viewer(vec)
    viewer.popGroup()
    vec.destroy()


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
        # DOFs = vertices — use gvec data directly
        mesh_var._sync_lvec_to_gvec()
        data = mesh_var._gvec.array.reshape(-1, nc).copy()
    else:
        # Build scratch DM at degree 1, interpolate, extract array
        options = PETSc.Options()
        prefix = "_fieldproj_"
        options.setValue(f"{prefix}petscspace_degree", 1)
        options.setValue(f"{prefix}petscdualspace_lagrange_continuity", True)
        options.setValue(f"{prefix}petscdualspace_lagrange_node_endpoints", False)

        fe_target = PETSc.FE().createDefault(
            mesh.dim, nc, mesh.isSimplex, mesh.qdegree, prefix, PETSc.COMM_SELF,
        )

        dm_scratch = mesh.dm.clone()
        dm_scratch.addField(fe_target)
        dm_scratch.createDS()

        iset, subdm_src = mesh.dm.createSubDM(mesh_var.field_id)
        interp, _scale = subdm_src.createInterpolation(dm_scratch)

        g_src = subdm_src.createGlobalVec()
        subdm_src.localToGlobal(mesh_var._lvec, g_src)

        g_dst = dm_scratch.createGlobalVec()
        interp.mult(g_src, g_dst)

        data = g_dst.array.reshape(-1, nc).copy()

        for obj in (interp, g_src, g_dst, dm_scratch, fe_target):
            obj.destroy()
        if _scale is not None:
            _scale.destroy()

    _write_vec_to_group(viewer, data, name, group, PETSc.COMM_WORLD)


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
    _write_vec_to_group(viewer, data, name, group, PETSc.COMM_WORLD)
