"""Project or prolong MeshVariable data to a different polynomial degree.

Uses a scratch PETSc DM with ``createInterpolation`` — no MeshVariable
is created, the mesh DM is never modified, and all scratch objects are
destroyed before returning.

Typical use cases
-----------------
* Down-sample a P2 field to P1 vertex values for visualisation / XDMF output.
* Prolong a P1 field to P2 DOF values for initialisation.
* Obtain vertex values (degree-1) from any higher-order variable.
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
