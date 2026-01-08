"""
Gradient evaluation at arbitrary coordinates.

This module provides functions for evaluating gradients of mesh variables
at arbitrary coordinates without requiring explicit gradient projection solves.

Methods
-------
The primary method uses PETSc's Clement interpolant for gradient recovery:

1. **Clement Interpolation**: Computes L2 projection of cellwise gradients onto
   continuous P1 (linear) space by averaging cell gradients at shared vertices.

   - Accuracy: O(h) - first order convergence
   - Cost: No linear solve required, just local averaging
   - Reference: Clément, P. (1975). "Approximation by finite element functions
     using local regularization". RAIRO Analyse numérique, 9(R-2), 77-84.

For higher accuracy gradient evaluation, use explicit projection:

2. **L2 Projection**: Solve a mass matrix system for optimal L2 gradient.

   - Accuracy: O(h²) for smooth solutions
   - Cost: Requires solving linear system
   - Use: `uw.systems.Projection` with gradient expression

Comparison
----------
| Method    | Accuracy | Cost        | Use Case                          |
|-----------|----------|-------------|-----------------------------------|
| Clement   | O(h)     | No solve    | Quick estimates, error indicators |
| L2 Proj   | O(h²)    | Linear solve| High accuracy requirements        |

Implementation Notes
--------------------
The evaluate_gradient function uses a "scratch DM" approach to avoid polluting
the main mesh's DM with temporary fields:

1. Clone mesh DM (gets topology, 0 fields)
2. Add P1 linear FE field matching Clement output layout
3. Populate with Clement gradient data
4. Interpolate to requested coordinates using DMInterpolation
5. Destroy scratch objects - main DM unchanged

This allows ephemeral gradient evaluation without side effects on the mesh.
"""

import numpy as np
from petsc4py import PETSc


def evaluate_gradient(scalar_var, coords, method="clement"):
    """
    Evaluate gradient of a scalar field at arbitrary coordinates.

    Computes the gradient of a scalar MeshVariable and evaluates it at
    the specified coordinates without adding permanent fields to the mesh.

    Parameters
    ----------
    scalar_var : MeshVariable
        Scalar field (num_components=1) to compute gradient of.
    coords : array-like
        Coordinates at which to evaluate gradient, shape (n_points, dim).
        Can be numpy array or UnitAwareArray.
    method : str, optional
        Gradient computation method. Currently supported:
        - "clement": Clement interpolant (O(h) accurate, no solve). Default.

    Returns
    -------
    ndarray
        Gradient values at requested coordinates, shape (n_points, dim).
        gradient[i, j] = ∂f/∂xⱼ at coords[i].

    Notes
    -----
    **Clement Method**: Uses PETSc's `DMPlexComputeGradientClementInterpolant`
    which averages cell-wise gradients at vertices. This is O(h) accurate -
    error halves when mesh resolution doubles.

    The function uses a scratch DM internally to avoid adding fields to the
    mesh's DM. All temporary PETSc objects are destroyed after evaluation.

    For O(h²) accuracy, use explicit L2 projection instead:

    ```python
    grad_proj = uw.systems.Projection(mesh, grad_var)
    grad_proj.uw_function = scalar_var.sym.diff(mesh.X)
    grad_proj.solve()
    result = uw.function.evaluate(grad_var.sym, coords)
    ```

    Examples
    --------
    >>> mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    >>> T = uw.discretisation.MeshVariable('T', mesh, 1)
    >>> T.array[:, 0, 0] = T.coords[:, 0]**2  # T = x²
    >>>
    >>> # Evaluate gradient at center
    >>> coords = np.array([[0.5, 0.5]])
    >>> grad = evaluate_gradient(T, coords)
    >>> # grad ≈ [[1.0, 0.0]]  (∂T/∂x = 2x = 1 at x=0.5)

    See Also
    --------
    uw.systems.Projection : For O(h²) accurate gradient via L2 projection
    uw.function.evaluate : General function evaluation

    References
    ----------
    .. [1] Clément, P. (1975). "Approximation by finite element functions
       using local regularization". RAIRO Analyse numérique, 9(R-2), 77-84.
    .. [2] PETSc DMPlexComputeGradientClementInterpolant in plexfem.c
    """
    import underworld3 as uw
    from underworld3.function._dminterp_wrapper import CachedDMInterpolationInfo

    if method != "clement":
        raise ValueError(f"Unknown gradient method: {method}. Supported: 'clement'")

    mesh = scalar_var.mesh
    dm = mesh.dm

    # Convert coords to numpy array if needed
    if hasattr(coords, 'magnitude'):
        # UnitAwareArray or UWQuantity - non-dimensionalise
        coords_nd = uw.scaling.non_dimensionalise(coords)
        coords_array = np.asarray(coords_nd, dtype=np.float64)
    else:
        coords_array = np.asarray(coords, dtype=np.float64)

    # Ensure 2D
    if coords_array.ndim == 1:
        coords_array = coords_array.reshape(1, -1)

    n_points = coords_array.shape[0]

    # Step 1: Create scratch DM with single scalar field for Clement computation
    # This is necessary because computeGradientClementInterpolant expects
    # a DM with a single field matching the input vector layout
    scalar_scratch_dm = dm.clone()

    options = PETSc.Options()
    options.setValue("_scratch_scalar_petscspace_degree", 1)
    options.setValue("_scratch_scalar_petscdualspace_lagrange_continuity", True)

    scalar_fe = PETSc.FE().createDefault(
        mesh.dim,
        1,  # scalar field
        mesh.isSimplex,
        mesh.qdegree,
        "_scratch_scalar_",
        PETSc.COMM_SELF,
    )

    scalar_scratch_dm.addField(scalar_fe)
    scalar_scratch_dm.createDS()

    # Populate scalar scratch vector with variable data
    scalar_lvec = scalar_scratch_dm.createLocalVec()
    scalar_lvec.setArray(scalar_var._lvec.getArray().copy())

    # Step 2: Compute Clement gradient at mesh nodes
    cdm = scalar_scratch_dm.getCoordinateDM()
    grad_vec = cdm.createLocalVec()
    grad_vec.zeroEntries()

    result = scalar_scratch_dm.computeGradientClementInterpolant(scalar_lvec, grad_vec)
    gradient_at_nodes = result.getArray().copy()

    # Cleanup scalar scratch objects
    scalar_lvec.destroy()
    scalar_scratch_dm.destroy()
    scalar_fe.destroy()

    # Step 2: Set up scratch DM for interpolation
    scratch_dm = dm.clone()

    # Configure P1 linear FE to match Clement output (DOFs at vertices only)
    options = PETSc.Options()
    options.setValue("_scratch_grad_petscspace_degree", 1)
    options.setValue("_scratch_grad_petscdualspace_lagrange_continuity", True)

    petsc_fe = PETSc.FE().createDefault(
        mesh.dim,
        mesh.dim,  # vector field for gradient
        mesh.isSimplex,
        mesh.qdegree,
        "_scratch_grad_",
        PETSc.COMM_SELF,
    )

    scratch_dm.addField(petsc_fe)
    scratch_dm.createDS()

    # Create local vector and populate with gradient data
    scratch_lvec = scratch_dm.createLocalVec()
    scratch_lvec.setArray(gradient_at_nodes)

    # Step 3: Interpolate to requested coordinates
    # Get cell hints from mesh's kdtree
    cells = mesh.get_closest_cells(coords_array)

    # Minimal mesh-like object for interpolation wrapper
    class _ScratchMesh:
        def __init__(self, dm, lvec):
            self.dm = dm
            self.lvec = lvec

    scratch_mesh = _ScratchMesh(scratch_dm, scratch_lvec)

    # Set up and evaluate interpolation
    interp_info = CachedDMInterpolationInfo()
    interp_info.create_structure(scratch_mesh, coords_array, cells, dofcount=mesh.dim)

    outarray = np.zeros((n_points, mesh.dim), dtype=np.float64)
    interp_info.evaluate(scratch_mesh, outarray)

    # Step 4: Cleanup - main DM unchanged
    scratch_lvec.destroy()
    scratch_dm.destroy()
    petsc_fe.destroy()
    grad_vec.destroy()

    return outarray


def interpolate_gradients_at_coords(source_vars, coords, mesh):
    """
    Compute Clement gradients for multiple source variables and interpolate.

    Computes gradients for each source variable using the Clement interpolant
    and evaluates at the specified coordinates.

    Parameters
    ----------
    source_vars : list of MeshVariable
        Scalar fields needing gradient computation.
    coords : array-like
        Coordinates at which to evaluate gradients, shape (n_points, dim).
    mesh : Mesh
        The mesh containing the variables.

    Returns
    -------
    dict
        Mapping from source MeshVariable to gradient array (n_points, dim).
        result[var][i, j] = ∂var/∂xⱼ at coords[i].

    Notes
    -----
    For an expression with multiple derivatives like `T.diff(x) + T.diff(y) + A.diff(x)`:
    - Groups by source variable: {T, A}
    - Computes gradient once per source variable
    - Each derivative component (T.diff(x), T.diff(y)) extracted from same gradient

    This is called internally by evaluate_nd when derivatives are detected.
    """
    if not source_vars:
        return {}

    # Evaluate gradient for each source variable
    # Each call to evaluate_gradient uses its own scratch DM
    result = {}
    for var in source_vars:
        result[var] = evaluate_gradient(var, coords)

    return result


def compute_clement_gradient_at_nodes(scalar_var):
    """
    Compute Clement gradient at mesh nodes only (no interpolation).

    This is the raw Clement interpolant without arbitrary point evaluation.
    Useful when you only need values at mesh nodes, e.g., for error estimation
    or visualization at node locations.

    Parameters
    ----------
    scalar_var : MeshVariable
        Scalar field (num_components=1) to compute gradient of.

    Returns
    -------
    ndarray
        Gradient at mesh nodes, shape (n_nodes, dim).
        gradient[i, j] = ∂f/∂xⱼ at node i.

    Notes
    -----
    Uses PETSc's `DMPlexComputeGradientClementInterpolant` which computes
    the L2 projection of cell-wise constant gradients onto a continuous
    P1 (linear) space by averaging at shared vertices.

    This is O(h) accurate - suitable for error estimation, quick visualization,
    or when higher accuracy is not required.

    Examples
    --------
    >>> T.array[:, 0, 0] = T.coords[:, 0]**2 + T.coords[:, 1]**2
    >>> grad = compute_clement_gradient_at_nodes(T)
    >>> # grad[i] ≈ [2*x_i, 2*y_i] at each node
    """
    mesh = scalar_var.mesh
    dm = mesh.dm

    # Create scratch DM with single scalar field for Clement computation
    # This is necessary because computeGradientClementInterpolant expects
    # a DM with a single field matching the input vector layout
    scalar_scratch_dm = dm.clone()

    options = PETSc.Options()
    options.setValue("_scratch_clement_petscspace_degree", 1)
    options.setValue("_scratch_clement_petscdualspace_lagrange_continuity", True)

    scalar_fe = PETSc.FE().createDefault(
        mesh.dim,
        1,  # scalar field
        mesh.isSimplex,
        mesh.qdegree,
        "_scratch_clement_",
        PETSc.COMM_SELF,
    )

    scalar_scratch_dm.addField(scalar_fe)
    scalar_scratch_dm.createDS()

    # Populate scratch vector with variable data
    scalar_lvec = scalar_scratch_dm.createLocalVec()
    scalar_lvec.setArray(scalar_var._lvec.getArray().copy())

    # Compute Clement gradient
    cdm = scalar_scratch_dm.getCoordinateDM()
    grad_vec = cdm.createLocalVec()
    grad_vec.zeroEntries()

    result = scalar_scratch_dm.computeGradientClementInterpolant(scalar_lvec, grad_vec)
    arr = result.getArray().copy()

    # Cleanup
    grad_vec.destroy()
    scalar_lvec.destroy()
    scalar_scratch_dm.destroy()
    scalar_fe.destroy()

    num_nodes = len(arr) // mesh.dim
    return arr.reshape(num_nodes, mesh.dim)
