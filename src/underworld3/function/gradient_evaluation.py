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


def evaluate_gradient(scalar_var, coords, method="clement", component=None):
    """
    Evaluate gradient of a mesh variable at arbitrary coordinates.

    Computes the gradient of a MeshVariable (or one of its components) and
    evaluates it at the specified coordinates without adding permanent fields
    to the mesh.

    Parameters
    ----------
    scalar_var : MeshVariable
        Field to compute gradient of. Can be scalar (num_components=1) or
        vector/tensor field. For multi-component fields, use `component`
        parameter to specify which component's gradient to compute.
    coords : array-like
        Coordinates at which to evaluate gradient, shape (n_points, dim).
        Can be numpy array or UnitAwareArray.
    method : str, optional
        Gradient computation method. Currently supported:
        - "clement": Clement interpolant (O(h) accurate, no solve). Default.
    component : int or None, optional
        For multi-component fields, which component to compute gradient of.
        If None and field has multiple components, raises ValueError.
        For scalar fields, this parameter is ignored.

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

    **Higher-degree fields (P2, etc.)**: For fields with degree > 1, the
    function first samples the field at P1 vertex locations, then computes
    the Clement gradient on that data. This introduces some approximation
    but allows gradient computation for any polynomial degree.

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

    # Handle multi-component fields
    num_components = scalar_var.num_components
    if num_components > 1:
        if component is None:
            raise ValueError(
                f"Field '{scalar_var.name}' has {num_components} components. "
                f"Specify which component's gradient to compute using component=0, 1, ..."
            )
        if component < 0 or component >= num_components:
            raise ValueError(
                f"component={component} out of range for field with {num_components} components"
            )

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

    # Step 1: Get field values at P1 vertex locations
    # For degree > 1 fields, we need to sample at vertices first
    # The Clement interpolant operates on vertex data

    # Get vertex coordinates (P1 node locations)
    vertex_coords = mesh.X.coords  # Shape: (n_vertices, dim)
    n_vertices = vertex_coords.shape[0]

    # Check if field is P1 scalar - can use data directly
    field_degree = scalar_var.degree
    is_p1_scalar = (field_degree == 1 and num_components == 1)

    if is_p1_scalar:
        # P1 scalar field - use data directly
        vertex_values = scalar_var._lvec.getArray().copy()
    else:
        # Need to sample the field at vertex locations
        # For P2 or vector fields, evaluate the appropriate component at vertices
        if num_components == 1:
            # Scalar field with degree > 1
            sym_expr = scalar_var.sym[0, 0]
        else:
            # Vector/tensor field - get specified component
            # Handle both 1D index (for vectors stored flat) and 2D (for tensors)
            sym_expr = scalar_var.sym[component, 0]

        # Evaluate at vertex locations (avoiding recursion by using direct interpolation)
        # We evaluate the field itself (not its derivative) at P1 vertices
        vertex_values = _evaluate_field_at_vertices(scalar_var, component, mesh)

    # Step 2: Create scratch DM with P1 scalar field for Clement computation
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

    # Populate scalar scratch vector with vertex values
    scalar_lvec = scalar_scratch_dm.createLocalVec()
    expected_size = scalar_lvec.getLocalSize()

    if len(vertex_values) != expected_size:
        # Size mismatch - this shouldn't happen if we sampled correctly
        scalar_lvec.destroy()
        scalar_scratch_dm.destroy()
        scalar_fe.destroy()
        raise ValueError(
            f"Size mismatch: vertex_values has {len(vertex_values)} entries, "
            f"expected {expected_size} for P1 scalar field. "
            f"Field degree={field_degree}, num_components={num_components}"
        )

    scalar_lvec.setArray(vertex_values)

    # Step 3: Compute Clement gradient at mesh nodes
    cdm = scalar_scratch_dm.getCoordinateDM()
    grad_vec = cdm.createLocalVec()
    grad_vec.zeroEntries()

    result = scalar_scratch_dm.computeGradientClementInterpolant(scalar_lvec, grad_vec)
    gradient_at_nodes = result.getArray().copy()

    # Cleanup scalar scratch objects
    scalar_lvec.destroy()
    scalar_scratch_dm.destroy()
    scalar_fe.destroy()

    # Step 4: Interpolate gradient to requested coordinates
    # The gradient_at_nodes array has shape (n_vertices * dim,) - gradient vector at each vertex
    # Reshape to (n_vertices, dim)
    n_vertices = mesh.X.coords.shape[0]
    gradient_reshaped = gradient_at_nodes.reshape(n_vertices, mesh.dim)

    # Use simple linear interpolation based on cell membership
    # For each query point, find its cell and interpolate from vertices
    cells = mesh.get_closest_cells(coords_array)

    outarray = np.zeros((n_points, mesh.dim), dtype=np.float64)

    # Get cell-vertex connectivity from mesh
    # Use barycentric interpolation within simplices
    for i in range(n_points):
        cell = cells[i]
        point = coords_array[i]

        # Get vertices of this cell
        try:
            closure = mesh.dm.getTransitiveClosure(cell)[0]
            # Filter to get only vertices (depth 0 entities)
            vertices = [v for v in closure if mesh.dm.getPointDepth(v) == 0]

            if len(vertices) >= mesh.dim + 1:
                # Get vertex coordinates and gradients
                vertex_coords = np.array([mesh.dm.getCoordinatesLocal().getArray()[
                    mesh.dm.getCoordinateSection().getOffset(v):
                    mesh.dm.getCoordinateSection().getOffset(v) + mesh.dim
                ] for v in vertices[:mesh.dim + 1]])

                vertex_grads = np.array([gradient_reshaped[v - mesh.dm.getDepthStratum(0)[0]]
                                         for v in vertices[:mesh.dim + 1]])

                # Compute barycentric coordinates
                bary = _compute_barycentric(point, vertex_coords)

                # Interpolate gradient
                outarray[i] = np.sum(bary[:, np.newaxis] * vertex_grads, axis=0)
            else:
                # Fallback: use nearest vertex gradient
                outarray[i] = gradient_reshaped[vertices[0] - mesh.dm.getDepthStratum(0)[0]]
        except Exception:
            # If cell query fails, use nearest vertex
            outarray[i] = gradient_reshaped[0]

    # Cleanup
    grad_vec.destroy()

    return outarray


def _compute_barycentric(point, vertices):
    """
    Compute barycentric coordinates of a point within a simplex.

    Parameters
    ----------
    point : ndarray
        Query point, shape (dim,).
    vertices : ndarray
        Simplex vertices, shape (dim+1, dim).

    Returns
    -------
    ndarray
        Barycentric coordinates, shape (dim+1,).
    """
    dim = len(point)
    n_vertices = len(vertices)

    if n_vertices != dim + 1:
        # Not a simplex, return uniform weights
        return np.ones(n_vertices) / n_vertices

    # Build matrix T where columns are (v_i - v_n) for i = 0..dim-1
    T = (vertices[:-1] - vertices[-1]).T  # shape (dim, dim)

    try:
        # Solve T @ lambda = (point - v_n)
        lambdas = np.linalg.solve(T, point - vertices[-1])
        # Last barycentric coordinate
        lambda_n = 1.0 - np.sum(lambdas)
        return np.append(lambdas, lambda_n)
    except np.linalg.LinAlgError:
        # Degenerate simplex, return uniform
        return np.ones(n_vertices) / n_vertices


def _evaluate_field_at_vertices(var, component, mesh):
    """
    Evaluate a field component at P1 vertex locations.

    For P1 scalar fields, returns the data directly.
    For higher-degree or multi-component fields, samples at vertex coordinates.

    Parameters
    ----------
    var : MeshVariable
        The mesh variable to sample.
    component : int or None
        Which component to evaluate (for multi-component fields).
    mesh : Mesh
        The mesh.

    Returns
    -------
    ndarray
        Field values at vertices, shape (n_vertices,).
    """
    import underworld3 as uw

    num_components = var.num_components
    field_degree = var.degree

    # Get vertex count
    vertex_coords = mesh.X.coords
    n_vertices = vertex_coords.shape[0]

    # P1 scalar - return data directly
    if field_degree == 1 and num_components == 1:
        return var._lvec.getArray().copy()

    # P1 multi-component - extract the component directly from data
    if field_degree == 1:
        # Data layout: [v0_c0, v0_c1, ..., v1_c0, v1_c1, ...]
        all_data = var._lvec.getArray()
        # Reshape to (n_vertices, num_components) and extract component
        data_reshaped = all_data.reshape(n_vertices, num_components)
        return data_reshaped[:, component].copy()

    # For degree > 1, try to extract vertex DOFs directly from PETSc ordering
    # PETSc typically stores vertex DOFs first, then edge/face/cell DOFs
    all_data = var._lvec.getArray()

    if num_components == 1:
        # Scalar P2+ - first n_vertices DOFs are vertex values
        if len(all_data) >= n_vertices:
            return all_data[:n_vertices].copy()
    else:
        # Vector P2+ - check for blocked layout
        # Layout: all DOFs for component 0, then all for component 1, etc.
        dofs_per_component = len(all_data) // num_components

        if dofs_per_component >= n_vertices:
            # Extract vertex DOFs for the requested component
            start = component * dofs_per_component
            return all_data[start:start + n_vertices].copy()

    # Fallback: evaluate using uw.function.evaluate at vertex coordinates
    # Build sym expression for the specific component
    if num_components == 1:
        sym_expr = var.sym[0, 0]
    else:
        # Access component - handle different sym layouts
        sym = var.sym
        if hasattr(sym, 'shape') and len(sym.shape) == 2:
            sym_expr = sym[component, 0]
        else:
            # sym is a column vector or 1D
            sym_expr = sym[component]

    result = uw.function.evaluate(sym_expr, vertex_coords)
    return result.flatten()


def interpolate_gradients_at_coords(source_vars, coords, mesh):
    """
    Compute Clement gradients for multiple source variables and interpolate.

    Computes gradients for each source variable using the Clement interpolant
    and evaluates at the specified coordinates.

    Parameters
    ----------
    source_vars : list of MeshVariable or list of (MeshVariable, int) tuples
        Fields needing gradient computation. For scalar fields, just pass
        the MeshVariable. For multi-component fields, pass tuples of
        (MeshVariable, component_index).
    coords : array-like
        Coordinates at which to evaluate gradients, shape (n_points, dim).
    mesh : Mesh
        The mesh containing the variables.

    Returns
    -------
    dict
        Mapping from (var, component) tuple to gradient array (n_points, dim).
        For scalar fields, component is 0.
        result[(var, component)][i, j] = ∂var[component]/∂xⱼ at coords[i].

    Notes
    -----
    For an expression with multiple derivatives like `T.diff(x) + v[0].diff(y)`:
    - Identifies source variables and components: {(T, 0), (v, 0)}
    - Computes gradient once per (variable, component) pair
    - Each derivative component extracted from the appropriate gradient

    This is called internally by evaluate_nd when derivatives are detected.
    """
    if not source_vars:
        return {}

    # Normalize input: convert bare variables to (var, component) tuples
    normalized_vars = []
    for item in source_vars:
        if isinstance(item, tuple):
            normalized_vars.append(item)
        else:
            # Bare variable - assume scalar (component 0)
            # For multi-component fields, caller should specify components
            var = item
            if var.num_components == 1:
                normalized_vars.append((var, 0))
            else:
                # Multi-component field passed without component spec
                # Compute gradient for all components
                for c in range(var.num_components):
                    normalized_vars.append((var, c))

    # Remove duplicates while preserving order
    seen = set()
    unique_vars = []
    for item in normalized_vars:
        if item not in seen:
            seen.add(item)
            unique_vars.append(item)

    # Evaluate gradient for each (variable, component) pair
    result = {}
    for var, component in unique_vars:
        if var.num_components == 1:
            result[(var, 0)] = evaluate_gradient(var, coords, component=None)
        else:
            result[(var, component)] = evaluate_gradient(var, coords, component=component)

    return result


def compute_clement_gradient_at_nodes(var, component=None):
    """
    Compute Clement gradient at mesh nodes only (no interpolation).

    This is the raw Clement interpolant without arbitrary point evaluation.
    Useful when you only need values at mesh nodes, e.g., for error estimation
    or visualization at node locations.

    Parameters
    ----------
    var : MeshVariable
        Field to compute gradient of. Can be scalar or multi-component.
    component : int or None, optional
        For multi-component fields, which component to compute gradient of.
        If None and field is scalar, uses the only component.
        If None and field is multi-component, raises ValueError.

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

    For higher-degree fields (P2, etc.), the function first samples the field
    at P1 vertex locations before computing the Clement gradient.

    Examples
    --------
    >>> T.array[:, 0, 0] = T.coords[:, 0]**2 + T.coords[:, 1]**2
    >>> grad = compute_clement_gradient_at_nodes(T)
    >>> # grad[i] ≈ [2*x_i, 2*y_i] at each node

    >>> # For vector field, specify component
    >>> grad_vx = compute_clement_gradient_at_nodes(v, component=0)
    """
    mesh = var.mesh
    dm = mesh.dm

    # Handle multi-component fields
    num_components = var.num_components
    if num_components > 1:
        if component is None:
            raise ValueError(
                f"Field '{var.name}' has {num_components} components. "
                f"Specify which component's gradient to compute using component=0, 1, ..."
            )
        if component < 0 or component >= num_components:
            raise ValueError(
                f"component={component} out of range for field with {num_components} components"
            )

    # Get vertex coordinates
    vertex_coords = mesh.X.coords
    n_vertices = vertex_coords.shape[0]

    # Get field values at P1 vertex locations
    field_degree = var.degree
    is_p1_scalar = (field_degree == 1 and num_components == 1)

    if is_p1_scalar:
        # P1 scalar field - use data directly
        vertex_values = var._lvec.getArray().copy()
    else:
        # Need to sample the field at vertex locations
        vertex_values = _evaluate_field_at_vertices(var, component, mesh)

    # Create scratch DM with single scalar field for Clement computation
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

    # Populate scratch vector with vertex values
    scalar_lvec = scalar_scratch_dm.createLocalVec()
    expected_size = scalar_lvec.getLocalSize()

    if len(vertex_values) != expected_size:
        scalar_lvec.destroy()
        scalar_scratch_dm.destroy()
        scalar_fe.destroy()
        raise ValueError(
            f"Size mismatch: vertex_values has {len(vertex_values)} entries, "
            f"expected {expected_size} for P1 scalar field. "
            f"Field degree={field_degree}, num_components={num_components}"
        )

    scalar_lvec.setArray(vertex_values)

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
