"""
Gradient evaluation at arbitrary coordinates.

This module provides functions for evaluating gradients of mesh variables
at arbitrary coordinates, with two methods offering different accuracy/cost tradeoffs.

Methods
-------
Two gradient evaluation methods are available:

1. **Interpolant (Clement)**: Fast gradient recovery via vertex averaging.

   - Accuracy: O(h) - first order convergence
   - Cost: No linear solve required, just local averaging
   - Best for: Quick estimates, RBF evaluation, error indicators
   - Reference: Clément, P. (1975). "Approximation by finite element functions
     using local regularization". RAIRO Analyse numérique, 9(R-2), 77-84.

2. **Projection (L2)**: Solve a mass matrix system for optimal L2 gradient.

   - Accuracy: O(h²) for smooth solutions
   - Cost: Requires solving linear system (cached for repeated calls)
   - Best for: High accuracy requirements, PETSc evaluation path

Comparison
----------
| Method       | Name          | Accuracy | Cost         | Use Case                |
|--------------|---------------|----------|--------------|-------------------------|
| Interpolant  | "interpolant" | O(h)     | No solve     | Quick estimates, RBF    |
| Projection   | "projection"  | O(h²)    | Linear solve | High accuracy, PETSc    |

Default Routing
---------------
When called via `uw.function.evaluate()`:
- `rbf=True` (RBF path): defaults to "interpolant" (fast)
- `rbf=False` (PETSc path): defaults to "projection" (accurate)

Caching
-------
The projection method caches gradient variables on the mesh to avoid repeated
setup costs. Subsequent solves use `zero_init_guess=False` for warm-starting.
"""

import numpy as np
from petsc4py import PETSc


def evaluate_gradient(scalar_var, coords, method="interpolant", component=None):
    """
    Evaluate gradient of a mesh variable at arbitrary coordinates.

    Computes the gradient of a MeshVariable (or one of its components) and
    evaluates it at the specified coordinates.

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
        Gradient computation method:
        - "interpolant": Clement interpolant (O(h) accurate, no solve). Default.
        - "projection": L2 projection (O(h²) accurate, requires solve).
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
    **Interpolant (Clement) Method**: Uses PETSc's
    `DMPlexComputeGradientClementInterpolant` which averages cell-wise gradients
    at vertices. This is O(h) accurate - error halves when mesh resolution doubles.
    Fast but limited to first-order accuracy.

    **Projection (L2) Method**: Solves a mass matrix system to find the optimal
    L2 projection of the gradient onto the finite element space. This is O(h²)
    accurate for smooth solutions. The projection is cached on the mesh for
    repeated calls, using the previous solution as initial guess.

    Examples
    --------
    >>> mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    >>> T = uw.discretisation.MeshVariable('T', mesh, 1)
    >>> T.array[:, 0, 0] = T.coords[:, 0]**2  # T = x²
    >>>
    >>> # Fast gradient (O(h))
    >>> grad_fast = evaluate_gradient(T, coords, method="interpolant")
    >>>
    >>> # Accurate gradient (O(h²))
    >>> grad_accurate = evaluate_gradient(T, coords, method="projection")

    See Also
    --------
    uw.systems.Projection : Direct L2 projection for explicit control
    uw.function.evaluate : General function evaluation

    References
    ----------
    .. [1] Clément, P. (1975). "Approximation by finite element functions
       using local regularization". RAIRO Analyse numérique, 9(R-2), 77-84.
    """
    if method == "interpolant":
        return _evaluate_gradient_interpolant(scalar_var, coords, component)
    elif method == "projection":
        return _evaluate_gradient_projection(scalar_var, coords, component)
    else:
        raise ValueError(
            f"Unknown gradient method: '{method}'. "
            f"Use 'interpolant' (fast, O(h)) or 'projection' (accurate, O(h²))"
        )


def _evaluate_gradient_interpolant(scalar_var, coords, component=None):
    """
    Evaluate gradient via Clement interpolant (fast, O(h) accurate).

    Uses PETSc's DMPlexComputeGradientClementInterpolant which averages
    cell-wise gradients at vertices. A scratch DM is used internally to
    avoid polluting the mesh's DM.

    Parameters
    ----------
    scalar_var : MeshVariable
        Field to compute gradient of.
    coords : array-like
        Coordinates at which to evaluate gradient.
    component : int or None
        For multi-component fields, which component.

    Returns
    -------
    ndarray
        Gradient values, shape (n_points, dim).
    """
    import underworld3 as uw

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
        coords_nd = uw.scaling.non_dimensionalise(coords)
        coords_array = np.asarray(coords_nd, dtype=np.float64)
    else:
        coords_array = np.asarray(coords, dtype=np.float64)

    if coords_array.ndim == 1:
        coords_array = coords_array.reshape(1, -1)

    n_points = coords_array.shape[0]

    # Get vertex coordinates (P1 node locations)
    vertex_coords = mesh.X.coords
    n_vertices = vertex_coords.shape[0]

    # Check if field is P1 scalar - can use data directly
    field_degree = scalar_var.degree
    is_p1_scalar = (field_degree == 1 and num_components == 1)

    if is_p1_scalar:
        vertex_values = scalar_var._lvec.getArray().copy()
    else:
        vertex_values = _evaluate_field_at_vertices(scalar_var, component, mesh)

    # Create scratch DM with P1 scalar field for Clement computation
    scalar_scratch_dm = dm.clone()

    options = PETSc.Options()
    options.setValue("_scratch_scalar_petscspace_degree", 1)
    options.setValue("_scratch_scalar_petscdualspace_lagrange_continuity", True)

    scalar_fe = PETSc.FE().createDefault(
        mesh.dim,
        1,
        mesh.isSimplex,
        mesh.qdegree,
        "_scratch_scalar_",
        PETSc.COMM_SELF,
    )

    scalar_scratch_dm.addField(scalar_fe)
    scalar_scratch_dm.createDS()

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

    # Compute Clement gradient at mesh nodes
    cdm = scalar_scratch_dm.getCoordinateDM()
    grad_vec = cdm.createLocalVec()
    grad_vec.zeroEntries()

    result = scalar_scratch_dm.computeGradientClementInterpolant(scalar_lvec, grad_vec)
    gradient_at_nodes = result.getArray().copy()

    # Cleanup scratch objects
    scalar_lvec.destroy()
    scalar_scratch_dm.destroy()
    scalar_fe.destroy()

    # Interpolate gradient to requested coordinates
    n_vertices = mesh.X.coords.shape[0]
    gradient_reshaped = gradient_at_nodes.reshape(n_vertices, mesh.dim)

    cells = mesh.get_closest_cells(coords_array)
    outarray = np.zeros((n_points, mesh.dim), dtype=np.float64)

    for i in range(n_points):
        cell = cells[i]
        point = coords_array[i]

        try:
            closure = mesh.dm.getTransitiveClosure(cell)[0]
            vertices = [v for v in closure if mesh.dm.getPointDepth(v) == 0]

            if len(vertices) >= mesh.dim + 1:
                vertex_coords_cell = np.array([mesh.dm.getCoordinatesLocal().getArray()[
                    mesh.dm.getCoordinateSection().getOffset(v):
                    mesh.dm.getCoordinateSection().getOffset(v) + mesh.dim
                ] for v in vertices[:mesh.dim + 1]])

                vertex_grads = np.array([gradient_reshaped[v - mesh.dm.getDepthStratum(0)[0]]
                                         for v in vertices[:mesh.dim + 1]])

                bary = _compute_barycentric(point, vertex_coords_cell)
                outarray[i] = np.sum(bary[:, np.newaxis] * vertex_grads, axis=0)
            else:
                outarray[i] = gradient_reshaped[vertices[0] - mesh.dm.getDepthStratum(0)[0]]
        except Exception:
            outarray[i] = gradient_reshaped[0]

    grad_vec.destroy()
    return outarray


def _evaluate_gradient_projection(scalar_var, coords, component=None):
    """
    Evaluate gradient via L2 projection (accurate, O(h²)).

    Creates a cached gradient MeshVariable and projector on the mesh.
    Subsequent calls reuse the cached objects and warm-start the solve
    from the previous solution.

    Parameters
    ----------
    scalar_var : MeshVariable
        Field to compute gradient of.
    coords : array-like
        Coordinates at which to evaluate gradient.
    component : int or None
        For multi-component fields, which component.

    Returns
    -------
    ndarray
        Gradient values, shape (n_points, dim).
    """
    import underworld3 as uw

    mesh = scalar_var.mesh
    dim = mesh.dim

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
        coords_nd = uw.scaling.non_dimensionalise(coords)
        coords_array = np.asarray(coords_nd, dtype=np.float64)
    else:
        coords_array = np.asarray(coords, dtype=np.float64)

    if coords_array.ndim == 1:
        coords_array = coords_array.reshape(1, -1)

    n_points = coords_array.shape[0]

    # Cache key based on variable and component
    comp_suffix = f"_c{component}" if component is not None else ""
    cache_name = f"_grad_proj_{scalar_var.name}{comp_suffix}"

    # Initialize gradient cache on mesh if not present
    if not hasattr(mesh, '_gradient_cache'):
        mesh._gradient_cache = {}

    # Get or create cached projectors for each gradient component
    if cache_name not in mesh._gradient_cache:
        # Determine degree for gradient variable
        # For P1 source, gradient is P0 (constant per cell) but we project to P1
        # For higher order, use degree-1 but at least 1
        grad_degree = max(1, scalar_var.degree - 1)

        # Build symbolic expression for the source field component
        if num_components == 1:
            source_sym = scalar_var.sym[0, 0]
        else:
            source_sym = scalar_var.sym[component, 0]

        # Create projector for each spatial dimension
        projectors = []
        for d in range(dim):
            # Create gradient component variable
            proj_var = uw.discretisation.MeshVariable(
                f"{cache_name}_d{d}", mesh, num_components=1, degree=grad_degree
            )

            # Create projector
            projector = uw.systems.Projection(mesh, proj_var)
            projector.uw_function = source_sym.diff(mesh.X[d])
            projector.smoothing = 0.0

            projectors.append((proj_var, projector))

        mesh._gradient_cache[cache_name] = {
            'projectors': projectors,
            'source_var': scalar_var,
            'component': component,
        }

    cache = mesh._gradient_cache[cache_name]

    # Solve projections (warm-start from previous solution)
    for proj_var, projector in cache['projectors']:
        projector.solve(zero_init_guess=False)

    # Evaluate gradient components at coordinates
    result = np.zeros((n_points, dim), dtype=np.float64)
    for d, (proj_var, _) in enumerate(cache['projectors']):
        result[:, d] = uw.function.evaluate(proj_var.sym, coords_array).flatten()

    return result


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


def interpolate_gradients_at_coords(source_vars, coords, mesh, method="interpolant"):
    """
    Compute gradients for multiple source variables and interpolate.

    Computes gradients for each source variable using the specified method
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
    method : str, optional
        Gradient computation method:
        - "interpolant": Clement interpolant (O(h) accurate, no solve). Default.
        - "projection": L2 projection (O(h²) accurate, requires solve).

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
            result[(var, 0)] = evaluate_gradient(var, coords, method=method, component=None)
        else:
            result[(var, component)] = evaluate_gradient(var, coords, method=method, component=component)

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
