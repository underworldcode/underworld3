"""
Optimized evaluation for pure sympy expressions.

This module provides automatic detection and fast evaluation of expressions
that contain only pure sympy symbols (no UW3 MeshVariable symbols).

For pure sympy expressions, we use cached lambdified functions instead of
the full RBF interpolation machinery, providing 10,000x+ speedups.
"""

import sympy
import numpy as np
from functools import lru_cache
import hashlib

# Global cache for lambdified functions
# Key: (expr_hash, symbols_tuple, modules_tuple)
# Value: lambdified function
_lambdify_cache = {}


def is_pure_sympy_expression(expr):
    """
    Detect if an expression contains only pure sympy symbols or mesh coordinates (no UW3 variable data).

    Expressions are considered "pure" (lambdifiable) if they contain:
    - Only sympy.Symbol objects
    - Only mesh coordinate BaseScalars (mesh.X[0], mesh.X[1], etc.)
    - Mix of Symbols and BaseScalars
    - No UW3 MeshVariable or SwarmVariable data

    Parameters
    ----------
    expr : sympy expression
        Expression to check

    Returns
    -------
    bool
        True if expression can be lambdified without mesh data interpolation
    symbols : set or None
        Set of free symbols if pure, None otherwise
    symbol_type : str or None
        'symbol', 'coordinate', or 'mixed' indicating what symbols were found

    Examples
    --------
    >>> import sympy
    >>> x = sympy.Symbol('x')
    >>> t = sympy.Symbol('t')
    >>> is_pure_sympy_expression(x**2 + t)
    (True, {x, t}, 'symbol')

    >>> # With mesh coordinates
    >>> x_mesh = mesh.X[0]
    >>> is_pure_sympy_expression(sympy.erf(x_mesh - 0.5))
    (True, {N.x}, 'coordinate')

    >>> # With UW3 variable data - NOT pure
    >>> T = uw.discretisation.MeshVariable("T", mesh, 1)
    >>> is_pure_sympy_expression(T.sym[0] + x)
    (False, None, None)
    """
    import underworld3 as uw

    # Get all free symbols
    try:
        free_symbols = expr.free_symbols
    except AttributeError:
        # Not a sympy expression
        return False, None, None

    if len(free_symbols) == 0:
        # Constant expression - can be handled efficiently
        return True, set(), 'constant'

    # CRITICAL: Check for UW3 Functions (MeshVariable/SwarmVariable symbols)
    # These appear as Function instances in the expression tree, not necessarily in free_symbols
    # Example: T.sym[0] creates T(N.x, N.y) where T is a Function
    #
    # However, we must distinguish between:
    # - UW3 Functions: T(N.x, N.y) with func.__module__ = None (custom function classes)
    # - SymPy Functions: erf(x), sin(x), etc. with func.__module__ from sympy (can be lambdified!)
    function_atoms = list(expr.atoms(sympy.Function))
    if function_atoms:
        # Check if any are UW3 functions (module is None or not from sympy)
        uw_functions = [
            f for f in function_atoms
            if f.func.__module__ is None or (
                f.func.__module__ is not None and 'sympy' not in f.func.__module__
            )
        ]
        if uw_functions:
            # Expression contains UW3 variable data - NOT pure, must use RBF interpolation
            return False, None, None
        # Otherwise, all functions are from SymPy (erf, sin, etc.) - these can be lambdified!

    # Classify remaining symbols
    has_base_scalars = False
    has_pure_symbols = False

    for symbol in free_symbols:
        # Check for mesh coordinate BaseScalars
        if isinstance(symbol, sympy.vector.scalar.BaseScalar):
            has_base_scalars = True

        # Pure sympy Symbol
        if isinstance(symbol, sympy.Symbol):
            has_pure_symbols = True

    # Determine symbol type
    if has_base_scalars and has_pure_symbols:
        symbol_type = 'mixed'
    elif has_base_scalars:
        symbol_type = 'coordinate'
    elif has_pure_symbols:
        symbol_type = 'symbol'
    else:
        # Unknown symbol type
        return False, None, None

    # Expression contains only Symbols and/or BaseScalars - can be lambdified!
    return True, free_symbols, symbol_type


def _expr_hash(expr):
    """
    Generate a hash for a sympy expression for caching.

    Parameters
    ----------
    expr : sympy expression
        Expression to hash

    Returns
    -------
    str
        Hash string
    """
    # Use sympy's srepr for consistent string representation
    expr_str = sympy.srepr(expr)
    return hashlib.md5(expr_str.encode()).hexdigest()


def get_cached_lambdified(expr, symbols, modules=('scipy', 'numpy')):
    """
    Get a cached lambdified function for an expression.

    Uses an LRU cache to avoid recompiling the same expression multiple times.

    Parameters
    ----------
    expr : sympy expression
        Expression to lambdify
    symbols : tuple of sympy.Symbol
        Symbols in order for lambdify
    modules : tuple of str, optional
        Modules to use for lambdify. Default: ('scipy', 'numpy')
        scipy is required for special functions like erf, gamma, etc.

    Returns
    -------
    callable
        Lambdified function

    Notes
    -----
    Cache key is based on:
    - Expression structure (hash of srepr)
    - Symbol names and order
    - Modules used
    """
    # Create cache key
    expr_h = _expr_hash(expr)
    symbols_tuple = tuple(str(s) for s in symbols)
    modules_tuple = tuple(modules) if isinstance(modules, (list, tuple)) else (modules,)

    cache_key = (expr_h, symbols_tuple, modules_tuple)

    # Check cache
    if cache_key in _lambdify_cache:
        return _lambdify_cache[cache_key]

    # Lambdify with scipy for special functions
    try:
        func = sympy.lambdify(symbols, expr, modules=modules)
        _lambdify_cache[cache_key] = func
        return func
    except Exception as e:
        # Fallback to numpy only if scipy fails
        if 'scipy' in modules:
            func = sympy.lambdify(symbols, expr, modules='numpy')
            _lambdify_cache[cache_key] = func
            return func
        else:
            raise


def evaluate_pure_sympy(expr, coords, coord_symbols=None):
    """
    Fast evaluation of pure sympy expressions using cached lambdified functions.

    This function provides optimized evaluation for expressions containing only
    pure sympy symbols (no UW3 MeshVariable symbols). It automatically:
    1. Detects the free symbols in the expression
    2. Maps coordinate columns to symbols
    3. Uses cached lambdified functions for efficiency

    Parameters
    ----------
    expr : sympy expression
        Pure sympy expression to evaluate
    coords : np.ndarray
        Coordinates at which to evaluate, shape (n_points, n_dims)
        For 2D: coords[:, 0] are x values, coords[:, 1] are y values
        For 3D: coords[:, 0] are x, coords[:, 1] are y, coords[:, 2] are z
    coord_symbols : tuple of sympy.Symbol, optional
        Symbols representing coordinates in order (x, y, z)
        If None, will try to infer from expression's free symbols

    Returns
    -------
    np.ndarray
        Evaluated expression values, shape depends on expression:
        - Scalar expr: (n_points,) or (n_points, 1, 1)
        - Vector expr: (n_points, n_components)
        - Matrix expr: (n_points, n_rows, n_cols)

    Examples
    --------
    >>> import sympy
    >>> import numpy as np
    >>> x, y = sympy.symbols('x y')
    >>> expr = sympy.sqrt(x**2 + y**2)  # Distance from origin
    >>> coords = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])
    >>> result = evaluate_pure_sympy(expr, coords, coord_symbols=(x, y))
    >>> # result = [1.0, 1.0, 5.0]

    Notes
    -----
    - Uses scipy for special functions (erf, gamma, etc.)
    - Caches lambdified functions for repeated evaluations
    - ~10,000x faster than sympy.subs() for many points
    """
    # Ensure coords is 2D numpy array
    coords_array = np.asarray(coords, dtype=np.double)
    if coords_array.ndim == 1:
        coords_array = coords_array.reshape(1, -1)

    n_points, n_dims = coords_array.shape

    # Handle Matrix expressions
    is_matrix = isinstance(expr, sympy.Matrix)
    if is_matrix:
        expr_shape = expr.shape
        # For matrices, we'll evaluate element-wise and reshape
        elements = []
        results_list = []

        for i in range(expr_shape[0]):
            row = []
            for j in range(expr_shape[1]):
                elem = expr[i, j]
                elem_result = evaluate_pure_sympy(elem, coords_array, coord_symbols)
                row.append(elem_result)
            results_list.append(row)

        # Stack results
        # results_list is shape [rows][cols] where each element is (n_points,)
        # We want output shape (n_points, rows, cols)
        result = np.zeros((n_points, expr_shape[0], expr_shape[1]))
        for i in range(expr_shape[0]):
            for j in range(expr_shape[1]):
                result[:, i, j] = results_list[i][j].flatten()

        return result

    # Scalar expression - get free symbols
    free_symbols = expr.free_symbols

    if len(free_symbols) == 0:
        # Constant expression - evaluate once and broadcast
        const_val = float(expr.evalf())
        return np.full((n_points, 1, 1), const_val)

    # Import UWCoordinate and UWexpression for symbol processing
    from underworld3.coordinates import UWCoordinate
    from underworld3.function.expressions import UWexpression, _unwrap_for_compilation

    # =========================================================================
    # STEP 1: First, unwrap any UWexpressions in the expression
    # This reveals hidden UWCoordinate symbols inside composite expressions
    # like r = sqrt(x² + y² + z²) where x, y, z are UWCoordinates
    # =========================================================================

    # Check if there are any UWexpressions that need unwrapping
    uw_expr_atoms = [s for s in free_symbols if isinstance(s, UWexpression)]
    if uw_expr_atoms:
        # Unwrap UWexpressions to reveal nested coordinates
        expr = _unwrap_for_compilation(expr, keep_constants=False, return_self=False)

    # =========================================================================
    # STEP 2: Substitute all UWCoordinate atoms with their underlying BaseScalar
    # This must happen AFTER unwrapping so we catch ALL UWCoordinates,
    # including those that were hidden inside UWexpressions
    # =========================================================================

    uw_coords = list(expr.atoms(UWCoordinate))
    if uw_coords:
        coord_subs = {uc: uc._original_base_scalar for uc in uw_coords}
        expr = expr.subs(coord_subs)

    # =========================================================================
    # STEP 3: Now extract coord_symbols from the fully processed expression
    # This happens AFTER unwrapping and substitution so we see ALL coordinates
    # =========================================================================

    if coord_symbols is None:
        # Extract BaseScalar atoms from the processed expression
        base_scalars = list(expr.atoms(sympy.vector.scalar.BaseScalar))

        if base_scalars:
            # We have mesh coordinates - sort them by their string representation
            # to get consistent ordering (N.x, N.y, N.z)
            coord_symbols = tuple(sorted(base_scalars, key=str)[:n_dims])

        else:
            # Pure sympy.Symbol - extract Symbol atoms from expression
            symbol_atoms = list(expr.atoms(sympy.Symbol))

            if symbol_atoms:
                # Try to match common coordinate names first for better ordering
                coord_names = ['x', 'y', 'z', 'r', 'theta', 'phi']
                coord_symbols_list = []

                for name in coord_names:
                    matching = [s for s in symbol_atoms if str(s) == name]
                    if matching:
                        coord_symbols_list.append(matching[0])
                        if len(coord_symbols_list) == n_dims:
                            break

                if len(coord_symbols_list) < n_dims:
                    # Add remaining symbols in alphabetical order
                    remaining = [s for s in symbol_atoms if s not in coord_symbols_list]
                    coord_symbols_list.extend(sorted(remaining, key=str)[:n_dims - len(coord_symbols_list)])

                coord_symbols = tuple(coord_symbols_list[:n_dims])
            else:
                # No symbols found - expression might be constant after unwrapping
                free_after = expr.free_symbols
                if len(free_after) == 0:
                    # Became constant after unwrapping
                    const_val = float(expr.evalf())
                    return np.full((n_points, 1, 1), const_val)
                raise ValueError("No coordinate symbols found in expression")
    else:
        # Convert to tuple if needed
        coord_symbols = tuple(coord_symbols) if isinstance(coord_symbols, (list, tuple)) else (coord_symbols,)

    # =========================================================================
    # STEP 4: Final validation - check for any remaining non-coordinate symbols
    # =========================================================================

    remaining_symbols = expr.free_symbols - set(coord_symbols)
    if remaining_symbols:
        # Check if they're UWexpressions we missed (shouldn't happen after unwrap)
        non_coord_params = set()
        for sym in remaining_symbols:
            if isinstance(sym, UWexpression):
                # Try to unwrap it
                expr = _unwrap_for_compilation(expr, keep_constants=False, return_self=False)
                break
            elif not isinstance(sym, (sympy.vector.scalar.BaseScalar, UWCoordinate)):
                non_coord_params.add(sym)

        if non_coord_params:
            raise ValueError(
                f"Expression contains symbols beyond coordinates: {non_coord_params}. "
                f"Please substitute parameter values before calling evaluate()."
            )

    # Get cached lambdified function
    # Use scipy for special functions (erf, gamma, etc.)
    func = get_cached_lambdified(expr, coord_symbols, modules=('scipy', 'numpy'))

    # Prepare coordinate arrays for evaluation
    # CRITICAL: For BaseScalar symbols, use their _id[0] to get correct column
    # For example, N.y has _id=(1, N), so we extract coords[:, 1] not coords[:, 0]
    coord_indices = []
    for symbol in coord_symbols:
        if isinstance(symbol, sympy.vector.scalar.BaseScalar):
            # BaseScalar has _id = (index, coordinate_system)
            # Extract the index (0 for x, 1 for y, 2 for z)
            coord_indices.append(symbol._id[0])
        else:
            # Regular Symbol - use position in coord_symbols
            # This maintains backward compatibility for non-BaseScalar cases
            coord_indices.append(len(coord_indices))

    # Extract coordinate arrays using correct column indices
    coord_arrays = [coords_array[:, idx] for idx in coord_indices]

    # Evaluate
    try:
        result = func(*coord_arrays)
    except TypeError as e:
        # May need to handle scalar vs array differently
        if n_points == 1:
            # Single point evaluation - use same column indices
            coord_values = [coords_array[0, idx] for idx in coord_indices]
            result = np.array([func(*coord_values)])
        else:
            raise

    # Ensure result is at least 1D
    result = np.atleast_1d(result)

    # Reshape to (n_points, 1, 1) for scalar expressions (consistency with UW3)
    if result.shape == (n_points,):
        result = result.reshape(n_points, 1, 1)

    return result


def clear_lambdify_cache():
    """
    Clear the cached lambdified functions.

    Useful for testing or if memory usage becomes a concern.
    """
    global _lambdify_cache
    _lambdify_cache.clear()
