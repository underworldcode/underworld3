"""
Clean unit-aware wrapper functions for evaluate and global_evaluate.

This module handles all unit conversion logic at the Python layer, providing
a clean interface that automatically converts between dimensional and
non-dimensional coordinate systems.

Architecture:
- Cython layer (_function.pyx): Works ONLY with non-dimensional [0-1] coordinates
  - Functions: evaluate_nd(), global_evaluate_nd()
  - Contract: Expects plain numpy arrays in [0-1] space, returns [0-1] results

- Python wrapper layer (this module): Handles ALL unit conversions
  - Functions: evaluate(), global_evaluate()
  - Responsibilities:
    1) Convert dimensional coords → non-dimensional [0-1]
    2) Call Cython functions with plain arrays
    3) Dimensionalise results [0-1] → dimensional values
    4) Wrap with unit-aware types (UWQuantity or UnitAwareArray)

This clean separation ensures:
- No spaghetti code between Python and Cython layers
- Clear contract: Cython = computation, Python = user interface
- Easy to maintain and test each layer independently
- Single source of truth for unit handling logic
"""

import numpy as np
import underworld3 as uw


def evaluate(
    expr,
    coords,
    coord_sys=None,
    other_arguments=None,
    simplify=True,
    verbose=False,
    evalf=False,
    rbf=False,
    data_layout=None,
    check_extrapolated=False,
):
    """
    Evaluate expression at coordinates with automatic unit handling.

    This function wraps the Cython evaluate_nd implementation to automatically
    handle unit conversions and return unit-aware results.

    When non-dimensional scaling is active (via use_nondimensional_scaling(True)),
    returns raw non-dimensional results without unit wrapping.

    Parameters
    ----------
    expr : sympy expression or UWexpression
        Expression to evaluate
    coords : array-like
        Coordinates at which to evaluate. Can be:
        - numpy array of doubles (shape: n_points x n_dims) in non-dimensional form
        - UnitAwareArray with dimensional coordinates (e.g., from mesh.X.coords)
        - Both work transparently - dimensional coords are auto-converted
    coord_sys : mesh.N vector coordinate system, optional
        Coordinate system to use (default: None)
    other_arguments : dict, optional
        Additional arguments for evaluation (default: None)
    simplify : bool, optional
        Whether to simplify expression (default: True)
    verbose : bool, optional
        Verbose output (default: False)
    evalf : bool, optional
        Force numerical evaluation (default: False)
    rbf : bool, optional
        Use RBF interpolation (default: False)
    data_layout : callable, optional
        Data layout specification (default: None)
    check_extrapolated : bool, optional
        Check for extrapolated values (default: False)

    Returns
    -------
    UWQuantity, UnitAwareArray, or ndarray
        - If non-dimensional scaling is active: plain ndarray (non-dimensional)
        - If expression has units and result is scalar: UWQuantity
        - If expression has units and result is array: UnitAwareArray
        - If expression has no units: plain ndarray (as before)

    Examples
    --------
    >>> # Works with both dimensional and non-dimensional coords
    >>> result = uw.function.evaluate(T.sym, T.coords)  # dimensional coords
    >>> result = uw.function.evaluate(T.sym, mesh.data[:, :2])  # non-dimensional
    >>> if hasattr(result, 'to'):
    ...     result_K = result.to('K')  # Unit conversion
    """
    from ._function import evaluate_nd as _evaluate_nd
    from .unit_conversion import has_units
    from underworld3.units import get_units
    from .expressions import unwrap as fn_unwrap
    from .quantities import quantity, UWQuantity
    from ..utilities.unit_aware_array import UnitAwareArray

    # Step 1: Auto-unwrap EnhancedMeshVariable wrappers
    # Check if expr has .sym property (MeshVariables, etc.) and use that
    # IMPORTANT: Get units BEFORE unwrapping to preserve unit context
    if hasattr(expr, 'sym'):
        # For MeshVariables, get units from the variable itself, not from .sym
        # (since .sym might be a Matrix which doesn't carry units)
        result_units = get_units(expr)  # Get from MeshVariable
        expr_unwrapped = fn_unwrap(expr.sym)
    else:
        result_units = get_units(expr)  # Check units BEFORE unwrap
        expr_unwrapped = fn_unwrap(expr)

    # DEBUG: Print what we got
    if verbose:
        print(f"[evaluate DEBUG] result_units = {result_units}, type = {type(result_units)}")

    # Step 2: Convert dimensional coordinates to non-dimensional [0-1] form
    # Internal mesh KDTrees are built with non-dimensional [0-1] coordinates from PETSc
    # User-facing var.coords properties may have dimensional units (meters)

    # Handle different coordinate input types
    # IMPORTANT: Check for UnitAwareArray BEFORE np.ndarray since it inherits from ndarray
    if isinstance(coords, UnitAwareArray):
        # Unit-aware array - need to non-dimensionalise
        coords_nondim = uw.non_dimensionalise(coords)
        coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
    elif isinstance(coords, np.ndarray):
        # Plain numpy array - assume it's already [0-1] non-dimensional
        coords_for_eval = np.asarray(coords, dtype=np.double)
    else:
        # Other type - try to non-dimensionalise
        coords_nondim = uw.non_dimensionalise(coords)
        coords_for_eval = np.asarray(coords_nondim, dtype=np.double)

    # Step 3: Call Cython implementation with plain non-dimensional arrays
    # NOTE: Returns NON-DIMENSIONAL values [0-1] since we queried
    # the KDTree with non-dimensional coordinates
    raw_result_nondim = _evaluate_nd(
        expr_unwrapped,
        coords_for_eval,
        coord_sys=coord_sys,
        other_arguments=other_arguments,
        simplify=simplify,
        verbose=verbose,
        evalf=evalf,
        rbf=rbf,
        data_layout=data_layout,
        check_extrapolated=check_extrapolated,
    )

    # Step 4: Check if non-dimensional scaling is active
    # When active, return raw results without unit wrapping
    if uw.is_nondimensional_scaling_active():
        if check_extrapolated:
            return raw_result_nondim[0], raw_result_nondim[1]
        else:
            return raw_result_nondim

    # Step 5: Use result_units obtained before unwrapping (Step 1)
    if result_units is None:
        # No units - return as-is (backward compatible)
        if check_extrapolated:
            return raw_result_nondim[0], raw_result_nondim[1]
        else:
            return raw_result_nondim

    # Step 6: Expression has units - need to dimensionalise results
    # The KDTree evaluation returned non-dimensional values [0-1],
    # but we need to convert them to dimensional values (e.g., 3000 K)
    # before wrapping with units

    # Unpack extrapolation flag if needed
    if check_extrapolated:
        raw_values, extrapolated = raw_result_nondim
    else:
        raw_values = raw_result_nondim
        extrapolated = None

    # Convert result_units string to Pint Unit object if needed
    from ..scaling import units as ureg
    if isinstance(result_units, str):
        result_units = ureg(result_units)

    # Get dimensionality from units (result_units is now a Pint Unit object)
    pint_qty = 1.0 * result_units
    dimensionality = dict(pint_qty.dimensionality)

    # Dimensionalise the non-dimensional result
    # This converts [0-1] → physical values using model scales
    raw_result_dimensional = uw.dimensionalise(
        raw_values,
        target_dimensionality=dimensionality,
    )

    # Step 7: Wrap the dimensional result with units
    if verbose:
        print(f"[evaluate DEBUG] Before wrapping:")
        print(f"  raw_result_dimensional type: {type(raw_result_dimensional)}")
        print(f"  raw_result_dimensional shape: {np.asarray(raw_result_dimensional).shape if not np.isscalar(raw_result_dimensional) else 'scalar'}")
        print(f"  result_units: {result_units}")

    # Check if dimensionalise() already returned a unit-aware object
    # If so, use it directly instead of wrapping again
    if hasattr(raw_result_dimensional, 'units') and raw_result_dimensional.units is not None:
        # Already unit-aware (from dimensionalise)
        wrapped_result = raw_result_dimensional
    elif np.isscalar(raw_result_dimensional):
        # Scalar result - wrap as UWQuantity
        wrapped_result = quantity(float(raw_result_dimensional), result_units)
    else:
        # Array result - wrap as UnitAwareArray
        wrapped_result = UnitAwareArray(
            np.asarray(raw_result_dimensional), units=result_units
        )

    if verbose:
        print(f"[evaluate DEBUG] After wrapping:")
        print(f"  wrapped_result type: {type(wrapped_result)}")
        print(f"  wrapped_result.units: {getattr(wrapped_result, 'units', 'NO ATTRIBUTE')}")

    # Return result with extrapolation flag if requested
    if check_extrapolated:
        return wrapped_result, extrapolated
    else:
        return wrapped_result


def global_evaluate(
    expr,
    coords=None,
    coord_sys=None,
    other_arguments=None,
    simplify=True,
    verbose=False,
    evalf=False,
    rbf=False,
    data_layout=None,
    check_extrapolated=False,
):
    """
    Global evaluate with automatic unit-aware results.

    Similar to evaluate() but performs global evaluation across all processes.
    Returns unit-aware objects when expression has units.

    When non-dimensional scaling is active (via use_nondimensional_scaling(True)),
    returns raw non-dimensional results without unit wrapping.

    Parameters
    ----------
    expr : sympy expression or UWexpression
        Expression to evaluate
    coords : array-like, optional
        Coordinates at which to evaluate (default: None)
    coord_sys : CoordinateSystem, optional
        Coordinate system to use (default: None)
    other_arguments : dict, optional
        Additional arguments for evaluation (default: None)
    simplify : bool, optional
        Whether to simplify expression (default: True)
    verbose : bool, optional
        Verbose output (default: False)
    evalf : bool, optional
        Force numerical evaluation (default: False)
    rbf : bool, optional
        Use RBF interpolation (default: False)
    data_layout : callable, optional
        Data layout specification (default: None)
    check_extrapolated : bool, optional
        Check for extrapolated values (default: False)

    Returns
    -------
    UWQuantity, UnitAwareArray, or ndarray
        - If non-dimensional scaling is active: plain ndarray (non-dimensional)
        - Otherwise: result with appropriate unit tracking
    """
    from ._function import global_evaluate_nd as _global_evaluate_nd
    from ..units import get_units
    from .quantities import quantity, UWQuantity
    from ..utilities.unit_aware_array import UnitAwareArray

    # Call the original Cython implementation with all parameters
    raw_result = _global_evaluate_nd(
        expr,
        coords=coords,
        coord_sys=coord_sys,
        other_arguments=other_arguments,
        simplify=simplify,
        verbose=verbose,
        evalf=evalf,
        rbf=rbf,
        data_layout=data_layout,
        check_extrapolated=check_extrapolated,
    )

    # Check if non-dimensional scaling is active
    # When active, return raw results without unit wrapping
    if uw.is_nondimensional_scaling_active():
        return raw_result

    # Try to get units from the expression
    result_units = get_units(expr)

    if result_units is None:
        # No units - return as-is
        return raw_result

    # Expression has units - wrap result
    if np.isscalar(raw_result):
        # Scalar result - wrap as UWQuantity
        return quantity(float(raw_result), result_units)
    else:
        # Array result - wrap as UnitAwareArray
        return UnitAwareArray(raw_result, units=result_units)
