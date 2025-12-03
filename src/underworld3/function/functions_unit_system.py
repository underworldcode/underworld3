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


@uw.timing.routine_timer_decorator
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
    from .expressions import unwrap_for_evaluate  # NEW: Use evaluate-specific unwrapper
    from .quantities import quantity, UWQuantity
    from ..utilities.unit_aware_array import UnitAwareArray
    from .pure_sympy_evaluator import is_pure_sympy_expression, evaluate_pure_sympy
    from .expressions import UWexpression

    # Step 1: UNWRAP to canonical form (preprocessing/compiler IR)
    # This converts ALL expressions to a standardized form:
    # - UWexpressions substituted with base SI numeric values
    # - Result dimensionality tracked for correct unit wrapping
    # This is the single source of truth for unit handling!
    from .expressions import unwrap_for_evaluate

    scaling_is_active = uw.is_nondimensional_scaling_active()
    expr_unwrapped, result_dimensionality = unwrap_for_evaluate(expr, scaling_active=scaling_is_active)

    # IMPORTANT: Check if the unwrapped value is already in dimensional form.
    # This only applies when scaling is NOT active - in that case, unwrap_for_evaluate()
    # returns expr.value (dimensional), not expr.data (non-dimensional).
    #
    # When scaling IS active, unwrap_for_evaluate() always returns ND values (expr.data),
    # so we MUST re-dimensionalize regardless of whether the original expression has units.
    #
    # The case where this matters:
    #   T_anal = T_left + (T_right - T_left) * sympy.erf(theta)
    # where T_left, T_right are UWQuantity with units 'K'.
    # Without scaling: unwrap returns dimensional values (300 K) - don't re-dim
    # With scaling: unwrap returns ND values (0.3 if T_scale=1000K) - MUST re-dim
    expr_already_dimensional = (
        not scaling_is_active
        and isinstance(expr, UWexpression)
        and expr.has_units
    )

    # Step 2: OPTIMIZE - Check if unwrapped form is pure sympy (no mesh data)
    # If so, use fast lambdified evaluation instead of full RBF machinery
    is_pure_sympy, free_symbols, symbol_type = is_pure_sympy_expression(expr_unwrapped)

    if is_pure_sympy:
        # Pure sympy expression - use optimized lambdify path
        # Note: expr_unwrapped is already in canonical form (base SI units)

        # Convert coordinates
        if isinstance(coords, UnitAwareArray):
            coords_nondim = uw.non_dimensionalise(coords)
            coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, UWQuantity):
            coords_nondim = uw.non_dimensionalise(coords)
            if hasattr(coords_nondim, 'value'):
                coords_for_eval = np.asarray(coords_nondim.value, dtype=np.double)
            else:
                coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, np.ndarray):
            coords_for_eval = np.asarray(coords, dtype=np.double)
        else:
            coords_nondim = uw.non_dimensionalise(coords)
            coords_for_eval = np.asarray(coords_nondim, dtype=np.double)

        # Evaluate using optimized lambdification
        raw_values = evaluate_pure_sympy(expr_unwrapped, coords_for_eval)

        # Step 3: Re-dimensionalize and wrap with units
        # GATEWAY PRINCIPLE: evaluate() ALWAYS returns dimensional values when units are known
        result_units = get_units(expr)
        if result_units is not None:
            if expr_already_dimensional:
                # Expression was built from UWQuantity arithmetic - values are ALREADY dimensional
                # Just wrap with units, do NOT re-dimensionalise
                if np.isscalar(raw_values):
                    result_with_units = quantity(float(raw_values), result_units)
                else:
                    result_with_units = UnitAwareArray(raw_values, units=result_units)

                if check_extrapolated:
                    extrapolated = np.full((coords_for_eval.shape[0],), False, dtype=bool)
                    return result_with_units, extrapolated
                else:
                    return result_with_units
            elif result_dimensionality is not None:
                # Values are non-dimensional - need to re-dimensionalise
                try:
                    # Re-dimensionalize: ND value → dimensional value
                    # This multiplies by the appropriate reference scale
                    dimensional_result = uw.dimensionalise(raw_values, target_dimensionality=result_dimensionality)

                    if check_extrapolated:
                        extrapolated = np.full((coords_for_eval.shape[0],), False, dtype=bool)
                        return dimensional_result, extrapolated
                    else:
                        return dimensional_result
                except Exception:
                    # If dimensionalise fails, wrap raw values with units
                    if np.isscalar(raw_values):
                        result_with_units = quantity(float(raw_values), result_units)
                    else:
                        result_with_units = UnitAwareArray(raw_values, units=result_units)

                    if check_extrapolated:
                        extrapolated = np.full((coords_for_eval.shape[0],), False, dtype=bool)
                        return result_with_units, extrapolated
                    else:
                        return result_with_units
            else:
                # Has units but no dimensionality - just wrap with units
                if np.isscalar(raw_values):
                    result_with_units = quantity(float(raw_values), result_units)
                else:
                    result_with_units = UnitAwareArray(raw_values, units=result_units)

                if check_extrapolated:
                    extrapolated = np.full((coords_for_eval.shape[0],), False, dtype=bool)
                    return result_with_units, extrapolated
                else:
                    return result_with_units

        # No dimensionality - return plain array
        if check_extrapolated:
            extrapolated = np.full((coords_for_eval.shape[0],), False, dtype=bool)
            return raw_values, extrapolated
        else:
            return raw_values

    # Step 1: Unwrap expression with proper unit handling for evaluate path
    # NEW: unwrap_for_evaluate() handles:
    # - UWexpression composition: Recursively unwraps (rho0 * alpha) etc.
    # - Constants: Non-dimensionalized (e.g., 1500 K → 1.0)
    # - Variables: Preserved as symbols (already ND in PETSc)
    # - Dimensionality: Tracked for re-dimensionalization
    scaling_is_active = uw.is_nondimensional_scaling_active()
    expr_unwrapped, result_dimensionality = unwrap_for_evaluate(expr, scaling_active=scaling_is_active)

    # Step 2: Convert dimensional coordinates to non-dimensional [0-1] form
    # Internal mesh KDTrees are built with non-dimensional [0-1] coordinates from PETSc
    # User-facing var.coords properties may have dimensional units (meters)

    # Handle different coordinate input types
    # IMPORTANT: Check for UnitAwareArray BEFORE np.ndarray since it inherits from ndarray
    if isinstance(coords, UnitAwareArray):
        # Unit-aware array - need to non-dimensionalise
        coords_nondim = uw.non_dimensionalise(coords)
        coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
    elif isinstance(coords, UWQuantity):
        # UWQuantity from arithmetic operations (e.g., coords - dt * velocity)
        coords_nondim = uw.non_dimensionalise(coords)
        if hasattr(coords_nondim, 'value'):
            coords_for_eval = np.asarray(coords_nondim.value, dtype=np.double)
        else:
            coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
    elif isinstance(coords, np.ndarray):
        # Plain numpy array - assume it's already [0-1] non-dimensional
        coords_for_eval = np.asarray(coords, dtype=np.double)
    else:
        # Other type - try to non-dimensionalise
        coords_nondim = uw.non_dimensionalise(coords)
        coords_for_eval = np.asarray(coords_nondim, dtype=np.double)

    # Ensure coordinates are 2D: shape (N, ndim) not (ndim,)
    # This handles single coordinate evaluation: coords[60] -> shape (2,) -> (1, 2)
    if coords_for_eval.ndim == 1:
        coords_for_eval = np.atleast_2d(coords_for_eval)

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

    # Step 4: Unpack extrapolation flag if needed
    if check_extrapolated:
        raw_values, extrapolated = raw_result_nondim
    else:
        raw_values = raw_result_nondim
        extrapolated = None

    # Step 5: Re-dimensionalize and wrap with units
    # GATEWAY PRINCIPLE: evaluate() ALWAYS returns dimensional values when units are known
    # The user sees dimensional results; non-dimensional is for internal solver use only

    if result_dimensionality is not None:
        # We have unit information - re-dimensionalize and wrap the result
        result_units = get_units(expr)
        if result_units is not None:
            try:
                # Re-dimensionalize: ND value → dimensional value
                dimensional_result = uw.dimensionalise(raw_values, target_dimensionality=result_dimensionality)

                if check_extrapolated:
                    return dimensional_result, extrapolated
                else:
                    return dimensional_result
            except Exception:
                # If dimensionalise fails, wrap raw values with units
                if np.isscalar(raw_values):
                    result_with_units = quantity(float(raw_values), result_units)
                else:
                    result_with_units = UnitAwareArray(raw_values, units=result_units)

                if check_extrapolated:
                    return result_with_units, extrapolated
                else:
                    return result_with_units

    # No dimensionality info - return plain array (no units)
    if check_extrapolated:
        return raw_values, extrapolated
    else:
        return raw_values


@uw.timing.routine_timer_decorator
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
    from .pure_sympy_evaluator import is_pure_sympy_expression, evaluate_pure_sympy

    # OPTIMIZATION: Check if this is a pure sympy expression (no UW3 variable data)
    # If so, use fast lambdified evaluation instead of full RBF machinery
    # This includes expressions with mesh coordinates (BaseScalar) or pure symbols
    is_pure_sympy, free_symbols, symbol_type = is_pure_sympy_expression(expr)

    if is_pure_sympy and coords is not None and (rbf or evalf):
        # Pure sympy expression with explicit coords - use optimized path
        # This is the same as evaluate() but for global context

        # Convert coordinates
        if isinstance(coords, UnitAwareArray):
            coords_nondim = uw.non_dimensionalise(coords)
            coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, UWQuantity):
            coords_nondim = uw.non_dimensionalise(coords)
            if hasattr(coords_nondim, 'value'):
                coords_for_eval = np.asarray(coords_nondim.value, dtype=np.double)
            else:
                coords_for_eval = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, np.ndarray):
            coords_for_eval = np.asarray(coords, dtype=np.double)
        else:
            coords_nondim = uw.non_dimensionalise(coords)
            coords_for_eval = np.asarray(coords_nondim, dtype=np.double)

        # Evaluate using optimized lambdification
        raw_result = evaluate_pure_sympy(expr, coords_for_eval)

        # Handle units (same logic as evaluate())
        if uw.is_nondimensional_scaling_active():
            return raw_result

        result_units = get_units(expr)
        if result_units is None:
            return raw_result

        if np.isscalar(raw_result):
            return quantity(float(raw_result), result_units)
        else:
            return UnitAwareArray(raw_result, units=result_units)

    # CRITICAL: Extract base numpy array from UnitAwareArray before passing to Cython
    # Cython code uses typed memory views that don't work with numpy subclasses
    if coords is not None:
        if isinstance(coords, UnitAwareArray):
            # Extract base array and non-dimensionalize if needed
            coords_nondim = uw.non_dimensionalise(coords)
            coords_for_cython = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, UWQuantity):
            # UWQuantity from arithmetic operations (e.g., coords - dt * velocity)
            # Extract the underlying value and non-dimensionalize
            coords_nondim = uw.non_dimensionalise(coords)
            # coords_nondim might be a scalar or array - ensure it's an array
            if hasattr(coords_nondim, 'value'):
                coords_for_cython = np.asarray(coords_nondim.value, dtype=np.double)
            else:
                coords_for_cython = np.asarray(coords_nondim, dtype=np.double)
        elif isinstance(coords, np.ndarray):
            coords_for_cython = np.asarray(coords, dtype=np.double)
        else:
            coords_for_cython = coords
    else:
        coords_for_cython = coords

    # CRITICAL FIX (2025-11-28): Match evaluate() gateway behavior.
    # global_evaluate must return dimensional results, just like evaluate().
    # Previously returned raw ND values when scaling was active - this was WRONG.

    # Step 1: Track dimensionality for re-dimensionalization (same as evaluate())
    # Note: We pass the ORIGINAL expression to _global_evaluate_nd, not an unwrapped version,
    # because _global_evaluate_nd expects the raw expression and handles evaluation internally.
    from .expressions import unwrap_for_evaluate
    scaling_is_active = uw.is_nondimensional_scaling_active()
    _, result_dimensionality = unwrap_for_evaluate(expr, scaling_active=scaling_is_active)

    # Call the original Cython implementation with all parameters
    # Note: Pass original expr, not unwrapped - _global_evaluate_nd handles the expression directly
    raw_result = _global_evaluate_nd(
        expr,  # Use original expression - Cython handles it
        coords=coords_for_cython,
        coord_sys=coord_sys,
        other_arguments=other_arguments,
        simplify=simplify,
        verbose=verbose,
        evalf=evalf,
        rbf=rbf,
        data_layout=data_layout,
        check_extrapolated=check_extrapolated,
    )

    # Step 2: Re-dimensionalize and wrap with units (GATEWAY PRINCIPLE)
    # global_evaluate() ALWAYS returns dimensional values when units are known,
    # exactly like evaluate(). The user sees dimensional results.

    if result_dimensionality is not None:
        # We have unit information - re-dimensionalize and wrap the result
        result_units = get_units(expr)
        if result_units is not None:
            try:
                # Re-dimensionalize: ND value → dimensional value
                dimensional_result = uw.dimensionalise(raw_result, target_dimensionality=result_dimensionality)
                return dimensional_result
            except Exception:
                # Fall through to simple wrapping if dimensionalise fails
                pass

    # Fallback: Try to get units from expression and wrap
    result_units = get_units(expr)

    if result_units is None:
        # No units - return as-is
        return raw_result

    # Expression has units but couldn't dimensionalize - wrap with units
    if np.isscalar(raw_result):
        # Scalar result - wrap as UWQuantity
        return quantity(float(raw_result), result_units)
    else:
        # Array result - wrap as UnitAwareArray
        return UnitAwareArray(raw_result, units=result_units)
