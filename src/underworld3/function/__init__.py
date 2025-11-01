from . import analytic
from ._function import (
    UnderworldFunction,
    global_evaluate as _global_evaluate_original,
    evaluate as _evaluate_original,
    dm_swarm_get_migrate_type,
    dm_swarm_set_migrate_type,
    _dmswarm_get_migrate_type,
    _dmswarm_set_migrate_type,
    # evalf_at_coords,
    # _interpolate_all_vars_on_mesh,
)

# Import unit conversion utilities
from .unit_conversion import (
    make_evaluate_unit_aware,
    # Expose utility functions for user convenience
    convert_quantity_units,
    detect_quantity_units,
    make_dimensionless,
    convert_array_units,
    auto_convert_to_mesh_units,
    convert_evaluation_result,
    add_units,
    has_units,
    get_units,
)


# Wrap evaluate to return unit-aware objects
def evaluate(expr, coords, **kwargs):
    """
    Evaluate expression at coordinates, returning unit-aware results.

    This function wraps the Cython evaluate implementation to automatically
    return UWQuantity or UnitAwareArray objects when the expression has units.

    When non-dimensional scaling is active (via use_nondimensional_scaling(True)),
    returns raw non-dimensional results without unit wrapping, as required by
    solver operations like semi-Lagrangian advection.

    **Smart Unit Handling**: This function intelligently handles coordinates with
    or without units. If dimensional coordinates are passed (e.g., from T.coords
    or mesh.X.coords), they are automatically converted to non-dimensional form
    for evaluation. The symbolic expressions remain purely dimensionless.

    Parameters
    ----------
    expr : sympy expression or UWexpression
        Expression to evaluate
    coords : array-like
        Coordinates at which to evaluate. Can be dimensional (with units) or
        non-dimensional (plain arrays). Both work transparently.
    **kwargs : dict
        Additional arguments passed to underlying evaluate function

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
    >>> result = uw.function.evaluate(T.sym, mesh.data[:,  :2])  # non-dimensional
    >>> result.to('K')  # Convert to Kelvin
    >>> print(f"Temperature: {result}")
    """
    import numpy as np
    import underworld3 as uw

    # SMART COORDINATE HANDLING: Always non-dimensionalise coords before evaluation
    # This is safe because non_dimensionalise() is idempotent:
    #   - If coords are already dimensionless → returns them unchanged
    #   - If coords have units → converts to dimensionless
    # Symbolic expressions are purely dimensionless, so coords must match
    coords_for_eval = uw.non_dimensionalise(coords)

    # Ensure we have a plain array (non_dimensionalise might return UnitAwareArray)
    if not isinstance(coords_for_eval, np.ndarray):
        coords_for_eval = np.asarray(coords_for_eval)

    # Call the original Cython implementation with non-dimensional coords
    raw_result = _evaluate_original(expr, coords_for_eval, **kwargs)

    # Check if non-dimensional scaling is active
    # When active, return raw results without unit wrapping
    if uw.is_nondimensional_scaling_active():
        return raw_result

    # Try to get units from the expression
    result_units = get_units(expr)

    if result_units is None:
        # No units - return as-is (backward compatible)
        return raw_result

    # Expression has units - wrap result
    if np.isscalar(raw_result):
        # Scalar result - wrap as UWQuantity
        return quantity(float(raw_result), result_units)
    else:
        # Array result - wrap as UnitAwareArray
        from ..utilities.unit_aware_array import UnitAwareArray

        return UnitAwareArray(raw_result, units=result_units)


# Wrap global_evaluate similarly
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
    Global evaluate with unit-aware results.

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
    data_layout : str, optional
        Data layout specification (default: None)
    check_extrapolated : bool, optional
        Check for extrapolated values (default: False)

    Returns
    -------
    UWQuantity, UnitAwareArray, or ndarray
        - If non-dimensional scaling is active: plain ndarray (non-dimensional)
        - Otherwise: result with appropriate unit tracking
    """
    import numpy as np
    import underworld3 as uw

    # Call the original Cython implementation with all parameters
    raw_result = _global_evaluate_original(
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
        from ..utilities.unit_aware_array import UnitAwareArray

        return UnitAwareArray(raw_result, units=result_units)


from .expressions import UWexpression as expression
from .expressions import UWDerivativeExpression as _derivative_expression
from .quantities import quantity, UWQuantity
from .expressions import substitute as fn_substitute_expressions
from .expressions import unwrap as fn_unwrap
from .expressions import substitute_expr as fn_substitute_one_expression
from .expressions import is_constant_expr as fn_is_constant_expr
from .expressions import extract_expressions as fn_extract_expressions
from .expressions import extract_expressions as fn_extract_expressions_and_functions
from .expressions import mesh_vars_in_expression as fn_mesh_vars_in_expression


def with_units(sympy_expr, name=None):
    """
    Wrap a SymPy expression in a unit-aware object with .units and .to() methods.

    This is particularly useful for derivatives and other SymPy operations that
    return plain SymPy expressions instead of unit-aware objects.

    Parameters
    ----------
    sympy_expr : sympy expression
        The SymPy expression to wrap (e.g., from temperature.diff(y))
    name : str, optional
        Optional name for the expression (auto-generated if not provided)

    Returns
    -------
    UWexpression
        Unit-aware expression with .units and .to() methods

    Examples
    --------
    >>> # Derivative returns plain SymPy
    >>> dTdy_sympy = temperature.diff(y)[0]
    >>>
    >>> # Wrap it to get unit-aware object
    >>> dTdy = uw.with_units(dTdy_sympy)
    >>> dTdy.units  # kelvin / kilometer
    >>> dTdy.to("K/mm")  # Unit conversion works!

    >>> # Also works with other SymPy operations
    >>> combined = uw.with_units(2 * temperature.sym + pressure.sym)
    """
    import sympy

    # Auto-generate name if not provided
    if name is None:
        # Try to create a reasonable name from the expression
        name = str(sympy_expr)[:50]  # Truncate if too long

    # Extract units from the expression
    units_str = get_units(sympy_expr)

    # Create description from units if available
    if units_str:
        description = f"Expression with units: {units_str}"
    else:
        description = "Dimensionless expression"

    # Wrap in UWexpression
    return expression(name, sympy_expr, description, units=units_str)


# from .expressions import UWconstant_expression as constant


def derivative(expression, variable, evaluate=True):
    """
    Obtain symbolic derivatives of any underworld function, correctly handling sub-expressions / constants.

    Note: This function is maintained for backward compatibility. The recommended approach
    is to use the natural syntax: expression.diff(variable, evaluate=evaluate)

    Args:
        expression: The expression to differentiate
        variable: The variable to differentiate with respect to
        evaluate (bool): If True, evaluate immediately. If False, return deferred derivative.

    Returns:
        The derivative (evaluated SymPy expression or UWDerivativeExpression)
    """
    import sympy

    if evaluate:
        subbed_expr = fn_unwrap(
            expression,
            keep_constants=False,
            return_self=False,
        )

        derivative = sympy.diff(
            subbed_expr,
            variable,
            evaluate=True,
        )

    else:
        # Use the new natural syntax internally for deferred derivatives
        if hasattr(expression, "diff"):
            derivative = expression.diff(variable, evaluate=False)
        else:
            # Fallback for non-UWexpression objects
            import sympy

            latex_expr = sympy.latex(expression)
            latex_diff_variable = sympy.latex(variable)
            latex = r"\partial \left[" + latex_expr + r"\right] / \partial " + latex_diff_variable

            # Handle vector derivatives
            try:
                rows, cols = sympy.Matrix(variable).shape
            except TypeError:
                rows, cols = (1, 1)

            if rows == 1 and cols == 1:
                derivative = _derivative_expression(latex, expression, variable)
            else:
                derivative = sympy.Matrix.zeros(rows=rows, cols=cols)
                for i in range(rows):
                    for j in range(cols):
                        latex = (
                            r"\partial \left["
                            + sympy.latex(expression)
                            + r"\right] / \partial "
                            + sympy.latex(variable[i, j])
                        )
                        derivative[i, j] = _derivative_expression(latex, expression, variable[i, j])

    return derivative
