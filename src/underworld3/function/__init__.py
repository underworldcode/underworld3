from . import analytic
# Import the _function module to expose it in the namespace (needed by expressions.py)
from . import _function
from ._function import (
    UnderworldFunction,
    global_evaluate_nd as _global_evaluate_nd,
    evaluate_nd as _evaluate_nd,
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
    # NOTE: get_units has been moved to units module - use uw.get_units() or import from units
)


# Import clean unit-aware wrapper functions
from .functions_unit_system import (
    evaluate,
    global_evaluate,
)


from .expressions import UWexpression as expression
from .expressions import UWDerivativeExpression as _derivative_expression
from .quantities import quantity, UWQuantity
from .expressions import substitute as fn_substitute_expressions
from .expressions import unwrap as fn_unwrap
from .expressions import expand as fn_expand
from .expressions import substitute_expr as fn_substitute_one_expression
from .expressions import is_constant_expr as fn_is_constant_expr
from .expressions import extract_expressions as fn_extract_expressions
from .expressions import extract_expressions as fn_extract_expressions_and_functions
from .expressions import mesh_vars_in_expression as fn_mesh_vars_in_expression


def with_units(sympy_expr, name=None, units=None):
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
    units : str, optional
        Explicit units for the expression. If provided, these units are used
        instead of trying to extract units from the expression. This is useful
        for derivatives where units are known from the derivative operation.

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

    >>> # Explicit units for derivatives
    >>> dTdy_elem = uw.with_units(derivative_expr, units="kelvin / kilometer")
    """
    import sympy

    # Auto-generate name if not provided
    if name is None:
        # Try to create a reasonable name from the expression
        name = str(sympy_expr)[:50]  # Truncate if too long

    # Use explicit units if provided, otherwise extract from expression
    if units is None:
        # Extract units from the expression (import from unified units module)
        from ..units import get_units
        units_str = get_units(sympy_expr)
    else:
        units_str = units

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
