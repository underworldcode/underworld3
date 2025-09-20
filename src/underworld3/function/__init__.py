from . import analytic
from ._function import (
    UnderworldFunction,
    global_evaluate,
    evaluate,
    dm_swarm_get_migrate_type,
    dm_swarm_set_migrate_type,
    _dmswarm_get_migrate_type,
    _dmswarm_set_migrate_type,
    # evalf_at_coords,
    # _interpolate_all_vars_on_mesh,
)

from .expressions import UWexpression as expression
from .expressions import UWDerivativeExpression as _derivative_expression
from .expressions import substitute as fn_substitute_expressions
from .expressions import unwrap as fn_unwrap
from .expressions import substitute_expr as fn_substitute_one_expression
from .expressions import is_constant_expr as fn_is_constant_expr
from .expressions import extract_expressions as fn_extract_expressions
from .expressions import extract_expressions as fn_extract_expressions_and_functions
from .expressions import mesh_vars_in_expression as fn_mesh_vars_in_expression

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
        if hasattr(expression, 'diff'):
            derivative = expression.diff(variable, evaluate=False)
        else:
            # Fallback for non-UWexpression objects
            import sympy
            latex_expr = sympy.latex(expression)
            latex_diff_variable = sympy.latex(variable)
            latex = (
                r"\partial \left[" + latex_expr + r"\right] / \partial " + latex_diff_variable
            )
            
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
