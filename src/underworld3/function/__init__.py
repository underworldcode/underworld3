from . import analytic
from ._function import (
    UnderworldFunction,
    evaluate,
    evalf,
    dm_swarm_get_migrate_type,
    dm_swarm_set_migrate_type,
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

# from .expressions import UWconstant_expression as constant


def derivative(expression, variable, evaluate=True):
    """Obtain symbolic derivatives of any underworld function, correctly handling sub-expressions / constants"""

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
        derivative = deferred_derivative(expression, variable)

    return derivative


def deferred_derivative(expr, diff_variable):

    import sympy

    latex_expr = sympy.latex(expr)
    latex_diff_variable = sympy.latex(diff_variable)
    latex = (
        r"\partial \left[" + latex_expr + r"\right] / \partial " + latex_diff_variable
    )

    if isinstance(diff_variable, expression):
        diff_variable = diff_variable.sym

    # We need to return a Matrix of \partial Expr \partial {diff_variable_i}

    try:
        rows, cols = sympy.Matrix(diff_variable).shape
    except TypeError:
        rows, cols = (1, 1)

    # Good question: should we return a 1x1 matrix or the actual derivative ?
    if rows == 1 and cols == 1:
        # ddx = sympy.Matrix((_derivative_expression(latex, expr, diff_variable),))
        ddx = _derivative_expression(latex, expr, diff_variable)
    else:
        ddx = sympy.Matrix.zeros(rows=rows, cols=cols)
        for i in range(rows):
            for j in range(cols):
                latex = (
                    r"\partial \left["
                    + sympy.latex(expr)
                    + r"\right] / \partial "
                    + sympy.latex(diff_variable[i, j])
                )
                ddx[i, j] = _derivative_expression(latex, expr, diff_variable[i, j])

    return ddx
