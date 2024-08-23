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
from .expressions import substitute as fn_substitute_expressions
from .expressions import substitute_expr as fn_substitute_one_expression
from .expressions import is_constant_expr as fn_is_constant_expr
from .expressions import extract_expressions as fn_extract_expressions

# from .expressions import UWconstant_expression as constant


def derivative(expression, *args, **kwargs):
    """Obtain symbolic derivatives of any underworld function, correctly handling sub-expressions / constants"""

    import sympy

    subbed_expr = fn_substitute_expressions(
        expression,
        keep_constants=True,
        return_self=True,
    )

    # # Substitution may return another expression
    # if isinstance(subbed_expr, expression):
    #     subbed_expr = subbed_expr.value

    subbed_derivative = sympy.Derivative(
        subbed_expr,
        *args,
        **kwargs,
        evaluate=True,
    )

    return subbed_derivative
