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
        if isinstance(expression, (sympy.Matrix, sympy.ImmutableMatrix)):
            def f(x):
                d = sympy.sympify(0)
                for t in x.as_ordered_terms():
                    d += sympy.diff(t, variable, evaluate=False)

                return

            derivative = expression.applyfunc(f).as_mutable()

            # f = lambda x: sympy.diff(
            #         x, variable, evaluate=False
            #     )
            # derivative = expression.applyfunc(f)
        else:
            derivative = sympy.diff(
                    expression,
                    variable,
                    evaluate=False,
                )



    return derivative
