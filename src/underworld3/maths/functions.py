"""tensor / matrix operations on meshes"""

import sympy
from sympy import sympify
from typing import Optional, Callable
from underworld3 import function
from underworld3 import maths


def delta(
    x: sympy.Basic,
    epsilon: float,
):
    sqrt_2pi = sympy.sqrt(2 * sympy.pi)
    delta_fn = sympy.exp(-(x**2) / (2 * epsilon**2)) / (epsilon * sqrt_2pi)

    return delta_fn


def L2_norm(n_s, a_s, mesh):
    """
    Compute the L2 norm (Euclidean norm) of the difference between numerical and analytical solutions.

    Parameters:
    n_s   : Numeric solution (scalar or vector field)
    a_s   : Analytic solution (scalar or vector field)
    mesh  : The mesh used for integration

    Returns:
    L2 norm value (for scalar or vector)
    """
    # Check if the input is a vector (SymPy Matrix)
    if isinstance(n_s, sympy.Matrix) and n_s.shape[1] > 1:
        # Compute squared difference using dot product for vectors
        squared_difference = (n_s - a_s).dot(n_s - a_s)
    else:
        # Compute squared difference for scalars
        squared_difference = (n_s - a_s) ** 2

    # Integral over the domain
    I = maths.Integral(mesh, squared_difference)

    # Compute the L2 norm
    L2 = sympy.sqrt(I.evaluate())

    return L2


# Definitions of expressions that are used in various places within underworld

vanishing = function.expressions.UWexpression(r"\varepsilon", 1e-18, "vanishingly small value")
