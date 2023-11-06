"""tensor / matrix operations on meshes"""

import sympy
from sympy import sympify
from typing import Optional, Callable
from underworld3 import function


def delta(
    x: sympy.Basic,
    epsilon: float,
):
    sqrt_2pi = sympy.sqrt(2 * sympy.pi)

    delta_fn = sympy.exp(-(x**2) / (2 * epsilon**2)) / (epsilon * sqrt_2pi)

    return delta_fn
