r"""
Mathematical helper functions for finite element analysis.

This module provides utility functions for common mathematical operations
in computational mechanics, including smoothed delta functions for
interface problems and error norms for solution verification.

See Also
--------
underworld3.maths.tensors : Tensor notation conversions.
underworld3.maths.vector_calculus : Differential operators.
"""

import sympy
from sympy import sympify
from typing import Optional, Callable
from underworld3 import function
from underworld3 import maths


def delta(
    x: sympy.Basic,
    epsilon: float,
):
    r"""
    Smoothed (Gaussian) approximation to the Dirac delta function.

    Returns a Gaussian with integral 1, approximating :math:`\delta(x)`
    as :math:`\epsilon \to 0`:

    .. math::

        \delta_\epsilon(x) = \frac{1}{\epsilon\sqrt{2\pi}}
        \exp\left(-\frac{x^2}{2\epsilon^2}\right)

    Parameters
    ----------
    x : sympy.Basic
        Symbolic expression (typically a coordinate or distance function).
    epsilon : float
        Smoothing width. Smaller values give sharper peaks.

    Returns
    -------
    sympy.Expr
        Gaussian approximation to the delta function.

    Notes
    -----
    Useful for representing interfaces, point sources, or boundary layers
    in a regularized form suitable for finite element integration.
    """
    sqrt_2pi = sympy.sqrt(2 * sympy.pi)
    delta_fn = sympy.exp(-(x**2) / (2 * epsilon**2)) / (epsilon * sqrt_2pi)

    return delta_fn


def L2_norm(n_s, a_s, mesh):
    r"""
    L2 norm of the difference between numerical and analytical solutions.

    Computes:

    .. math::

        \|n - a\|_{L^2} = \sqrt{\int_\Omega (n - a)^2 \, d\Omega}

    For vector fields, uses the dot product:

    .. math::

        \|\mathbf{n} - \mathbf{a}\|_{L^2} = \sqrt{\int_\Omega
        (\mathbf{n} - \mathbf{a}) \cdot (\mathbf{n} - \mathbf{a}) \, d\Omega}

    Parameters
    ----------
    n_s : sympy.Expr or sympy.Matrix
        Numerical solution (scalar or vector field).
    a_s : sympy.Expr or sympy.Matrix
        Analytical solution (scalar or vector field).
    mesh : Mesh
        The mesh over which to integrate.

    Returns
    -------
    float
        L2 norm of the error.
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
