"""tensor / matrix operations on meshes"""

import sympy
from sympy import sympify
from typing import Optional, Callable
from underworld3 import function
from underworld3.cython.petsc_maths import Integral


def delta(
    x: sympy.Basic,
    epsilon: float,
):
    sqrt_2pi = sympy.sqrt(2 * sympy.pi)

    delta_fn = sympy.exp(-(x**2) / (2 * epsilon**2)) / (epsilon * sqrt_2pi)

    return delta_fn



def L2_norm(U_exp, A_exp, mesh):
    '''
    Compute the L2 norm between the computed and analytical fields, both of which are sympy.Matrix objects.

    The L2-norm is defined as:

    .. math::
        L_2\text{-norm} = \left( \int_{\Omega} |u_{\text{numeric}}(x) - u_{\text{analytic}}(x)|^2 \, dx \right)^{1/2}

    Where:
    .. math::
        - \( u_{\text{numeric}}(x) \) is the numerical solution.
        - \( u_{\text{analytic}}(x) \) is the analytical solution.
        - \( \Omega \) is the domain.
        - The integral computes the total squared error across the entire domain, and the square root is applied after the integration to give the L2-norm.

    Parameters:
    U_exp : sympy.Matrix representing a scalar (1x1) or vector field (Nx1 or 1xN)
    A_exp : sympy.Matrix representing a scalar (1x1) or vector field (Nx1 or 1xN)
    mesh : the mesh over which to integrate
    
    Returns:
    L2_norm : the computed L2 norm
    '''
    # Check if the inputs are matrices
    if isinstance(U_exp, sympy.Matrix) and isinstance(A_exp, sympy.Matrix):
        # Ensure that the dimensions of U_exp and A_exp match
        assert U_exp.shape == A_exp.shape, "U_exp and A_exp must have the same shape."
        
        # Initialize the squared difference
        squared_diff = 0
        
        # Loop over the components of the matrices (scalar or vector case)
        for i in range(U_exp.shape[0]):
            squared_diff += (U_exp[i] - A_exp[i])**2

    else:
        raise TypeError("U_exp and A_exp must be sympy.Matrix objects.")
    
    # Perform the integral over the mesh
    I = Integral(mesh, squared_diff)
    
    # Compute the L2 norm (sqrt of the integral result)
    L2_norm = sympy.sqrt( I.evaluate() )
    
    return L2_norm
