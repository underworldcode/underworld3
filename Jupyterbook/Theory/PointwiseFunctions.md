# PETSc pointwise functions and UW3 functions

As we saw in [the Finite Element Pages], the finite element method provides a very general way to approach the numerical solution of a very wide variety of problems in continuum mechanics using a standardised, matrix-based formulation and with considerable flexibility in the choice of discretisation, mesh geometry, and the ability to deal very naturally with jumps in material properties. 

However, the FEM is based upon a variational, "weak" form of the governing equations of any problem while most publications outline the strong form of any problem and this can make it difficult for anyone without a strong mathematical background to translate quickly from publication to a new model.

PETSc provides a mechanism to automatically generate a finite element weak form from the strong (point-wise) form of the governing equations. This takes the form of a template equation and, in the case of non-linear problems, a template for the corresponding Jacobian derivatives. 

The PETSc strong-form interface asks the user to provide pre-compiled functions with a pre-defined pattern of arguments relating known nodal-point variables, shape-functions and derivatives at any point in the domain.  If the strong form is written 

$$ \mathbf{F}(\mathbf{u}) \sim \nabla.\mathbf{f}_1(u, \nabla u) - f_0 (u, \nabla u) = 0 $$ 

Where the $f_0$ term, generally speaking, represents the forces and $f_1$ comprises flux terms.
Then a corresponding weak form is 

$$ \phi^T \mathbf{F}(\mathbf{u}) \sim \int_\Omega \phi \cdot f_0 \left(u, \nabla u \right) + 
                                                   \nabla \phi \mathbf{f}_1 \left(u, \nabla u \right) = 0 $$


The discrete form of this equation has some obvious similarities to the standard finite element matrix form except that the functions $f_0$ and $\mathbf{f}_1$ are not directly expressed in terms of the basis function matrices: 

$$ \cal{F}(\cal{u}) \sim \sum_e \epsilon_e^T \left[ B^T W f_0(u^q, \nabla u^q) + 
                                                    \sum_k D_k^T W \mathbf{f}_1^k (u^q, \nabla u^q) \right] = 0 
$$

where $q$ represents a set of quadrature points, $W$ is a matrix of weights and $B$, $D$ are the usual basis function matrices but here evaluated at the quadrature points. $\epsilon$ restricts the terms to the element. See {cite}`knepleyAchievingHighPerformance2013` for details.

The user must provide a compiled representation of the terms $f_0$ and $\mathbf{f}_1$ but it is also necessary to provide the corresponding compiled representations of the Jacobians which satisfy

$$ \cal{F'}(\cal{u}) \sim \sum_e \epsilon_e^T 
                \left[ \begin{array}{cc}
                    B^T  & \mathbf{D}^T \\
                \end{array} \right]
                \mathbf{W}
                \left[ \begin{array}{cc} 
                    f_{0,0} & f_{0,1} \\
                    \mathbf{f}_{1,0} & \mathbf{f}_{1,1} \\
                \end{array}\right]
                 \left[ \begin{array}{c}
                    B^T  \\ 
                    \mathbf{D}^T 
                \end{array} \right] \epsilon_e
\quad \mathrm{and} \quad
                f_{[i,j]} = 
                \left[ \begin{array}{cc} 
                    \partial f_0 / \partial u & \partial f_0 / \partial \nabla u \\
                    \partial \mathbf{f}_1 / \partial u & \partial \mathbf{f}_1 / \partial \nabla u
                \end{array}\right]
$$

This formulation greatly simplifies writing optimal code as the user sets up a problem without touching the bulk of the implementation. The only difficulty is ensuring that the Jacobians are derived consistently and correctly. As the Jacobians need to be calculated and implemented as separate functions this introduces a potential point of failure. 

In `underworld`, we provide a fully symbolic approach to specifying the strong form terms, $f_0$ and $\mathbf{f}_1$ using the `sympy` symbolic algebra python package. `sympy` is also able to differentiate
$f_0$ and $\mathbf{f}_1$ to obtain the relevant Jacobian matrices in symbolic form. We also take
advantage of `sympy`s automatic code generation capabilities to compile functions that match 
the PETSc templates. 

This provides a very natural mapping between the formulation above and the `sympy` python code.

```python

        # X and U are sympy position vector, unknown vector 
        # that may include mesh variables 
 
        U_grad = sympy.derive_by_array(U, X)

        F0 = f
        F1 = kappa * U_grad

        # Jacobians are 

        J_00 = sympy.derive_by_array(F0, U)
        J_01 = sympy.derive_by_array(F0, U_grad)
        J_10 = sympy.derive_by_array(F1, U)
        J_11 = sympy.derive_by_array(F1, U_grad)

        # Pass F0, F1, J_xx to sympy code generation routines

        # ...

```

Note: some re-indexing may be necessary to interface between `sympy` and PETSc 
especially for vector problems.

Note: underworld provides a translation between mesh variables and their `sympy`
symbolic representation on the user-facing side that also needs to translate 
to PETSc data structures in the compiled code. 

## Underworld Solver Classes



## Example 1 - The Poisson Equation

## Example 2 - The Scalar Advection-diffusion Equation

## Example 3 - The Stokes Equation

## Example 3a - Navier-Stokes

## Example 4 - Non-linear Constraints


