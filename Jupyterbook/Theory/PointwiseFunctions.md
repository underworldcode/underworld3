# PETSc pointwise functions and solvers

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

We provide 3 base classes to build solvers. These are a scalar SNES solver, 
a vector SNES solver and a Vector SNES saddle point solver (constrained vector problem).
These are bare-bones classes that implement the pointwise function / sympy approach that
can then be used to build solvers for many common situations. 

A blank slate is a very scary thing and so we provide templates for some common equations
and examples to show how these can be extended. 

## Example 1 - The Poisson Equation

The classical form of the scalar Poisson Equation is 

$$ \alpha \nabla^2 \psi = f $$ 

Where $\psi$ is an unknown scalar quantity, $\alpha$ is 
a constitutive parameter that relates gradients to fluxes, and $f$ 
is a source term.

This equation is obtained by considering the divergence of fluxes needed to 
balance the sources. For example, in thermal diffusion we identify
$\psi$ with the temperature, $T$, and the constitutive parameter, $k$,
is a thermal conductivity. 

$$ \nabla \cdot k \nabla T = h $$

In this form, $\mathbf{q} = k \nabla T$ is Fourier's expression of the 
heat flux in terms of temperature gradients.

This form matches the template above if we identify:

$$ f_0 = -h \quad \textrm{and} \quad f_1 = k\nabla T$$ 

and, in fact, this is exactly what we need to specify in the underworld equation
system. 

```python 
        solver._L= sympy.derive_by_array(solver._U, solver._X).transpose()

        # f0 residual term (weighted integration) - scalar function
        solver.F0 = -h

        # f1 residual term (integration by parts / gradients)
        solver.F1 = k * solver._L
```

which means the user only needs to supply a mesh, a mesh variable to 
hold the solution and sympy expressions for $k$ and $h$ in order
to solve a Poisson equation. 

The `SNES_Poisson` class is a very lightweight wrapper on
the `SNES_Scalar` class which provides a template for the flux
term and very little else. 
$F_0$ and $F_1$ are inherited as an empty scalar and vector respectively. 
These are available in the template for the user to extend the equation as needed.

[This notebook](../Notebooks/Ex_Poisson_Cartesian_Generic) compares 
the generic class and the one with the flux templated. 

## Example 2 - Projections and Evaluations

PETSc has a very general concept of discretisation spaces that do not 
necessarily admit to continuous interpolation to or from arbitrary points.
For this reason, a more general concept is to create projections that map
between representations of the data. For example, in Finite Elements, 
fluxes are generally not available at nodal points because shape functions
have discontinuous gradients there. To compute fluxes at nodal points, we 
would establish a projection problem to form a best fitting continous function
to the values at points where we can evaluate the fluxes. In addition, 
sympy functions (including those for fluxes) that contain derivatives of finite element variables 
can not be evaluated numerically by sympy but can be evaluated as compiled
functions in the context of a solver. 

We write these evaluations using the `Projection` solver classes. This is
the simplest of the solvers and we are only discussing it second because it
is almost too simple to be instructive (and because the weak form of this
equation is the natural one to work with).

We would like to solve for a continuous, nodal point solution $u$ that 
satisfies as best possible,

$$ \int_\Omega \phi u d\Omega = \int_\Omega \phi \tilde{u} d\Omega $$

where $\tilde{u}$ is a function with unknown continuity that we
are able to evaluate at integration points in the mesh. 

The generic solver specification in underworld looks like this

```python 
=
        # f0 residual term (weighted integration) - scalar function
        solver.F0 = solver.u.fn - user_uw_function

        # f1 residual term (integration by parts / gradients)
        solver.F1 = 0.0
```
where `user_uw_function` is some sympy expression in terms of spatial 
coordinates that can include mesh or swarm variables. `solver.u.fn` is the
mesh variable (in function form) where the solution will reside.

In principle we could add a smoothing term via `solver.F1` and, importantly,
we can also add boundary conditions or constraints that need to be satisfied in 
addition to fitting the integration point values. 

We provide projection operators for scalar fields, vector fields and 
solenoidal vector fields (ensuring that the projection remains divergence free). These
provide templates for the $F_0$ and $F_1$ terms with generic smoothing 

[This notebook](../Notebooks/Ex_Project_Function.md) has an example of 
each of these cases. 

## Example 3 - Incompressible Stokes Equation

A saddle point system in which we solve for a constraint parameter
as well as the primary unknown. We have to tweak the template a little
bit for this one.


## Example 4 - Advection without diffusion

The pure transport equation 

$$ \frac{D \psi}{D t} = H $$

(This is not especially well suited to the pointwise formulation )

## Example 4 - The Scalar Advection-diffusion Equation

The situation where advection and diffusion are in balance 

 - this means that grid methods / particle methods both have issues.
 
## Example 5 - Navier-Stokes

All the issues from example 4 but with even more non-linearity.

