# PETSc pointwise functions and PDE solvers

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


`````{tabbed} Poisson Equation

````{panels}
Equation
^^^
$$ \nabla \cdot (\alpha \nabla \psi) = \rho $$

---

`sympy` expression 
^^^
```python
grad_psi  =  sympy.vector.gradient(solver.psi)
solver.F0 =  uw_fn_rho 
solver.F1 = -uw_fn_alpha * grad_psi
```
````
`````

`````{tabbed} Projections
````{panels}
Equation
^^^
Solve for $u$ on the mesh unknowns that best satisfy $\tilde{u}$ 
evaluated on points within the mesh.

$$ \int_\Omega \phi u d\Omega = \int_\Omega \phi \tilde{u} d\Omega $$

---

`sympy` expression 
^^^
```python
solver.F0 = solver.u - uw_function_u_tilde
solver.F1 =  0.0
```
````
`````

`````{tabbed} Incompressible Stokes
````{panels}
Equation
^^^

The momentum balance equation is 

$$ \nabla \cdot \left(\mathbf{\tau} - p \mathbf{I}\right) = f_\textrm{buoy} $$

with an incompressible flow constraint 

$$ \nabla \cdot \mathbf{u} = 0 $$

and the deviatoric stress defined as

$$ \tau = \eta \left( \nabla \mathbf{u} +  \nabla \mathbf{u}^T \right) $$


---

`sympy` expression 
^^^
```python
grad_U = sympy.derive_by_array(solver.U, solver.X)
grad_U_T = grad_U.transpose()
epsdot = (grad_U + grad_U_T)

solver.UF0 = -uw_fn_buoyancy_force
solver.UF1 =  u_fn_viscosity * epsdot - \
              sympy.eye(dim) * solver.P

# constraint equation
solver.PF0 = sympy.vector.divergence(solver.U)
```
````

`````

`````{tabbed} Advection-Diffusion Equations
````{panels}
Equation
^^^

$$ \frac{\partial \psi}{\partial t} + \mathbf{u}\cdot\nabla\psi = \nabla \cdot \alpha \nabla \psi $$ 

or, in Lagrangian form (following $\mathbf{u}$),

$$ \frac{D \psi}{D t} = \nabla \cdot \alpha \nabla \psi $$ 

and this approximation for the time derivative

$$ \frac{D \psi}{Dt}  \approx \frac{\psi_p - \psi^*}{\Delta t}$$

where $\psi^*$ is the value upstream at $t-\Delta t$



---

`sympy` expression 
^^^
```python

grad_U  = sympy.derive_by_array(solver.U, solver.X)
grad_Us = sympy.derive_by_array(solver.U_star, solver.X)
DUDt   = (solver.U - solver.U_star) / delta_t

solver.F0 = DUdt 
solver.F1 = uw_fn_alpha * (grad_U + grad_Us) / 2  

```
````
`````

## Implementation & Examples

### Poisson Solvers

(link to another document)
Diffusion

Darcy flow

Advection-diffusion (SLCN)

### Advection dominated flow

(link to another document)

Swarm-based problems

Projection / swarm evaluation

Advection-diffusion (Swarm)

Material point methods

### Incompressible Stokes 

(link to another document)

Saddle point problems

Stokes, boundary conditions, constraints

Navier-Stokes (Swarm)

Viscoelasticity

## Remarks

The generic solver classes can be used to construct all of the examples above. The equation-system classes that we provide help to provide a template or scaffolding for a less experienced user and they also help to orchestrate cases where multiple solvers come together in a specific order (e.g. the Navier-Stokes case where history variable 
projections need to be evaluated during the solve).

Creating sub-classes from the equation systems or the generic solvers is an excellent way to build workflows whenever there is a risk of exposing some fragile construction at the user level. 

Some of the need for these templates is a result of inconsistencies in the way `sympy` treats matrices, vectors and tensor (array) objects. We expect this to change over time.


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
heat flux in terms of temperature gradients.This form matches the template above if we identify:

$$ f_0 = -h \quad \textrm{and} \quad f_1 = k\nabla T$$ 

and, in fact, this is exactly what we need to specify in the underworld equation
system. 

```python 
        solver.L= sympy.derive_by_array(solver.U, solver.X).transpose()

        # f0 residual term (weighted integration) - scalar function
        solver.F0 = -h

        # f1 residual term (integration by parts / gradients)
        solver.F1 = k * solver.L
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
provide templates for the $F_0$ and $F_1$ terms with generic smoothing.
For an explanation of the divergence free projection methodology, see the next 
example on the incompressible Stokes problem.

[This notebook](../Notebooks/Ex_Project_Function.md) has an example of 
each of these cases. 

## Example 3 - Incompressible Stokes Equation

The incompressible Stokes Equation is an example of a problem with an
additional constraint equation that must be satisfied by the solution.
Variational formulations (the weak form of
classical finite elements is one) naturally lend themselves 
to the addition of multiple constraints into the functional to be minimised. 

Unfortunately, the incompressiblity constraint needs to be enforced very strongly
to obtain reasonable solutions and this can lead to unacceptably 
ill conditioned systems that are slow or impossible to solve.

Alternatively, we solve a coupled problem in which additional, kinematic, 
parameters of the constraint term are introduced. This forms a new block system of
equations that is a general expression for a large range of constrained problems. 

These are often known as *saddle point problems* which represents the trade-off
between satisfying the original equations and the constraint equations. 
(Saddle point here refers to the curvature of the functional we are 
optimising) See: M. Benzi, G. Golub and J. Liesen, 
Numerical solution of saddle point problems, Acta Numerica 14 (2005),
pp. 1–137. for a general discussion. 

The coupled equation system we want to solve is

$$ \nabla \cdot \mathbf{\tau} - \nabla p = f_\textrm{buoy} $$

with the constraint 

$$ \nabla \cdot \mathbf{u} = 0 $$

The saddle-point solver requires us to specify both of these equations and 
to provide two solution vectors $\mathbf{u}$ and $\mathbf{p}$. In this 
system, $\mathbf{p}$ is the parameter that enforces the incompressiblity
constraint equation and is physically identifiable as a pressure. 

```python 

        # definitions 

        U_grad = sympy.derive_by_array(solver.U, solver.X)

        strainrate = (sympy.Matrix(U_grad) + sympy.Matrix(U_grad).T)/2
        stress     = 2*solver.viscosity*solver.strainrate

        # set up equation terms

        # u f0 residual term (weighted integration) - vector function
        solver.UF0 = - solver.bodyforce

        # u f1 residual term (integration by parts / gradients) - tensor (sympy.array) term
        solver.UF1 = stress

        # p f0 residual term (these are the constraints) - vector function

        solver.PF0 = sympy.vector.divergence(solver.U)
```

In `underworld`, the `SNES_Stokes` solver class is responsible for managing the 
user interface to the saddle point system for incompressible Stokes flow.

## Example 4 - Advection in the absence of diffusion

The pure transport equation can be written in Lagrangian

$$ \frac{D \psi}{D t} = 0 $$ 

or in Eulerian form

$$ \frac{\partial \psi}{\partial t} + \mathbf{u} \cdot \nabla \psi = 0 $$ 

In the Lagrangian form, there is nothing to solve, provided the fluid-transported reference frame is available. In the Eulerian form, the non-linear *advection* term 
$\mathbf{u} \cdot \nabla \psi$ is reknowned for being difficult to solve, especially in the pure-transport form. 

Underworld provides discrete Lagrangian `swarm` variables [ CROSSREF ] that make it straightforward to work with transported quantities
on a collection of moving sample points that we normally refer to as *particles*. 
Behind the scenes, there is complexity in 1) following the Lagrangian reference frame accurately, 
2) mapping the fluid-deformed reference frame to the stationary mesh, and 3) for
parallel meshes, migrating particles (and their data) across the decomposed domain.

The `swarm` that manages the variables is able to update the locations of the particles
when provided with a velocity vector field and a time increment and will handle the
particle re-distribution in the process. 

Each variable on a swarm has a corresponding mesh variable (a *proxy* variable) that 
is automatically updated when the particle locations change. The proxy variable is
computed through a projection (see above). 

*Note:* If specific boundary conditions need to be applied, it is necessary for the user
to define their own projection operator, apply the boundary conditions, and solve when needed.
(*Feature request: allow user control over the projection, including
boundary conditions / constraints, so that this is not part of the user's responsibility*)

## Example 5 - The Scalar Advection-diffusion Equation

The situation where a quantity is diffusing through a moving fluid. 

$$ \frac{\partial \psi}{\partial t} + \mathbf{u}\cdot\nabla\psi = \nabla \cdot \alpha \nabla \psi + f$$ 

where $\mathbf{u}$ is a (velocity) vector that transports $\psi$ and $\alpha$ is a
diffusivity. In Lagrangian form (following $\mathbf{u}$),

$$ \frac{D \psi}{D t} = \nabla \cdot \alpha \nabla \psi + f$$ 

As before, the advection terms are greatly simplified in a Lagrangian reference
frame but now we also have diffusion terms and boundary conditions that are easy
to solve accurately in an Eulerian mesh but which must also be applied to variables
that derive from a Lagrangian swarm (which has no boundary conditions of its own).

Advection-diffusion equations are often dominated by the presence of boundary layers where
advection of a quantity (along the direction of flow) is balanced by a diffusive flux
in the cross-stream direction. Under these conditions, there is some work to be done to
ensure that these two terms are calculated consistently and this is particularly important
close to regions where boundary conditions need to be applied.

The approach in `underworld` is to provide a solver structure to manage 
advection-diffusion problems on behalf of the user. We use
a swarm-variable for tracking the history of the $\psi$ as it is transported
by $\mathbf{u}$ and we allow the user to specify (solve for) this flow, and
to update the swarm positions accordingly. The history variable, $\psi^*$ 
is the value of $\psi$ upstream at an earlier timestep and allows us to 
approximate $D \psi/Dt$ as a finite difference approximation along the
characteristics of the advection operator:

$$\left. \frac{D \psi}{Dt} \right|_{p} \approx \frac{\psi_p - \psi^*_p}{\Delta t}$$

Here, the subscript $p$ indicates a value at a particle in the Lagrangian swarm. 

This approach leads to a very natural problem description in python that corresponds closely to the mathematical formulation, namely:

```python 
        solver.L     = sympy.derive_by_array(solver.U,      solver.X).transpose()
        solver.Lstar = sympy.derive_by_array(solver.U_star, solver.X).transpose()

        # f0 residual term
        solver._f0 = -solver.f + (solver.U.fn - solver.U_star.fn) / solver.delta_t

        # f1 residual term (backward Euler)  
        solver._f1 =  solver.L * solver.k

        ## OR 

        # f1 residual term (Crank-Nicholson)
        solver._f1 =  0.5 * (solver.L + solver.Lstar) * solver.k
```

In the above, the `U_star` variable is a projection of the Lagrangian history variable
$\psi^*_p$ onto the mesh *subject to the same boundary conditions as* $\psi$.

In the `SNES_AdvectionDiffusion_Swarm` class (which is derived from `SNES_Poisson`),
the `solve` method solves for `U_star` using an in-built projection and boundary
conditions copied from the parent, before calling a standard Poisson solver. This class manages every aspect of the creation, refresh and solution of the necessary
projection subroutines Lagrangian history term, but not the update of this variable or the advection. 

*Caveat emptor:* In the Crank-Nicholson stiffness matrix terms above, we form the derivatives in both the flux and the flux history with the same operator where, strictly, we should transport the derivatives (or form derivatives with respect to the transported coordinate system).  
 
## Example 6 - Navier-Stokes

The incompressible Navier-Stokes equation of fluid dynamics is essentially the vector equivalent of the 
scalar advection-diffusion equation above, in which the transported quantity is the velocity (strictly momentum) vector that is also responsible for the transport.

$$ \rho \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u}\cdot\nabla\mathbf{u} = \nabla \cdot \eta \left( \nabla \mathbf{u} + \nabla \mathbf{u}^T \right)/2 + \rho \mathbf{g}$$ 

$$ \nabla \cdot \mathbf{u} = 0 $$

Obviously this is a strongly non-linear problem, but simply introduce the time dependence to the Stokes equation in the same way as we did for the Poisson equation above. A finite difference representation of the Lagrangian derivative of the velocity is defined using a vector swarm variable  

$$\left. \frac{D \mathbf{u}}{Dt} \right|_{p} \approx \frac{\mathbf{u}_p - \mathbf{u}^*_p}{\Delta t}$$

And the python problem description becomes:

```python
        # definitions 

        U_grad      = sympy.derive_by_array(solver.U,     solver.X)
        U_grad_star = sympy.derive_by_array(solver.Ustar, solver.X)

        strainrate = (sympy.Matrix(U_grad) + sympy.Matrix(U_grad).T)/2
        stress     = 2*solver.viscosity*solver.strainrate

        strainrate_star = (sympy.Matrix(U_grad_star) + sympy.Matrix(U_grad_star).T)/2
        stress_star     = 2*solver.viscosity*solver.strainrate_star

        # set up equation terms

        # u f0 residual term (weighted integration) - vector function
        solver.UF0 = - solver.bodyforce + solver.rho * (solver.U.fn - solver.U_star.fn) / solver.delta_t

        # u f1 residual term (integration by parts / gradients) - tensor (sympy.array) term
        solver.UF1 = 0.5 * stress * 0.5 * stress_star

        # p f0 residual term (these are the constraints) - vector function
        solver.PF0 = sympy.vector.divergence(solver.U)
```

Note, again, that, formulated in this way, the stress and strain-rate history variables neglect terms resulting from the deformation of the coordinate system over the timestep, $\Delta t$. We could instead transport the strain rate or stress 

