---
title: "Underworld 3: Systems of Equations"
keywords: [underworld3, sympy, PETSc, SNES, PDEs]
authors: 
  - name: Underworld Team
exports:
  - format: pdf
    template: lapreprint
    theme_color: blue
---

# Equation systems in Underworld3 

Implementations dealt with here - how we build a solver from the equation fragments. Which parts are symbolic

HERE: show how we rewrite the equations in the SNES classes to include time dependence / transport terms.

PIC as motivation for this approach




## Template problem

IMAGE

In classical fluid dynamics we write evolution equations by considering the conservation of a given quantity with the explicit assumption of fluid-mediated transport when we derive the partial differential equations. 


For example, energy conservation leads to a partial differential equation for $T$

$$
\color{DarkGreen}{\underbrace{ \Bigl[ \frac{\partial T}{\partial t} + \left( \mathbf{v} \cdot \nabla \right) T \Bigr]}_{\dot{\mathbf{f}}}} -
 \color{Blue}{
\nabla \cdot
       \underbrace{\Bigl[ \boldsymbol\kappa \nabla T \Bigr]}_{\mathbf{F}}} -
\color{Maroon}{\underbrace{\Bigl[ H \Bigl] }_{\mathbf{f}}} 
{\color{Black} = 0}

\label{eqn-template}
$$

To match the template form provided by the PETSc pointwise interface, we need to express  the term labeled as $\dot{\mathbf{f}}$ in terms of $f_0$ and $\mathbf{f}_1$ from equation [%s](#eqn-template).

We can express the material derivate term ${\partial T}/{\partial t} + \left( \mathbf{v} \cdot \nabla \right) T$ in (first-order) finite difference form:

$$
\frac{\partial T}{\partial t} + \left( \mathbf{v} \cdot \nabla \right) T \sim \frac{T(\mathbf{x}) - T^*(\mathbf{x}^*)}{t-t^*}
$$

Here $T^*$ is the value of $T$ at the earlier time ($t^*$) measured at the location $\mathbf{x}^*$ which was transported to $\mathbf{x}$ by the velocity field, $\mathbf{v}$ over the timestep.



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

