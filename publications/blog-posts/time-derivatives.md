---
title: "Symbolic Time Derivatives in Underworld3"
status: published
published: 2026-04-16
url: https://www.underworldcode.org/symbolic-time-derivatives-in-underworld3/
feeds_into: [paper-1]
target: underworldcode.org (Ghost)
tags: [underworld, time-derivatives, BDF, advection, geodynamics, SymPy]
---

# Symbolic Time Derivatives in Underworld3

In Underworld3, the time derivative is a symbolic object. It appears in the solver's weak form as a SymPy expression, alongside the constitutive stress and the body force. You can inspect it, display it in a notebook, and verify that the time discretisation is doing what you expect. And you can swap between Lagrangian, Semi-Lagrangian, and Eulerian approaches without rewriting the solver.

## The Problem

Many geodynamics equations involve a material derivative. Temperature advection-diffusion, viscoelastic stress transport, Navier-Stokes momentum. The material derivative $D\phi/Dt$ combines the time rate of change with advection by the flow:

$$
\frac{D\phi}{Dt} = \frac{\partial \phi}{\partial t} + \mathbf{v} \cdot \nabla\phi
$$

Discretising this in time requires choices. How do you handle the advection term? How many previous timesteps do you use? How do you deal with variable timestep sizes? These choices affect accuracy, stability, and computational cost.

In many finite element codes, these choices are baked into the solver implementation. A classic example is Crank-Nicolson time stepping: the flux must be evaluated at both the current and previous timestep, which means adding history terms to the left-hand side of the assembled system as well as introducing a time-derivative term on the right-hand side. The time discretisation reaches into both sides of the equation, and changing it means restructuring the solver. In UW3, the time derivative ***offers*** flux and force terms to the solver to handle symbolically as it sees fit. 

## The DDt Hierarchy

UW3 provides four implementations of the time derivative, all sharing the same calling interface and working for scalar, vector and tensor quantities.

**Eulerian** stores history on the mesh. The material derivative is approximated by finite differences in time at fixed grid points, with an advection correction. This is the classical approach: simple, mesh-based, but subject to CFL stability constraints when advection dominates.

**Semi-Lagrangian** traces characteristics backward in time. At each mesh node, it asks: where was this material parcel at the previous timestep? It then interpolates the previous solution at that departure point. This is unconditionally stable because the material derivative is evaluated along the characteristic, not at a fixed grid point. There is no CFL constraint, though the user needs to be conscious of accuracy trade-offs inherent in the scheme.

**Lagrangian** follows particles through the flow. The history is stored on swarm variables and advected with the particles. This is the natural choice when material history matters physically, as in viscoelastic stress transport where the stress tensor must be advected and rotated with the material. Accuracy of this scheme depends upon the quality of the particle-layout (density of particles, presence of gaps).

**Symbolic** provides pure symbolic history without mesh or swarm storage. It is used internally by the constitutive model system for building expressions that involve time derivatives.

All four produce the same thing: a symbolic expression that can be embedded in a solver's weak form. The solver does not need to know which implementation is providing the time derivative. It sees a SymPy expression for $D\phi/Dt$ and includes those terms when it differentiates the equation system to determine the Jacobians.

## BDF Schemes: Multi-Step Time Integration

The time discretisation uses backward differentiation formulas (BDF). These are implicit multi-step methods that use solution values from previous timesteps to approximate the time derivative at the current step.

At order 1, this is the backward Euler method:

$$
\frac{D\phi}{Dt} \approx \frac{\phi^n - \phi^{n-1}}{\Delta t}
$$

At order 2, BDF-2 uses two previous values for second-order accuracy:

$$
\frac{D\phi}{Dt} \approx \frac{\frac{3}{2}\phi^n - 2\phi^{n-1} + \frac{1}{2}\phi^{n-2}}{\Delta t}
$$

The coefficients $[3/2, -2, 1/2]$ are for constant timesteps. When the timestep varies, the coefficients adapt. If the current timestep is $\Delta t _ n$ and the previous was $\Delta t _ {n-1}$, with ratio $r = \Delta t _ n / \Delta t _ {n-1}$, the BDF-2 coefficients become:

$$
c _ 0 = \frac{1+2r}{1+r}, \quad c _ 1 = -(1+r), \quad c _ 2 = \frac{r^2}{1+r}
$$

This matters in practice. Underworld simulations usually adjust the timestep as the flow evolves.

The coefficients are computed as exact `sympy.Rational` values and wrapped in UWexpression objects. This means they are routed through PETSc's constants array, just like viscosity or density parameters. When the timestep changes, the coefficient values update without recompilation. The compiled C code has the same structure at every step, only the values in the constants array will change.

## Adams-Moulton Weighting for Fluxes

BDF handles the time derivative of the unknown. But transient problems may also need a time-weighted evaluation of the flux (the spatial operator). The Adams-Moulton (AM) family provides this.

At order 0, the flux is evaluated purely at the current time (fully implicit). At order 1 with $\theta = 1/2$, we have the *Crank-Nicolson* scheme: the flux is averaged between the current and previous timestep. Higher order variants use more history values.

In UW3, the solver's flux time derivative (`DFDt`) provides an `adams _ moulton _ flux()` method that returns the appropriately weighted combination of the current flux and previous flux values as symbolic forms backed by stored evaluations. This expression then appears in the solver's $F _ 1$ template as a symbolic expression. Like the BDF coefficients, the AM weights are UWexpressions that update between timesteps.

The combination of BDF for the time derivative and AM for the flux evaluation gives a family of time integration schemes. BDF-1 with AM-0 is backward Euler. BDF-2 with AM-1 gives second-order accuracy in both the time derivative and the flux evaluation. The user controls this through the `order` parameter when creating the solver.

## Order Ramping at Startup

A BDF-2 scheme needs two previous solutions. At the start of a simulation, there is only one (the initial condition). UW3 handles this through automatic order ramping.

The DDt object tracks an `effective _ order` that starts at 1 and increases with each completed solve, up to the requested order. On the first timestep, BDF-1 is used regardless of what order was requested. On the second timestep, BDF-2 becomes available. On the third, BDF-3. The transition is automatic and the coefficients adjust accordingly.

History slots are initialised with the current solution value on the first call. This means the first BDF-1 step sees $\phi^{n-1} = \phi^n$, giving a zero time derivative. This is correct for a system starting from rest or from a steady-state initial condition. For impulsive starts, the first-order accuracy of the initial step is usually acceptable because the solution is changing rapidly and the timestep is typically small.

## What the Solver Sees

Consider advection-diffusion of temperature:

$$
\frac{D T}{Dt} = \nabla \cdot (k \nabla T) + H
$$

The solver setup looks like this:

```python
adv_diff = uw.systems.AdvDiffusion(
    mesh, u_Field=T, V_fn=v,
    order=2,                    # BDF-2 / AM-1
)
adv_diff.constitutive_model.Parameters.diffusivity = k
adv_diff.f = H
adv_diff.solve(timestep=dt)
```

The `order` parameter controls the time discretisation. The solver builds its weak form from two template expressions, $F _ 0$ (force-like, paired with the test function) and $F _ 1$ (flux-like, paired with the test function gradient):

```python
F0 = DuDt.bdf() / delta_t - H
F1 = DFDt.adams_moulton_flux()
```

At **order 1** (backward Euler / fully implicit), these expand to:

$$
F _ 0 = \frac{T^n - T^{n-1}}{\Delta t} - H
$$

$$
F _ 1 = k \nabla T^n
$$

The flux is evaluated entirely at the current timestep. The system is first-order accurate in time.

At **order 2** (BDF-2 / Crank-Nicolson), the expressions become:

$$
F _ 0 = \frac{\frac{3}{2}T^n - 2T^{n-1} + \frac{1}{2}T^{n-2}}{\Delta t} - H
$$

$$
F _ 1 = \frac{1}{2} k \nabla T^n + \frac{1}{2} k \nabla T^{n-1}
$$

The flux is now averaged between the current and previous timesteps. Both the time derivative and the flux evaluation are second-order accurate. The history terms $T^{n-1}$ and $T^{n-2}$ are mesh variable symbols managed by the DDt objects. The coefficients ($3/2$, $-2$, $1/2$, etc.) are UWexpression symbolic constants whose values update each step, if $\Delta t$ changes, and no recompilation is required.

Note: terms that contain $T^n$ automatically populate the implicit parts of the solver, while history terms are automatically treated explicitly. They are also automatically combined with history terms originating, for example, from the constitutive law. 

The solver differentiates through all of this for the Jacobian. The JIT compiler handles it like any other symbolic expression. From PETSc's perspective, the time-stepping machinery is invisible. It sees compiled C functions at quadrature points, with coefficient values arriving through the constants array. 

The solve sequence for each timestep is:

1. **Pre-solve**: Update BDF/AM coefficients for the current timestep. For Semi-Lagrangian, trace characteristics to find departure points and sample history values.
2. **Solve**: PETSc SNES solves the assembled system. The compiled C callbacks evaluate the symbolic expressions at quadrature points.
3. **Post-solve**: Shift the history chain. What was $\phi^{n-1}$ becomes $\phi^{n-2}$. The current solution becomes the new $\phi^{n-1}$. Increment the solve counter for order ramping.

## Choosing a Time Derivative

The choice of DDt type is a one-parameter decision at solver construction:

```python
# Default for advection-diffusion: Semi-Lagrangian (unconditionally stable)
adv_diff = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v)

# Override with Lagrangian (particle-based, requires a swarm)
DTdt = uw.systems.Lagrangian_Swarm_DDt(
    swarm, psi_fn=T.sym,
    vtype=uw.VarType.SCALAR, degree=T.degree,
    continuous=True, order=2
)
adv_diff = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v, DuDt=DTdt)

# Or Eulerian (mesh-based, for problems without strong advection)
DTdt = uw.systems.Eulerian_DDt(
    mesh, T, vtype=uw.VarType.SCALAR,
    degree=T.degree, continuous=True, order=2
)
adv_diff = uw.systems.AdvDiffusion(mesh, u_Field=T, V_fn=v, DuDt=DTdt)
```

The solver does not need to be made aware of which DDt type you chose. It calls `bdf()` and `adams _ moulton _ flux()` and gets SymPy expressions. The physics of the time discretisation is encapsulated in the DDt object. The numerics of the spatial discretisation are encapsulated in the solver. They communicate through symbolic expressions.

Each solver type has a sensible default. Advection-diffusion and Stokes default to Semi-Lagrangian. Pure diffusion defaults to Eulerian. Viscoelastic solvers use the DFDt infrastructure for stress history on particles. You only need to override the default when your problem requires it.

## Why This Matters

In UW2, if you wanted to change from explicit particle advection to a semi-Lagrangian scheme, you would need to plug in a different solver. The time discretisation was part of the solver's implementation, not a separable component. When history terms appeared through the constitutive model, we had to develop a matrix of all potential interactions and code them independently. 

In UW3, the time derivative is an object you can create, configure, inspect, and swap. The BDF coefficients are visible as symbolic expressions. The history terms are mesh variables you can plot. The AM flux weighting is a symbolic combination you can display in a notebook.

This is the same design principle we described in the [constitutive models post](/constitutive-models-in-symbolic-form/): separate the physics from the numerics, connect them through symbolic expressions, and make both sides inspectable. For constitutive models, the boundary is the stress tensor. For time derivatives, it is the BDF/AM expression. In both cases, the solver sees a SymPy expression and does not need to know how it was constructed.



<!-- April 14, 16:41pm -->

---

*The Underworld project is supported by AuScope and the Australian Government through the National Collaborative Research Infrastructure Strategy (NCRIS). Source code: [github.com/underworldcode/underworld3](https://github.com/underworldcode/underworld3)*
