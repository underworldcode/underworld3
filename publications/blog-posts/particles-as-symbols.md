---
title: "Particles That Know Calculus"
status: draft
feeds_into: [paper-2]
target: underworldcode.org (Ghost)
tags: [underworld, particles, swarm, proxy, symbolic, geodynamics]
---

# Particles as first-class participants in underworld3 symbolic algebra. 

Lagrangian particles in a finite element code are used to carry material properties with the flow. Composition, strain history, stress memory, damage. Finite element solvers work with fields defined on the mesh, not scattered data on particles. In Underworld 1 and Underworld 2, we created dynamic integration schemes for particle swarms on an element-by-element basis. This option is not available with the PETSc point wise-function approach. For this, we need to project particle data onto the available interpolating functions before each solve. 

We also need to be able to represent the particle-based data and its derivatives symbolically for the underworld3 representation to be composable with mesh-based data when we contruct the weak form. 

In Underworld3, swarm variables are symbolic objects. A particle-carried quantity has a `.sym` property that returns a SymPy symbol, just like a mesh variable. That symbol participates in the solver's weak form, the constitutive model, the boundary conditions. The solver does not know whether it is reading a field computed on the mesh or a field projected from particles. The distinction is invisible at the symbolic level.

## The Problem

A Stokes solver assembles the weak form by evaluating expressions at quadrature points inside each element. The expressions reference field variables defined at mesh nodes. If the viscosity depends on a material property stored on particles, that property must first be available as a mesh field.

In UW2, the user managed this explicitly. You would call a projection routine before each solve, mapping particle data onto the mesh. Forget the projection, and the solver uses stale data. Call it too often, and you waste time on redundant work.

UW3 automates this through the proxy mesh variable pattern.

## Swarm Variables

Particles carry data through swarm variables:

```python
swarm = uw.swarm.Swarm(mesh)
swarm.populate(fill_param=3)

material = uw.swarm.SwarmVariable("material", swarm, size=1)
material.data[:] = initial_values
```

Each swarm variable is stored as a PETSc field on the DMSwarm. When particles migrate between processors, their variable data travels with them automatically. The `.data` property provides direct access to the underlying array, with the same NDArray_With_Callback mechanism described in the [arrays-in-sync post](/mesh-variables-and-petsc-vectors-keeping-arrays-in-sync/).

A swarm variable can be scalar, vector, or tensor:

```python
stress_history = uw.swarm.SwarmVariable(
    "stress", swarm, vtype=uw.VarType.SYM_TENSOR, proxy_degree=2
)
```

The `proxy_degree` parameter is the key. When set, the system creates a companion mesh variable and manages the projection between particles and mesh automatically.

## The Proxy Mesh Variable

When you create a swarm variable with `proxy_degree > 0`, UW3 creates a companion mesh variable behind the scenes. This proxy is a standard finite element field defined on the mesh nodes, with the specified polynomial degree.

The projection from particles to mesh uses inverse-distance-weighted interpolation from the nearest particles to each mesh node, implemented through a KDTree of particle positions:

1. Build a KDTree of all particle positions on the local rank.
2. For each mesh node, find the $k$ nearest particles ($k = \text{dim} + 1$ by default).
3. Compute inverse-distance weights and interpolate the swarm variable values.
4. Store the result on the proxy mesh variable.

The proxy has a `.sym` property that returns a SymPy symbol. This is the same interface as any mesh variable. You use it in expressions, constitutive models, boundary conditions, source terms. The solver sees a mesh variable symbol and does not need to know that the data originated on particles.

## Lazy Evaluation

The projection is not free. Building a KDTree, querying nearest neighbours, computing weights, and writing to the mesh variable all take time. Doing this before every solver assembly would be wasteful when the particle data has not changed.

UW3 handles this through lazy evaluation. The swarm variable tracks whether its data has been modified since the last projection. When you access `.sym`, the system checks this flag. If the data is stale, it triggers a fresh projection. If not, it returns the cached proxy symbol immediately.

```python
# Modify particle data
material.data[some_particles] = new_values   # marks proxy as stale

# Next access to .sym triggers projection
viscosity_fn = eta_0 * sympy.exp(-material.sym)   # projection happens here
```

During solver assembly, the same symbol may be evaluated many times at different quadrature points. The projection happens once, on the first access after a modification. Subsequent accesses within the same solve use the cached mesh field.

This means the user never calls a projection routine. You modify particle data, you use the symbol, the framework handles the rest.

## Particles in Expressions

Because `.sym` returns a standard SymPy symbol, particle data composes with everything else in UW3's symbolic layer:

```python
# Material-dependent viscosity from particle data
viscosity_fn = eta_0 * sympy.exp(-gamma * material.sym)

# Use in constitutive model
stokes.constitutive_model.Parameters.shear_viscosity_0 = viscosity_fn

# The solver differentiates through this for the Jacobian
# It does not know that material.sym comes from particles
```

The same pattern works for stress history in viscoelastic models, where the DFDt infrastructure stores previous stress values on swarm variables with proxies. The constitutive model reads `stress_star.sym` as part of its effective strain rate expression. The projection from particles to mesh is lazy, the symbol is SymPy, and the Jacobian derivation is automatic.

This is the same design principle that applies throughout UW3: separate the physics from the numerics through symbolic expressions. For [constitutive models](/constitutive-models-in-symbolic-form/), the boundary is the stress tensor. For [time derivatives](/symbolic-time-derivatives-in-underworld3/), the boundary is the BDF/AM expression. For particles, the boundary is the proxy mesh variable symbol.

## Batching Updates

Migration involves MPI communication and KDTree reconstruction. If you are updating particle coordinates and modifying variable data in the same operation, you do not want to trigger migration and projection after each step. The `access()` context batches changes:

```python
with swarm.access():
    swarm.particle_coordinates.data[...] = new_coords
    material.data[:] = new_values
# Migration and cache invalidation happen here, once
```

This is the particle analogue of `uw.synchronised_array_update()` for mesh variables. Batch your changes, let the framework handle the communication once.

## What the Solver Sees

From the solver's perspective, there are only mesh variable symbols. The JIT compiler generates C code that reads field values at quadrature points. Whether those values were computed by the solver, set by the user, or projected from particles is an implementation detail behind the symbol.

This uniformity is important. It means you can start with a mesh-based viscosity field, decide later to track viscosity on particles for better advection properties, and the solver code does not change. You swap the symbol from a mesh variable to a swarm variable proxy. The constitutive model, the weak form, the Jacobian, the compiled C all remain the same.

---

*The Underworld project is supported by AuScope and the Australian Government through the National Collaborative Research Infrastructure Strategy (NCRIS). Source code: [github.com/underworldcode/underworld3](https://github.com/underworldcode/underworld3)*
