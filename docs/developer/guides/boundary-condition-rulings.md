---
name: boundary-condition-rulings
description: Which boundary treatment to use in Underworld3 and why — rotated strong free-slip over Nitsche or penalty, Nitsche only for a condition that must evolve, natural tractions, essential data as controls in an adjoint, what happens at corners, and the mesh-side traps that leave a condition silently empty. Read with the class-level view of a solver, which lists the mechanisms it accepts.
families: [Stokes, Stokes_Constrained, VE_Stokes, NavierStokes, Poisson, Diffusion, AdvDiffusion, SteadyStateDarcy, TransientDarcy, Richards]
kind: guide
---

# Boundary conditions: the rulings

A solver lists the mechanisms it accepts (`uw.systems.Stokes.view()`:
essential, natural, Nitsche, rotated free-slip, fault contact, and for a
saddle point the multiplier constraint). This page says which to choose,
and what each one commits you to.

## Free-slip: rotated strong free-slip, not Nitsche or penalty

`solver.add_rotated_freeslip_bc(conds, boundary, normal=None)`, value first:
`conds=0` is free-slip, a scalar or expression prescribes the wall-normal
datum strongly.

- It enforces $v\cdot\hat n = 0$ to machine precision; Nitsche and penalty
  leak at about $10^{-3}$.
- It is correct on curved, tilted and deformed boundaries: the normal is
  taken per node, measure-weighted to match the facet integral the
  assembler evaluates (#560). Leave `normal=None` unless the constraint
  must follow the true surface rather than the mesh; an analytic normal is
  exact for the geometry but carries a consistency error against the
  faceted assembly.
- It works inside the nonlinear SNES and with geometric multigrid, and is
  transparent to the tangent (`consistent_jacobian` behaves as it would
  without it).
- Its reaction is the boundary normal traction, read with
  `solver.boundary_normal_traction(boundary)`; no augmented-Lagrangian
  splitting is involved.

Governing document: [rotated free-slip](../subsystems/rotated-freeslip.md).

## Nitsche: only for a condition that must evolve

A hard rotated constraint cannot morph. A condition that changes character
in time, a Dirichlet-to-Neumann ramp or a traction that switches on, is a
Nitsche condition (`add_nitsche_bc`). Two rulings come with it:

- The penalty scale must be recalibrated whenever the cell size is
  redefined (#734). A value of 10 was a cliff, a hundred times worse than
  12.5, after the cell size changed under it; the scale is tied to the
  cell size by #697.
- Nitsche is the one mechanism whose boundary Jacobian reads the unknown,
  so a solver carrying it takes the matrix route for its adjoint and
  refuses a parameter in a Dirichlet datum until the boundary tangent is
  in the reaction term.

## Natural conditions: tractions and fluxes

`add_natural_bc(value, boundary)` is a facet load. `mesh.Gamma` in the
expression resolves to the facet normal: exact per quadrature point on an
external boundary, the declared analytic normal on an internal one, where
PETSc's normal is orientation-ambiguous (#327). A parameter in a natural
condition is differentiated by the adjoint as a facet part with nothing
further from the user.

## Essential data

`add_essential_bc` and `add_dirichlet_bc` are the same call. Components
given as `None` (or `sympy.oo`) are left free. Three things to know:

- PETSc constrains the closure of each labelled boundary, corner vertices
  included, and inserts the data boundary by boundary in the order they
  were registered, so where two boundaries meet the later one's datum is
  the one applied. Register the lid last if the lid's value is the one the
  corner should carry.
- A named parameter in a datum is a control: `solver.gradient(...)`
  carries its reaction term (#762). A bare number is not a control, since
  it has no name.
- A datum given as a plain number is scaled by the model's units at
  registration; a symbolic datum is scaled when it is compiled. Either way
  the transcript's key shows the value as applied.

## The mesh side: conditions that come out empty

- Gmsh physical groups must be numbered in the order of the boundary
  enumeration, or the conditions attach to the wrong stratum and are
  silently empty. Check `mesh.view()`, which lists the boundaries with
  their sizes at level 1.
- A patch on a fault must be at least three cells across to be split;
  smaller, the split refuses.
- A boundary condition on a field that is not the solver's unknown (a
  pressure datum on a Stokes solver) is accepted by the solver but not by
  the adjoint, which refuses rather than drop it.
