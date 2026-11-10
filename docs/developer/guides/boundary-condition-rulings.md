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
- Where the rotated wall meets an essential wall, a component the essential
  condition constrains contributes zero to σ_nn, and the free components keep
  their load (Louis, 2026-11-04, #936).

Governing document: [rotated free-slip](../subsystems/rotated-freeslip.md).

Rotated linear-solver defaults are unchanged unless explicitly overridden.
The `rotated_fieldsplit_velocity_ksp_type`,
`rotated_fieldsplit_pressure_ksp_type` and `rotated_ksp_max_it` options on
`solver.petsc_options` select the benchmark's inner solvers and outer iteration
allowance; see the governing document's explicit linear-solver options.

For Stokes, `petsc_use_pressure_nullspace=True` requests a pressure gauge;
the rotated solver verifies it against both sides of the actual constrained
operator before projecting any residual. Analytic radial normals on faceted
spherical walls can make pressure physically determined in the discrete system.
That pressure component is then retained. When both operator actions are
nonzero, the native iterative route applies a balanced two-vector pressure
correction around its independent fieldsplit preconditioner. It changes neither
the original equations nor their residual target. Verified pressure gauges and
explicitly disabled requests retain the existing preconditioner.

`solve_report.config['gauge']` records requested, verified, attached and active
pressure projection separately. `config['rotated_preconditioner']` records the
correction and retained base PC; `config['rotated_residual']` records fresh
residuals after finalizing the fields. Solver convergence establishes the
algebraic residual, while physical-wall accuracy still requires benchmark checks.

## Physical circular walls on affine annulus meshes

`Stokes.add_cylindrical_wall_transfer` registers a lagged rotated normal
constraint and tangential natural load for free slip on the true circle.
`Stokes.add_cylindrical_zero_slip_transfer` registers a full-vector shifted
Dirichlet condition. These replace the ordinary wall registration on each
boundary. Both require fixed, nondimensional affine triangles, continuous
P2 velocity/P1 regular pressure, and a complete inner/outer circle pair.

Call `solve_cylindrical_wall_updates(walls=walls, updates=...)` for free slip,
or `solve_cylindrical_zero_slip_updates` for zero slip, with the pressure
nullspace enabled. The explicit update count performs one initial solve plus
that many numerical wall updates. Check the returned physical-circle residuals
and update changes; a solver convergence flag alone does not establish wall
iteration convergence. Free-slip transfer also requires constant viscosity.

Pressure splitting remains separate. For free-slip transfer, pass the known
`pressure_jump` expression from `cylindrical_pressure_lifting`, or the scalar
DG P1 `pressure_jump_field` from `pressure_split`; pass zero without splitting.
The latter uses one-sided cell coefficients, without averaging a jump at
facets. Keep the split's exterior natural datum on natural components.
Zero-slip transfer needs only velocity, so it accepts either pressure-split
workflow without a wall pressure input. No topography or gravity calculation
is needed. Method docstrings give the equations, references and limitations.

`Stokes.solve_cylindrical_gap` is an explicitly selected research alternative
for the same constant-viscosity P2/P1 annulus. It assembles momentum and
continuity on polygon + outer caps - inner caps and solves a physical-circle
normal multiplier with velocity and pressure. The normal condition is weak,
not pointwise rotated free slip; its KSP options are supplied explicitly.
Do not combine it with ordinary wall registrations. The method docstring
identifies the Gap-SBM and weak-slip multiplier references and distinguishes
the annulus construction from their published methods. No uniform inf-sup
stability theorem is claimed for this particular P2/P1/P2 space.

`Stokes.solve_cylindrical_gap_zero_slip` adds a tangential circle multiplier
and enforces both velocity components weakly. It returns normal and tangential
reactions. Rotation is physically constrained, so no rotation gauge is removed;
nonzero applied torque is balanced by wall reactions. Both gap backends use
the ordinary mesh FE quadrature for volume forcing and circle-cap quadrature
for its extension, keeping bulk load conventions matched in comparisons.
Their distributed matrix graph is preallocated from element and circle-multiplier
connectivity with PETSc `MATPREALLOCATOR`, including off-process rows. Zero-filling
that graph prevents storage growth during numerical insertion; allocation errors
remain enabled to catch a missing coupling. This changes storage, not the weak form.

`Stokes.solve_spherical_gap` and `solve_spherical_gap_zero_slip` integrate
inner-cut tetrahedra on their physical restriction outside the inner ball.
An inner radial gap may leave its boundary owner's tetrahedron, so its owner's
polynomial must not be subtracted throughout that gap. The cell-local rule uses
the actual tetrahedral faces, splits their circle intersections, and integrates
each cell polynomial consistently in momentum, continuity, forces and gauges.
Wholly void cells contribute zero; unknowns without physical support receive
identity completion. Inner-wall velocity traces use these same cell polynomials
and the original radially projected multiplier space. Only outer continuations
retain artificial radial curtains. The corrected inner geometry does not prove
global mixed stability or convergence of a particular iterative preconditioner.

The optional spherical norm preconditioner uses full background viscous and mass
terms on physically supported cells, including cut cells. Wholly void cells do
not contribute to its velocity or pressure norms. Exact-zero unsupported velocity
rows and pressure diagonals receive identity completion, matching their algebraic
removal from the physical system. The velocity action remains one GAMG cycle;
pressure and wall multipliers retain their separate Jacobi actions. This changes
the preconditioning norm, not the mixed equations, boundary constraints or their
original residual target. It does not establish uniform mixed stability.


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

`Stokes.pressure_split` lifts prescribed scalar pressure jumps on named interior
facets of affine triangle/tetrahedron meshes with continuous P1 pressure. It
accepts analytical expressions or numerical jump data, including compatible
finite and intersecting interfaces. Replace the normal sheet load with its
regular force; reconstruct total pressure for diagnostics and apply its datum
on exterior natural components. See
[prescribed pressure jumps](../../advanced/prescribed-pressure-jumps.md).

For a prescribed circular pressure jump in a nondimensional P2/continuous-P1
annulus, `Stokes.cylindrical_pressure_lifting` separates physical pressure into
regular pressure and a known jump. Replace the sheet with its returned regular
force and register its `natural_datum` on free outer-wall components. Fully
prescribed zero-slip walls need no natural datum. The method docstring defines
the load sign and pressure reconstruction.

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

## Spherical solve observations

The optional [spherical diagnostics API](../subsystems/spherical-stokes-diagnostics.md)
provides physical error/reaction integration and shared cylindrical/spherical `Stokes.observe_stokes`
callbacks. Observations do not change wall constraints or the original equations.
