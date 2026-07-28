# Rotated strong free-slip and the prescribed-normal datum

The rotated free-slip boundary condition imposes

$$\mathbf{u}\cdot\hat{\mathbf{n}} = \tilde u_n$$

strongly — as an exact per-node constraint, not a weak (Nitsche/penalty) term —
by rotating each boundary node's velocity components into a (normal,
tangential) frame and constraining the rotated normal component. `conds = 0`
(the default datum) is free-slip; a non-zero scalar `conds` prescribes the
wall-normal velocity, which is what the free-surface manager uses to keep the
surface a material boundary.

```python
stokes.add_rotated_freeslip_bc(0, "Upper", normal=nhat)          # free-slip
stokes.add_rotated_freeslip_bc(h_dot.sym[0], "Upper", normal=nhat)  # u·n̂ = field
```

`normal=None` uses the geometric facet normal; a sympy `1×dim` matrix in
`mesh.X` supplies an analytic normal (exact `X/|X|` on curved boundaries — the
preferred choice there); a constant array is also accepted. The datum must be a
*scalar* (a number, an expression of `mesh.X`, or a scalar field read); on an
enclosed boundary it must be discretely flux-free for incompressibility. A
corner or 3D-edge node shared between rotated boundaries has no single normal
and stays at the free-slip pinning (the datum is ignored there).

Why strong rather than Nitsche/penalty: the constraint holds to machine
precision (a penalty leaks ~1e-3, and the leak grows exactly where anisotropy
makes the boundary condition matter), it is correct on curved/tilted/deformed
boundaries, and the constraint **reaction is the boundary normal traction**
σ_nn (`solver.boundary_normal_traction`, `solver.dynamic_topography`) — the
quantity the free-surface machinery consumes. Reserve Nitsche for conditions
that must morph in time (Dirichlet→Neumann ramps).

**Implementation**: `src/underworld3/utilities/rotated_bc.py`; registration and
dispatch in `petsc_generic_snes_solvers.pyx` (`add_rotated_freeslip_bc`).

## One solve path

There is a single driver, `solve_rotated_freeslip`: a manual outer
Newton/Picard loop that rotates the residual and tangent every iteration
(`F̂ = Q F`, `Ĵ = Q J Qᵀ`), imposes the constraint on the rotated normal rows,
and solves each increment with a self-contained fieldsplit-Schur KSP (geometric
FMG on the custom prolongation when a hierarchy is registered, else GAMG; the
native 1/μ pressure mass as the Schur preconditioner). There is no separate
linear path and no up-front nonlinearity probe: a linear model converges after
its first increment and the loop self-terminates.

The manual loop exists because the rotated operator `Q A Qᵀ` carries no DM
field information, so PETSc's DM-coupled fieldsplit cannot precondition it;
the increment is solved by an IS-built fieldsplit instead, driven from the
loop. The loop honours `zero_init_guess`, `picard`, and the solver's
`consistent_jacobian` tangent policy (frozen / Newton / continuation).

### How the datum passes through Newton

The constraint is affine, and the design keeps every **accepted iterate
feasible** (`u·n̂ = ũ_n` exactly, via an affine snap in the rotated frame), so
each Newton increment satisfies the *homogeneous* constraint `n̂·δ = 0` — the
datum never touches the tangent, the increment right-hand side, the FMG
prolongation, or the null-space handling.

The one deliberate exception is a **cold start with a non-zero datum**: the
first increment carries the datum jump through the affine lift
(`zeroRowsColumns(rows, diag, x̂, b̂)`) at the rest-state tangent, and is
accepted without a line search. Snapping the zero state onto the datum instead
manufactures an extreme boundary-strip strain state whose shear-thinning
tangent produces a first step orders of magnitude too large — no line-searched
step descends (measured). A warm start is assumed smooth and takes the exact
snap directly.

### Convergence semantics

The loop converges on `‖F̂‖ ≤ rtol·ref + atol` with
`ref = max(‖F̂(u₀)‖, ‖F̂(0)‖)`: the **rest-state residual is the intrinsic
forcing scale**, so a good warm start (whose own initial residual is small) is
not punished with an ever-stricter absolute target. A step-norm exit (tiny
Newton step) is *verified* against the same reference before it may report
convergence — a stiff tangent also produces tiny steps far from the solution,
and an unverified tiny step goes back through the line search (a slow crawl
continues; a genuine stall ends the loop with an explicit warning). An
unconverged exit always warns; the fields hold the last iterate.

### Null-space handling

Rigid-rotation candidates (one mode on a closed circle, three on a spherical
shell) are admitted to the increment null space and the post-solve gauge
removal only if they pass **two** tests: tangential to every rotated
constraint row (`Q·m ≈ 0` there), **and** a null vector of the assembled
operator (`‖J·m‖ ≈ 0` against the velocity diagonal scale). The second test is
what catches pinning the first cannot see — an essential condition on another
boundary leaves the mode tangential to the rotated wall while the eliminated
operator is *not* null on it, and admitting it projects an irreducible
component out of every increment (the residual then floors far above
tolerance instead of converging). The null space is therefore built after the
first Jacobian assembly.

### Parallel notes

- Every branch the loop takes is decided **collectively**. The datum-activity
  flag rides a PETSc `Vec` norm of the lift vector (the datum bookkeeping
  itself is rank-local — only owners of datum boundary nodes hold entries —
  and branching on it per-rank desynchronises the collective sequences: the
  np>1 deadlock class).
- All row surgery on vectors uses ownership-relative indices
  (`_zero_rows_local` / `_set_rows_local`); indexing a local slice with global
  rows is the np>1 crash class.
- Direct LU per increment (`solver._rotated_use_lu = True`) is a **serial**
  preconditioner-free diagnostic (the pressure-gauge pin is a naive per-rank
  scan, marked `TODO(BUG)`); use it to separate "the operator/constraint is
  wrong" from "the preconditioner is struggling".

## Result and reaction

The solve fills the solver's fields and stores a result dict
(`solver._rotated_freeslip_info`) with the rotation, the constrained rows, the
per-increment KSP iteration counts, the convergence verdict, and the
**reaction** — the converged Cartesian residual `F(u)`, which for a linear
residual equals `A·u − b` exactly. `boundary_normal_traction` projects the
nodal reaction onto the boundary normal (corner-correct) and de-smears it with
the shared boundary-mass machinery in `utilities/boundary_flux.py`;
`dynamic_topography_field` writes `h = −(σ_nn − σ̄_nn)/(Δρ g)` onto a surface
field for the free-surface integrator.

## Tests

`tests/test_1018_rotated_freeslip.py` (serial: essential-equivalence, FMG,
tangent policies, datum linear + nonlinear),
`tests/parallel/test_1066_rotated_datum_parallel.py` (np≥2: partition
independence of the linear datum and the nonlinear Newton datum path).
