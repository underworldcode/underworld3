# Rotated Free-Slip Workspace Reuse Across Solves

## Scope

This note documents the cross-solve workspace cache for rotated strong
free-slip Stokes solves. It addresses
[issue #417](https://github.com/underworldcode/underworld3/issues/417):
repeated transient solves rebuilt the rotated operator, the fieldsplit Schur
preconditioner and the GAMG hierarchy on every timestep, growing resident
memory until out-of-memory and paying distributed assembly and `PCSetUp`
every step.

The change does not alter the rotated boundary condition or its discrete
equations. It changes PETSc object ownership across solves, and decides when
per-solve work can be skipped.

## Where the reuse lives now

The rotated solve is a single manual Newton/Picard loop
(`rotated_bc.solve_rotated_freeslip`) — there is no separate linear path.
Within one solve the loop already reuses its operator and KSP context between
Newton iterations (ptap-with-result, `setOperators` refresh). The cache
extends exactly that pattern across solves, split by what each piece depends
on:

- **Geometry tier** — the rotation `Q`/`Q.T`, the constrained normal rows,
  the fault contact pair blocks, and the custom-FMG prolongation depend only
  on the mesh, the boundary specs and the fault registration. They are reused
  whenever the geometry signature matches (boundary names + normals, fault
  registration, DM identity).
- **Structure tier** — the transformed operator `Ahat`, the pressure-mass
  Schur block `Mp`, the nullspaces and the fieldsplit KSP/PC are reused as
  *objects* with their values refreshed in place — the same operation
  sequence the Newton loop performs between its own iterations, so it carries
  the same (production-validated) risk profile. This tier is always correct
  regardless of any change detection, because values are reassembled.
- **Iteration-0 fast path** — when the operator key proves nothing
  operator-relevant changed, the first Newton increment additionally skips
  Jacobian assembly, the ptap and `PCSetUp` entirely. This is the timestep
  fast path: a body-force (temperature) change alters only the residual, so
  repeated constant-viscosity Stokes solves pay one residual assembly and one
  Krylov solve per step.

## The structural safety net

A wrong fast-path decision cannot produce a wrong answer. The loop measures
the **true residual** (fresh kernels, current constants) at every iterate,
convergence is declared only on that measurement, and every iteration after
the first always reassembles the operator. A stale cached operator therefore
costs one extra Newton increment; it can never return a stale solution. This
is deliberately independent of the invalidation key — a safety net gated by
the trigger it guards is not a safety net, which was the unresolved finding
of the original PR #418 review (the state-counter key was blind to rampable
constants, and the same "unchanged" verdict disabled the matrix probe that
was supposed to catch it, returning a bit-identical stale solution flagged
as legitimately reused).

## Invalidation

The iteration-0 fast path requires ALL of:

- Operator coefficient **mesh variables** — collected from the constitutive
  parameters, constraint term, penalty and saddle preconditioner (expressions
  unwrapped first, unknowns excluded). Their base `MeshVariable._state`
  counters must match.
- The **constants signature** — the packed `constants[]` values the compiled
  kernels will assemble with, plus the JIT bundle key. This is what sees a
  rampable UWexpression constant (the #416 contract: a value change bumps no
  state counter — the original PR's blind spot) and an in-place kernel
  rewire. It deliberately over-invalidates on RHS-only constant changes:
  reassembly is the safe default.
- A **linear hint** — the previous solve on this workspace converged in at
  most one increment. A nonlinear model's cached operator is last solve's
  tangent, not this iterate's; the hint is self-measured by the loop, so no
  up-front nonlinearity probe is paid.
- If the coefficient enumeration or the constants manifest cannot be read,
  the fast path is forfeited and every solve reassembles (structure-tier
  reuse only) — correctness first.
- An explicit `solve(time=...)` vetoes the fast path: `petsc_t` reaches the
  kernels through the DM, invisible to any counter or constant value.
- The match verdict is made **collective** (allgather + unanimity) before it
  gates any collective PETSc call — state counters follow rank-local writes,
  and a rank-divergent verdict would be a deadlock, not a wrong answer.
- Mesh deform, field-layout, boundary-condition or forced-setup changes route
  through the solver's full-rebuild teardown, which destroys the whole
  workspace (`_reset_rotated_solver_cache`, called from `_reset()` and from
  the `_build()` full-rebuild branch, before the SNES/DM are destroyed).
- A geometry-signature mismatch (different boundaries or fault registration
  on the same solver) destroys and rebuilds the workspace.

The cache is forfeited entirely for direct-LU solves, prescribed-datum solves
(the datum is re-evaluated from possibly-changed fields inside
`build_rotation`), and fault **interface-law** solves (the interface tangent
and the Picard-lagged normal stress are solution-dependent). Frictionless
split-fault contacts cache normally: their pair blocks are geometry.

The high-level `uw.systems.Stokes.solve()` wrapper exposes and forwards
`time=` for the veto. It remains distinct from the viscoelastic integration
`timestep=`.

## PETSc ownership

`_destroy_rotated_ksp_ctx` destroys the KSP, the pressure-mass block and the
owned nullspaces. `_destroy_rotated_linear_cache` additionally destroys the
transformed operator and the rotation matrices, and *dereferences* (does not
destroy) the custom-FMG prolongation list, whose coarse matrices are shared
with the solver's registered multigrid hierarchy. The result dict shares the
`Q`/`Q.T` Python wrappers with the cache, so garbage collection and explicit
teardown compose without double-destroys.

## Validation

The focused regression (`test_1018_rotated_freeslip.py`) verifies that:

- a body-force-only change preserves the `Q`, `Ahat` and KSP handles and
  rides the fast path (`workspace_reused`), with the velocity scaling exactly
  with the right-hand side;
- a viscosity mesh-variable change is detected and refreshes the operator
  values in place on the same objects;
- an explicit `time=` solve vetoes the fast path.

Production validation for the original mechanism: 310 guarded Zhong A1
steps (8 ranks, `cellsize=1/8`) with flat RSS and a continuous physical
trajectory — see the PR #418 thread.

Always launch MPI tests with the worktree MPI executable:

```bash
.pixi/envs/amr-dev/bin/mpirun -np 2 \
  .pixi/envs/amr-dev/bin/python <script.py>
```

Using a system `mpirun` from a different Open MPI installation can stall
during initialization and is not a solver failure.
