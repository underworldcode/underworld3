# Multigrid with minimal user control — design note

**Status**: PROPOSED (2026-07). No implementation in this PR — this note
surveys what exists after #369 and asks the maintainer to settle the
default-behaviour questions at the end.

**Maintainer direction** (2026-07): *"the laplacian solvers all benefit
from MG; the real question is how much control is handed to / required
from the user? I would prefer that this is, by default, minimal."*

---

## 1. What exists today (post-#369)

Five distinct ways a UW3 solver ends up with (or without) multigrid:

### 1.1 `solver.preconditioner` — the one documented knob

`"auto"` (default) / `"fmg"` / `"gamg"` on `SolverBaseClass`
(`cython/petsc_generic_snes_solvers.pyx`). Resolution happens at
`_build` time in `_apply_preconditioner_options()`:

- `"auto"`: **native PETSc geometric FMG** when the mesh carries a
  genuine refinement hierarchy (`len(mesh.dm_hierarchy) > 1`, i.e.
  built with `refinement >= 1`), otherwise GAMG. Deliberately
  conservative: it only ever *adds* geometric MG, never rewrites a PC
  the user (or an internal utility solver) configured — an explicit
  `pc_type` in the options DB latches `_pc_user_override` and wins
  forever after.
- Native FMG is **locked out for single-field solvers** (#276): PETSc's
  `DMCreateInjection` fails unpredictably (geometry × degree ×
  refinement) on refined DMPlex for scalar/vector discretisations, so
  `"auto"`/`"fmg"` on a scalar solver falls back to GAMG with a
  warning. The Stokes **velocity block** (`fieldsplit_velocity_`) is
  the validated native-FMG path.

### 1.2 Raw PETSc options — the expert escape hatch

`solver.petsc_options[...]` reaches everything (`pc_type=mg`, SNESFAS,
smoother choices, level counts). The `_pc_user_override` latch above is
what keeps 1.1 from fighting this. This is also how internal utility
solvers (projections, smoothers) carry their own tuned PC bundles.

### 1.3 `solver.set_custom_fmg(coarse_meshes, ...)` — custom-P geometric MG

The generalized custom-prolongation hierarchy
(`utilities/custom_mg.py`, `CustomMGHierarchy`): barycentric/RBF
node-level transfers, BC-per-level reduction, Galerkin (RAP) coarse
operators. Works for **any coarse/fine mesh pair** (nested or not),
**any single field** (`field_id=0` targets the Stokes velocity block),
and — since #369 — **in parallel** (co-partitioned rank-local fast path
plus a cross-partition fallback selected automatically on the
zero-column signature). Notably it does *not* need `DMCreateInjection`,
so it does not suffer #276's single-field lock-out.

### 1.4 Mesh-owned auto-injection on `adapt()` children (#369)

A refinement child from `mesh.adapt(...)` carries
`mesh._custom_mg_coarse_meshes` (the static coarse tail). The first
`solve()` of *any* solver on such a mesh lazily builds and installs a
custom-P hierarchy (`auto_inject_custom_mg`) — geometric MG on the
refined operator with **no user call at all**. It is opportunistic and
fail-safe: any build/validation failure warns and falls back to the
solver's default PC (a solve must never crash because an optional
preconditioner upgrade failed).

### 1.5 Legacy `solver.set_custom_mg(...)` — deprecated

Serial-only, single-field, finest-only reduction (no BC-per-level).
Deprecated in favour of `set_custom_fmg`; carries a
`TODO(deprecate)` lifespan marker (this PR, D-87).

**Adjacent fact**: SNESFAS (nonlinear multigrid) is reachable through
1.2 alone today (validated feasible via options; no UW3 surface).

## 2. The problem

The capability matrix is excellent but the *decision burden* sits with
the user: whether their Poisson solve gets MG depends on how the mesh
was built (`refinement=`), which solver class they used (velocity block
vs scalar), whether the mesh is an adapt child, and whether they know
`set_custom_fmg` exists. The #276 lock-out means the most common case —
**a scalar Laplacian on an ordinary refined mesh** — silently gets
GAMG even though a robust geometric path (1.3) exists one call away.

## 3. Proposal — minimal-control default

One principle: **solvers auto-configure the best available MG; the user
supplies nothing.** Expert control remains via exactly two escape
hatches: `solver.petsc_options` (always wins, latched) and
`set_custom_fmg` (explicit hierarchy).

Proposed `"auto"` resolution ladder, per solver, at `_build` time:

1. **User options present** (`_pc_user_override`) → hands off.
2. **Explicit `set_custom_fmg` hierarchy** → build + install it (loud
   failure — the user asked for it).
3. **Adapt-child mesh-owned hierarchy** → auto-inject custom-P
   (fail-safe, as today).
4. **Refinement hierarchy + multi-field velocity block** → native
   PETSc FMG (as today).
5. **Refinement hierarchy + single-field solver** → *NEW*:
   auto-build a custom-P FMG from `mesh.dm_hierarchy`'s coarse levels
   (closing the #276 gap with the injection-free path) instead of
   falling back to GAMG.
6. **No hierarchy** → GAMG (as today).

Steps 1–4 and 6 are current behaviour; only step 5 is new machinery
(and it reuses 1.3 wholesale — the coarse meshes already exist as
`mesh._coarse_level_meshes()`).

Under this ladder `solver.preconditioner` keeps its three values but
`"fmg"` on a single-field solver stops meaning "warn and use GAMG" and
starts meaning "custom-P geometric FMG".

## 4. Questions for the maintainer

1. **Adopt step 5?** Should `"auto"` route single-field solvers with a
   refinement hierarchy to custom-P FMG (closing the #276 GAMG
   fallback), or does custom-P need more production mileage before it
   is a silent default? (An intermediate: `"fmg"` explicitly requests
   it, `"auto"` keeps GAMG.)
2. **Failure loudness.** Auto-injection today warns-and-falls-back
   (1.4). Is that the right contract for *every* automatic MG decision,
   or should an explicit `preconditioner="fmg"` request fail hard when
   the hierarchy cannot be built?
3. **Utility solvers.** Internal solvers (projections, mesh-smoothing
   Poisson solves) are currently excluded via their own tuned options +
   the override latch. Keep the latch as the exclusion mechanism, or
   add an explicit `wants_mg = False` class attribute so exclusion is
   visible in the solver class rather than in options state?
4. **Naming.** `solver.preconditioner` takes algorithm names
   (`"fmg"`/`"gamg"`). Under the purposeful-naming ruling, should the
   user surface become capability-shaped —
   e.g. `solver.multigrid = "auto" | "geometric" | "algebraic" | "off"`
   — with the current spellings kept as accepted aliases? Or is
   `preconditioner` (an established PETSc term) exempt?
5. **Legacy `set_custom_mg` removal.** It is deprecated with a
   lifespan marker. Remove in the next release cycle (with test_1015's
   legacy cases), or keep one more cycle?
6. **SNESFAS scope.** Nonlinear MG stays options-only (out of the
   minimal-control surface), or should a future `"auto"` consider it
   for strongly nonlinear Laplacian-family solves?
7. **`mesh.dm_hierarchy` as the trigger.** The whole ladder keys off
   `refinement >= 1` at mesh build. Is it acceptable that a mesh built
   without `refinement=` never gets geometric MG silently (GAMG only),
   or should UW3 build a default hierarchy (e.g. `refinement=1`) for
   the common constructors so step 5 usually has something to work
   with?

## 5. References

- `docs/developer/design/GENERALIZED_FMG_HIERARCHY_AND_ADAPT.md` —
  the custom-P hierarchy design (BC-per-level invariant).
- `docs/advanced/multigrid-preconditioning.md` — user-facing guide.
- `utilities/custom_mg.py` — implementation (readability pass D-79..87
  in the PR that carries this note).
- #276 (single-field native-FMG lock-out), #369 (parallel custom-MG +
  auto-injection), #290 (custom prolongation).
