---
title: "Solvers Subsystem"
---

# Solvers System Documentation

```{note} Well-Documented Subsystem
**Module**: `systems/solvers.py` (2,255 lines)  
**Priority**: 🟢 Low - already well documented  
**Current Status**: Good documentation ✅

Could benefit from performance tuning guidance.
```

## Overview

The solvers subsystem provides numerical solvers for PDEs using PETSc's SNES and linear solvers.

### Current State
- **Files**:
  - `solvers.py`: 2,255 lines - 10 main solver classes  
  - `solver_template.py`: 411 lines - Base solver framework
  - `ddt.py`: 1,241 lines - Time derivative implementations
- **Complexity**: Very High - mathematical solver implementations
- **Documentation Quality**: Good ✅

### Core Solvers

```python
# Primary solver types - all well documented
SNES_Poisson          # Elliptic problems
SNES_Stokes           # Incompressible flow
AdvDiffSLCN           # Advection-diffusion (SLCN)
AdvDiffHamilton       # Hamiltonian advection
SteadyStateHeat       # Thermal diffusion
NavierStokesSLCN      # Navier-Stokes flow
```

## Current Documentation Status

### Strengths
- ✅ Mathematical formulations in docstrings
- ✅ Boundary condition examples
- ✅ Solver parameter descriptions
- ✅ Integration with constitutive models

### Enhancement Opportunities
- ⚠️ Performance tuning guidance needed

## Critical Architecture: Solver-Authoritative Unknowns

```{important} Fundamental Design Principle
**The solver holds the authoritative copies of all unknowns and their histories.**

This is a critical architectural insight that affects all constitutive model implementations, especially multi-material systems.
```

### Core Architecture Principle

**Ownership Hierarchy:**
```
Solver
├── Unknowns (authoritative)
│   ├── u (velocity/temperature field)
│   ├── DuDt (field time derivatives)  
│   └── DFDt (flux time derivatives - STRESS HISTORY)
│
└── Constitutive Model
    ├── References to solver's Unknowns
    └── flux property (computed from unknowns)
```

**Key Insight:** Individual constitutive models do **NOT** maintain independent copies of field histories. They read from the **shared solver state**.

### History Variable Management

**Stress History Flow:**
1. **Solver Setup**: `DFDt.psi_fn = constitutive_model.flux`
2. **Pre-Solve**: `DFDt.update_pre_solve()` → Updates $\psi^*[0]$ (history)
3. **Model Access**: `model.stress_star` → Reads $\psi^*[0]$
4. **Post-Solve**: `DFDt.update_post_solve()` → Current flux becomes next history

```python
# Example from SNES_ViscoElastic solver setup:
self.DFDt.psi_fn = self.constitutive_model.flux.T  # Set flux expression

# Individual model reads shared history:
@property  
def stress_star(self):
    if self.Unknowns.DFDt is not None:
        self._stress_star.sym = self.Unknowns.DFDt.psi_star[0].sym  # Shared state
    return self._stress_star
```

### Multi-Material Implications

**Critical for Multi-Material Models:**
- All constituent models must share the **same Unknowns object**
- Stress history is the **composite flux**, not individual model fluxes  
- Each material experiences the **same stress history** (physically correct)

**Incorrect Approach:** ❌
```python
# DON'T: Independent histories per material
material_0.Unknowns = unknowns_0  # Separate DFDt
material_1.Unknowns = unknowns_1  # Separate DFDt  
# Result: Each material only sees its own stress history
```

**Correct Approach:** ✅
```python
# DO: Shared unknowns system
for model in constituent_models:
    model.Unknowns = self.Unknowns  # Share solver's authoritative state
# Result: All materials see composite stress history
```

### Implementation Guidelines

**For Constitutive Model Developers:**

1. **Never create independent unknowns**: Always use solver-provided unknowns
2. **Read, don't store**: Access $\psi^*[0]$ via `self.Unknowns.DFDt.psi_star[0]` for history
3. **Trust solver state**: Don't cache or duplicate field derivatives
4. **Validate sharing**: Ensure multi-material models share unknowns

**For Solver Developers:**

1. **Maintain single source of truth**: Solver owns all field state
2. **Update histories consistently**: Use DDT update sequence
3. **Share unknowns objects**: Don't create duplicates for different models
4. **Document state ownership**: Make clear what solver vs model owns

### Performance Benefits

**Memory Efficiency:**
- Single $D\mathbf{F}/Dt$ system regardless of material count
- No duplication of field histories
- Shared state reduces memory fragmentation

**Computational Efficiency:**  
- One history update per time step (not per material)
- Consistent field access patterns
- Better cache locality for field operations

### Debugging and Validation

**Common Issues:**
```python
# Symptom: Multi-material elastic response seems wrong
# Cause: Models have separate unknowns (independent histories)
# Fix: Ensure all models share solver.Unknowns

# Symptom: Memory usage scales with material count  
# Cause: Each material creating own $D\mathbf{F}/Dt$ system
# Fix: Share unknowns object across all materials

# Symptom: History seems inconsistent between materials
# Cause: Reading from different $\psi^*$ arrays
# Fix: All models read from same shared $D\mathbf{F}/Dt$
```

**Validation Checks:**
```python  
def validate_unknowns_sharing(multi_material_model):
    """Verify all constituent models share the same unknowns"""
    reference_unknowns = multi_material_model.Unknowns
    
    for i, model in enumerate(multi_material_model._constitutive_models):
        assert model.Unknowns is reference_unknowns, \
            f"Model {i} has independent unknowns - should share solver unknowns"
        
        # Verify $D\mathbf{F}/Dt$ sharing
        if hasattr(model, '_stress_star'):
            assert model.Unknowns.DFDt is reference_unknowns.DFDt, \
                f"Model {i} $D\mathbf{{F}}/Dt$ not shared - stress history will be wrong"
```
- ⚠️ Preconditioner *selection* is partly covered — see "Choosing the Krylov method for a fieldsplit sub-solve" and "The multigrid option bundle, and its one owner" below; Schur preconditioner choice is still undocumented
- Could benefit from optimization examples

## Choosing the Krylov method for a fieldsplit sub-solve

This choice gets re-argued periodically. The confusion is that it looks like three
separate questions — flexible or not, `preonly` or not, how tight — when in fact the
first and third are **two halves of one design decision**, and only the middle one is
independent.

### The design, and where it comes from

The Stokes configuration descends from the Citcom solver of Moresi & Solomatov
(1995). Its central choice is that the **inner solves are deliberately inexact** —
you do not solve the velocity block exactly to apply the Schur complement, because
that would be ruinous and is unnecessary. Two consequences follow, and they must
hold together:

1. **Because the inner solves are inexact, the search directions are perturbed, so
   the outer/Schur Krylov must be flexible** (`fgmres` or similar). A non-flexible
   method assumes a fixed preconditioner; its residual recurrence is invalidated by
   a search direction that drifts.
2. **Inexact is not unbounded.** The inner solves must still converge to *well
   below* the final tolerance required of the outer solve. That margin is what the
   `0.033` and `0.1` factors encode — they are a safety margin, not a tuned constant.

Flexibility buys tolerance of inexactness; the margin bounds how inexact. Neither
works without the other, which is why arguing them separately never settles.

```{note} Guardrail policy
Defaults err on the side of robust generality. A default that is slower but
survives configurations nobody has tested yet is the right default; loosening it is
the caller's decision and the caller's risk.
```

### Axis 1 — flexible or not (`fgmres` vs `gmres` / `cg` / `fcg`)

Settled, and settled by the design above: the inner solves are inexact by
construction, so the outer and Schur Krylov methods must be flexible. The question
is **is the preconditioner stationary?**, not whether the operator is symmetric.

The velocity block of Stokes *is* SPD, so `cg` is admissible on symmetry grounds.
It still fails, because GAMG with `mg_levels_ksp_converged_maxits` performs a
variable number of smoothing iterations, making the preconditioner application
non-linear. `cg`/`fcg` residual recurrences and standard `gmres` cannot accommodate
that. FGMRES is right-preconditioned by construction and can.

Settled empirically in [#147](https://github.com/underworldcode/underworld3/issues/147)
(spherical Kramer `case1`, free-slip Nitsche, Gadi at np=144, cellsize 1/32):
`fcg` reported an indefinite matrix / `DIVERGED_PC_FAILED`; `gmres` failed with a
residual-recursion mismatch; the same configuration converged on macOS, so this is
a robustness cliff exposed by scale, not a formulation error. The rationale is
recorded inline at `petsc_generic_snes_solvers.pyx` (velocity sub-solve block).

**Default: `fgmres` on both sub-solves.** Anything non-flexible is only safe if you
can show your preconditioner is stationary, and ours generally is not.

### Axis 2 — `preonly` or an iterative wrapper

This is not a Krylov-taste question at all. It is **positional**: is this KSP a
*preconditioner application*, or an *operator inverse*?

- **Top-level PC** (scalar and single-field vector solvers, empty
  `_pc_option_prefix`): the multigrid *is* the preconditioner and an outer Krylov
  cleans up after it. One cycle per application is a legitimate design.
- **Velocity block under `pc_fieldsplit_schur_fact_type=full`**: `PCFieldSplit`
  forms `S = A₁₁ − A₁₀ A₀₀⁻¹ A₀₁` and applies `A₀₀⁻¹` *through this KSP*. It is no
  longer preconditioning anything — it **defines the operator the pressure Krylov
  iterates against**. `preonly` there does not degrade conditioning; it changes
  which system is being solved, to `S̃ ≠ S`, while the 1/μ pressure mass still
  preconditions `S`.

The failure is quiet. Measured on the rotated free-slip path (annulus, transversely
isotropic, weak plane reaching the constrained boundary, η₁/η₀ = 1e-3), the pressure
residual under `preonly` falls 4.4e4 in ~16 iterations and then **stagnates at a
floor ≈ 3.1e-7**, burning the remaining 184 iterations of its cap on every outer
iteration: 9 outer iterations and 2.77 s, against 1 outer, 17 pressure iterations
and 0.67 s once the same multigrid is wrapped in FGMRES. The isotropic control moves
identically (5 → 1 outer), so this is the Schur application and not the anisotropy.

Note the operator is *not* moving between applications — applying the Schur operator
twice to the same vector is bitwise identical under both settings. `S̃` is a fixed
linear operator, just the wrong one. (Why the floor sits where it does is not
isolated; a range/null-space inconsistency between `S̃` and the constant-pressure
null space attached to `S` is the obvious suspect.)

**Rule: `preonly` is fine as a preconditioner application and never fine as an
inverse underneath a Schur complement.** Multigrid is an excellent preconditioner;
it cannot be the whole solve when a Schur complement is applied through it.

In the terms of the design above, `preonly` is the degenerate case of axis 3: it has
no tolerance at all, so there is no margin below the outer solve for the flexible
outer Krylov to work with. Flexibility tolerates inexactness; it cannot manufacture
a margin that was never there. That is exactly the stagnation floor measured above.

### Axis 3 — how far below the outer tolerance the inner solves must go

Not a free parameter. The invariant from the design above is that **every inner
solve reaches well below the tolerance demanded of the outer solve**; the only
judgment is how much margin, and the guardrail policy decides that.

The failure this catches is concrete. The rotated free-slip path ran its outer KSP
at `rtol = tolerance` and its pressure sub-solve *also* at `rtol = tolerance` — no
margin whatsoever, the inner solve asked to be no better than the answer it feeds.
That is the invariant broken outright rather than a tuning disagreement.

Loosening the velocity sub-solve likewise does not degrade gracefully — it walks
back toward the `preonly` failure above:

| velocity sub-KSP rtol | outer its |
|---|---|
| 1e-1 | 6 |
| 1e-2 | 4 |
| 0.033 × tolerance | 1 |

Current defaults, matched across the native and rotated paths:

| sub-solve | rtol | max_it |
|---|---|---|
| velocity | `0.033 × tolerance` | 200 |
| pressure | `0.1 × tolerance` | 200 |

The rotated path previously used `0.1 × tolerance` (velocity) and `1.0 × tolerance`
(pressure). Adopting the native values costs ~17% wall clock with identical outer
iteration counts (measured on both velocity-block routes, isotropic and TI) and
reduced a transversely isotropic fault smoke test from 24 to 15 nonlinear
iterations. Cheaper is available to anyone who measures their own configuration; it
is not the default.

The `0.033` and `0.1` factors themselves are inherited from the Citcom
configuration and have no derivation recorded here beyond "well below the outer
tolerance". They are a margin whose *existence* is principled and whose *size* is
convention.

### Detecting a degraded sub-solve

Two properties make this family of problems hard to see, and both need instrumenting
rather than eyeballing:

- **The outer iteration count does not reveal it.** A full Schur factorisation with a
  good pressure mass still reports ~1 outer iteration while the inner solve grinds
  against its cap underneath. Read the sub-KSP counts.
- **An exhausted cap does not raise.** A sub-KSP that hits `max_it` returns
  `KSP_DIVERGED_ITS`, which `KSPCheckSolve` deliberately does not escalate, so it
  degrades silently. `preonly` could never fail this way, so switching to an
  iterative wrapper introduces a failure mode that must be checked for explicitly.

Beware also that `KSPGetIterationNumber` on a sub-KSP reports only its **most recent**
application, and the velocity KSP is applied once per Schur `MatMult` — i.e. once per
pressure Krylov iteration. That number is a sample, not work. `SolverInstrumentation`
(`systems/solver_health.py`) is the mechanism that sums properly.

## The multigrid option bundle, and its one owner

Three routes reach a multigrid velocity block, and they are **the same
preconditioner reached three ways**, not three alternatives:

| route | when | prolongation |
|---|---|---|
| native | mesh built with `refinement >= 1`, ordinary BCs | PETSc `DMCreateInterpolation` between refined DMPlex levels |
| custom-P, standard path | `set_custom_fmg`, or an `adapt()` child's mesh-owned coarse tail | barycentric / RBF, Galerkin coarse operators |
| custom-P, rotated path | rotated free-slip, via `rotated_bc` | as above, with the fine prolongation rotated, `P̂ = Q_v·P` |

custom-P is **mandatory** wherever native cannot go: rotated boundary conditions
(the DM-coupled hierarchy cannot express a per-node rotation) and non-nested
grids (`adapt()` children have no DMPlex refinement relation). So the routes are
not ranked — the one that matters most for adapted and curved-boundary work is
custom-P.

The option *values* live in one module, `utilities/multigrid_options.py`. Every
writer reads a bundle from there and applies it to its own options object under
its own prefix; nobody writes a multigrid option value anywhere else. That is
structural rather than stylistic: when the bundle was written in two places, the
native path was deliberately moved to a measured `gmres`+`sor` smoother and the
custom-P path was not, and the custom-P routes ran `richardson` at an iteration
count **nobody had set** — inherited from whatever last wrote that options
prefix (3 left behind by the GAMG bundle on the standard path, PETSc's own PCMG
default of 2 on the rotated path). The same function smoothed differently
depending on what had run before it.

### What a bundle carries

A bundle is the settings it sets *and* the keys it must clear. The clear-list is
derived, not hand-written: it is every key any sibling bundle sets that this one
does not. These bundles share an options prefix, so switching a block from GAMG
to geometric MG leaves the GAMG-only keys behind and `setFromOptions` will
happily re-read them. Deriving the list means a key added to one bundle
automatically becomes stale for the others.

Two consequences worth stating outright:

- **Set the smoother iteration count, never inherit it.** A bundle that omits
  `mg_levels_ksp_max_it` is not "using the default" — it is using whatever the
  last writer left.
- **The measured smoother is `gmres`+`sor` at 4 iterations.** Chebyshev needs
  eigenvalue estimates of the smoothed operator, which are fragile on the
  indefinite / variable-viscosity velocity block. Richardson is stationary and
  degrades on the non-symmetric operator the consistent-Newton tangent produces:
  measured on the Spiegelman notch (Drucker–Prager, η contrast 1e26, nested
  4-level hierarchy) the per-cycle contraction is ρ = 0.75 richardson against
  0.56 gmres at the *same* four iterations, and the gmres margin **grows with
  depth** (5% at 3 levels, 25% at 4).

### The one legitimate per-route difference

The coarse solve. The Galerkin-coarsened **rotated** velocity block inherits the
rigid-rotation null space of the constrained problem (a closed circle: one mode;
a spherical shell: three), and `redundant`/LU hits a zero pivot there —
`SUBPC_ERROR`, outer reason −11. So the rotated route asks for
`geometric_mg_bundle(coarse="svd")`, which is null-space robust and cheap on a
small coarse level. This is a named variant of the shared bundle, not a
call-site override, so it is visible in the same place as everything else.

The other native/custom-P asymmetry — that native FMG is unusable for
single-field solvers because `DMCreateInjection` cannot reliably be built on a
refined DMPlex (#276) — is deliberately *not* in the bundle. It is a routing
decision (which route a solver may take), not an option value, and lives with
the route choice in `_apply_preconditioner_options`.

### Testing it

`tests/test_1021_mg_option_bundle.py` reads the smoother configuration back off
the **live PETSc objects** after setup — not out of the options database — for
all three routes and asserts they agree, with the coarse-solve difference
asserted rather than tolerated. Options-database assertions would not have caught
the original drift, because the drift was precisely a key that was never written.

## Critical Stability Note

```{warning} Solver Stability is Paramount
**DO NOT MODIFY** solver internals without extensive benchmarking. These have been optimized over years and are the core of the system. Any documentation additions should focus on usage patterns rather than implementation changes.
```

## Implementation Tasks

```{note} For Contributors
This well-documented subsystem could benefit from:
- Preconditioner selection guidance (Krylov sub-solve choice and the multigrid option bundle are now covered; Schur preconditioner selection is not)
- Performance tuning documentation  
- Convergence analysis examples
- Scaling studies and optimization
- Advanced usage patterns
```

---

*This subsystem demonstrates good documentation practices for complex mathematical code.*