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
- ⚠️ Preconditioner selection missing
- Could benefit from optimization examples

## The Stokes fieldsplit: two nested Krylov loops

The Stokes saddle-point solver is configured as

```
pc_type                          = fieldsplit
pc_fieldsplit_type               = schur
pc_fieldsplit_schur_fact_type    = full
pc_fieldsplit_schur_precondition = a11
```

which is easy to read as one solver and is in fact **two nested Krylov loops with a third
inside them**. Knowing which loop does the work is the difference between tuning the
solver and permuting options.

For the saddle system

$$
\begin{bmatrix} A & B^{T} \\ B & 0 \end{bmatrix}
\begin{bmatrix} u \\ p \end{bmatrix} =
\begin{bmatrix} f \\ g \end{bmatrix}
$$

`fieldsplit` is a **preconditioner for a Krylov method on the full coupled system**. With
`schur_fact_type = full`, one application of that preconditioner is the block
factorisation, which needs $A^{-1}$ twice and $\hat{S}^{-1}$ once, where
$S = -B A^{-1} B^{T}$ is the Schur complement and $\hat S$ is its preconditioner — with
`schur_precondition = a11`, the $1/\mu$-weighted pressure mass matrix that
`saddle_preconditioner` supplies.

The three solves:

| loop | iterates on | cost of one iteration |
|---|---|---|
| **outer KSP** | the full coupled residual | one preconditioner application |
| **pressure sub-KSP** | the Schur system $S p = r$ | a MatMult by $S$ — i.e. **a velocity solve** |
| **velocity sub-KSP** | $A u = r$, preconditioned by FMG or GAMG | one multigrid cycle per Krylov step |

### Two designs, and the same work in different loops

The pressure block's `ksp_type` chooses between two genuinely different algorithms.

**A pressure Krylov solve gives you Citcom.** The Schur system is actually solved, so the
block preconditioner is close to the exact inverse and the outer KSP converges in one or
two iterations — a formality wrapped around a Uzawa solve. This is the classical
arrangement (Moresi & Solomatov 1995): an outer *pressure* iteration with an inexact
multigrid velocity solve inside it. UW3's inner-tolerance margins (`0.033` velocity,
`0.1` pressure, applied by the `tolerance` setter) were designed for this shape.

**`fieldsplit_pressure_ksp_type = preonly` gives you a block preconditioner.** The Schur
system is never solved; $\hat S^{-1}$ is a single mass-matrix application, the fieldsplit
becomes a cheap approximate preconditioner, and the **outer** Krylov does all the coupling
work. This is the Elman–Silvester–Wathen approach, standard in the finite-element
literature. It is not a degenerate configuration — but it is not Citcom, and the margins
above are not aimed at it.

Neither is universally better, and UW3 does not take a position: the choice is
problem-dependent and worth measuring on the problem at hand. What matters is knowing
which one is in force, because the diagnostics differ.

```{important} The two choices interact with the outer restart
Under `preonly`, *every* coupling iteration is an **outer** iteration — and the outer loop
is where GMRES's restart lives. A problem needing more outer iterations than
`ksp_gmres_restart` (PETSc default 30) discards its Krylov space and stagnates, exactly
where a deep residual is being ground out. Under a pressure Krylov the outer count stays
at one or two, so the restart never bites there; it moves into the pressure KSP, which can
be given its own.

A `preonly` configuration therefore wants a **flexible outer method and a generous
restart**. Note also that the outer preconditioner *varies between applications* whenever
a sub-block is itself a Krylov solve — which a non-flexible GMRES assumes away.
```

```{tip} Report the outer iteration count
`solve_report.sub["velocity"].its` is the total multigrid cycle count — a **cost** measure.
A restart stagnation looks like a large **outer** count (`solve_report.ksp_its`) with a
poor residual, and is invisible in the velocity total. When diagnosing a Stokes solve that
grinds, print both, plus `stokes.snes.getKSP().getConvergedReason()`.
```

### Inner tolerances are deliberately inexact, and bounded

The inner solves are inexact on purpose — you do not solve the velocity block exactly in
order to apply the Schur complement. Two consequences hold together:

1. inexact inner solves perturb the outer search directions, so **the outer Krylov must be
   flexible**; and
2. the inexactness is **bounded** — the inner solves must still converge well below the
   tolerance demanded of the outer solve. The `0.033` and `0.1` factors are that margin.

"Flexible or not" and "how tight is the inner tolerance" are two halves of one decision,
not independent axes: flexibility buys tolerance of inexactness, and the margin bounds how
inexact. `preonly` is the degenerate end — no tolerance, hence no margin — which is why it
belongs to a design where the *outer* loop is doing the converging.

Both `fieldsplit_velocity_ksp_rtol` and `fieldsplit_pressure_ksp_rtol` are settable, and
setting `tolerance` re-derives them from it — so set `tolerance` first, then any override.

```{warning} `preonly` under the Schur is a different matter
`preonly` on the **pressure** block is a design choice, as described above. `preonly` on
the **velocity** block, under `schur_fact_type = full`, is a defect: PCFieldSplit applies
$A^{-1}$ *through* the velocity sub-KSP when forming the Schur action, so an inexact
velocity solve hands the pressure Krylov a different operator than the one its
preconditioner was built for.
```

## Critical Stability Note

```{warning} Solver Stability is Paramount
**DO NOT MODIFY** solver internals without extensive benchmarking. These have been optimized over years and are the core of the system. Any documentation additions should focus on usage patterns rather than implementation changes.
```

## Implementation Tasks

```{note} For Contributors
This well-documented subsystem could benefit from:
- Preconditioner selection guidance
- Performance tuning documentation  
- Convergence analysis examples
- Scaling studies and optimization
- Advanced usage patterns
```

---

*This subsystem demonstrates good documentation practices for complex mathematical code.*