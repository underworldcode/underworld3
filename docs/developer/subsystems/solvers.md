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