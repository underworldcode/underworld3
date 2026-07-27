---
title: "Constitutive Models Subsystem"
---

# Constitutive Models Documentation

```{note} Well-Documented Subsystem
**Module**: `constitutive_models.py` (1,967 lines)  
**Priority**: 🟢 Low - already well documented  
**Current Status**: Good documentation ✅
```

## Overview

Material physics and rheology implementations with 10 material models covering various physical behaviors.

### Current State
- **Lines of Code**: 1,967
- **Material Models**: 10 types (ViscousFlowModel, ViscoPlasticFlowModel, etc.)
- **Complexity**: High - requires physics knowledge
- **Documentation Quality**: Good ✅

## ⚠️ Critical Architecture: Solver-Authoritative State

```{important} Fundamental Design Constraint
**Constitutive models do NOT own field histories. The solver owns all unknowns and their time derivatives.**

This is essential for multi-material systems where all materials must experience the same composite stress history.
```

### State Ownership Architecture

**Solver Responsibilities:**
- Owns `Unknowns` object (velocity/temperature fields)
- Manages $D\mathbf{F}/Dt$ system (flux/stress history tracking)
- Updates field histories via DDT pre-solve/post-solve cycle
- Provides single source of truth for all field state

**Constitutive Model Responsibilities:**
- Receives reference to solver's `Unknowns`
- Computes `flux` property from current field state
- Reads (never writes) stress history $\psi^*[0]$ via `self.Unknowns.DFDt.psi_star[0]`
- Must NOT create independent field state

### History Variable Pattern

**Correct Implementation:**
```python
class ViscoElasticModel(Constitutive_Model):
    def __init__(self, unknowns):
        super().__init__(unknowns)  # Accept solver's unknowns
        # DON'T create own $D\mathbf{F}/Dt$ - use solver's
        
    @property
    def stress_star(self):
        # Read shared history from solver's $D\mathbf{F}/Dt$
        if self.Unknowns.DFDt is not None:
            self._stress_star.sym = self.Unknowns.DFDt.psi_star[0].sym
        return self._stress_star
```

**Multi-Material Pattern:**
```python
class MultiMaterialConstitutiveModel(Constitutive_Model):
    def __init__(self, unknowns, material_var, constituent_models):
        super().__init__(unknowns)
        
        # CRITICAL: All constituent models share same unknowns
        for model in constituent_models:
            model.Unknowns = self.Unknowns  # Share solver's state
            
    @property
    def flux(self):
        # Composite flux - becomes history for all materials
        return sum(level_set_i * model_i.flux for i in range(n_materials))
```

```{warning} Common Pitfalls
**❌ Never Do This:**
```python
# DON'T: Create independent histories per material
for model in constituent_models:
    model.Unknowns = create_new_unknowns()  # WRONG!

# DON'T: Cache or store field derivatives
self._cached_stress_history = DFDt.psi_star[0].copy()  # WRONG!
```

**✅ Always Do This:**
```python
# DO: Share solver's authoritative state
for model in constituent_models:
    model.Unknowns = solver.Unknowns  # Correct

# DO: Read current state, don't cache
stress_history = self.Unknowns.DFDt.psi_star[0]  # Correct $\psi^*[0]$
```
```

### Retargeting a model to another solver

Because a constitutive model reads the solver's unknowns, its parameter expressions
carry that solver's *identity*. A strain-rate-dependent viscosity — say
`shear_viscosity_0 = eps_II(v)**(1/n - 1)` — is built from a specific solver's
velocity `u` and velocity-gradient `Unknowns.L`, and every one of those atoms is
tagged with that solver's variable id.

So you **cannot copy a nonlinear model's parameters onto another solver by value**.
Doing so silently binds the new solver's viscosity to the *source* solver's solution:
the copied expression still references the source velocity, so it evaluates the
viscosity from whatever the source last solved, not the target's own iterate. The
symptom is subtle — the target solve still converges, to the wrong (frozen-viscosity)
problem.

The fix is a **rebind**: substitute the source solver's `u` and `Unknowns.L` atoms with
the target solver's, then set the result on the target model. Two gotchas:

- The atoms are hidden inside a wrapped `UWexpression`, so `.subs(...)` reaches nothing
  until you `unwrap(...)` the value first (the same "unwrap before extracting atoms"
  rule that applies to JIT compilation).
- Velocity and its first gradient cover strain-rate rheology; a pressure- or
  higher-gradient-dependent law would extend the substitution map.

This is a general **solver-copy** operation, not a free-surface quirk. It is needed
anywhere one solver is derived from another that shares a rheology — the
`FreeSurface` held/consistent solves, and equally an **adjoint operator** built to
share the forward model's viscosity. Reference implementation:
`FreeSurface._velocity_rebind_map` / `_copy_constitutive_model` in
`src/underworld3/systems/free_surface.py`.

### Key Material Models

| Model Class | Physics | Documentation |
|------------|---------|---------------|
| `ViscousFlowModel` | Newtonian viscosity | Good |
| `ViscoPlasticFlowModel` | Yield stress materials | Good |
| `ViscoplasticPlateauFlowModel` | Complex yielding | Partial |
| `ViscoElasticFlowModel` | Elastic deformation | Good |
| `Anisotropic_FlowModel` | Directional properties | Minimal |

## Documentation Status

### Strengths
- ✅ Mathematical formulations included
- ✅ Physical parameters explained
- ✅ Unit requirements documented
- ✅ Good API documentation in source code

### Enhancement Opportunities
- ⚠️ Parameter selection guidance needed
- Could benefit from more usage examples

## Comprehensive Theory Documentation

### Additional Resources

For detailed theoretical background on constitutive models, see:

- **[Constitutive Models Theory](constitutive-models-theory.md)** - Fundamental concepts, tensor formulations, Voigt notation
- **[Anisotropic Constitutive Models](constitutive-models-anisotropy.md)** - Advanced anisotropic materials, canonical forms, rotation tensors

These companion documents provide the mathematical foundation and theoretical framework for understanding and implementing constitutive models in Underworld3.

## Implementation Status

```{note} For Contributors
This subsystem already has good documentation. Potential improvements:
- Additional parameter selection examples
- Performance optimization guidance for complex models
- Integration examples with solvers
- Advanced anisotropic model usage patterns
- Validation and benchmarking examples
```

---

*This subsystem demonstrates excellent documentation practices for complex physics implementations.*