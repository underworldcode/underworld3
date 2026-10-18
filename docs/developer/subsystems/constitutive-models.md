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

### Viscoelastic constituents in a multi-material model

The history stores the composite stress $\sigma^* = \sum_i \phi_i \sigma_i^*$, which
inside material $i$ is that material's own stress. Each constituent advances it with
its own law, integrator and coefficients, so the composite is the set of Maxwell laws,
each integrated where it applies. `MultiMaterialConstitutiveModel` forwards the
stress-history interface to its constituents:

| Interface | Composite behaviour |
|---|---|
| `requires_stress_history` | true when any constituent integrates a history |
| `order` / `effective_order` | the highest over the viscoelastic constituents (the history is allocated at that order; a rebuild follows the start-up ramp of the highest-order one). Setting `order` (the solver does, when it already holds a history) leaves each constituent's own order alone and refuses a history of lower order than a constituent needs; the solver then keeps the model it had. A single viscoelastic model is different: the solver sets its order to the history's, so an order-1 model given to an order-2 history integrates at order 2 |
| `stress_history_ddt_kwargs` | every slot any constituent asks for; contradictory requests raise |
| `Parameters.dt_elastic` | set on every viscoelastic constituent; read from the first |
| `_update_history_coefficients`, `_update_history_post_solve` | called on every constituent |
| `history_flux` | $\sum_i \phi_i h_i$, with $h_i$ the constituent's `history_flux` (its `flux` when it has none) |
| `Parameters._solver`, `_solver_is_setup` | passed to / gathered from every constituent, so a constituent's parameter change rebuilds the solver's functions |

**What combines.** Any mix of `ViscousFlowModel`, `ViscoElasticPlasticFlowModel`,
`TransverseIsotropicVEPFlowModel` (and the split variant), with `integrator="bdf"`
or `"etd"`, first or second order, with or without yield. Mixing integrators is
consistent because the shared history is only the stress: a BDF constituent reads
$\sigma^*$ through its own BDF weights, an exponential one through its own
$\alpha = e^{-\Delta t/\tau_i}$, $\varphi$. Each viscoelastic model therefore owns its
`(alpha, phi)` coefficients (`_ExponentialCoefficients`), formed from its own Maxwell
time; one pair on the history would carry the Maxwell time of whichever constituent
wrote it last. A BDF-1 constituent beside a BDF-2 one reads a second-order history with
a zero second weight; an ETD-1 constituent ignores the forcing slot an ETD-2 one asks
for ($\varphi = \alpha$ removes that term).

**Refused.** A constituent with `stress_history="log_conformation"` (the history
would no longer be the stress), and constituents whose history requests contradict
each other.

**Accuracy.** The history is a continuous projection of a stress that jumps at the
interface, so it is smeared there, and the error reaching a point grows with the
stress jump measured against the local stress and falls with the distance in cells.
Measured in pure shear on an 8x8 mesh, at 0.35 from the interface (about 2.8 cells),
relative to the local peak stress: two VEP materials with a 4:1 Maxwell-time
contrast, 6e-4; VEP beside TI-VEP (stresses 1.55 and 0.47), 1.4e-3 in the TI
material; a viscous material ($\eta = 4$, stress 8) beside a VEP one (stress up to
1.55), 7.9e-3 (BDF) and 7.3e-3 (ETD) in the VEP material. Three materials in thirds
(about 1.3 cells from each interface in the middle band) reach 4.1e-2. Where the
stress jump is large against the stress of interest, resolve the interface or read
the stress further from it (`tests/test_1054_multimaterial_viscoelastic.py`).

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