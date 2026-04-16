---
title: "Expressions & Functions System"
---

# Expressions & Functions

## Overview

The `UWexpression` class is the symbolic backbone of Underworld3. It wraps
SymPy symbols with metadata (units, values, descriptions) while remaining
fully compatible with SymPy arithmetic and the JIT compilation pipeline.

**Key files:**

| File | Purpose |
|------|---------|
| `function/expressions.py` | `UWexpression`, unwrapping, constant detection |
| `function/_function.pyx` | `UnderworldFunction` — mesh variable symbols |
| `utilities/_jitextension.py` | JIT compiler, constants extraction, C code generation |

## Creating Expressions

```python
import underworld3 as uw

# Scalar constant
viscosity = uw.expression("eta", 1e21)

# With units (when scaling is active)
viscosity = uw.expression("eta", uw.quantity(1e21, "Pa*s"))

# Composite expression — built from other expressions
Ra = uw.expression("Ra", rho * alpha * g * DeltaT * L**3 / (eta * kappa))
```

Expressions are SymPy `Symbol` subclasses, so they work naturally in
equations:

```python
# Arithmetic produces new SymPy expressions (not raw floats)
flux = viscosity * strain_rate        # viscosity stays symbolic
buoyancy = Ra * temperature * unit_z  # Ra stays symbolic
```

## Why Expressions Matter for Performance

**Expressions are the preferred way to pass parameters to solvers.**
When a solver parameter is a `UWexpression`, changing its value between
time steps does not trigger JIT recompilation. When the parameter is a
raw Python number, changing it requires a full rebuild of the compiled
C extension (~5–15 seconds per solve).

```python
# GOOD — expression parameter, no recompilation on change
eta = uw.expression("eta", 1e21)
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta

for step in range(100):
    eta.sym = compute_new_viscosity(step)  # Just updates constants[]
    stokes.solve()                          # ~0.3s per solve

# SLOW — raw number, forces recompilation every step
for step in range(100):
    stokes.constitutive_model.Parameters.shear_viscosity_0 = new_value
    stokes.solve(_force_setup=True)  # ~15s per solve (JIT rebuild)
```

This is especially important for:

- **Viscoelastic solvers** — `dt_elastic` changes every step
- **Parameter sweeps** — varying viscosity, yield stress, etc.
- **Time-dependent BCs** — oscillatory or ramped boundary conditions
- **Navier-Stokes** — any time-varying forcing or material property

## How It Works: The Constants Mechanism

### The Problem

The JIT compiler translates SymPy expressions into C code for PETSc's
pointwise function interface. Previously, all constant values were baked
as C literals:

```c
// Old: value baked into compiled code
double result = 1e+21 * velocity_gradient;  // must recompile to change
```

### The Solution

Every PETSc pointwise function signature includes `numConstants` and
`constants[]` parameters that were previously unused. Now, `UWexpression`
atoms that are spatially constant (no dependence on coordinates or field
variables) are automatically routed through this array:

```c
// New: value read from constants array at runtime
double result = constants[0] * velocity_gradient;  // update via PetscDSSetConstants()
```

### What Happens Automatically

1. **Constant detection** — Before JIT compilation, `_extract_constants()`
   scans all expression trees for `UWexpression` atoms whose fully-unwrapped
   value is a pure number. This works at any nesting depth (user expression →
   constitutive model parameter → solver template).

2. **Structural hashing** — The JIT cache key is computed from the
   *structural* form of expressions (constants replaced with placeholders).
   Changing a constant value produces the same hash → cache hit → no
   recompilation.

3. **Two-phase unwrap** — During code generation:
   - Phase 1: constant UWexpressions → `_JITConstant` symbols (render as `constants[i]`)
   - Phase 2: remaining UWexpressions → numerical values (baked into C code)

4. **Runtime update** — Before every `snes.solve()`, the solver calls
   `_update_constants()` which packs current values from the manifest
   and calls `PetscDSSetConstants()`. This propagates to all levels
   of the multigrid hierarchy.

### What Goes Through Constants

Any `UWexpression` that resolves to a number when fully unwrapped:

| Example | In constants[]? | Why |
|---------|-----------------|-----|
| `uw.expression("eta", 1e21)` | Yes | Pure number |
| `uw.expression("Ra", rho*g*alpha*...)` | Yes | Composite of numbers |
| `constitutive_model.Parameters.shear_viscosity_0` | Yes | Wraps user expression |
| `uw.expression("f", sin(x))` | No | Depends on coordinate `x` |
| `velocity.sym[0]` | No | Mesh variable (field dependency) |

### Inspecting the Constants Manifest

After the first solve, the constants manifest is available:

```python
stokes.solve()
for idx, expr in stokes.constants_manifest:
    print(f"constants[{idx}] = {expr.name} = {expr.sym}")
```

## Expression Unwrapping

The `unwrap()` function resolves nested `UWexpression` atoms to their
underlying values. Two modes are used internally:

| Mode | Purpose | Used by |
|------|---------|---------|
| `nondimensional` | Numeric values for JIT/evaluate | `_createext()`, `evaluate()` |
| `dimensional` | Display values with units | `print()`, notebooks |

```python
# Nested expressions
alpha = uw.expression("alpha", 3e-5)
DeltaT = uw.expression("DeltaT", 1000)
buoyancy = alpha * DeltaT  # SymPy expression, not a float

# Unwrap reveals the numeric value
from underworld3.function.expressions import unwrap
unwrap(buoyancy, keep_constants=False)  # → 0.03
```

## Integration with Constitutive Models

Constitutive model parameters are themselves `UWexpression` objects.
When you assign a user expression to a parameter, it becomes nested:

```
User: K = uw.expression("K", 1.0)
     ↓ assign to constitutive model
Model: \upkappa.sym = K        (UWexpression wrapping UWexpression)
     ↓ used in solver template
Solver: F1.sym = \upkappa * grad(u)
     ↓ constants extraction finds \upkappa
JIT: F1 → constants[0] * petsc_u_x[0]
```

Changing `K.sym = 2.0` propagates through the chain: `\upkappa` still
wraps `K`, so `_pack_constants()` reads the new value automatically.

## Key Functions

| Function | Location | Purpose |
|----------|----------|---------|
| `uw.expression(name, value)` | `expressions.py` | Create a named expression |
| `unwrap(expr, mode)` | `expressions.py` | Resolve nested expressions |
| `is_constant_expr(expr)` | `expressions.py` | Check for spatial dependencies |
| `_extract_constants(fns)` | `_jitextension.py` | Find constants in expression trees |
| `_pack_constants(manifest)` | `_jitextension.py` | Get current values for PetscDS |
| `getext(...)` | `_jitextension.py` | Full JIT pipeline: extract → compile → return |

## Related Systems

- [Mathematical Objects](../UW3_Developers_MathematicalObjects.md) — `MathematicalMixin` for natural syntax
- [Template Expressions](../TEMPLATE_EXPRESSION_PATTERN.md) — `ExpressionProperty` for solver templates
- [Solvers](solvers.md) — consume compiled expressions via PetscDS
- [Constitutive Models](constitutive-models.md) — parameter expressions feed into solver templates
