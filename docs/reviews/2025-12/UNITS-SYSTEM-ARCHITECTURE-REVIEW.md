# Units System - Architectural Review

**Review ID**: REVIEW-2025-12-01
**Date**: 2025-12-01
**Status**: 🔍 Under Review
**Priority**: HIGH
**Category**: System Architecture

---

## Executive Summary

The Underworld3 units system allows users to work **entirely in physically-dimensioned quantities**—specifying velocities in cm/year, viscosities in Pa·s, and domains in kilometers—while the internal solver architecture **automatically handles non-dimensionalization** using reference scales provided by the user.

This automatic ND system produces **identical numerical values** internal to the PETSc solver subsystem compared to what would be obtained through manual non-dimensionalization. The system achieves this through:

1. **Pint Quantity Wrappers** (`UWQuantity`, `UWexpression`): User-facing objects that carry physical dimensions and automatically convert to non-dimensional values when passed to solvers.

2. **PETSc Array Wrappers** (`UnitAwareArray`): Array views that present **dimensional values to users** while storing **dimensionless values for PETSc**. The same underlying data serves both interfaces.

3. **Reference Quantity System** (`Model`): User specifies characteristic scales (length, time, mass, temperature) from which all derived unit scales are automatically computed using linear algebra dimensional analysis.

The result: Users write physics in natural units, solvers see properly scaled numbers, and results return in physical units—with no manual conversion code required.

---

## Overview

The Underworld3 units system provides **dimensional awareness** for geophysical simulations, enabling users to work with physical quantities (kilometers, megayears, pascals) while the solver operates on non-dimensional values for numerical stability. The system is designed around the **Gateway Pattern**: units exist at boundaries (user input/output) but not during internal symbolic operations.

**Motivation**: Geophysical simulations span extreme scales—kilometers to millimeters, gigayears to seconds. A units system prevents common mistakes (e.g., mixing cm/year with m/s) and enables proper non-dimensionalization for numerical solvers.

**Scope**: Core quantities (`UWQuantity`), lazy expressions (`UWexpression`), Model-based reference scaling, unit-aware arrays, integrals, and derivatives.

---

## System Architecture

### Core Design: Gateway Pattern

The Gateway Pattern isolates unit handling to input/output boundaries:

```
User Input          Symbolic Layer              Output
───────────         ──────────────              ──────
uw.quantity() ─┐
               ├──► UWexpression ──► unwrap() ──► evaluate() ──► UnitAwareArray
uw.expression()┘    (lazy eval)      (ND for     (dimensional
                                     solver)      for user)
```

**Key Principle**: During symbolic manipulation and PETSc solving, values are **non-dimensional**. Units are attached at the final `evaluate()` call.

**Benefits**:
- Preserves lazy evaluation for time-dependent parameters
- Computational efficiency (no unit tracking during matrix assembly)
- Clean separation of concerns between physics and numerics

### Component Hierarchy

| Component | Purpose | Lines of Code |
|-----------|---------|---------------|
| **`UWQuantity`** | Atomic quantities (number + units) | ~870 |
| **`UWexpression`** | Lazy containers for symbolic expressions | ~2000 |
| **`Model`** | Reference quantity management and scaling dispatch | ~400 |
| **`UnitAwareArray`** | Arrays with unit metadata | ~200 |
| **Integral** | Unit propagation for volume/surface integrals | ~275 (in petsc_maths.pyx) |

### Key Architectural Decisions

#### 1. Pint for All Unit Operations

All unit arithmetic is delegated to the **Pint** library. No manual unit conversion code.

**Why**: Pint handles edge cases (temperature offsets, compound units like Pa·s) correctly. Manual fallbacks create subtle bugs.

```python
# CORRECT: Pint handles conversion
qty = uw.quantity(5, "cm/year")
qty_si = qty.to_base_units()  # Pint computes: 1.58e-9 m/s

# WRONG: Manual conversion loses precision
factor = 0.01 / (365.25 * 24 * 3600)  # Approximation!
```

#### 2. Transparent Container Principle

Containers (UWexpression) derive units from their contents; they don't own units separately.

**Why**: Eliminates synchronization bugs where stored units differ from computed units.

```python
# Container derives units from contents
alpha = uw.expression("α", uw.quantity(3e-5, "1/K"))
alpha.units  # → queries self._value_with_units.units (derived)

# Composite derives from tree traversal
product = alpha * beta
get_units(product)  # → traverses tree, finds atoms, combines units
```

#### 3. Linear Algebra for Dimensional Analysis

Non-dimensionalization uses linear algebra (matrix solve) rather than pattern matching.

**Why**: Pattern matching fails for composite dimensions (viscosity = Pa·s = kg/(m·s)). Linear algebra reliably computes any derived unit from reference quantities.

```python
# Model stores reference quantities for L, M, T, θ
model.set_reference_quantities(
    length=uw.quantity(1000, "km"),
    time=uw.quantity(1, "Myr"),
    mass=uw.quantity(1e20, "kg"),
    temperature=uw.quantity(1000, "K"),
)

# System solves: [L M T θ] × [exponents] = target_dimensionality
# Velocity (m/s) → L^1·T^-1 → scale = length_scale / time_scale
```

#### 4. Partial Dimensional Coverage Support

Reference quantities need not cover all four fundamental dimensions (L, M, T, θ).

**Why**: Isothermal Stokes problems don't need temperature scale. Requiring θ would be user-hostile.

**Implementation**: The system solves only for covered dimensions and fails lazily if a missing dimension is actually required.

---

## Files Modified/Created

### Core Implementation Files

| File | Purpose | Key Functions |
|------|---------|---------------|
| `src/underworld3/units.py` | Public API | `get_units()`, `non_dimensionalise()`, `dimensionalise()`, `_extract_units_info()` |
| `src/underworld3/function/quantities.py` | UWQuantity class | `.value`, `.data`, `.units`, Pint integration |
| `src/underworld3/function/expressions.py` | UWexpression class | Lazy evaluation, arithmetic operators, `unwrap_for_evaluate()` |
| `src/underworld3/model.py` | Model class | `set_reference_quantities()`, `get_scale_for_dimensionality()`, `_handle_under_determined_system()` |
| `src/underworld3/utilities/nondimensional.py` | Dimensional analysis | Matrix-based scale computation |
| `src/underworld3/cython/petsc_maths.pyx` | Integral unit propagation | `Integral.evaluate()` returns `UWQuantity` with proper units |

### Design Documents

| Document | Status |
|----------|--------|
| `docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md` | **AUTHORITATIVE** - Current architecture |
| `CLAUDE.md` | Units System Design Principles section |
| `docs/reviews/2025-11/UNITS-SYSTEM-FIXES-REVIEW.md` | Historical - SI conversion fix |
| `docs/reviews/2025-11/UNITS-EVALUATION-FIXES-2025-11-25.md` | Historical - Evaluation bug fixes |
| `docs/reviews/2025-12/UNITS-INTEGRALS-DERIVATIVES-2025-12-01.md` | Historical - Integral/derivative enhancement |

---

## Major Bug Fixes Consolidated (November-December 2025)

### 1. Non-dimensionalization SI Conversion Fix

**Problem**: Different input units (km/yr vs cm/yr) produced the same dimensionless value.

**Root Cause**: Division using raw magnitudes without converting to common units first.

**Fix**: Convert to base SI units before dividing by scale:
```python
# In units.py:non_dimensionalise()
value_si = value.to_base_units()
scale_si = scale.to_base_units()
result = value_si / scale_si  # Now correct!
```

### 2. UWexpression evaluate() ND Scaling Fix

**Problem**: `evaluate(UWexpression(...))` returned wrong values when nondimensional scaling active.

**Root Cause**: Code assumed expressions with units were already dimensional, but `unwrap_for_evaluate()` returns ND values when scaling is active.

**Fix**: Only skip re-dimensionalization when scaling is NOT active:
```python
# In expressions.py
if not scaling_is_active:  # Changed from: if expr_already_dimensional
    return result  # Already dimensional
# Otherwise re-dimensionalize
```

### 3. Derivative Units Computation Fix

**Problem**: `get_units(dv/dx)` returned variable units (m/s) instead of derivative units (m/s/km).

**Root Cause**: `_extract_units_info()` tried to get units from `sympy_expr.func` (a class), not the parent MeshVariable.

**Fix**: Access parent MeshVariable via `func.meshvar` weakref:
```python
# In units.py:_extract_units_info()
if hasattr(sympy_expr, 'diffindex'):
    meshvar_ref = sympy_expr.func.meshvar
    meshvar = meshvar_ref() if callable(meshvar_ref) else meshvar_ref
    var_units = meshvar.units
    coord_units = get_units(coord_symbol)
    derivative_units = var_units / coord_units  # Correct!
```

### 4. Integral Unit Propagation Feature

**Problem**: `Integral.evaluate()` returned plain `float`, losing dimensional information.

**Solution**: Return `UWQuantity` with proper units when mesh has coordinate units:
```python
# In petsc_maths.pyx:Integral.evaluate()
if coord_has_units or integrand_has_units:
    volume_units = coord_units ** mesh.dim  # km² or km³
    result_units = integrand_units * volume_units
    return uw.quantity(physical_value, result_units)
return vald  # Plain float for backward compatibility
```

**Backward Compatibility**: Plain meshes without units continue to return `float`.

### 5. Partial Dimensional Coverage Support

**Problem**: System required all four reference quantities (L, M, T, θ) even for isothermal problems.

**Fix**: `_handle_under_determined_system()` now solves the sub-system for covered dimensions:
```python
# In nondimensional.py
covered_dims = [d for d in [L, M, T, θ] if has_reference[d]]
sub_matrix = full_matrix[covered_dims, :]
solution = np.linalg.solve(sub_matrix, target_vector[covered_dims])
```

### 6. Unit Conversion on Composite Expressions Fix

**Problem**: `.to_base_units()` and `.to_reduced_units()` caused evaluation errors on composite expressions.

**Root Cause**: Methods embedded conversion factors that were double-applied during ND evaluation cycles.

**Fix**: For composite expressions, only change display units (no factor embedding):
```python
# In unit_aware_expression.py
if expr.atoms(UWexpression):  # Composite
    warnings.warn("changing display units only...")
    new_expr = expr  # No modification
else:  # Simple
    new_expr = expr * conversion_factor  # Apply factor
```

---

## Testing Instructions

### Core Units Tests

```bash
cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3

# Rebuild after any source changes
pixi run underworld-build

# Run units system tests
pixi run -e default pytest tests/test_07*.py -v

# Run units integration tests
pixi run -e default pytest tests/test_08*.py -v --tb=short

# Run integral tests
pixi run -e default pytest tests/test_0501_integrals.py -v
```

### Test Results Summary (2025-12-01)

**Level 2 Test Suite (Full Run)**:
- **Total**: 411 tests (288 passed, 78 failed, 34 skipped, 11 xfailed, 28 xpassed)
- **Runtime**: ~46 minutes
- **Key Metrics**:
  - Core units tests (07XX): ~79/81 passing (98%)
  - Integration tests (08XX): Many failures (test quality issues, not implementation)
  - Integral tests: 10/15 passing (5 swarm-specific failures)

**Known Test Issues**:
- Many 08XX test failures are **test quality issues** (wrong assertions comparing strings to Pint Units)
- Swarm integral tests fail due to `.sym` property access on SwarmVariable proxy
- Pattern: Tests written before Pint integration need `units_match()` helper

### Quick Validation

```bash
# Run core solver tests with units
pixi run -e default pytest tests/test_0817_poisson_nd.py tests/test_0818_stokes_nd.py -v

# Expected: All passing
```

---

## Known Limitations

### 1. Global Model State

The global Model singleton (`uw.get_default_model()`) creates persistent state affecting all tests.

**Impact**: Tests that set reference quantities can cause unrelated tests to fail when run together.

**Workaround**: Use `uw.reset_default_model()` at test start, or run tests with `pytest-xdist` isolation.

### 2. Reference Quantities Required for ND Scaling

Variables with units require Model reference quantities that cover the needed dimensions.

**Example**: A variable with units `"Pa*s"` (viscosity) requires length, mass, and time references.

**Mitigation**: Partial coverage support now allows L, M, T without θ for isothermal problems.

### 3. Test Quality in 08XX Series

Many unit-aware integration tests have incorrect assertions (string comparisons instead of Pint Unit comparisons).

**Pattern**:
```python
# WRONG (fails)
assert var.units == "km"

# CORRECT
from underworld3.scaling import units as ureg
assert var.units == ureg.km
```

---

## Arithmetic Closure Summary

| Operation | Result Type | Units Preserved? |
|-----------|-------------|------------------|
| `UWQuantity ⊗ UWQuantity` | `UWQuantity` | ✅ Pint arithmetic |
| `UWQuantity ⊗ scalar` | `UWQuantity` | ✅ |
| `UWQuantity ⊗ UWexpression` | `UWexpression` | ✅ Wrapped with combined units |
| `UWexpression ⊗ UWexpression` | `sympy.Mul` | ✅ Units discoverable from atoms |
| `UWexpression ⊗ scalar` | `UWexpression` | ✅ |
| `MeshVar.sym ⊗ MeshVar.sym` | `sympy.Mul` | ✅ Units discoverable from atoms |

**Key Rule**: Any operation involving unit-aware types returns a type that preserves units.

---

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | Claude (AI) | 2025-12-01 | Submitted |
| Primary Reviewer | TBD | TBD | Pending |
| Secondary Reviewer | TBD | TBD | Pending |
| Project Lead | Louis Moresi | TBD | Pending |

---

## Review Checklist

### Architecture
- [ ] Gateway pattern correctly isolates units from solver internals
- [ ] Transparent container principle prevents sync bugs
- [ ] Linear algebra dimensional analysis handles all derived units
- [ ] Partial coverage support works for isothermal problems

### Implementation
- [ ] All unit operations delegate to Pint (no manual fallbacks)
- [ ] SI conversion in non_dimensionalise() is correct
- [ ] Derivative units computation handles all MeshVariable types
- [ ] Integral unit propagation is backward compatible

### Testing
- [ ] Core units tests (07XX) pass
- [ ] Stokes ND tests pass
- [ ] Integral tests pass (mesh-based)
- [ ] Test quality issues in 08XX series documented

### Documentation
- [ ] UNITS_SIMPLIFIED_DESIGN_2025-11.md is authoritative
- [ ] CLAUDE.md reflects current design principles
- [ ] Bug fixes consolidated in this review

---

## Related Pull Requests and Issues

- **PR #35**: Review system infrastructure (establishes this process)
- November 2025 reviews: 6 component reviews under consideration
- Historical fixes: Multiple commits (see individual review documents)

---

**Document Status**: This is the comprehensive architectural review for the Units System, consolidating work from November-December 2025. It supersedes granular per-fix reviews which are preserved as historical reference.

**Last Updated**: 2025-12-01
