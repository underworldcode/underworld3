# Units System Architectural Review

**Review ID**: UW3-2026-02-001
**Date**: 2026-02-01
**Status**: Submitted for Review
**Component**: Units and Dimensional Analysis
**Reviewer**: [To be assigned]

## Overview

This architectural review documents the current state of Underworld3's units system as of February 2026. The system has undergone significant consolidation since the November 2025 reviews, with ~940 lines of deprecated mixin code removed in January 2026. The current implementation follows a **Gateway Pattern** where units are handled at boundaries (input/output), not during symbolic manipulation.

## Changes Made

### January 2026 Consolidation

**Removed Components (~940 lines)**:
- `DimensionalityMixin` deprecated code paths
- Duplicate unit-tracking in expression arithmetic
- Legacy `UnitAwareExpression` class (~1500 lines reduced to current UWexpression)
- Complex unit-tracking inheritance chains

**Resulting Clean Architecture**:
- Single point of unit handling: `unwrap()` / `evaluate()` gateway
- Simpler inheritance: No dual mixin patterns
- Consistent pattern: Same as MeshVariable template
- Better lazy evaluation: No cached state on composites

### Core Files Modified

| File | Changes |
|------|---------|
| `src/underworld3/scaling/_scaling.py` | Model scaling infrastructure (~656 LOC) |
| `src/underworld3/scaling/units.py` | Pint registry configuration (~1,993 LOC) |
| `src/underworld3/function/quantities.py` | UWQuantity implementation (~859 LOC) |
| `src/underworld3/function/expressions.py` | UWexpression implementation (~1,797 LOC) |
| `src/underworld3/function/nondimensional.py` | Non-dimensionalization (~350 LOC) |
| `src/underworld3/function/unit_conversion.py` | get_units(), has_units() (~500 LOC) |

## System Architecture

### Codebase Metrics

| Module | Lines of Code | Purpose |
|--------|---------------|---------|
| `scaling/_scaling.py` | ~656 | Model scaling infrastructure |
| `scaling/units.py` | ~1,993 | Pint registry, unit definitions |
| `function/quantities.py` | ~859 | UWQuantity class |
| `function/expressions.py` | ~1,797 | UWexpression (lazy evaluation) |
| `function/nondimensional.py` | ~350 | Non-dimensionalization utilities |
| `function/unit_conversion.py` | ~500 | Conversion utilities, get_units() |
| **Total** | **~6,155** | Core units infrastructure |

### Gateway Pattern Design

The units system follows the **Gateway Pattern** documented in `UNITS_SIMPLIFIED_DESIGN_2025-11.md`:

```
User Input          Symbolic Layer              Output
───────────         ──────────────              ──────
uw.quantity() ─┐
               ├──► UWexpression ──► unwrap() ──► evaluate() ──► UnitAwareArray
uw.expression()┘    (lazy eval)      (ND for     (dimensional
                                     solver)      for user)
```

### Key Principle: Transparent Container

**Atomic vs Container Distinction**:

| Type | Role | What it stores |
|------|------|----------------|
| **UWQuantity** | Atomic leaf node | Value + Units (indivisible) |
| **UWexpression** | Container | Reference to contents only (derives everything) |

The container never "owns" units or values - it provides access to what's inside via lazy property evaluation.

## Type Hierarchy

### 1. UWQuantity (`function/quantities.py`)

Lightweight number with units, backed by Pint:

```python
# User creates with string (convenience)
viscosity = uw.quantity(1e21, "Pa*s")

# Properties
viscosity.value      # → 1e21 (dimensional)
viscosity.data       # → non-dimensional (after scaling)
viscosity.units      # → Pint Unit object (NOT string!)

# Arithmetic works via Pint
Ra = (rho0 * alpha * g * DeltaT * L**3) / (eta0 * kappa)
```

**Key Implementation Details**:
- Wraps Pint quantities internally (`_pint_qty`)
- `.units` returns Pint Unit object for dimensional analysis
- Supports conversion via `.to()` method

### 2. UWexpression (`function/expressions.py`)

Lazy-evaluated wrapper providing the preferred user-facing interface:

```python
alpha = uw.expression(r"\alpha", uw.quantity(3e-5, "1/K"), "thermal expansivity")

# Properties derived from contents
alpha.value   # → 3e-5 (dimensional)
alpha.data    # → non-dimensional value
alpha.units   # → discovered from wrapped thing
alpha.sym     # → the wrapped SymPy expression or value
```

**Transparent Container Implementation**:
```python
class UWexpression:
    @property
    def units(self):
        # Always derived, never stored separately
        if self._value_with_units is not None:
            return self._value_with_units.units  # From contained atom
        return get_units(self._sym)  # From contained tree
```

### 3. MeshVariable Integration

MeshVariables follow the same pattern as template:

```python
temperature = uw.discretisation.MeshVariable("T", mesh, 1, units="K")

# Properties
temperature.array   # → UnitAwareArray (dimensional)
temperature.data    # → non-dimensional values
temperature.sym     # → SymPy Function for symbolic use
temperature.units   # → Pint Unit object
```

### 4. UnitAwareArray (`utilities/unit_aware_array.py`)

NumPy array subclass with units attached:

```python
# Returned by evaluate(), mesh.X.coords
coords = mesh.X.coords          # → UnitAwareArray
coords.units                    # → Pint Unit object
coords.to("m")                  # → convert to different units
```

## Arithmetic Closure

The system maintains unit closure through arithmetic operations:

| Left ⊗ Right | Result Type | Units Preserved? |
|--------------|-------------|------------------|
| `UWQuantity * UWQuantity` | `UWQuantity` | ✅ Pint arithmetic |
| `UWQuantity * scalar` | `UWQuantity` | ✅ Pint arithmetic |
| `UWQuantity * UWexpression` | `UWexpression` | ✅ Wrapped with combined units |
| `UWexpression * UWexpression` | `sympy.Mul` | ✅ Discoverable via get_units() |
| `UWexpression * scalar` | `UWexpression` | ✅ Preserves expression units |
| `MeshVar.sym * MeshVar.sym` | `sympy.Mul` | ✅ Discoverable from atoms |

## Scaling and Non-Dimensionalization

### Model Reference Quantities

```python
model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(2900, "km"),
    time=uw.quantity(1, "Myr"),
    temperature=uw.quantity(1, "K"),
    mass=uw.quantity(1e21, "kg")
)
```

### Non-Dimensionalization Pipeline

```python
# User specifies dimensional value
eta = uw.quantity(1e21, "Pa*s")

# System computes non-dimensional for solver
eta_nd = uw.scaling.non_dimensionalise(eta)  # → scalar for PETSc

# Results returned dimensional
velocity = stokes.solve()
v_mps = velocity.to("m/s")  # → UnitAwareArray
```

### Coordinate Units

Mesh coordinates carry units via patching approach:

```python
x, y = mesh.X
uw.get_units(x)  # → 'kilometer' (discovered from expression tree)
```

**Implementation**: `patch_coordinate_units()` adds `_units` attribute to SymPy BaseScalar objects, with enhanced `get_units()` that searches inside expressions.

## Testing Instructions

### Core Tests (test_07xx)

- `test_0700_units_system.py`: Core functionality (21 tests)
- `test_0701_units_dimensionless.py`: Dimensionless handling
- `test_0702_units_temperature.py`: Temperature conversions
- `test_0703_units_pressure.py`: Pressure units

### Integration Tests (test_08xx)

- `test_0801_units_utilities.py`: Utility functions (10 tests)
- `test_0803_units_workflow_integration.py`: Workflow tests
- `test_0818_stokes_nd.py`: Stokes solver with ND (5 tests, all passing)

## Known Limitations

### Current Issues

| Issue | Status | Notes |
|-------|--------|-------|
| Poisson `add_natural_bc()` | Bug | PETSc error 73, being investigated |
| Swarm advection CI | Environment | Pint version mismatch, fix applied |

### Architectural Limitations

1. **Serialization**: Unit metadata not yet preserved in HDF5 save/load operations
2. **Complex Expressions**: Deeply nested expressions may not propagate units through all SymPy operations
3. **Performance**: Unit checking adds minor overhead (~5-10%) for large array operations

## Key Files

| File | Purpose |
|------|---------|
| `src/underworld3/scaling/_scaling.py` | Model scaling infrastructure |
| `src/underworld3/scaling/units.py` | Pint registry configuration |
| `src/underworld3/function/quantities.py` | UWQuantity implementation |
| `src/underworld3/function/expressions.py` | UWexpression implementation |
| `src/underworld3/function/nondimensional.py` | Non-dimensionalization |
| `src/underworld3/function/unit_conversion.py` | get_units(), has_units() |
| `docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md` | Authoritative design doc |

## Benefits of Current Architecture

### For Users
1. **Natural Units**: Work with geological units (km, Myr, GPa) directly
2. **Safety**: Dimensional consistency checking prevents errors
3. **Clarity**: Unit metadata makes code self-documenting
4. **Correctness**: Gradient operations return physically meaningful units

### For Maintenance
1. **Single Point of Truth**: Units handled in gateway functions
2. **No Cached State**: Transparent container eliminates sync issues
3. **Clean Inheritance**: No complex mixin chains
4. **Well-Tested**: Comprehensive test coverage at 07xx/08xx levels

## Recommendations

### Short-Term
1. Complete Tier B→A promotion for stable units tests
2. Investigate Poisson natural BC bug (issue in planning file)
3. Verify CI environment after Pint version fix

### Medium-Term
1. Add serialization support for unit metadata in HDF5
2. Performance optimization for large array operations
3. Enhanced expression unit propagation for complex trees

## Related Documentation

- `docs/developer/design/UNITS_SIMPLIFIED_DESIGN_2025-11.md` - Authoritative design
- `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md` - Coordinate handling
- `docs/developer/design/WHY_UNITS_NOT_DIMENSIONALITY.md` - Terminology rationale
- `docs/beginner/tutorials/12-Units_System.ipynb` - User tutorial

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | AI Assistant (Claude) | 2026-02-01 | Submitted |
| Primary Reviewer | [To be assigned] | | Pending |
| Secondary Reviewer | [To be assigned] | | Pending |
| Project Lead | [To be assigned] | | Pending |

## Review Comments and Resolutions

[To be filled during review process]

---

**Review Status**: Awaiting assignment of reviewers
**Priority**: HIGH
**Supersedes**: UW3-2025-11-003 (UNITS-AWARENESS-SYSTEM-REVIEW.md)

This architectural review documents the consolidated and streamlined units system following the January 2026 cleanup, reflecting the current production-ready state of the codebase.
