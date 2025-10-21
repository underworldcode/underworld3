# Unit Tracking Systems in Underworld3 - Comparison

## Overview

Underworld3 now has **two complementary unit tracking systems** serving different purposes:

1. **`UnitAwareMixin`** (existing) - For **variables** (MeshVariable, SwarmVariable)
2. **`UnitAwareArray`** (new) - For **evaluated results** from `uw.function.evaluate()`

These systems are **complementary, not duplicative** - they work together to provide complete unit tracking throughout the workflow.

## System 1: `UnitAwareMixin` (Existing)

**Location**: `src/underworld3/utilities/units_mixin.py`

**Purpose**: Add unit awareness to **variable objects** (MeshVariable, SwarmVariable)

**Key Features**:
- Full-featured units system with dimensional analysis
- Backend-agnostic (supports Pint)
- Provides unit conversion, scaling, dimensional compatibility checking
- Integrates with MathematicalMixin for symbolic operations
- Used via multiple inheritance: `class EnhancedMeshVariable(UnitAwareMixin, MeshVariable)`

**Interface**:
```python
velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="m/s")

# Query units
velocity.units                 # "m/s" (Pint Unit object)
velocity.dimensionality        # {'[length]': 1, '[time]': -1}
velocity.has_units             # True

# Operations
velocity.create_quantity(value)              # Create Pint quantity
velocity.non_dimensional_value()             # For solvers
velocity.check_units_compatibility(other)    # Check compatibility
velocity.to_units("km/s")                    # Convert units

# Mathematical operations with unit checking
momentum = density * velocity  # Checks unit compatibility
```

**What it provides**:
- ✅ Complete dimensional analysis system
- ✅ Unit conversion capabilities
- ✅ Non-dimensionalization for solvers
- ✅ Unit compatibility checking
- ✅ Integration with symbolic math
- ✅ Backend abstraction (Pint, potentially others)

**What it's NOT for**:
- ❌ Plain numpy arrays (can't inherit from both ndarray and this mixin)
- ❌ Evaluate results (evaluate() returns arrays, not variable objects)
- ❌ Lightweight metadata (this is a heavyweight system with full features)

## System 2: `UnitAwareArray` (New - 2025-10-13)

**Location**: `src/underworld3/function/unit_conversion.py` lines 12-70

**Purpose**: Add unit metadata to **numpy arrays** returned by `uw.function.evaluate()`

**Key Features**:
- Lightweight numpy array subclass
- Carries unit metadata without changing behavior
- Works with all numpy operations (linalg.norm, max, min, etc.)
- Unit metadata is queryable via `uw.get_units(array)`
- Intentionally loses units through operations (since dimensions change)

**Interface**:
```python
vel = uw.function.evaluate(velocity, coords)

# Query units
uw.get_units(vel)      # "m/s" (string)
vel.units              # "m/s" (string)
vel._units             # "m/s" (string)

# All numpy operations work naturally
magnitudes = np.linalg.norm(vel, axis=1)  # Returns plain numpy array
max_vel = np.max(vel)
```

**What it provides**:
- ✅ Lightweight unit metadata attachment
- ✅ Complete numpy compatibility
- ✅ Queryable units via `uw.get_units()`
- ✅ Zero behavioral changes to arrays
- ✅ Natural numpy workflow

**What it's NOT for**:
- ❌ Unit conversions (returns simple strings, not Pint quantities)
- ❌ Dimensional analysis operations
- ❌ Unit propagation through operations (intentionally lost)
- ❌ Compatibility checking (that's done at the variable level)

## Comparison Table

| Feature | UnitAwareMixin (Variables) | UnitAwareArray (Results) |
|---------|---------------------------|-------------------------|
| **Purpose** | Full-featured units for variables | Lightweight metadata for arrays |
| **Applies to** | MeshVariable, SwarmVariable | evaluate() results |
| **Unit representation** | Pint Unit objects | String (e.g., "m/s") |
| **Dimensional analysis** | ✅ Full support | ❌ Metadata only |
| **Unit conversion** | ✅ `to_units()` | ❌ Not supported |
| **Numpy compatibility** | N/A (not arrays) | ✅ Perfect compatibility |
| **Query method** | `.units` property | `uw.get_units()` function |
| **Unit propagation** | ✅ Through operations | ❌ Intentionally lost |
| **Backend support** | ✅ Pint (extensible) | ❌ Simple strings |
| **Compatibility checking** | ✅ `check_units_compatibility()` | ❌ Not supported |
| **Non-dimensionalization** | ✅ For solvers | ❌ Not supported |
| **Weight** | Heavyweight (full features) | Lightweight (metadata only) |

## How They Work Together

The two systems complement each other in the workflow:

```python
# 1. Create variable with FULL unit support (UnitAwareMixin)
velocity = uw.discretisation.MeshVariable("V", mesh, 2, units="m/s")

# Variable-level operations use UnitAwareMixin:
print(velocity.units)                    # Pint Unit object
print(velocity.dimensionality)          # Full dimensional analysis
velocity.check_units_compatibility(other)  # Compatibility checking

# 2. Evaluate returns LIGHTWEIGHT unit-aware array (UnitAwareArray)
vel = uw.function.evaluate(velocity, coords)

# Array-level queries use simple string metadata:
print(uw.get_units(vel))                # "m/s" (string)

# 3. Numpy operations work naturally (return plain arrays)
magnitudes = np.linalg.norm(vel, axis=1)  # Plain numpy array, no units
print(uw.get_units(magnitudes))          # None

# 4. Assignment back to variable uses UnitAwareMixin for validation
temperature = uw.discretisation.MeshVariable("T", mesh, 1, units="K")
temperature.array[:, 0, 0] = magnitudes  # Validation at variable level
```

## Design Rationale

### Why Two Systems?

1. **Different Use Cases**:
   - **Variables** need full-featured units: conversion, scaling, dimensional analysis
   - **Evaluate results** need simple metadata: "what units does this array represent?"

2. **Technical Constraints**:
   - Can't inherit from both `np.ndarray` and `UnitAwareMixin` (multiple inheritance issues)
   - Arrays need to be lightweight for performance (millions of points)
   - Variables are few (dozens) so can be heavyweight with full features

3. **Numpy Compatibility**:
   - UnitAwareArray is designed to be numpy-transparent (subclass of ndarray)
   - UnitAwareMixin is designed for variable objects, not arrays

4. **Unit Propagation Philosophy**:
   - **Variables**: Units propagate through symbolic operations (`velocity * density`)
   - **Arrays**: Units DON'T propagate through numpy operations (dimensions change)
   - This difference requires different implementations

### No Duplication

While both systems track units, they don't duplicate functionality:

- **UnitAwareMixin** provides: Conversion, scaling, dimensional analysis, compatibility
- **UnitAwareArray** provides: Metadata attachment, query support, numpy compatibility

The overlap is intentional and minimal: both answer "what units?" but in appropriate ways for their context.

## Future Consolidation?

Could these be consolidated? Unlikely to be beneficial because:

1. **Inheritance constraints**: Can't use mixin approach for numpy arrays
2. **Weight tradeoff**: Variables benefit from full features, arrays need to be lightweight
3. **Use pattern differences**: Variables support operations, arrays support numpy
4. **Backend needs**: Variables need Pint backend, arrays just need strings

The current design is **correct separation of concerns**:
- **UnitAwareMixin**: Operational unit system for variables
- **UnitAwareArray**: Informational unit metadata for arrays

## Summary

**Not duplication, but complementary systems**:

- `UnitAwareMixin` = "I am a variable with units" (heavyweight, full-featured)
- `UnitAwareArray` = "I am an array that came from something with units" (lightweight, metadata-only)

Together they provide complete unit tracking:
1. Variables track units operationally (conversion, scaling, checking)
2. Arrays track units informationally (query, inspection)
3. Users can always ask "what units?" at any point in the workflow
4. Numpy operations work naturally without unit overhead
