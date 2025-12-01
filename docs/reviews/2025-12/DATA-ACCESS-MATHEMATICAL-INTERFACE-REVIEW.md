# Data Access & Mathematical Interface System - Architectural Review

**Review ID**: REVIEW-2025-12-02
**Date**: 2025-12-01
**Status**: 🔍 Under Review
**Priority**: HIGH
**Category**: System Architecture

---

## Executive Summary

The Data Access & Mathematical Interface System provides **transparent, reactive data access** for Underworld3 variables. Users can write natural code like `var.data[...] = values` or `velocity * 2` while the system **automatically handles PETSc synchronization** and **preserves dimensional information**.

This eliminates the previous requirement for explicit context managers (`with mesh.access(var):`) and enables **mathematical notation** that looks like the equations being modeled. The system achieves this through three tightly-integrated components:

1. **NDArray_With_Callback** (~940 lines): Numpy array subclass that triggers callbacks on modification, enabling automatic PETSc sync without user intervention.

2. **MathematicalMixin** (~700 lines): Mixin class that enables natural mathematical operations (`var * 2`, `var.norm()`, `var[0].diff(x)`) via SymPy integration.

3. **UnitAwareArray** (~1500 lines): Extension of callback arrays with comprehensive unit tracking, providing the dimensional/dimensionless dual-view interface.

The result: Users write physics in natural notation with physical units, and the system maintains parallel-safe, properly-scaled data for PETSc solvers.

---

## Overview

### Problem Statement

Scientific computing with PETSc requires careful management of:
- **Data synchronization**: Local array modifications must be communicated to distributed vectors
- **Access patterns**: Multiple variables sharing a DM need coordinated access
- **Dimensional analysis**: Users think in physical units; solvers need dimensionless values

The traditional approach required explicit context managers and manual unit handling, leading to verbose, error-prone code.

### Solution Architecture

The Data Access system implements a **reactive programming model**:

```
User Code                    Internal System                PETSc
──────────                   ───────────────                ─────
var.data[i] = value  ──►  NDArray_With_Callback  ──►  pack_to_petsc()
                              │                              │
                          callback()  ◄──────────────────────┘
                              │
                          parallel sync (MPI barrier)
```

### Key User Benefits

1. **No context managers needed**: `var.data[...] = values` just works
2. **Natural math syntax**: `velocity * 2`, `stress.norm()`, `var[0].diff(x)`
3. **Automatic unit handling**: Users see `km`, PETSc sees dimensionless
4. **Parallel safety**: MPI barriers ensure consistency across ranks

---

## System Architecture

### Component Hierarchy

```
UnitAwareArray
    │
    └── NDArray_With_Callback (base class)
            │
            └── numpy.ndarray (base class)

MeshVariable / SwarmVariable
    │
    ├── MathematicalMixin (mixin for math ops)
    │
    └── uses UnitAwareArray for .array property
```

### Component 1: NDArray_With_Callback

**Location**: `src/underworld3/utilities/nd_array_callback.py`
**Lines**: ~940

**Purpose**: Numpy ndarray subclass that triggers callbacks when array data is modified.

**Key Features**:

| Feature | Implementation | Lines |
|---------|---------------|-------|
| Callback registration | `set_callback()`, `add_callback()` | 283-318 |
| In-place operators | `__iadd__`, `__isub__`, `__imul__`, etc. | 545-748 |
| Method interception | `__setitem__`, `fill()`, `sort()` | 521-779 |
| Delayed callback batching | `delay_callbacks_global()` | 337-474 |
| Thread-safe manager | `DelayedCallbackManager` | 33-100 |
| Weak reference ownership | Prevents circular dependencies | 236 |

**Callback Interface**:
```python
def callback(array, change_info):
    """
    change_info = {
        "operation": str,        # "setitem", "iadd", "fill", etc.
        "indices": tuple/slice,  # Location of change
        "old_value": array-like, # Previous value
        "new_value": array-like, # New value
        "data_has_changed": bool # Whether to trigger sync
    }
    """
```

**Delayed Callback Pattern** (for batch operations):
```python
# Batch multiple updates, sync once at the end
with uw.synchronised_array_update():
    var1.array[...] = values1
    var2.array[...] = values2
    var3.array[...] = values3
# Single MPI barrier here, callbacks execute
```

### Component 2: MathematicalMixin

**Location**: `src/underworld3/utilities/mathematical_mixin.py`
**Lines**: ~700

**Purpose**: Enable natural mathematical notation for variables via SymPy integration.

**Key Features**:

| Feature | Method | Lines |
|---------|--------|-------|
| SymPy protocol | `_sympy_()` | 54-59 |
| Component access | `__getitem__()` | 61-96 |
| Arithmetic ops | `__add__`, `__mul__`, etc. | 226-500 |
| Method delegation | `__getattr__()` | 630-696 |
| Differentiation | `diff()` | 171-223 |
| Norm computation | `norm()` | 511-543 |

**Transparent Container Principle**:
```python
# Arithmetic returns raw SymPy - units derived on demand
def __mul__(self, other):
    if isinstance(other, MathematicalMixin):
        # Return raw SymPy product
        return self.sym * other.sym
    # Units discovered via get_units() when needed
```

**Usage Examples**:
```python
# Before (explicit .sym access)
momentum = density * velocity.sym
strain_rate = velocity.sym[0].diff(x)

# After (natural notation)
momentum = density * velocity
strain_rate = velocity[0].diff(x)
```

### Component 3: UnitAwareArray

**Location**: `src/underworld3/utilities/unit_aware_array.py`
**Lines**: ~1500

**Purpose**: Extend callback arrays with unit tracking and dimensional analysis.

**Key Features**:

| Feature | Method/Property | Lines |
|---------|-----------------|-------|
| Unit storage | `._units` (Pint Unit) | 71-120 |
| Unit compatibility | `_check_unit_compatibility()` | 280-457 |
| Reduction operations | `max()`, `min()`, `mean()`, etc. | 509-644 |
| Global MPI operations | `global_max()`, `global_mean()`, etc. | 646-1264 |
| Unit arithmetic | `__mul__`, `__truediv__` with units | 1313-1439 |
| NumPy integration | `__array_function__()` protocol | 1564-1780 |

**Unit Compatibility Rules**:
```python
# Addition/Subtraction: Units must match (auto-convert if compatible)
length_m + length_km  # → converts km to m, returns m

# Multiplication/Division: Units combine
length_m * time_s  # → returns m·s
length_m / time_s  # → returns m/s

# Scalar operations: Preserve units
length_m * 2  # → returns m
```

**Dual-View Interface**:
```python
# User sees dimensional values
T.array  # → UnitAwareArray with units="K", values in Kelvin

# PETSc sees dimensionless values (internally)
T._lvec.array  # → raw ndarray, dimensionless
```

---

## Integration Points

### MeshVariable Integration

**Location**: `src/underworld3/discretisation/discretisation_mesh_variables.py`

**Key Integration**:

1. **Canonical Data Array** (Lines 2133-2197):
   ```python
   def _create_canonical_data_array(self):
       # Single source of truth for all array operations
       arr = NDArray_With_Callback(...)
       arr.set_callback(self._canonical_data_callback)
       return arr
   ```

2. **Canonical Callback** (Lines 2158-2196):
   ```python
   def _canonical_data_callback(self, change_info):
       # Sync to PETSc on any modification
       self.pack_raw_data_to_petsc()
       # Handle mesh update locking
       # Call _on_data_changed() for variable-specific updates
   ```

3. **Array Property** (Lines 1442-1451):
   ```python
   @property
   def array(self):
       # Returns UnitAwareArray wrapping canonical data
       return self._array_view  # Shape: (N, a, b)
   ```

### SwarmVariable Integration

**Location**: `src/underworld3/swarm.py`

**Key Integration**:

1. **MathematicalMixin Inheritance** (Line 40):
   ```python
   class SwarmVariable(DimensionalityMixin, MathematicalMixin, Stateful, uw_object):
   ```

2. **Variable Array Creation** (Lines 270-299):
   ```python
   def _create_variable_array(self):
       arr = NDArray_With_Callback(...)
       arr.set_callback(self.variable_update_callback)
       return arr
   ```

3. **Lazy Proxy Evaluation** (Lines 218-244):
   ```python
   # Proxy only updates when sym property accessed
   @property
   def sym(self):
       if self._proxy_stale:
           self._update()
       return self._proxy.sym
   ```

---

## Data Flow Diagrams

### MeshVariable Data Flow

```
User: var.data[i] = value
         │
         ▼
NDArray_With_Callback.__setitem__()
         │
         ▼
_trigger_callback(change_info)
         │
         ▼
_canonical_data_callback()
         │
         ▼
pack_raw_data_to_petsc()
         │
         ▼
PETSc vector updated (parallel sync)
```

### SwarmVariable Data Flow

```
User: swarm_var.data[i] = value
         │
         ▼
NDArray_With_Callback.__setitem__()
         │
         ▼
variable_update_callback()
         │
         ▼
unpack_uw_data_to_petsc()
         │
         ▼
PETSc DM field updated
         │
         ▼
_proxy_stale = True (lazy update)
```

### Mathematical Operations Flow

```
User: var1 + var2
         │
         ▼
MathematicalMixin.__add__()
         │
         ▼
Returns: var1.sym + var2.sym (raw SymPy)
         │
         ▼
Units derived on demand via uw.get_units()
```

---

## Key Architectural Decisions

### 1. Single Canonical Data Source

**Decision**: MeshVariable has ONE `_canonical_data` array, all views derive from it.

**Why**: Prevents field access conflicts where multiple arrays could have different states.

**Implementation**: `_create_canonical_data_array()` creates the single source; `.array` and `.data` are views.

### 2. Callback-Based Reactivity

**Decision**: Use callbacks instead of explicit sync calls.

**Why**:
- Users don't need to remember to sync
- Automatic MPI barriers ensure parallel safety
- Batch operations via `delay_callbacks_global()`

**Trade-off**: Small overhead per modification (callback invocation).

### 3. Weak Reference Ownership

**Decision**: NDArray_With_Callback stores owner as `weakref`.

**Why**: Prevents circular dependencies between arrays and their owning variables.

**Implementation**: Line 236 in `nd_array_callback.py`.

### 4. Transparent Container Principle (MathematicalMixin)

**Decision**: Arithmetic returns raw SymPy; units are derived on demand.

**Why**:
- Eliminates unit sync issues
- Simplifies arithmetic implementation
- Works with SymPy's expression tree model

**Implementation**: `get_units()` traverses expression tree to find atoms with `.units`.

### 5. Pint Unit Storage (Not Strings)

**Decision**: Store Pint `Unit` objects internally, accept strings at user boundary.

**Why**: Pint objects enable dimensional analysis; strings are for user convenience.

**Example**:
```python
# User provides string
arr = UnitAwareArray([1, 2, 3], units="m/s")

# Internally stored as Pint Unit
arr._units  # → <Unit('meter / second')>
```

---

## Testing Instructions

### Core Data Access Tests

```bash
cd /Users/lmoresi/+Underworld/underworld-pixi-2/underworld3

# Rebuild after source changes
pixi run underworld-build

# Run data access tests
pixi run -e default pytest tests/test_0100_backward_compatible_data.py -v
pixi run -e default pytest tests/test_0120_data_property_access.py -v
pixi run -e default pytest tests/test_0140_synchronised_updates.py -v

# Run mathematical mixin tests
pixi run -e default pytest tests/test_0520_mathematical_mixin_enhanced.py -v

# Run unit-aware array tests
pixi run -e default pytest tests/test_0802_unit_aware_arrays.py -v
```

### Quick Validation

```python
import underworld3 as uw
import numpy as np

# Test 1: Direct data access (no context manager)
mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# This should just work - no 'with mesh.access()' needed
T.array[...] = 300.0
print(f"✅ Direct assignment: T.max() = {T.array.max()}")

# Test 2: Mathematical notation
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
v.array[...] = [[1.0, 0.0]]  # Velocity field

# Natural math syntax
speed = v.norm()  # Works without .sym
print(f"✅ Math notation: v.norm() type = {type(speed)}")

# Test 3: Batch updates with delayed callbacks
with uw.synchronised_array_update():
    T.array[0] = 400.0
    T.array[1] = 350.0
print("✅ Batch updates: Single sync at context exit")
```

---

## Known Limitations

### 1. Callback Overhead

Each array modification triggers a callback. For tight loops with many small updates, this adds overhead.

**Mitigation**: Use `delay_callbacks_global()` to batch updates:
```python
with uw.synchronised_array_update():
    for i in range(1000):
        var.array[i] = compute(i)
# Single callback/sync here
```

### 2. View vs Copy Semantics

NumPy views (slices, reshapes) share underlying data but may not trigger callbacks correctly in all cases.

**Recommendation**: Modify the canonical `.array` or `.data` properties directly.

### 3. SwarmVariable Proxy Staleness

SwarmVariable proxy mesh variables are lazily updated. Accessing `.sym` after data changes but before proxy update may show stale values.

**Current Behavior**: Proxy updates when `.sym` is accessed.

### 4. Thread Safety

The `DelayedCallbackManager` uses thread-local storage, which works for most use cases but may have edge cases in complex threading scenarios.

---

## Relationship to Units System

This system integrates with the Units System (reviewed in PR #36) at the `UnitAwareArray` level:

| Units System Component | Data Access Integration |
|------------------------|-------------------------|
| `UWQuantity` | Values assigned to `UnitAwareArray` |
| `get_units()` | Called by MathematicalMixin for unit discovery |
| Model scaling | `UnitAwareArray` presents dimensional view |
| Pint integration | `UnitAwareArray._units` stores Pint Unit |

The **dual-view interface** is the key integration point:
- Users see dimensional values via `var.array` (UnitAwareArray)
- PETSc sees dimensionless values via internal vectors
- Conversion happens automatically at the boundary

---

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `utilities/nd_array_callback.py` | ~940 | Reactive array with callbacks |
| `utilities/mathematical_mixin.py` | ~700 | Natural math notation via SymPy |
| `utilities/unit_aware_array.py` | ~1500 | Unit-aware arrays |
| `discretisation/discretisation_mesh_variables.py` | Integration | MeshVariable uses all three |
| `swarm.py` | Integration | SwarmVariable uses MathematicalMixin |

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
- [ ] Single canonical data source prevents field access conflicts
- [ ] Callback-based reactivity eliminates need for explicit sync
- [ ] Weak reference ownership prevents circular dependencies
- [ ] Transparent container principle simplifies unit handling

### Implementation
- [ ] NDArray_With_Callback intercepts all mutating operations
- [ ] MathematicalMixin provides complete arithmetic operator set
- [ ] UnitAwareArray correctly handles unit compatibility
- [ ] DelayedCallbackManager is thread-safe

### Integration
- [ ] MeshVariable uses canonical data pattern correctly
- [ ] SwarmVariable proxy lazy evaluation works
- [ ] Units system integration via UnitAwareArray

### Testing
- [ ] Direct data access tests pass
- [ ] Mathematical notation tests pass
- [ ] Unit-aware array tests pass
- [ ] Parallel safety verified

---

**Document Status**: Comprehensive architectural review of the Data Access & Mathematical Interface System, documenting the reactive array system that eliminates context managers and enables natural mathematical notation.

**Last Updated**: 2025-12-01
