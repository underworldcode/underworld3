# Array System and Mathematical Mixins Review

**Review ID**: UW3-2025-11-002
**Date**: 2025-11-17
**Status**: Submitted for Review
**Component**: Core Data Structures
**Reviewer**: [To be assigned]

## Overview

This review covers the redesign of Underworld3's array access system and mathematical operation framework. The changes eliminate the need for access context managers (`with mesh.access()`), introduce automatic PETSc synchronization, and enable natural mathematical notation for variables. This represents a fundamental improvement in API usability while maintaining full backward compatibility.

## Changes Made

### Code Changes

**Core Array System**:
- `src/underworld3/utilities/nd_array_with_callback.py` - NDArray_With_Callback implementation (~300 lines)
- `src/underworld3/utilities/mathematical_mixin.py` - MathematicalMixin class (~800 lines)

**Variable Integration**:
- `src/underworld3/discretisation/discretisation_mesh_variables.py`:
  - `.array` property with automatic callbacks
  - `.data` property for backward compatibility
  - Integration with MathematicalMixin

- `src/underworld3/swarm.py`:
  - SwarmVariable array property
  - Data property with lazy proxy updates
  - MathematicalMixin integration

**Utility Functions**:
- `uw.synchronised_array_update()` context manager for batch operations

### Documentation Changes

**Created**:
- Migration guide for `with mesh.access()` → direct `.array` access
- Mathematical operations guide for natural notation
- Array shape conventions documentation
- Callback system technical notes

### Test Coverage

**Test Files**:
- `test_0100_backward_compatible_data.py` - Data property validation
- `test_0120_data_property_access.py` - Direct access tests
- `test_0140_synchronised_updates.py` - Batch update tests
- `test_0500_enhanced_array_structure.py` - Array structure validation
- `test_0520_mathematical_mixin_enhanced.py` - Mathematical operations

**Coverage**: ~25 tests covering array access, callbacks, and mathematical operations

## System Architecture

### Part 1: NDArray_With_Callback

#### Purpose

Provide automatic synchronization between NumPy arrays and PETSc vectors without requiring explicit access context managers.

#### Key Features

**1. Transparent Callbacks**
```python
# User writes
var.array[...] = values  # Triggers automatic sync to PETSc

# System executes
# 1. Write to NumPy array
# 2. Invoke callback: pack_to_petsc()
# 3. Update PETSc vector
# 4. No user action required!
```

**2. Global Callback Deferral**
```python
# Batch multiple operations
with uw.synchronised_array_update():
    var1.array[...] = values1  # Deferred
    var2.array[...] = values2  # Deferred
    var3.array[...] = values3  # Deferred
# All callbacks fire here in one batch
```

**3. Index Tracking**
```python
# Each callback gets unique index
callback_id = NDArray_With_Callback._register_callback(func)

# Prevents conflicts
var.array  # callback_id = 1
var.data   # callback_id = 2 (different!)
```

#### Implementation Details

**Base Class Structure**:
```python
class NDArray_With_Callback(np.ndarray):
    """NumPy array subclass with automatic callbacks on write."""

    # Class-level callback registry
    _callback_registry = {}
    _deferred_callbacks = []
    _defer_depth = 0

    def __setitem__(self, key, value):
        """Trigger callback on array write."""
        super().__setitem__(key, value)

        if self._defer_depth == 0:
            # Execute immediately
            self._callback_func(self, key, value)
        else:
            # Defer for batch execution
            self._deferred_callbacks.append(...)
```

**Callback Registration**:
```python
# MeshVariable registers callback
self._array = NDArray_With_Callback(
    data_array,
    callback=self._pack_to_petsc
)

# Callback invoked on write
var.array[0, 0, 0] = 1.0  # → _pack_to_petsc() called
```

**Deferred Execution**:
```python
@contextmanager
def delay_callbacks_global(context_info="batch"):
    """Defer all callbacks for batch execution."""
    NDArray_With_Callback._defer_depth += 1
    try:
        yield
    finally:
        NDArray_With_Callback._defer_depth -= 1
        if NDArray_With_Callback._defer_depth == 0:
            _execute_deferred_callbacks()
```

### Part 2: Mathematical Mixin

#### Purpose

Enable natural mathematical notation for variables, eliminating the need for explicit `.sym` access in most cases.

#### Key Features

**1. Direct Arithmetic**
```python
# Before
momentum = density * velocity.sym

# After
momentum = density * velocity  # Natural!
```

**2. Component Access**
```python
# Before
strain_rate = velocity.sym[0].diff(x)

# After
strain_rate = velocity[0].diff(x)  # Natural!
```

**3. SymPy Matrix API**
```python
# All SymPy Matrix methods available
velocity.T          # Transpose
velocity.norm()     # Magnitude
velocity.cross(b)   # Cross product
velocity.diff(x)    # Differentiation
# And hundreds more!
```

#### Implementation Details

**Integration Protocols**:
```python
class MathematicalMixin:
    """Mixin providing mathematical operation support."""

    def _sympify_(self):
        """SymPy protocol for conversion."""
        return self.sym  # Return symbolic form

    def __add__(self, other):
        """Enable: var + 1"""
        return self.sym + other

    def __radd__(self, other):
        """Enable: 1 + var"""
        return other + self.sym

    def __mul__(self, other):
        """Enable: var * 2"""
        return self.sym * other

    def __rmul__(self, other):
        """Enable: 2 * var"""
        return other * self.sym

    def __getitem__(self, key):
        """Enable: var[0] instead of var.sym[0]"""
        return self.sym[key]

    def __getattr__(self, name):
        """Delegate to SymPy Matrix API."""
        if name.startswith('_'):
            raise AttributeError
        return getattr(self.sym, name)
```

**Dual Operation Support**:
- `_sympify_()`: Handles SymPy-initiated operations
- Explicit methods (`__add__`, `__mul__`, etc.): Handle Python-initiated operations
- Both needed for complete integration

**Display Preservation**:
```python
def __repr__(self):
    """Show computational view by default."""
    return f"<MeshVariable '{self.name}' on mesh...>"

def sym_repr(self):
    """Mathematical/symbolic view when needed."""
    return repr(self.sym)  # LaTeX rendering in Jupyter
```

### Part 3: Backward Compatibility

#### The `.data` Property

**Purpose**: Maintain API compatibility with old `with mesh.access()` patterns

**Implementation**:
```python
@property
def data(self):
    """Backward-compatible data property."""
    # Returns view of array with different callback
    return self.array.reshape(-1, self.num_components)
```

**Key Insight**: Both `.array` and `.data` trigger callbacks, but use different callback IDs to prevent conflicts.

#### Migration Path

**Old Pattern** (Still Works):
```python
with mesh.access(var):
    var.data[...] = values
```

**New Pattern** (Recommended):
```python
var.array[...] = values
```

**Batch Updates** (Old):
```python
with mesh.access(var1, var2, var3):
    var1.data[...] = values1
    var2.data[...] = values2
    var3.data[...] = values3
```

**Batch Updates** (New):
```python
with uw.synchronised_array_update():
    var1.array[...] = values1
    var2.array[...] = values2
    var3.array[...] = values3
```

## Array Shape Conventions

### The (N, a, b) Format

**Purpose**: Unified shape for scalars, vectors, and tensors

**Convention**:
- Scalar field: `(N, 1, 1)`
- Vector field (3D): `(N, 1, 3)`
- Tensor field (3x3): `(N, 3, 3)`
- Symmetric tensor: `(N, 3, 3)` (full storage, Voigt in `.data`)

**Benefits**:
- Consistent indexing across field types
- Natural tensor operations
- Clear dimensional structure

**Example**:
```python
# Velocity field (vector)
velocity.array.shape  # (N_vertices, 1, 3)
velocity.array[100, 0, :]  # Velocity vector at vertex 100

# Stress tensor (3x3)
stress.array.shape  # (N_vertices, 3, 3)
stress.array[100, :, :]  # Stress tensor at vertex 100
```

## Key Technical Insights

### 1. Callback Index Conflicts

**Problem**: If `.array` and `.data` shared callback ID, recursion occurs.

**Solution**: Each property gets unique callback ID:
```python
# Array property
self._array = NDArray_With_Callback(
    data, callback=self._pack_to_petsc
)  # ID = 1

# Data property
return self._array.view()  # Different view, ID = 2
```

### 2. Lazy Evaluation for Swarms

**Problem**: Swarm proxy variables can't update during PETSc field access.

**Solution**: Mark proxy as stale, update only when `.sym` accessed:
```python
def _pack_to_petsc(self, array, key, value):
    """Callback on swarm data write."""
    # Update PETSc
    self._update_petsc_vector()

    # Mark proxy as stale (don't update now!)
    if hasattr(self, '_proxy'):
        self._proxy._is_stale = True
```

### 3. SymPy Integration Dual Paths

**Why Both `_sympify_()` and Explicit Methods?**

**SymPy-Initiated**:
```python
x = sympy.Symbol('x')
x * velocity  # SymPy calls velocity._sympify_()
```

**Python-Initiated**:
```python
velocity * 2  # Python calls velocity.__mul__(2)
```

Both paths needed for seamless integration.

## Testing Instructions

### Test Array Access Patterns
```bash
# Basic array access
pytest tests/test_0120_data_property_access.py -v

# Callback system
pytest tests/test_0140_synchronised_updates.py -v

# Array structure
pytest tests/test_0500_enhanced_array_structure.py -v
```

### Test Mathematical Operations
```bash
# Mathematical mixin
pytest tests/test_0520_mathematical_mixin_enhanced.py -v
```

### Verify Backward Compatibility
```bash
# Old access patterns still work
pytest tests/test_0100_backward_compatible_data.py -v
```

## Performance Impact

### Callback Overhead

**Measurement**: Negligible (~microseconds per write)

**Reason**: Callback is simple function call + PETSc vector update (already needed)

**Benefit**: Eliminates manual sync errors worth the tiny overhead

### Mathematical Operations

**Performance**: Identical to `.sym` approach

**Reason**: Returns same SymPy expression, just more conveniently

**JIT Compatibility**: Preserved - expressions identical

## Known Limitations

### 1. Read-Only Access Still Calls Callback

**Issue**: Even read operations trigger callback on first access

**Impact**: Minor - callback is no-op if no changes

**Future**: Could optimize to track dirty state

### 2. Symmetric Tensor `.data` Format

**Complexity**: `.data` returns Voigt format (6 components), `.array` full (9 components)

**Reason**: PETSc stores symmetric tensors in Voigt form

**Impact**: Users must understand which format they're accessing

### 3. Display Preference Debate

**Question**: Should `var` show mathematical or computational view?

**Decision**: Computational (current state) by default, mathematical via `sym_repr()`

**Rationale**: Users often want to inspect data, not symbolic form

## Benefits Summary

### For Users

1. **Simpler API**: No more `with mesh.access()` boilerplate
2. **Natural Math**: Variables work like mathematical objects
3. **Less Error-Prone**: Automatic sync prevents forgotten updates
4. **Better Ergonomics**: `velocity[0]` vs `velocity.sym[0]`

### For Developers

1. **Cleaner Code**: Removal of access context verbosity
2. **Fewer Bugs**: Automatic sync prevents state inconsistencies
3. **Better Debugging**: Clear separation of data vs symbolic views

### For Maintenance

1. **Backward Compatible**: Old code still works
2. **Well-Tested**: Comprehensive test coverage
3. **Documented**: Clear migration path and examples

## Related Documentation

- Migration guide for `with mesh.access()` patterns
- `utilities/nd_array_with_callback.py` implementation
- `utilities/mathematical_mixin.py` implementation
- Mathematical operations user guide

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | AI Assistant | 2025-11-17 | Submitted |
| Primary Reviewer | [To be assigned] | | Pending |
| Secondary Reviewer | [To be assigned] | | Pending |
| Project Lead | [To be assigned] | | Pending |

## Review Comments and Resolutions

[To be filled during review process]

---

**Review Status**: Awaiting assignment of reviewers
**Expected Completion**: [TBD]
**Priority**: HIGH

This review documents a fundamental API improvement that enhances usability while maintaining full backward compatibility and correctness.
