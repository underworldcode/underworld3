---
title: "Data Access Patterns"
---

## Purpose

The data access system provides a uniform interface for reading and writing field data on meshes and swarms while maintaining compatibility with PETSc's parallel infrastructure. The design enables direct array manipulation with automatic synchronization, eliminating the need for explicit context managers while preserving solver performance.

## Current Implementation (2025+)

### Direct Array Access

Variables now provide direct array access through the `array` property, which returns an `NDArray_With_Callback` object that automatically handles PETSc synchronization:

```python
# MeshVariable - direct access
temperature.array[...] = initial_conditions

# SwarmVariable - direct access  
material.array[:, 0, 0] = material_indices

# No context manager needed
velocity.array[mask] = boundary_values
```

The `NDArray_With_Callback` class wraps NumPy arrays with callbacks that synchronize changes to PETSc's internal storage. This provides NumPy's convenience while maintaining PETSc's parallel correctness.

### Array Shapes

Variables follow a consistent shape convention:

- **array property**: `(N, a, b)` format
  - Scalars: `(N, 1, 1)`
  - Vectors: `(N, 1, dim)` where dim=2 or 3
  - Tensors: `(N, dim, dim)`
  - Symmetric tensors: `(N, dim, dim)` (unpacked)

- **data property**: `(-1, num_components)` flat format
  - Scalars: `(N, 1)`
  - Vectors: `(N, dim)`
  - Tensors: `(N, dim²)`
  - Symmetric tensors: `(N, 6)` for 3D (packed)

The `data` property provides backward compatibility with code expecting flat arrays.

### Batch Operations

For operations updating multiple variables simultaneously, use the `synchronised_array_update()` context manager:

```python
with uw.synchronised_array_update():
    temperature.array[...] = T_values
    velocity.array[...] = V_values
    pressure.array[...] = P_values
# All synchronization happens here atomically
```

This defers PETSc synchronization until all assignments complete, ensuring atomic updates and better performance for batch operations.

### Solver Integration

Solvers access data through the `vec` property, which provides direct PETSc vector access:

```python
# Solver internal code
petsc_vec = variable.vec
petsc_vec.assemble()
```

This separation preserves solver performance. User code uses `array`, solver code uses `vec`, and the system handles synchronization between them.

## Architecture Details

### NDArray_With_Callback

The callback mechanism wraps NumPy arrays with synchronization logic:

```python
class NDArray_With_Callback:
    def __init__(self, data, callback):
        self._data = data
        self._callback = callback
    
    def __setitem__(self, key, value):
        self._data[key] = value
        if self._callback:
            self._callback(self._data)
```

When array data changes, the callback updates the corresponding PETSc vector. This ensures consistency between NumPy-style manipulation and PETSc's parallel infrastructure.

### Lazy Proxy Updates (Swarm Variables)

Swarm variables with proxy mesh variables use lazy evaluation to avoid PETSc access conflicts:

```python
# Data write triggers lazy update
swarm_var.array[...] = values  # Marks proxy as stale

# Proxy updates only when accessed
symbolic_expr = swarm_var.sym  # Triggers RBF interpolation if stale
```

This pattern prevents circular dependencies when callbacks would trigger nested PETSc field access.

### Vector Availability

Variables set `_available=True` by default, ensuring solvers can access vectors without modification. Lazy initialization creates vectors on first access:

```python
@property
def vec(self):
    if self._lvec is None:
        self._set_vec(available=True)
    return self._lvec
```

This guarantees solver compatibility while maintaining the simple user interface.

## Performance Considerations

### Direct Access Overhead

The `NDArray_With_Callback` approach adds minimal overhead. Each array assignment triggers a callback, but this is necessary regardless of API style to maintain PETSc consistency. Benchmarks show negligible performance difference compared to context manager patterns.

### Batch Update Benefits

The `synchronised_array_update()` context provides performance benefits by:

1. Deferring PETSc synchronization until batch completion
2. Reducing redundant parallel communication
3. Ensuring atomic updates across multiple variables
4. Including MPI barriers for proper parallel coordination

### Solver Performance Preservation

```{important} Solver Code Unchanged
Solver internals continue using direct `vec` property access. No performance impact from data access API changes.
```

## Implementation Files

Key implementation locations:

- **MeshVariable**: `discretisation_mesh_variables.py`
  - `array` property (lines 700-720)
  - `data` property for compatibility (lines 680-700)
  - Callback registration in `_create_variable_array()`

- **SwarmVariable**: `swarm.py`
  - `array` property with caching (lines 857-871)
  - Lazy proxy updates via `_update_proxy_if_stale()` (lines 424-445)
  - Flat data property for compatibility (lines 832-854)

- **NDArray_With_Callback**: `utilities/_utils.py`
  - Core wrapper class
  - Global callback delay mechanism
  - Data synchronization callbacks

## Migration Guide (Legacy Patterns)

### Access Context Manager (Pre-2025)

Previously, data access required explicit context managers:

```python
# OLD: Context manager required
with mesh.access(temperature):
    temperature.data[...] = values

with swarm.access(material):
    material.data[...] = indices
```

The context manager handled PETSc vector retrieval and restoration. This pattern is no longer necessary - direct array access is now the standard approach.

### Migration Steps

To update legacy code:

1. **Remove context managers**:
   ```python
   # Before
   with mesh.access(var):
       var.data[...] = values
   
   # After
   var.array[...] = values
   ```

2. **Update multi-variable operations**:
   ```python
   # Before
   with mesh.access(var1, var2, var3):
       var1.data[...] = values1
       var2.data[...] = values2
       var3.data[...] = values3
   
   # After
   with uw.synchronised_array_update():
       var1.array[...] = values1
       var2.array[...] = values2
       var3.array[...] = values3
   ```

3. **Shape adjustments** (if using data property):
   ```python
   # data property: flat (N, components)
   var.data[:, 0] = scalars
   
   # array property: shaped (N, a, b)
   var.array[:, 0, 0] = scalars
   ```

### Compatibility Layer

The legacy `mesh.access()` method remains available but now acts as a no-op dummy wrapper. It defers callbacks but does not manage PETSc access. This provides compatibility for old code while allowing gradual migration.

The `data` property continues to work, providing flat array views with automatic synchronization. It exists solely for backward compatibility - new code should use `array`.

### Deprecated Methods

The following methods exist only for compatibility with migration test scaffolding and should not be used:

- `variable.use_legacy_array()` - No-op placeholder
- `variable.use_enhanced_array()` - No-op placeholder

These will be removed in a future cleanup phase.