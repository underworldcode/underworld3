# Underworld3 Style and Patterns Guide

This guide documents the established patterns, conventions, and architectural decisions for Underworld3 development. It serves as a reference for maintaining consistency across the codebase.

## Table of Contents

1. [Code Organization](#code-organization)
2. [Property Patterns](#property-patterns)
3. [Documentation Style](#documentation-style)
4. [Array and Data Management](#array-and-data-management)
5. [Context Managers](#context-managers)
6. [MPI and Parallel Patterns](#mpi-and-parallel-patterns)
7. [Callback and Event Systems](#callback-and-event-systems)
8. [Testing Patterns](#testing-patterns)

---

## Code Organization

### Directory Structure
- **Source code**: `underworld3/src/underworld3/`
- **Documentation**: `underworld3-documentation/Notebooks/`
- **Tests**: `underworld3/tests/`
- **Utilities**: `underworld3/src/underworld3/utilities/`

### Import Patterns
```python
# Utilities are imported and made available
from underworld3.utilities import NDArray_With_Callback

# MPI access pattern
import underworld3 as uw
if hasattr(uw, 'mpi') and hasattr(uw.mpi, 'barrier'):
    uw.mpi.barrier()
```

### Naming Conventions
- **Private attributes**: Use `_` prefix (e.g., `_particle_coordinates`, `_clip_to_mesh`)
- **Internal methods**: Use `_` prefix (e.g., `_trigger_callback`, `_on_data_changed`)
- **Public properties**: No prefix, use descriptive names (e.g., `data`, `clip_to_mesh`)
- **Context managers**: Use descriptive names (e.g., `delay_callback`, `dont_clip_to_mesh`)

---

## Property Patterns

### Reactive Data Properties
Properties should return array-like objects that can trigger updates when modified:

```python
class Mesh:
    @property
    def data(self):
        """Mesh coordinate data with reactive callbacks."""
        if self._cached_data is None:
            self._cached_data = NDArray_With_Callback(
                self._coordinates,
                owner=self
            )
            self._cached_data.set_callback(self._on_coordinates_changed)
        return self._cached_data
    
    def _on_coordinates_changed(self, array, change_info):
        # Invalidate cached computations
        self._jacobians = None
        self._mesh_quality = None
```

### Property with Getter/Setter Pattern
```python
@property
def clip_to_mesh(self):
    return self._clip_to_mesh

@clip_to_mesh.setter
def clip_to_mesh(self, value):
    self._clip_to_mesh = bool(value)
```

### Array-like Property Access
When properties need to behave like arrays but with additional functionality:
```python
# Users should access: mesh.data[...] instead of mesh.data
# Properties return NDArray_With_Callback for transparent numpy compatibility
```

---

## Documentation Style

### Markdown Docstrings for pdoc/pdoc3
Use markdown format with mathematics support:

```python
class MyClass:
    """
    # MyClass

    Brief description with **bold** and *italic* formatting.

    ## Mathematical Representation

    Given an array $\\mathbf{A} \\in \\mathbb{R}^{n \\times m}$, operations follow:

    $$\\mathbf{A}' = \\mathcal{O}(\\mathbf{A}) \\implies \\text{callback}(\\mathbf{A}', \\text{info})$$

    ## Usage Examples

    ### Basic Usage
    ```python
    obj = MyClass([1, 2, 3])
    obj.set_callback(my_callback)
    ```

    ## Advanced Features

    - **Feature 1**: Description
    - **Feature 2**: Description

    ## Performance Notes

    - **Zero overhead** when disabled
    - **Minimal impact** during normal operations
    """
```

### Key Documentation Elements
- Use `#` headers for structure
- Include mathematical notation with LaTeX
- Provide complete, runnable examples
- Use tables for parameter documentation
- Include performance considerations

---

## Array and Data Management

### NDArray_With_Callback Pattern
For reactive array data that needs to trigger updates:

```python
# Constructor pattern (array data first, like numpy)
arr = NDArray_With_Callback([1, 2, 3])  # Basic usage
arr = NDArray_With_Callback(data, owner=self)  # With ownership
arr = NDArray_With_Callback(data, owner=self, callback=func)  # With callback

# Callback signature
def callback(array: NDArray_With_Callback, change_info: dict) -> None:
    # change_info contains: operation, indices, old_value, new_value, array_shape, array_dtype
    pass
```

### Data Access Patterns
```python
# Preferred: Direct array access
mesh.data[0] = new_position
swarm.data += displacement

# Avoid: Property without indexing (when using array-like wrapper)
# coords = mesh.data  # May not work as expected with new array-like properties
```

### Coordinate System Transformations
```python
# Reference changes throughout codebase
# OLD: swarm.particle_coordinates
# NEW: swarm._particle_coordinates
# OLD: mesh.deform_mesh
# NEW: mesh._deform_mesh
```

---

## Context Managers

### Access Context Pattern
For expensive operations requiring careful state management:

```python
with swarm.access():
    # Expensive setup occurs here
    data = swarm.data
    data[0] = new_value
# Cleanup and finalization occurs here
```

### Delay Callback Pattern
For batching operations and MPI synchronization:

```python
# Single array
with arr.delay_callback("batch update"):
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
# All callbacks fire here with MPI barriers

# Global coordination
with NDArray_With_Callback.delay_callbacks_global("mesh deformation"):
    mesh.data += displacement
    swarm.data += velocity * dt
# Synchronized execution across all arrays
```

### Custom Context Managers
```python
def dont_clip_to_mesh(self):
    """Context manager that temporarily disables mesh clipping."""
    class _ClipToggleContext:
        def __init__(self, swarm):
            self.swarm = swarm
            self.original_value = None
            
        def __enter__(self):
            self.original_value = self.swarm._clip_to_mesh
            self.swarm._clip_to_mesh = False
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            self.swarm._clip_to_mesh = self.original_value
            
    return _ClipToggleContext(self)
```

---

## MPI and Parallel Patterns

### MPI Integration
```python
# Safe MPI import pattern
try:
    import underworld3 as uw
    _has_uw_mpi = hasattr(uw, 'mpi') and hasattr(uw.mpi, 'barrier')
except ImportError:
    _has_uw_mpi = False
    uw = None

# MPI barrier usage in context managers
if _has_uw_mpi:
    try:
        uw.mpi.barrier()
    except Exception as e:
        logger.warning(f"MPI barrier failed: {e}")
```

### Parallel Context Synchronization
- **Entry barrier**: All processes enter context together
- **Pre-callback barrier**: All processes finish operations before callbacks
- **Exit barrier**: All processes complete callbacks before context exit

### Thread Safety
- Use `threading.local()` for thread-local storage
- Implement proper locking for shared resources
- Use weak references to prevent circular dependencies

---

## Callback and Event Systems

### Callback Registration Patterns
```python
# Multiple callback support
arr.set_callback(callback)          # Replace existing
arr.add_callback(callback)          # Add additional  
arr.remove_callback(callback)       # Remove specific
arr.clear_callbacks()               # Remove all

# Enable/disable for performance
arr.disable_callbacks()             # Batch operations
arr.enable_callbacks()              # Re-enable
```

### Error Handling in Callbacks
```python
for callback in self._callbacks.copy():
    try:
        callback(self, change_info)
    except Exception as e:
        logger.warning(f"Callback error in {callback}: {e}")
        # Continue with other callbacks
```

### Owner Pattern
```python
# Weak reference to owner
self._owner = weakref.ref(owner) if owner is not None else None

# Safe owner access
@property
def owner(self):
    return self._owner() if self._owner is not None else None
```

---

## Testing Patterns

### Test Structure
```python
def test_feature_name(setup_data):
    # Arrange
    obj = setup_data
    obj.configure_for_test()
    
    # Act
    result = obj.perform_operation()
    
    # Assert
    assert result.meets_expectations()
    np.testing.assert_allclose(expected, actual, rtol=1e-15)
```

### Callback Testing
```python
def test_callback_triggering():
    execution_log = []
    
    def test_callback(array, info):
        execution_log.append(f"{info['operation']} at {info['indices']}")
    
    arr = NDArray_With_Callback([1, 2, 3])
    arr.set_callback(test_callback)
    
    arr[0] = 99
    
    assert len(execution_log) == 1
    assert "setitem at 0" in execution_log[0]
```

---

## File and Directory Conventions

### New Utility Files
- Location: `underworld3/src/underworld3/utilities/`
- Import: Add to `utilities/__init__.py`
- Pattern: `from .filename import ClassName`

### Documentation Files
- Notebooks: `underworld3-documentation/Notebooks/Developers/WIP/`
- Format: Python percent format (`# %%` cells)
- Naming: Descriptive names with purpose (e.g., `NDArray_With_Callback_Demo.py`)

### Test Files  
- Location: `underworld3/tests/`
- Naming: `test_NNNN_description.py`
- Use fixtures for setup/teardown

---

## Performance Considerations

### Callback Performance
- **Zero overhead** when callbacks disabled
- **Minimal impact** (< 5% typical) when enabled
- Use delayed contexts for batch operations
- Disable callbacks during bulk modifications

### Memory Management
- Use weak references for owner relationships
- Clean up cached data appropriately
- Avoid circular dependencies

### MPI Performance  
- Batch operations within delay contexts
- Minimize barrier frequency
- Use appropriate synchronization points

---

## Common Patterns Summary

1. **Reactive Properties**: Return NDArray_With_Callback with owner and callbacks
2. **Context Managers**: Use for state management and batch operations
3. **MPI Integration**: Always include barriers with error handling
4. **Documentation**: Markdown with mathematics for pdoc/jupyter compatibility
5. **Testing**: Comprehensive callback and functionality testing
6. **Error Handling**: Graceful degradation and logging
7. **Performance**: Provide enable/disable mechanisms for expensive operations

This guide should be updated as new patterns emerge and existing patterns evolve.

---

*Last updated: Based on NDArray_With_Callback implementation and Underworld3 development session*