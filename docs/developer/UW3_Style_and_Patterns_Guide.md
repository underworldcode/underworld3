---
title: "Underworld3 Style and Patterns Guide"
subtitle: "Development Standards and Architectural Patterns"
author: "Underworld Development Team"
date: today
execute:
  enabled: false
format:
  html:
    toc: true
    toc-depth: 3
    toc-location: left
    number-sections: true
    code-fold: true
    theme: cosmo
  pdf:
    documentclass: report
    geometry: margin=1in
    toc: true
    number-sections: true
---

```{note} Document Purpose
This guide documents the established patterns, conventions, and architectural decisions for Underworld3 development. It serves as a reference for maintaining consistency across the codebase.
```

# Code Organization {#sec-organization}

## Directory Structure

- **Source code**: `underworld3/src/underworld3/`
- **Documentation**: `underworld3/docs/`
- **Tests**: `underworld3/tests/`
- **Utilities**: `underworld3/src/underworld3/utilities/`

## Import Patterns

```{python}
#| eval: false

# Utilities are imported and made available
from underworld3.utilities import NDArray_With_Callback

# MPI access pattern
import underworld3 as uw
if hasattr(uw, 'mpi') and hasattr(uw.mpi, 'barrier'):
    uw.mpi.barrier()

# Synchronised updates pattern
import underworld3 as uw
with uw.synchronised_array_update():
    # Batch operations here
    pass
```

## Naming Conventions

- **Private attributes**: Use `_` prefix (e.g., `_particle_coordinates`, `_clip_to_mesh`)
- **Internal methods**: Use `_` prefix (e.g., `_trigger_callback`, `_on_data_changed`)
- **Public properties**: No prefix, use descriptive names (e.g., `data`, `clip_to_mesh`)
- **Context managers**: Use descriptive names (e.g., `delay_callback`, `dont_clip_to_mesh`)

# Property Patterns {#sec-properties}

## Reactive Data Properties

Properties should return array-like objects that can trigger updates when modified:

```{python}
#| eval: false

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

## Property with Getter/Setter Pattern

```{python}
#| eval: false

@property
def clip_to_mesh(self):
    return self._clip_to_mesh

@clip_to_mesh.setter
def clip_to_mesh(self, value):
    self._clip_to_mesh = bool(value)
```

## Array-like Property Access

When properties need to behave like arrays but with additional functionality:

```{python}
#| eval: false

# Users should access: mesh.data[...] instead of mesh.data
# Properties return NDArray_With_Callback for transparent numpy compatibility
```

# Documentation Style {#sec-documentation}

## Markdown Docstrings for pdoc/pdoc3

Use markdown format with mathematics support:

```{python}
#| eval: false

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

## Key Documentation Elements

- Use `#` headers for structure
- Include mathematical notation with LaTeX
- Provide complete, runnable examples
- Use tables for parameter documentation
- Include performance considerations

# Array and Data Management {#sec-arrays}

## NDArray_With_Callback Pattern

For reactive array data that needs to trigger updates:

```{python}
#| eval: false

# Constructor pattern (array data first, like numpy)
arr = NDArray_With_Callback([1, 2, 3])  # Basic usage
arr = NDArray_With_Callback(data, owner=self)  # With ownership
arr = NDArray_With_Callback(data, owner=self, callback=func)  # With callback

# Callback signature
def callback(array: NDArray_With_Callback, change_info: dict) -> None:
    # change_info contains: operation, indices, old_value, new_value, array_shape, array_dtype
    pass
```

## Array vs Data Property Shapes

```{python}
#| eval: false

# Array property: (N, a, b) format - PREFERRED
scalar.array.shape      # (N, 1, 1)
vector.array.shape      # (N, 1, dim)  
tensor.array.shape      # (N, dim, dim)

# Data property: (-1, components) format - BACKWARD COMPATIBILITY
scalar.data.shape       # (N, 1)
vector.data.shape       # (N, dim)
tensor.data.shape       # (N, 6) for symmetric

# Indexing patterns
scalar.array[:, 0, 0] = values        # Scalar assignment
vector.array[:, 0, i] = component_i   # Vector component
vector.array[:, 0, :] = all_components # Full vector
```

## Data Access Patterns

```{python}
#| eval: false

# Preferred: Direct array access with proper indexing
temperature.array[:, 0, 0] = temp_values  # Scalar
velocity.array[:, 0, :] = vel_field      # Vector
mesh.data[0] = new_position               # Mesh coordinates
swarm.data += displacement                # Swarm positions

# Avoid: Incorrect indexing
# scalar.array[:, 0] = values  # Missing third index!
# vector.array[:, i] = values  # Missing middle index!
```

## Coordinate System Transformations

```{python}
#| eval: false

# Reference changes throughout codebase
# OLD: swarm.particle_coordinates
# NEW: swarm._particle_coordinates
# OLD: mesh.deform_mesh
# NEW: mesh._deform_mesh
```

# Context Managers {#sec-context}

## Direct Array Access Pattern (Preferred)

For most operations, use direct array access without context managers:

```{python}
#| eval: false

# Single variable - no context needed
temperature.array[:, 0, 0] = initial_values
velocity.array[:, 0, :] = velocity_field

# Multiple variables - use synchronised update
with uw.synchronised_array_update():
    temperature.array[:, 0, 0] = temp_values
    velocity.array[:, 0, :] = vel_values
    pressure.array[:, 0, 0] = press_values
# All arrays synchronized here
```

## Legacy Access Context (Deprecated)

The old pattern still works but is no longer recommended:

```{python}
#| eval: false

# OLD - Still works but deprecated
with mesh.access(var):
    var.data[...] = values
```

## Delay Callback Pattern

For batching operations and MPI synchronization:

```{python}
#| eval: false

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

## Custom Context Managers

```{python}
#| eval: false

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

# MPI and Parallel Patterns {#sec-mpi}

## MPI Integration

```{python}
#| eval: false

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

## Parallel Context Synchronization

- **Entry barrier**: All processes enter context together
- **Pre-callback barrier**: All processes finish operations before callbacks
- **Exit barrier**: All processes complete callbacks before context exit

## Thread Safety

- Use `threading.local()` for thread-local storage
- Implement proper locking for shared resources
- Use weak references to prevent circular dependencies

# Callback and Event Systems {#sec-callbacks}

## Callback Registration Patterns

```{python}
#| eval: false

# Multiple callback support
arr.set_callback(callback)          # Replace existing
arr.add_callback(callback)          # Add additional  
arr.remove_callback(callback)       # Remove specific
arr.clear_callbacks()               # Remove all

# Enable/disable for performance
arr.disable_callbacks()             # Batch operations
arr.enable_callbacks()              # Re-enable
```

## Error Handling in Callbacks

```{python}
#| eval: false

for callback in self._callbacks.copy():
    try:
        callback(self, change_info)
    except Exception as e:
        logger.warning(f"Callback error in {callback}: {e}")
        # Continue with other callbacks
```

## Owner Pattern

```{python}
#| eval: false

# Weak reference to owner
self._owner = weakref.ref(owner) if owner is not None else None

# Safe owner access
@property
def owner(self):
    return self._owner() if self._owner is not None else None
```

# Testing Patterns {#sec-testing}

## Test Structure

```{python}
#| eval: false

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

## Callback Testing

```{python}
#| eval: false

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

# File and Directory Conventions {#sec-files}

## New Utility Files

- **Location**: `underworld3/src/underworld3/utilities/`
- **Import**: Add to `utilities/__init__.py`
- **Pattern**: `from .filename import ClassName`

## Documentation Files

- **Developer docs**: `underworld3/docs/developer/`
- **Format**: Quarto markdown (`.qmd`)
- **Naming**: Descriptive names with purpose (e.g., `UW3_Developers_NDArrays.qmd`)

## Test Files  

- **Location**: `underworld3/tests/`
- **Naming**: `test_NNNN_description.py`
- Use fixtures for setup/teardown

# Performance Considerations {#sec-performance}

## Callback Performance

- **Zero overhead** when callbacks disabled
- **Minimal impact** (< 5% typical) when enabled
- Use delayed contexts for batch operations
- Disable callbacks during bulk modifications

## Memory Management

- Use weak references for owner relationships
- Clean up cached data appropriately
- Avoid circular dependencies

## MPI Performance  

- Batch operations within delay contexts
- Minimize barrier frequency
- Use appropriate synchronization points

# Common Patterns Summary {#sec-summary}

## Essential Patterns

1. **Reactive Properties**: Return NDArray_With_Callback with owner and callbacks
2. **Context Managers**: Use for state management and batch operations
3. **MPI Integration**: Always include barriers with error handling
4. **Documentation**: Markdown with mathematics for pdoc/jupyter compatibility
5. **Testing**: Comprehensive callback and functionality testing
6. **Error Handling**: Graceful degradation and logging
7. **Performance**: Provide enable/disable mechanisms for expensive operations

## Migration Patterns

| Pattern | Legacy | Current | Future |
|---------|--------|---------|--------|
| **Array Access** | `with mesh.access(var): var.data[...] = values` | `var.array[:, 0, 0] = values` | Direct access preferred |
| **Multi-Variable** | `with mesh.access(var1, var2):` | `with uw.synchronised_array_update():` | Batch context |
| **Documentation** | Plain markdown | Quarto markdown | Enhanced features |
| **Testing** | Ad-hoc patterns | Structured fixtures | Comprehensive coverage |

## Quality Guidelines

```{tip} Code Quality Checklist
- [ ] Proper error handling with logging
- [ ] Thread-safe operations where needed
- [ ] MPI barriers for parallel coordination
- [ ] Comprehensive docstrings with examples
- [ ] Unit tests for new functionality
- [ ] Performance considerations documented
- [ ] Backward compatibility preserved
```

---

```{tip} Contributing
This guide should be updated as new patterns emerge and existing patterns evolve. For questions or suggestions, please see the Contributing Guidelines or open an issue on the Underworld3 repository.

*Last updated: After NDArray migration and synchronised_array_update implementation*
```