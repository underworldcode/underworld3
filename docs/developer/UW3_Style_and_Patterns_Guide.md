---
title: "Underworld3 Style and Patterns Guide"
subtitle: "Development Standards and Architectural Patterns"
---

```{note} Document Purpose
This guide documents the established patterns, conventions, and architectural decisions for Underworld3 development. It serves as a detailed reference for maintaining consistency across the codebase.

The normative style contract is the [UW3 Style Charter](UW3_STYLE_CHARTER.md) — where this guide and the Charter disagree, **the Charter wins**. See the authority map in [the developer documentation index](index.md).
```

# Code Organization

## Directory Structure

- **Source code**: `underworld3/src/underworld3/`
- **Documentation**: `underworld3/docs/`
- **Tests**: `underworld3/tests/`
- **Utilities**: `underworld3/src/underworld3/utilities/`

## Import Patterns

```python

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

# Property Patterns

## Reactive Data Properties

Properties should return array-like objects that can trigger updates when modified:

```python

class Field:
    @property
    def data(self):
        """Field data with reactive callbacks."""
        if self._cached_data is None:
            self._cached_data = NDArray_With_Callback(
                self._values,
                owner=self
            )
            self._cached_data.set_callback(self._on_values_changed)
        return self._cached_data

    def _on_values_changed(self, array, change_info):
        # Invalidate cached computations that depend on the values
        self._interpolant = None
        self._stats_cache = None
```

## Property with Getter/Setter Pattern

```python

@property
def clip_to_mesh(self):
    return self._clip_to_mesh

@clip_to_mesh.setter
def clip_to_mesh(self, value):
    self._clip_to_mesh = bool(value)
```

## Array-like Property Access

When properties need to behave like arrays but with additional functionality:

```python

# Users index into the property: var.data[...] rather than rebinding var.data
# Properties return NDArray_With_Callback for transparent numpy compatibility
```

# Documentation Style

## NumPy/Sphinx Docstrings

Docstrings use **NumPy style with RST markup** (`:math:`, ``double backticks``
for code). This is the settled standard (Style Charter §6); the docstrings turn
into the Sphinx documentation and render well in Jupyter. The conversion of older
docstrings is tracked in `docs/plans/docstring-conversion-plan.md`.

```python

def solve_diffusion(kappa, delta_t, monotone=False):
    r"""Advance the diffusion equation by one timestep.

    Integrates :math:`\partial_t T = \nabla \cdot (\kappa \nabla T)` with an
    implicit theta-scheme.

    Parameters
    ----------
    kappa : float or sympy expression
        Diffusivity :math:`\kappa`. May depend on position or on other
        mesh variables.
    delta_t : float
        Timestep. Non-dimensionalised internally when the model carries units.
    monotone : bool, default=False
        Clamp interpolation overshoot during the semi-Lagrangian trace-back.

    Returns
    -------
    MeshVariable
        The updated temperature field.

    Examples
    --------
    >>> T_new = solve_diffusion(1.0, 0.01)

    Notes
    -----
    We do not shy away from equations in docstrings: state the weak form or the
    scheme where it aids the reader.
    """
```

## Key Documentation Elements

- NumPy sections in the standard order: summary line, extended description,
  ``Parameters``, ``Returns``, ``Examples``, ``Notes``, ``See Also``
- Mathematical notation via RST ``:math:`` roles (raw strings ``r"""`` for backslashes)
- Provide complete, runnable examples
- Include performance considerations where they matter (in ``Notes``)

# Array and Data Management

## NDArray_With_Callback Pattern

For reactive array data that needs to trigger updates:

```python

# Constructor pattern (array data first, like numpy)
arr = NDArray_With_Callback([1, 2, 3])  # Basic usage
arr = NDArray_With_Callback(data, owner=self)  # With ownership
arr = NDArray_With_Callback(data, owner=self, callback=func)  # With callback

# Callback signature
def callback(array: NDArray_With_Callback, change_info: dict) -> None:
    # change_info contains: operation, indices, old_value (always None,
    # retained for compatibility), new_value, array_shape, array_dtype
    pass

# For callbacks that synchronise a variable's CANONICAL storage, register
# with add_canonical_callback(func): it guards against firing on derived
# views/copies (rank-asymmetric collectives, #376) and hands the callback
# the canonical array every time.
```

## Array vs Data Property Shapes

```python

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

```python

# Preferred: Direct array access with proper indexing
temperature.array[:, 0, 0] = temp_values   # Scalar
velocity.array[:, 0, :] = vel_field        # Vector

# Mesh coordinates: read via mesh.X.coords; deform via mesh.deform()
coords = mesh.X.coords                     # (N, dim) read access
mesh.deform(mesh.X.coords + displacement)  # coordinate changes go through deform()

# Swarm particle positions: the public coords property (getter and setter)
swarm.coords = swarm.coords + displacement

# Avoid: Incorrect indexing
# scalar.array[:, 0] = values  # Missing third index!
# vector.array[:, i] = values  # Missing middle index!

# Avoid: deprecated coordinate accessors (kept only so old code runs)
# mesh.data, mesh.points, swarm.data, swarm.points
```

# Context Managers

## Direct Array Access Pattern (Preferred)

For most operations, use direct array access without context managers:

```python

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

```python

# OLD - Still works but deprecated
with mesh.access(var):
    var.data[...] = values
```

## Delay Callback Pattern

For batching operations and MPI synchronization:

```python

# Single array
with arr.delay_callback("batch update"):
    arr[0] = 1
    arr[1] = 2
    arr[2] = 3
# All callbacks fire here with MPI barriers

# Global coordination across several variables
with NDArray_With_Callback.delay_callbacks_global("field update"):
    temperature.array[:, 0, 0] = temp_values
    velocity.array[:, 0, :] = vel_values
# Synchronized execution across all arrays
# (uw.synchronised_array_update() is the public spelling of this context)
```

## Custom Context Managers

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

# MPI and Parallel Patterns

## MPI Integration

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

## Parallel Context Synchronization

- **Entry barrier**: All processes enter context together
- **Pre-callback barrier**: All processes finish operations before callbacks
- **Exit barrier**: All processes complete callbacks before context exit

## Thread Safety

- Use `threading.local()` for thread-local storage
- Implement proper locking for shared resources
- Use weak references to prevent circular dependencies

# Callback and Event Systems

## Callback Registration Patterns

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

## Error Handling in Callbacks

```python

for callback in self._callbacks.copy():
    try:
        callback(self, change_info)
    except Exception as e:
        logger.warning(f"Callback error in {callback}: {e}")
        # Continue with other callbacks
```

## Owner Pattern

```python

# Weak reference to owner
self._owner = weakref.ref(owner) if owner is not None else None

# Safe owner access
@property
def owner(self):
    return self._owner() if self._owner is not None else None
```

# Testing Patterns

## Test Structure

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

## Callback Testing

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

# File and Directory Conventions

## New Utility Files

- **Location**: `underworld3/src/underworld3/utilities/`
- **Import**: Add to `utilities/__init__.py`
- **Pattern**: `from .filename import ClassName`

## Documentation Files

- **Developer docs**: `underworld3/docs/developer/`
- **Format**: MyST Markdown (`.md`), built with Sphinx — see the
  "Documentation Requests" section of `CLAUDE.md` for admonition/math syntax
- **Naming**: Descriptive names with purpose (e.g., `UW3_Developers_NDArrays.md`)
- **Integration**: add new documents to the appropriate `toctree` in the parent
  `index.md` and verify with `pixi run docs-build`

## Test Files  

- **Location**: `underworld3/tests/`
- **Naming**: `test_NNNN_description.py`
- Use fixtures for setup/teardown

# Performance Considerations

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

# Common Patterns Summary

## Essential Patterns

1. **Reactive Properties**: Return NDArray_With_Callback with owner and callbacks
2. **Context Managers**: Use for state management and batch operations
3. **MPI Integration**: Always include barriers with error handling
4. **Documentation**: NumPy/Sphinx docstrings with RST `:math:`; MyST `.md` docs
5. **Testing**: Comprehensive callback and functionality testing
6. **Error Handling**: Graceful degradation and logging
7. **Performance**: Provide enable/disable mechanisms for expensive operations

## Migration Patterns

| Pattern | Legacy | Current | Future |
|---------|--------|---------|--------|
| **Array Access** | `with mesh.access(var): var.data[...] = values` | `var.array[:, 0, 0] = values` | Direct access preferred |
| **Multi-Variable** | `with mesh.access(var1, var2):` | `with uw.synchronised_array_update():` | Batch context |
| **Documentation** | Plain markdown | MyST markdown (Sphinx) | Enhanced features |
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

*Last updated: 2026-07 (Wave E docs alignment — docstring/doc-format/coordinate sections brought in line with the UW3 Style Charter)*
```