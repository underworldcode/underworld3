# Coordinate Access Patterns Audit

**Date**: 2025-01-11
**Context**: Implementation of mesh.X as CoordinateSystem object with .coords and .units properties

## Executive Summary

This document audits coordinate access patterns across the Underworld3 codebase and provides migration guidance for transitioning to the new mesh.X interface.

### Pattern Summary

| Pattern | Files | Status | Recommendation |
|---------|-------|--------|----------------|
| `mesh.data` | 65 | ❌ DEPRECATED | → `mesh.X.coords` or `mesh.points` |
| `mesh.points` | 39 | ✅ RECOMMENDED | Keep (current best practice) |
| `mesh.X[i]` | 9 | ✅ CORRECT | Keep (symbolic coordinates) |
| `mesh._points` | 3 | ⚠️ INTERNAL | Only in internal documentation |
| `mesh.X.coords` | 1 | ✅ NEW | Preferred for new code |
| `mesh.X.units` | 1 | ✅ NEW | Preferred for new code |

### Implementation Status

✅ **COMPLETED**:
- CoordinateSystem class enhanced with `.coords`, `.units` properties
- CoordinateSystem implements `__getitem__`, `__iter__`, `__len__` for backward compatibility
- `mesh.X` now returns CoordinateSystem object instead of just symbolic matrix

## New Interface Design

### mesh.X as CoordinateSystem Object

```python
# mesh.X now returns a CoordinateSystem object with:
mesh.X          # CoordinateSystem object
mesh.X[0]       # Symbolic x-coordinate (backward compatible)
mesh.X.coords   # Coordinate data array (NEW - same as mesh.points)
mesh.X.units    # Coordinate units (NEW - same as mesh.units)
x, y = mesh.X   # Unpacking (backward compatible)
```

### Pattern Comparison

```python
# OLD PATTERN (deprecated)
coords = mesh.data
units = mesh.units

# NEW PATTERN (recommended)
coords = mesh.X.coords
units = mesh.X.units

# ALTERNATIVE (also recommended)
coords = mesh.points
units = mesh.units
```

## Recommended Patterns by Use Case

### 1. Accessing Coordinate Data

**Recommended**: Use `mesh.X.coords` or `mesh.points`
```python
# NEW: Consistent with variable pattern (var.array, var.units)
coords = mesh.X.coords

# ALSO GOOD: Explicit property name
coords = mesh.points
```

**Deprecated**: `mesh.data`
```python
# OLD: Ambiguous name (data of what?)
coords = mesh.data
```

### 2. Accessing Coordinate Units

**Recommended**: Use `mesh.X.units` or `mesh.units`
```python
# NEW: Consistent with coordinate system interface
units = mesh.X.units

# ALSO GOOD: Direct property
units = mesh.units
```

### 3. Symbolic Coordinates

**Correct**: Use `mesh.X[i]` or unpacking
```python
# Symbolic coordinate access (no change needed)
x, y = mesh.X
expr = 1000 + 500 * mesh.X[0]
temperature_gradient = temp.sym.diff(mesh.X[1])
```

### 4. Checking Coordinate Shape

**Recommended**: Use `mesh.X.coords.shape` or `mesh.points.shape`
```python
# NEW
n_nodes, n_dims = mesh.X.coords.shape

# ALSO GOOD
n_nodes, n_dims = mesh.points.shape
```

**Deprecated**: `mesh.data.shape`
```python
# OLD
n_nodes, n_dims = mesh.data.shape
```

### 5. Min/Max Coordinate Values

**Recommended**: Use `mesh.X.coords` or `mesh.points`
```python
# NEW
x_min, x_max = mesh.X.coords[:, 0].min(), mesh.X.coords[:, 0].max()

# ALSO GOOD
x_min, x_max = mesh.points[:, 0].min(), mesh.points[:, 0].max()
```

**Deprecated**: `mesh.data`
```python
# OLD
x_min, x_max = mesh.data[:, 0].min(), mesh.data[:, 0].max()
```

### 6. Coordinate-Based Initialization

**Recommended**: Use `mesh.X.coords` or `mesh.points`
```python
# NEW: Clear that we're using coordinate data
with uw.synchronised_array_update():
    temperature.array[:, 0, 0] = 1000 + 500 * mesh.X.coords[:, 0]

# ALSO GOOD
with uw.synchronised_array_update():
    temperature.array[:, 0, 0] = 1000 + 500 * mesh.points[:, 0]
```

**Deprecated**: `mesh.data`
```python
# OLD
with uw.synchronised_array_update():
    temperature.array[:, 0, 0] = 1000 + 500 * mesh.data[:, 0]
```

### 7. Passing Coordinates to Functions

**Recommended**: Use `mesh.X.coords` or `mesh.points`
```python
# NEW: Explicit coordinate data
result = uw.function.evaluate(expr, mesh.X.coords)

# ALSO GOOD
result = uw.function.evaluate(expr, mesh.points)
```

**Deprecated**: `mesh.data`
```python
# OLD
result = uw.function.evaluate(expr, mesh.data)
```

## Design Philosophy

### Why mesh.X.coords Instead of Just mesh.X.data?

The new interface follows the "honest asymmetry" principle from COORDINATE_INTERFACE_DESIGN.md:

**Meshes ARE Coordinate Systems**:
- Carry geometric structure (metric tensors, orientation)
- Have symbolic coordinate functions (mesh.X[0])
- Have coordinate data in physical space (mesh.X.coords)

**Pattern Consistency**:
```python
# Variables
temperature.sym         # Symbolic expression
temperature.array       # Field data
temperature.units       # Temperature units

# Coordinates (NEW)
mesh.X                  # Coordinate system object
mesh.X[0]               # Symbolic coordinate function
mesh.X.coords           # Coordinate data
mesh.X.units            # Coordinate units
```

### Why Not mesh.data?

1. **Ambiguous**: "data" of what? Field data? Coordinate data? Metadata?
2. **Inconsistent**: Variables use `.array`, not `.data`
3. **Non-geometric**: Doesn't reflect that meshes are coordinate systems
4. **Legacy**: Was created as alias before coordinate system interface existed

## Backward Compatibility Strategy

### Phase 1: Enhanced Interface (DONE ✅)
- Implement mesh.X as CoordinateSystem object
- Add `.coords` and `.units` properties
- Keep backward compatibility with `mesh.X[0]` and unpacking

### Phase 2: Documentation Update (CURRENT)
- Audit codebase for coordinate access patterns
- Document recommended patterns
- Create migration guide

### Phase 3: Gradual Migration (FUTURE)
- Update tests to use new patterns
- Update examples to use new patterns
- Update documentation/tutorials
- Keep `mesh.data` as deprecated alias

### Phase 4: Deprecation (DISTANT FUTURE)
- Add deprecation warnings for `mesh.data`
- Update all remaining code
- Eventually remove `mesh.data` (breaking change)

## Codebase Audit Results

### By File Type

#### Source Code (`src/`)

**`visualisation/visualisation.py` (2 uses)**
- Lines 86, 111: `mesh.data` passed to visualization functions
- **Status**: Low priority (internal conversion for PyVista)
- **Migration**: Could use `mesh.X.coords` but not urgent

**`cython/petsc_generic_snes_solvers.pyx` (3 uses)**
- Lines 822, 1423, 2664: `mesh.data` used for hash calculations
- **Status**: Low priority (solver internals)
- **Migration**: Could use `mesh.X.coords` for clarity

**`meshing/cartesian.py` (3 uses)**
- Documentation mentions only
- **Status**: No action needed

**`utilities/nd_array_callback.py` (2 uses)**
- Example code in docstrings
- **Status**: Update docstring examples to use `mesh.X.coords`

#### Tests (`tests/`)

**Units/Interface Tests**:
- `test_0620_mesh_units_interface.py` (6 uses) - Tests mesh.data interface
- `test_0630_mesh_units_demonstration.py` (8 uses) - Demonstrates mesh units
- `test_0720_coordinate_units_gradients.py` (3 uses) - Tests coordinate units
- `test_0730_variable_units_integration.py` (1 use) - Variable initialization
- `test_0803_units_workflow_integration.py` (3 uses) - Integration tests
- **Status**: HIGH PRIORITY - Update to demonstrate new interface

**Basic Tests**:
- `test_0101_kdtree.py` (5 uses) - KDTree construction with coordinates
- `test_0005_IndexSwarmVariable.py` (5 uses) - Material property initialization
- `test_0505_rbf_swarm_mesh.py` (4 uses) - RBF interpolation tests
- **Status**: MEDIUM PRIORITY - Good examples for new pattern

**Solver Tests**:
- `test_1120_SLVectorCartesian.py` (1 use) - Mesh bounds checking
- **Status**: LOW PRIORITY - Simple find/replace

#### Examples (`docs/examples/`)

**Heat Transfer** (17 uses across 6 files):
- Finding mesh bounds (min/max)
- Passing coordinates to evaluation functions
- Mesh transformations
- **Status**: MEDIUM PRIORITY - User-facing examples

**Porous Flow** (4 uses across 2 files):
- Mesh coordinate operations
- **Status**: MEDIUM PRIORITY

**Others**:
- Various examples across fluid mechanics, solid mechanics, etc.
- **Status**: MEDIUM PRIORITY - Update opportunistically

#### Tutorials (`docs/beginner/tutorials/`)

- `1-Meshes.ipynb` (2 uses) - Basic mesh introduction
- `10-Particle_Swarms.ipynb` (4 uses) - Swarm operations
- `11-Multi-Material_SolCx.ipynb` (1 use) - Multi-material setup
- `12-Units_System.ipynb` (2 uses) - Units system demonstration
- `13-Dimensional_Thermal_Convection.ipynb` (2 uses) - Convection example
- `14-Scaled_Thermal_Convection.ipynb` (1 use) - Scaled convection
- **Status**: HIGH PRIORITY - Entry point for new users

### By Usage Pattern

#### Pattern 1: Finding Mesh Bounds
**Count**: ~20 instances
**Current**: `mesh.data[:, i].min()`, `mesh.data[:, i].max()`
**Migration**: `mesh.X.coords[:, i].min()`, `mesh.X.coords[:, i].max()`

#### Pattern 2: Coordinate-Based Initialization
**Count**: ~15 instances
**Current**: `var.array[:] = f(mesh.data[:, 0], mesh.data[:, 1])`
**Migration**: `var.array[:] = f(mesh.X.coords[:, 0], mesh.X.coords[:, 1])`

#### Pattern 3: Passing to Evaluation
**Count**: ~10 instances
**Current**: `uw.function.evaluate(expr, mesh.data)`
**Migration**: `uw.function.evaluate(expr, mesh.X.coords)`

#### Pattern 4: Shape Checking
**Count**: ~8 instances
**Current**: `mesh.data.shape`
**Migration**: `mesh.X.coords.shape`

#### Pattern 5: Mesh Transformation
**Count**: ~5 instances
**Current**: `new_coords = mesh.data.copy()`
**Migration**: `new_coords = mesh.X.coords.copy()`

## Migration Priority

### High Priority (Update First)
1. **Tutorials** (12 uses) - User-facing entry points
2. **Units tests** (21 uses) - Demonstrate new interface patterns
3. **Documentation examples** in docstrings

### Medium Priority (Update Opportunistically)
1. **Example scripts** (25+ uses) - As we encounter them
2. **Basic tests** (14 uses) - When running test suite
3. **Solver tests** (1 use) - Simple find/replace

### Low Priority (Update Eventually)
1. **Internal visualization code** (2 uses) - Works fine as-is
2. **Solver internals** (3 uses) - Works fine as-is
3. **Planning documents** - Already reference new pattern

## Migration Script Template

For systematic migration, use this search/replace pattern:

```bash
# Pattern 1: Finding bounds
OLD: mesh.data[:, 0].min()
NEW: mesh.X.coords[:, 0].min()

# Pattern 2: Coordinate initialization
OLD: mesh.data[:, 0]
NEW: mesh.X.coords[:, 0]

# Pattern 3: Passing to functions
OLD: uw.function.evaluate(expr, mesh.data)
NEW: uw.function.evaluate(expr, mesh.X.coords)

# Pattern 4: Shape checking
OLD: mesh.data.shape
NEW: mesh.X.coords.shape

# Pattern 5: Copying
OLD: mesh.data.copy()
NEW: mesh.X.coords.copy()
```

## Testing Strategy

### Verify Backward Compatibility

```python
# Test that old patterns still work
coords_old = mesh.data
coords_new = mesh.X.coords
assert np.allclose(coords_old, coords_new)

# Test symbolic access still works
x_old = mesh.X[0]  # Should work
x, y = mesh.X      # Should work
```

### Verify New Interface

```python
# Test new properties exist
assert hasattr(mesh.X, 'coords')
assert hasattr(mesh.X, 'units')

# Test they return correct values
assert np.allclose(mesh.X.coords, mesh.points)
assert mesh.X.units == mesh.units
```

## Swarm Coordinates (For Reference)

**Note**: Swarms follow a different pattern as documented in COORDINATE_INTERFACE_DESIGN.md:

```python
# Swarms are NOT coordinate systems (they're point collections)
swarm.data          # Particle positions (CORRECT)
swarm.coords        # Alias for swarm.data (also correct)

# Swarms do NOT have .X (by design - they're not coordinate systems)
# swarm.X           # This does not exist
```

## Summary

The new `mesh.X` interface provides a clean, consistent pattern for accessing both symbolic coordinates and coordinate data:

- **Symbolic**: `mesh.X[0]`, `x, y = mesh.X` (unchanged)
- **Data**: `mesh.X.coords` (new, preferred) or `mesh.points` (also good)
- **Units**: `mesh.X.units` (new, preferred) or `mesh.units` (also good)
- **Deprecated**: `mesh.data` (keep as alias for backward compatibility)

This design reflects the mathematical reality that meshes are coordinate systems with geometric structure, not just collections of points.

## Related Documents

- `COORDINATE_INTERFACE_DESIGN.md` - Design rationale for mesh.X vs swarm.coords
- `UNITS_SYSTEM_DESIGN_PRINCIPLES.md` - Units system principles
- `mesh_coordinate_units_design.md` - Original coordinate units design

## Next Steps

1. ✅ Implement mesh.X as CoordinateSystem object (DONE)
2. ✅ Audit codebase for patterns (DONE)
3. ⏭️ Update high-priority files (tutorials, units tests)
4. ⏭️ Update medium-priority files (examples, basic tests)
5. ⏭️ Add deprecation warnings for mesh.data (future)
