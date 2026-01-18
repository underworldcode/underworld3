# Coordinate Access Migration Guide

**Date**: 2025-01-11
**Audience**: Developers updating Underworld3 code to use new mesh.X interface

## Quick Reference

### Pattern Migration Cheat Sheet

```python
# ❌ DEPRECATED          →  ✅ RECOMMENDED
mesh.data                 →  mesh.X.coords  or  mesh.points
mesh.data.shape           →  mesh.X.coords.shape
mesh.data[:, 0]           →  mesh.X.coords[:, 0]
mesh.data.min()           →  mesh.X.coords.min()
mesh.data.copy()          →  mesh.X.coords.copy()

# ✅ UNCHANGED (these work exactly as before)
mesh.X[0]                 →  mesh.X[0]  (symbolic x-coordinate)
x, y = mesh.X             →  x, y = mesh.X  (symbolic unpacking)
mesh.units                →  mesh.units  (coordinate units)
```

## Migration Examples by Use Case

### 1. Finding Mesh Bounds

**Before:**
```python
maxY = mesh.data[:, 1].max()
minY = mesh.data[:, 1].min()
x_range = (mesh.data[:, 0].min(), mesh.data[:, 0].max())
```

**After:**
```python
maxY = mesh.X.coords[:, 1].max()
minY = mesh.X.coords[:, 1].min()
x_range = (mesh.X.coords[:, 0].min(), mesh.X.coords[:, 0].max())
```

**Alternative (also recommended):**
```python
maxY = mesh.points[:, 1].max()
minY = mesh.points[:, 1].min()
x_range = (mesh.points[:, 0].min(), mesh.points[:, 0].max())
```

### 2. Coordinate-Based Field Initialization

**Before:**
```python
with uw.synchronised_array_update():
    temperature.array[:, 0, 0] = 1000 + 500 * mesh.data[:, 0]
    velocity.array[:, 0, 0] = mesh.data[:, 1]**2
```

**After:**
```python
with uw.synchronised_array_update():
    temperature.array[:, 0, 0] = 1000 + 500 * mesh.X.coords[:, 0]
    velocity.array[:, 0, 0] = mesh.X.coords[:, 1]**2
```

**Why it's better**: Makes it clear you're using coordinate data, not field data.

### 3. Passing Coordinates to Evaluation Functions

**Before:**
```python
numerical_soln = uw.function.evaluate(poisson.u.fn, mesh.data)
analytic_soln = uw.function.evaluate(1.0 - mesh.N.y, mesh.data)
```

**After:**
```python
numerical_soln = uw.function.evaluate(poisson.u.fn, mesh.X.coords)
analytic_soln = uw.function.evaluate(1.0 - mesh.N.y, mesh.X.coords)
```

### 4. Shape Checking and Mesh Info

**Before:**
```python
print(f"Mesh created with {mesh.data.shape[0]} nodes")
n_nodes, n_dims = mesh.data.shape
```

**After:**
```python
print(f"Mesh created with {mesh.X.coords.shape[0]} nodes")
n_nodes, n_dims = mesh.X.coords.shape
```

### 5. Mesh Transformations

**Before:**
```python
new_coords = mesh.data.copy()
new_coords[:, 1] = uw.function.evaluate(h_fn * y, mesh.data)
mesh.deform_mesh(new_coords)
```

**After:**
```python
new_coords = mesh.X.coords.copy()
new_coords[:, 1] = uw.function.evaluate(h_fn * y, mesh.X.coords)
mesh.deform_mesh(new_coords)
```

### 6. Conditional Logic Based on Position

**Before:**
```python
mask = (
    (mesh.data[:, 1] >= mesh.data[:, 1].min() + offset)
    & (mesh.data[:, 1] <= mesh.data[:, 1].max() - offset)
)
temperature.array[mask, 0, 0] = hot_value
```

**After:**
```python
mask = (
    (mesh.X.coords[:, 1] >= mesh.X.coords[:, 1].min() + offset)
    & (mesh.X.coords[:, 1] <= mesh.X.coords[:, 1].max() - offset)
)
temperature.array[mask, 0, 0] = hot_value
```

### 7. KDTree Construction

**Before:**
```python
index = uw.kdtree.KDTree(mesh.data)
coords = mesh.data + 0.5 * elsize * np.random.random(mesh.data.shape)
```

**After:**
```python
index = uw.kdtree.KDTree(mesh.X.coords)
coords = mesh.X.coords + 0.5 * elsize * np.random.random(mesh.X.coords.shape)
```

### 8. Working with Units

**Before:**
```python
coords = mesh.data  # Raw numpy array or UnitAwareArray
units = mesh.units
print(f"Mesh scale: {mesh.data.max()} {mesh.units}")
```

**After:**
```python
coords = mesh.X.coords  # Consistent with mesh.X.units
units = mesh.X.units
print(f"Mesh scale: {mesh.X.coords.max()} {mesh.X.units}")
```

**Why it's better**: Everything coordinate-related accessed through `mesh.X`.

### 9. Visualization Setup

**Before:**
```python
pvmesh.point_data["T"] = uw.function.evaluate(temperature.sym, mesh.data)
pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.data)
```

**After:**
```python
pvmesh.point_data["T"] = uw.function.evaluate(temperature.sym, mesh.X.coords)
pvmesh.point_data["rho"] = uw.function.evaluate(density, mesh.X.coords)
```

### 10. Mixed Symbolic and Data Operations

**Before:**
```python
# Symbolic gradient
dT_dx = temperature.sym.diff(mesh.X[0])

# Data-based initialization
temperature.array[:, 0, 0] = 300 + 100 * mesh.data[:, 0]
```

**After (no change to symbolic, update data access):**
```python
# Symbolic gradient (unchanged)
dT_dx = temperature.sym.diff(mesh.X[0])

# Data-based initialization (updated)
temperature.array[:, 0, 0] = 300 + 100 * mesh.X.coords[:, 0]
```

## File-by-File Migration Strategy

### Tutorial Files (High Priority)

#### `1-Meshes.ipynb`
**Lines to update**: Display cells showing `mesh.data`

**Before:**
```python
mesh.data
```

**After:**
```python
# Show both new interface and legacy
mesh.X.coords  # Recommended: access via coordinate system
mesh.points    # Also good: explicit property name
# mesh.data    # Deprecated: kept for backward compatibility
```

#### `12-Units_System.ipynb`
**Purpose**: Demonstrate coordinate units interface

**Before:**
```python
print(f"Mesh created with {mesh.data.shape[0]} nodes")
coords = mesh.data
```

**After:**
```python
print(f"Mesh created with {mesh.X.coords.shape[0]} nodes")
coords = mesh.X.coords  # Coordinates with units
units = mesh.X.units     # Coordinate units
```

### Test Files (High Priority)

#### `test_0620_mesh_units_interface.py`
**Purpose**: Test mesh coordinate units interface

**Update strategy**: Show both old and new patterns, emphasize new interface

**Before:**
```python
def test_mesh_data_units(self):
    """Test that mesh.data returns unit-aware coordinates."""
    data = mesh.data
    assert hasattr(data, '_pint_qty')
```

**After:**
```python
def test_mesh_X_coords_units(self):
    """Test that mesh.X.coords returns unit-aware coordinates."""
    coords = mesh.X.coords
    assert hasattr(coords, '_pint_qty')
    assert mesh.X.units is not None

    # Backward compatibility
    assert np.allclose(mesh.X.coords, mesh.data)
```

#### `test_0730_variable_units_integration.py`
**Purpose**: Variable initialization with units

**Before:**
```python
temperature.array[:, 0, 0] = 1000 + 500 * self.mesh.data[:, 0]
```

**After:**
```python
temperature.array[:, 0, 0] = 1000 + 500 * self.mesh.X.coords[:, 0]
```

### Example Files (Medium Priority)

#### Heat transfer examples
**Pattern**: Finding mesh bounds for sampling

**Before:**
```python
sample_y = np.linspace(
    mesh.data[:, 1].min(), mesh.data[:, 1].max(), num_samples
)
```

**After:**
```python
sample_y = np.linspace(
    mesh.X.coords[:, 1].min(), mesh.X.coords[:, 1].max(), num_samples
)
```

#### Mesh deformation examples
**Pattern**: Copying and modifying coordinates

**Before:**
```python
new_coords = mesh.data.copy()
new_coords[:, 1] = uw.function.evaluate(y - dy, mesh.data).squeeze()
```

**After:**
```python
new_coords = mesh.X.coords.copy()
new_coords[:, 1] = uw.function.evaluate(y - dy, mesh.X.coords).squeeze()
```

## Testing Your Migration

### 1. Verify Backward Compatibility

After making changes, verify old patterns still work:

```python
import numpy as np

# Both should return identical data
coords_old = mesh.data
coords_new = mesh.X.coords
assert np.allclose(coords_old, coords_new)

# Symbolic access should work
x, y = mesh.X
assert mesh.X[0] is x
```

### 2. Verify New Interface

Test that new properties exist and work correctly:

```python
# Test new properties
assert hasattr(mesh.X, 'coords')
assert hasattr(mesh.X, 'units')

# Test they return expected types
assert isinstance(mesh.X.coords, np.ndarray) or hasattr(mesh.X.coords, '_pint_qty')
assert mesh.X.units == mesh.units

# Test they're consistent with legacy
assert np.allclose(mesh.X.coords, mesh.points)
```

### 3. Run Relevant Tests

After migration, run tests that use the modified code:

```bash
# Run all tests
pixi run pytest tests/

# Run specific test file
pixi run pytest tests/test_0620_mesh_units_interface.py -v

# Run tests matching pattern
pixi run pytest tests/ -k "coordinate" -v
```

## Common Pitfalls

### Pitfall 1: Confusing Symbolic and Data Access

❌ **Wrong**:
```python
# Trying to use mesh.X for data (old pattern)
coords = mesh.X  # This is now CoordinateSystem, not data!
```

✅ **Right**:
```python
# Symbolic: mesh.X[i] or unpacking
x, y = mesh.X
expr = mesh.X[0]**2 + mesh.X[1]**2

# Data: mesh.X.coords
coords = mesh.X.coords
```

### Pitfall 2: Mixing mesh.data and mesh.X.coords

⚠️ **Inconsistent** (but works):
```python
x_min = mesh.data[:, 0].min()
x_max = mesh.X.coords[:, 0].max()
```

✅ **Better** (consistent):
```python
x_min = mesh.X.coords[:, 0].min()
x_max = mesh.X.coords[:, 0].max()
```

### Pitfall 3: Not Understanding Units

❌ **Wrong assumption**:
```python
# Assuming mesh.X.coords always returns plain numpy array
coords = mesh.X.coords
coords[0, 0] = 1.0  # Might fail if UnitAwareArray!
```

✅ **Right**:
```python
# Handle both cases
coords = mesh.X.coords
if hasattr(coords, '_pint_qty'):
    # It's a UnitAwareArray, work with units
    coords_magnitude = coords._pint_qty.magnitude
else:
    # It's a plain numpy array
    coords_magnitude = coords
```

### Pitfall 4: Internal Code Using mesh._points

⚠️ **Internal use only**:
```python
# mesh._points is internal PETSc data (model coordinates)
# Don't use in user-facing code!
raw_coords = mesh._points  # Internal only
```

✅ **Use public interface**:
```python
# Use mesh.X.coords (physical coordinates with scaling)
coords = mesh.X.coords  # Public interface
```

## Automated Migration Script

For bulk updates, you can use this regex-based search/replace:

```bash
#!/bin/bash
# migrate_coordinates.sh

# Find all Python files
find . -name "*.py" -type f | while read file; do
    # Skip if file is in .git or __pycache__
    if [[ $file == *".git"* ]] || [[ $file == *"__pycache__"* ]]; then
        continue
    fi

    # Create backup
    cp "$file" "$file.bak"

    # Perform replacements (most specific first)
    sed -i.tmp 's/mesh\.data\.shape/mesh.X.coords.shape/g' "$file"
    sed -i.tmp 's/mesh\.data\.copy()/mesh.X.coords.copy()/g' "$file"
    sed -i.tmp 's/mesh\.data\.min()/mesh.X.coords.min()/g' "$file"
    sed -i.tmp 's/mesh\.data\.max()/mesh.X.coords.max()/g' "$file"
    sed -i.tmp 's/mesh\.data\[/mesh.X.coords[/g' "$file"
    sed -i.tmp 's/mesh\.data,/mesh.X.coords,/g' "$file"
    sed -i.tmp 's/mesh\.data)/mesh.X.coords)/g' "$file"
    sed -i.tmp 's/(mesh\.data/(mesh.X.coords/g' "$file"

    # Clean up temp files
    rm "$file.tmp"

    echo "Processed: $file"
done
```

**Warning**: This script is aggressive! Review changes carefully before committing.

## Gradual vs. Bulk Migration

### Gradual Approach (Recommended)

**Pros**:
- Lower risk of breaking changes
- Can test incrementally
- Learn best practices as you go

**Cons**:
- Takes longer
- Codebase has mixed patterns temporarily

**Strategy**:
1. Start with tutorials (high visibility)
2. Update tests as you work on related features
3. Update examples opportunistically
4. Keep `mesh.data` as deprecated alias

### Bulk Approach (Not Recommended)

**Pros**:
- Consistent codebase quickly
- One-time effort

**Cons**:
- High risk of breaking something
- Harder to isolate issues
- Requires extensive testing

**Only use if**:
- You have comprehensive test coverage
- You can dedicate time to fixing issues
- You're doing it in a feature branch

## Communication Strategy

### For Users

**Documentation Update**:
```markdown
## Coordinate Access

The recommended way to access mesh coordinates is via the `mesh.X` coordinate system object:

```python
# Coordinate data
coords = mesh.X.coords  # Recommended
coords = mesh.points    # Also good

# Coordinate units
units = mesh.X.units

# Symbolic coordinates (for expressions)
x, y = mesh.X
expr = mesh.X[0]**2 + mesh.X[1]**2
```

**Note**: `mesh.data` is deprecated but still works for backward compatibility.
```

### For Contributors

**Contribution Guidelines Update**:
```markdown
## Coordinate Access Patterns

When accessing mesh coordinates:

✅ **Do**:
- Use `mesh.X.coords` for coordinate data
- Use `mesh.X.units` for coordinate units
- Use `mesh.X[i]` for symbolic coordinates

❌ **Don't**:
- Use `mesh.data` in new code (deprecated)
- Use `mesh._points` in user-facing code (internal only)
```

## Summary

The migration from `mesh.data` to `mesh.X.coords` provides:

1. **Clarity**: Explicit that you're accessing coordinate data
2. **Consistency**: Matches the pattern for variables (`.sym`, `.array`, `.units`)
3. **Correctness**: Reflects that meshes are coordinate systems with geometric structure
4. **Extensibility**: Natural place for future coordinate-related properties (e.g., `mesh.X.metric`)

The migration is **low risk** because:
- `mesh.data` still works (it's an alias for `mesh.points`)
- `mesh.X[i]` and unpacking work exactly as before
- All changes are additive (no breaking changes)

Take your time, update code opportunistically, and focus on high-visibility areas first (tutorials, documentation, test examples).

## Related Documents

- `COORDINATE_ACCESS_AUDIT.md` - Complete codebase audit
- `COORDINATE_INTERFACE_DESIGN.md` - Design rationale
- `UNITS_SYSTEM_DESIGN_PRINCIPLES.md` - Units system context
