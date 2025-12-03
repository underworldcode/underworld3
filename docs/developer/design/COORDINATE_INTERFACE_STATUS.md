# Coordinate Interface Implementation - Status Report

**Date**: 2025-01-11
**Status**: ✅ **Phase 1 & 2 COMPLETE** - Implementation and documentation done

## What Was Accomplished

### ✅ Phase 1: Implementation (COMPLETE)

**Files Modified**:

1. **`src/underworld3/coordinates.py`** (lines 346-385)
   - Added `__getitem__(self, idx)` method for `mesh.X[0]` support
   - Added `__iter__(self)` method for `x, y = mesh.X` unpacking
   - Added `__len__(self)` method for `len(mesh.X)` support
   - Added `.coords` property (returns `self.mesh.points`)
   - Added `.units` property (returns `self.mesh.units`)
   - These changes enable CoordinateSystem to act as both symbolic matrix and data container

2. **`src/underworld3/discretisation/discretisation_mesh.py`** (lines 1241-1258)
   - Changed `mesh.X` property to return CoordinateSystem object instead of symbolic matrix
   - Added comprehensive docstring explaining new interface
   - Maintains complete backward compatibility with existing code

### ✅ Phase 2: Documentation (COMPLETE)

**Documents Created**:

1. **`planning/COORDINATE_ACCESS_AUDIT.md`**
   - Complete audit of 65 files using `mesh.data`
   - Categorization by file type (tests, examples, tutorials, source)
   - Pattern analysis showing usage frequencies
   - Priority rankings for migration (High/Medium/Low)
   - Testing strategy for verifying backward compatibility

2. **`planning/COORDINATE_MIGRATION_GUIDE.md`**
   - Practical migration examples for 10 common patterns
   - File-by-file migration strategy with before/after code
   - Testing procedures to verify migrations
   - Common pitfalls and how to avoid them
   - Automated migration script template
   - Communication strategy for users and contributors

3. **`planning/COORDINATE_INTERFACE_STATUS.md`** (this document)
   - Overall status summary
   - What's complete, what's next
   - Design decisions and rationale

### Design Decisions Made

1. **mesh.X Returns CoordinateSystem Object**
   - **Why**: Provides both symbolic and data access through single interface
   - **Backward compatible**: `mesh.X[0]` and `x, y = mesh.X` still work via `__getitem__` and `__iter__`
   - **Future-proof**: Natural place for curvilinear coordinate features (metric, jacobian, etc.)

2. **Honest Asymmetry: mesh.X vs swarm.coords**
   - **Meshes ARE coordinate systems**: Carry geometric structure (metric tensors, orientation)
   - **Swarms are NOT coordinate systems**: Just point collections without geometric structure
   - **Interface reflects reality**: mesh.X for coordinate systems, swarm.coords for point data

3. **Deprecation Strategy: Keep mesh.data as Alias**
   - **Why**: Avoid breaking user code
   - **Approach**: Gradual migration, eventually add deprecation warning
   - **Timeline**: No rush - can live with mixed patterns during transition

## What Still Needs to Be Done

### 🔄 Phase 3: Gradual Codebase Migration (NOT STARTED)

**Priority 1: High-Visibility User-Facing Code**

These files are entry points for new users and should demonstrate best practices:

1. **Tutorials** (12 uses of `mesh.data`):
   - `docs/beginner/tutorials/1-Meshes.ipynb` - Show new mesh.X interface
   - `docs/beginner/tutorials/12-Units_System.ipynb` - Demonstrate coordinate units via mesh.X
   - `docs/beginner/tutorials/13-Dimensional_Thermal_Convection.ipynb`
   - `docs/beginner/tutorials/14-Scaled_Thermal_Convection.ipynb`
   - `docs/beginner/tutorials/10-Particle_Swarms.ipynb`
   - `docs/beginner/tutorials/11-Multi-Material_SolCx.ipynb`

2. **Units Test Suite** (21 uses):
   - `tests/test_0620_mesh_units_interface.py` - Test new mesh.X.coords/.units
   - `tests/test_0630_mesh_units_demonstration.py` - Demonstrate usage patterns
   - `tests/test_0720_coordinate_units_gradients.py` - Coordinate unit conversions
   - `tests/test_0730_variable_units_integration.py` - Variable initialization
   - `tests/test_0803_units_workflow_integration.py` - Integration workflows
   - `tests/test_0803_simple_workflow_demo.py` - Simple demos

**Priority 2: Examples and Basic Tests**

Update opportunistically when working in these areas:

3. **Example Scripts** (25+ uses):
   - Heat transfer examples (17 uses) - Finding bounds, evaluation
   - Porous flow examples (4 uses) - Mesh transformations
   - Various other examples - Update as encountered

4. **Basic Tests** (14 uses):
   - `tests/test_0101_kdtree.py` - KDTree with coordinates
   - `tests/test_0005_IndexSwarmVariable.py` - Material initialization
   - `tests/test_0505_rbf_swarm_mesh.py` - RBF interpolation

**Priority 3: Internal Code**

Update eventually, not urgent:

5. **Visualization Code** (2 uses):
   - `src/underworld3/visualisation/visualisation.py` - Works fine as-is

6. **Solver Internals** (3 uses):
   - `src/underworld3/cython/petsc_generic_snes_solvers.pyx` - Hash calculations

7. **Documentation Examples** (2 uses):
   - `src/underworld3/utilities/nd_array_callback.py` - Docstring examples

### Recommended Migration Approach

**Step 1**: Update tutorials (highest visibility)
```bash
# Work through tutorial notebooks one at a time
# Replace mesh.data with mesh.X.coords
# Test each tutorial after changes
```

**Step 2**: Update units tests (demonstrate new interface)
```bash
# Update test_0620, test_0630, etc.
# Add tests for mesh.X.coords and mesh.X.units
# Verify backward compatibility (mesh.data still works)
```

**Step 3**: Update examples opportunistically
```bash
# As you work on related features, update examples
# No need to do bulk migration - gradual is fine
```

**Step 4**: Eventually deprecate mesh.data
```bash
# After most code migrated, add deprecation warning
# Give users time to update (6-12 months)
# Finally remove in major version bump
```

## Implementation Details

### New Interface Structure

```python
mesh.X                      # CoordinateSystem object
  ├── __getitem__(idx)      # mesh.X[0] → symbolic x-coordinate
  ├── __iter__()            # x, y = mesh.X → unpacking
  ├── __len__()             # len(mesh.X) → coordinate dimension
  ├── .coords               # Coordinate data (mesh.points)
  ├── .units                # Coordinate units (mesh.units)
  └── .X                    # Symbolic matrix (internal)
```

### Backward Compatibility

All existing code patterns continue to work:

```python
# ✅ These all still work (no changes needed)
x, y = mesh.X               # Unpacking
mesh.X[0]                   # Indexing
mesh.data                   # Alias for mesh.points
mesh.points                 # Direct property
mesh.units                  # Direct property

# ✅ New patterns (preferred for new code)
mesh.X.coords               # Via coordinate system
mesh.X.units                # Via coordinate system
```

### Testing Verification

To verify the implementation is working:

```python
import numpy as np
import underworld3 as uw

# Create a test mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(4, 4),
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0)
)

# Test new interface exists
assert hasattr(mesh.X, 'coords')
assert hasattr(mesh.X, 'units')

# Test backward compatibility
x, y = mesh.X  # Should work
assert mesh.X[0] is x  # Should work

# Test consistency
assert np.allclose(mesh.X.coords, mesh.points)
assert np.allclose(mesh.X.coords, mesh.data)
assert mesh.X.units == mesh.units

print("✅ All interface tests passed!")
```

## Why This Design?

### Problem: Inconsistent Coordinate Access

**Before**, coordinate access was inconsistent:
- `mesh.data` - Ambiguous name, unclear what "data" means
- `mesh.points` - Better, but not connected to symbolic coordinates
- `mesh.X[0]` - Symbolic only, no way to get data from same interface
- `mesh.units` - Separate from coordinate access

### Solution: Unified Coordinate System Interface

**Now**, everything coordinate-related through `mesh.X`:
- `mesh.X[0]` - Symbolic x-coordinate function
- `mesh.X.coords` - Coordinate data array
- `mesh.X.units` - Coordinate units
- Follows same pattern as variables: `.sym`, `.array`, `.units`

### Benefits

1. **Consistency**: Matches variable interface pattern
2. **Clarity**: Everything coordinate-related in one place
3. **Extensibility**: Natural place for future features (metric, jacobian)
4. **Mathematical correctness**: Reflects that meshes ARE coordinate systems
5. **Backward compatible**: All existing code continues to work

## Related Design Documents

These documents provide the philosophical foundation for this implementation:

1. **`COORDINATE_INTERFACE_DESIGN.md`**
   - Design decision: mesh.X vs swarm.coords asymmetry
   - Mathematical rationale for treating meshes as coordinate systems
   - Pattern consistency across variables and coordinates
   - Future extensions (curvilinear coordinates, etc.)

2. **`UNITS_SYSTEM_DESIGN_PRINCIPLES.md`**
   - Coordinate units are transformations between unit systems
   - Model units vs user units (both dimensional)
   - Reference quantities as conversion factors
   - Why 0→1 range is not special

3. **`mesh_coordinate_units_design.md`**
   - Original coordinate units design (now superseded)
   - Historical context

## Critical Points for Future Work

### 1. Don't Rush the Migration

- **Gradual is better**: Update code opportunistically, not in bulk
- **Test as you go**: Each change should be tested individually
- **User code works**: No urgency since mesh.data still works
- **Learn patterns**: Understand best practices before mass migration

### 2. Maintain Backward Compatibility

- **Keep mesh.data**: It's an alias, costs nothing to keep
- **Warn before breaking**: Add deprecation warning well before removal
- **Give time**: Users need 6-12 months to update their code
- **Document clearly**: Show migration path in deprecation message

### 3. Focus on User-Facing Code First

- **Tutorials matter most**: Entry point for new users
- **Examples teach patterns**: Users copy example code
- **Tests demonstrate usage**: Developers learn from test patterns
- **Internal code later**: Solver internals less urgent

### 4. Test the Interface

Before migrating large amounts of code:
- Write comprehensive tests for mesh.X.coords and mesh.X.units
- Verify backward compatibility (mesh.data, mesh.X[0] still work)
- Test with units (UnitAwareArray handling)
- Test in parallel (MPI-safe operations)

## Next Session Checklist

When you return to this work:

1. **✅ Review this status document** - Understand what's done
2. **✅ Check related documents** - COORDINATE_ACCESS_AUDIT.md and COORDINATE_MIGRATION_GUIDE.md
3. **Start with tutorials** - Highest priority, highest visibility
4. **Test incrementally** - Don't batch changes, test as you go
5. **Update docs as you learn** - Add patterns to migration guide

## Questions to Consider

Before starting Phase 3:

1. **Do we need mesh.X.units?** Or is mesh.units sufficient?
   - **Answer**: Yes, for consistency. Everything coordinate-related through mesh.X.

2. **Should we add deprecation warning for mesh.data?**
   - **Answer**: Not yet. Wait until most tutorials/examples migrated.

3. **Do we need to support mesh.X.data as alias?**
   - **Answer**: No. Use mesh.X.coords to be clear it's coordinate data.

4. **What about swarm.X?**
   - **Answer**: Intentionally omitted. Swarms are not coordinate systems (see COORDINATE_INTERFACE_DESIGN.md).

5. **Future curvilinear features?**
   - **Answer**: mesh.X.metric, mesh.X.jacobian, mesh.X.basis would naturally fit the interface.

## Summary for Restart

**What we did**:
1. ✅ Enhanced CoordinateSystem class with .coords and .units properties
2. ✅ Made mesh.X return CoordinateSystem object (backward compatible)
3. ✅ Audited entire codebase (65 files with mesh.data)
4. ✅ Documented recommended patterns and migration strategies

**What's next**:
1. ⏭️ Update tutorials to demonstrate new interface (high priority)
2. ⏭️ Update units tests to test new interface (high priority)
3. ⏭️ Update examples and other tests opportunistically (medium priority)
4. ⏭️ Eventually add deprecation warning for mesh.data (low priority)

**No urgent actions needed** - existing code works fine. Migration can be gradual and thoughtful.

## Files Changed in This Session

### Modified Files
1. `src/underworld3/coordinates.py` (lines 346-385)
2. `src/underworld3/discretisation/discretisation_mesh.py` (lines 1241-1258)

### Created Files
1. `planning/COORDINATE_ACCESS_AUDIT.md` (comprehensive audit)
2. `planning/COORDINATE_MIGRATION_GUIDE.md` (practical migration guide)
3. `planning/COORDINATE_INTERFACE_STATUS.md` (this document)

All changes are backward compatible. No breaking changes. Ready for gradual migration.
