# PETSc DM Section Synchronization Fix

**Date**: 2025-10-11
**Status**: ✅ **IMPLEMENTED AND TESTED**
**Issue**: Variables created after solver runs caused "Invalid field number" error

## Problem Summary

### Original Error
```
PetscError: Invalid field number 1; not in [0, 1)
```

**Trigger Scenario**:
1. Create a mesh variable (e.g., `T`)
2. Run a solver (which accesses `T.vec` and calls `createDS()`)
3. Create a NEW variable (e.g., `gradT_y`)
4. Try to access the new variable's vector (`gradT_y.vec`)
5. **Error occurs** in `createSubDM(field_id=1)`

### Root Cause: Section Not Synchronized

**PETSc Architecture**:
- **DM (Distributed Mesh)**: Contains field list and discretization structure (DS)
- **Section**: Separate object tracking field layout (component counts, offsets)
- **Critical Gap**: Section is NOT automatically updated when fields are added to existing DM

**What Happened**:
- DM field list correctly shows 2 fields (`getNumFields() = 2`)
- Section still thinks there's only 1 field
- `createSubDM(field_id=1)` queries the Section
- Section errors: "field 1 not in [0, 1)" (only knows about field 0)

**Original Code Issue** (`_setup_ds()` lines 1196-1224):
```python
self.field_id = self.mesh.dm.getNumFields()
self.mesh.dm.addField(petsc_fe)           # Updates DM field list ✓
field, _ = self.mesh.dm.getField(self.field_id)
field.setName(self.clean_name)
self.mesh.dm.createDS()                    # Creates DS but doesn't rebuild Section! ✗
```

## Solution: DM Rebuild Pattern

### Implementation
Modified `_setup_ds()` in `discretisation_mesh_variables.py` (lines 1196-1251):

```python
def _setup_ds(self):
    options = PETSc.Options()
    name0 = "VAR"
    options.setValue(f"{name0}_petscspace_degree", self.degree)
    options.setValue(f"{name0}_petscdualspace_lagrange_continuity", self.continuous)
    options.setValue(f"{name0}_petscdualspace_lagrange_node_endpoints", False)

    dim = self.mesh.dm.getDimension()
    petsc_fe = PETSc.FE().createDefault(
        dim, self.num_components, self.mesh.isSimplex,
        self.mesh.qdegree, name0 + "_", PETSc.COMM_SELF,
    )

    # Check if this is the first field or if we need to rebuild the DM
    num_existing_fields = self.mesh.dm.getNumFields()

    if num_existing_fields > 0:
        # DM already has fields - need to rebuild to sync Section
        dm_old = self.mesh.dm
        dm_new = dm_old.clone()              # Clone DM
        dm_old.copyFields(dm_new)            # Copy existing fields

        field_id = dm_new.getNumFields()
        dm_new.addField(petsc_fe)            # Add new field
        field, _ = dm_new.getField(field_id)
        field.setName(self.clean_name)

        dm_new.createDS()                    # Create DS (builds fresh Section!)

        # Replace old DM with new one
        dm_old.destroy()
        self.mesh.dm = dm_new
        self.mesh.dm_hierarchy[-1] = dm_new
        self.field_id = field_id
    else:
        # First field - normal fast path
        self.field_id = self.mesh.dm.getNumFields()
        self.mesh.dm.addField(petsc_fe)
        field, _ = self.mesh.dm.getField(self.field_id)
        field.setName(self.clean_name)
        self.mesh.dm.createDS()
    return
```

### Why This Works
1. **Clone the DM**: Get a fresh DM structure
2. **Copy existing fields**: Preserve all previously added fields
3. **Add new field**: Add the new field to the cloned DM
4. **Create DS on new DM**: `createDS()` on fresh DM builds Section from scratch
5. **Replace old DM**: Swap out the old DM entirely

**Key Insight**: `createDS()` on a **fresh DM** builds a Section that knows about ALL fields (copied + new). The old stale Section is discarded with the old DM.

### Pattern Source
Borrowed from proven implementation in `__init__` method (lines 353-383) which successfully handles similar DM reconstruction when `old_gvec is not None`.

## Failed Approaches

### Attempt #1: clearDS() + createDS()
```python
self.mesh.dm.clearDS()  # Clear discretization structure
self.mesh.dm.createDS() # Recreate it
```

**Why it failed**: `clearDS()` only clears the discretization structure. The Section is a separate object that isn't automatically rebuilt by `createDS()` when fields already exist.

**Evidence**: Test still failed with same error after this change.

## Testing Results

### Race Condition Test
**File**: `/tmp/test_race_condition.py`
**Result**: ✅ **PASSED** - No errors

### Full Regression Suite
```bash
pixi run -e default pytest tests/test_06*_regression.py -v
```
**Result**: ✅ **59/59 tests PASSED** (9 harmless warnings)

## Performance Considerations

### Current Implementation
- **First variable**: Fast path (no DM rebuild)
- **Subsequent variables**: DM rebuild (expensive but safe)

### Cost Analysis
- **DM clone**: Moderate cost (copies structure)
- **Field copy**: Low cost (just field descriptors)
- **DS creation**: Moderate cost (rebuilds Section)
- **DM destroy**: Low cost (cleanup)

### Optimization Opportunities (Future Work)
Research PETSc Section API for more efficient solution:
- Can we explicitly rebuild just the Section?
- Is there a PETSc API to add fields to Section directly?
- Can we avoid full DM rebuild for subsequent variables?

**Decision**: Use safe approach now, optimize later if performance becomes an issue.

## Related Code Locations

- **Fixed method**: `discretisation_mesh_variables.py:1196-1251` (`_setup_ds()`)
- **Pattern source**: `discretisation_mesh_variables.py:353-383` (`__init__` method)
- **Error location**: `discretisation_mesh_variables.py:1228` (`_set_vec` method)
- **Field ID assignment**:
  - Rebuild path: line 1242
  - Fast path: line 1249

## Key Learnings

### PETSc Architecture Insights
1. **DM and Section are separate**: Field list updates don't automatically sync Section
2. **Section tracks layout**: Component counts, offsets, field numbering
3. **createDS() behavior varies**: Works differently on fresh DM vs existing DM with fields
4. **Clone pattern is safe**: Follow existing patterns for complex PETSc operations

### Design Principles Applied
1. **Conservative approach wins**: Expensive but safe > clever but broken
2. **Borrow proven patterns**: Don't reinvent - reuse working code patterns
3. **Test thoroughly**: Both unit tests and regression suite
4. **Document clearly**: Explain why unusual patterns are needed

## Future Work

### TODO: Optimize DM Rebuild
**Priority**: Low (only if performance becomes an issue)

**Research Questions**:
1. Does PETSc provide Section-level field addition API?
2. Can Section be rebuilt without full DM clone?
3. What's the actual performance impact of DM rebuild?
4. Are there PETSc best practices for dynamic field addition?

**Approach**:
1. Profile actual DM rebuild cost in typical workflows
2. Review PETSc documentation for Section manipulation
3. Contact PETSc community if needed
4. Implement optimization only if proven beneficial

### Potential Regression Test Addition
Consider adding `/tmp/test_race_condition.py` to test suite as:
- `tests/test_1300_dm_section_sync.py`
- Ensures this race condition doesn't reoccur
- Tests variable creation after solver execution

## References

### Investigation Documents (in /tmp)
- `FIX_COMPLETE_SUMMARY.md` - Comprehensive technical summary
- `INVESTIGATION_SUMMARY.md` - Root cause analysis with solution
- `FIX_ANALYSIS.md` - Analysis of failed clearDS() approach
- `test_race_condition.py` - Regression test script

### Related PETSc Concepts
- DM (Distributed Mesh): PETSc's mesh abstraction
- Section: Field layout and numbering
- DS (Discretization Structure): Finite element spaces
- Field: Individual variable in multi-field DM
