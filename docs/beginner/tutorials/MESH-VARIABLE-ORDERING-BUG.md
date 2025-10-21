# ~~CRITICAL BUG~~: Mesh Variable Ordering ~~Corrupts~~ Gradient Computations - FIXED

## Executive Summary

**Creating new MeshVariables on a mesh AFTER calling solve() previously corrupted mesh state and produced incorrect results in subsequent projection solvers.**

- **Status**: ✅ **FIXED** (2025-10-14)
- **Impact**: HIGH - Affected all gradient computations in notebooks and user code
- **Fix**: Properly invalidate and restore variable vectors when rebuilding DM after adding new fields

## The Fix

**Root Cause**: When a new MeshVariable was added to a mesh that already had variables (triggering a DM rebuild), the existing variables' PETSc vectors (`_lvec`, `_gvec`) remained pointing to the OLD DM structure. Subsequent operations used these stale vectors, producing incorrect results.

**Solution** (`discretisation_mesh_variables.py` lines 1220-1254):
1. **Save data** from all existing variables before destroying their vectors
2. **Invalidate vectors** by setting `_lvec = None` and `_gvec = None`
3. **Rebuild DM** with the new field
4. **Restore data** by recreating vectors from the NEW DM and copying back the saved data

**Test**: `tests/test_0813_mesh_variable_ordering_regression.py`
- Both tests (variable created before/after solve) now pass ✓
- Produces correct gradient: 2.600 K/m (was incorrectly 6.09 K/m)

## The Problem (Historical Documentation)

### Observed Behavior

When computing gradients after solving a Poisson equation:

```python
# THIS PRODUCES WRONG RESULTS (Bug)
mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable('T', mesh, 1, degree=2, units='kelvin')

# Solve Poisson
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.solve()  # Mesh state is now "locked"

# Create gradient variable AFTER solving
gradT = uw.discretisation.MeshVariable('gradT', mesh, 1, degree=1)  # BUG!

# Compute gradient
proj = uw.systems.Projection(mesh, gradT, degree=1)
proj.uw_function = T.diff(y)
proj.solve()

# Result: WRONG! Expected 2.6 K/m, Got 6.09 K/m
```

### The Workaround

```python
# THIS PRODUCES CORRECT RESULTS (Workaround)
mesh = uw.meshing.StructuredQuadBox(...)
T = uw.discretisation.MeshVariable('T', mesh, 1, degree=2, units='kelvin')

# Create gradient variable BEFORE solving
gradT = uw.discretisation.MeshVariable('gradT', mesh, 1, degree=1)  # ✓

# Solve Poisson
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.solve()

# Compute gradient
proj = uw.systems.Projection(mesh, gradT, degree=1)
proj.uw_function = T.diff(y)
proj.solve()

# Result: CORRECT! Got 2.6 K/m as expected
```

## Evidence

### Regression Test

`tests/test_0813_mesh_variable_ordering_regression.py` documents this bug:
- Test 1 (variable after solve): **XFAIL** - Expected gradient 2.6, got 6.09
- Test 2 (variable before solve): **PASS** - Got correct gradient 2.6

Run with: `pixi run -e default pytest tests/test_0813_mesh_variable_ordering_regression.py -v`

### Affected Code

1. **Notebook 13** (`docs/beginner/tutorials/13-Scaling-Physical-Problems.ipynb`)
   - Cell 10 and Cell 15 create gradient variables after solving
   - Produces wrong gradient values

2. **All gradient computation workflows** where variables are created after solving

## Root Cause (Hypothesis)

Related to PETSc DM (Distributed Mesh) initialization:

1. When `solve()` is called, PETSc builds internal DM structures for the mesh
2. The mesh enters a "finalized" state with specific field configurations
3. Adding new MeshVariables after this point corrupts the DM state
4. Subsequent solvers use the corrupted DM, producing wrong results

This is similar to the `_dm_initialized` flag issue we fixed previously, but adding variables after solve still causes problems.

## Impact Assessment

### HIGH PRIORITY - Affects Core Functionality

- **Notebooks**: Notebook 13 and any other notebooks computing gradients after solving
- **User Code**: Any workflow that creates variables after solving will silently produce wrong results
- **Tests**: Test suite may not catch this if tests follow the workaround pattern

### Silent Failure

This is particularly dangerous because:
- No error is raised
- Results "look reasonable" but are wrong by 2-3x
- Easy to miss in complex simulations

## Required Actions

### Immediate (Workaround)

1. ✅ **Document in regression test** (`test_0813_mesh_variable_ordering_regression.py`)
2. **Update Notebook 13** to create gradient variables before solving
3. **Document in CLAUDE.md** for AI assistant context
4. **Add warning to documentation** about variable creation ordering

### Long-term (Fix)

1. **Investigate DM state management** in `discretisation_mesh.py`
2. **Fix `_dm_initialized` flag** to properly handle adding variables after solve
3. **Add defensive checks** that warn or error when variables are added to finalized meshes
4. **Update all affected notebooks** once fix is validated

## Temporary Coding Guidelines

**CRITICAL RULE**: Always create ALL mesh variables BEFORE calling any solve() method.

```python
# CORRECT ORDER:
# 1. Create mesh
mesh = uw.meshing.StructuredQuadBox(...)

# 2. Create ALL variables (solution AND auxiliary)
T = uw.discretisation.MeshVariable('T', mesh, ...)
gradT = uw.discretisation.MeshVariable('gradT', mesh, ...)  # Create early!
flux = uw.discretisation.MeshVariable('flux', mesh, ...)     # Create early!

# 3. Set up and solve
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.solve()

# 4. Use auxiliary variables
proj = uw.systems.Projection(mesh, gradT, ...)
proj.solve()
```

## Related Issues

- Previous DM initialization bug (fixed with `_dm_initialized` flag)
- Variable availability issues with solvers
- Access context manager removal complications

## Files

- **Regression test**: `tests/test_0813_mesh_variable_ordering_regression.py`
- **Affected notebook**: `docs/beginner/tutorials/13-Scaling-Physical-Problems.ipynb`
- **Core issue**: `src/underworld3/discretisation/discretisation_mesh.py` (DM management)
- **This document**: `docs/beginner/tutorials/MESH-VARIABLE-ORDERING-BUG.md`


---

# Documentation Cleanup Summary (2025-10-14)

## Files Removed (Investigation/Debug Files)

The following files were investigation documents created while tracking down the variable ordering bug. They are now obsolete since the root cause was identified and fixed:

1. **GRADIENT-PROJECTION-ISSUE.md** - Obsolete investigation into gradient projection issues
   - Root cause was actually the variable ordering bug, not projection systems
   
2. **13-Scaling-Physical-Problems-FIXED.md** - Temporary fix instructions
   - Fixed cells from when we thought it was a BC or coordinate issue
   - Actual fix was the DM state corruption bug
   
3. **NOTEBOOK-13-FIX-SUMMARY.md** - BC unit conversion fix summary
   - BC unit handling is now well-documented in test_0812_poisson_with_units.py
   
4. **VECTOR-PROJECTION-NOT-FOR-GRADIENTS.md** - Mixed guidance
   - Some useful info about when to use Vector_Projection vs scalar Projection
   - However, the "wrong results" were actually due to the variable ordering bug

## Files Kept

1. **MESH-VARIABLE-ORDERING-BUG.md** - Main bug documentation (KEEP)
   - Documents the DM state corruption bug and fix
   - Historical record with "FIXED" status
   - Reference for understanding the fix in discretisation_mesh_variables.py

## Key Information Preserved

All useful information has been consolidated into:
- `MESH-VARIABLE-ORDERING-BUG.md` - Bug history and fix
- `CLAUDE.md` - "NO BATMAN" anti-pattern documentation
- `tests/test_0813_mesh_variable_ordering_regression.py` - Regression test
- `tests/test_0812_poisson_with_units.py` - BC unit handling tests

## Cleanup Date
2025-10-14 - After fixing the DM state corruption bug
