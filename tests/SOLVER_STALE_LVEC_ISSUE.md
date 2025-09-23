# Solver _stale_lvec Flag Issue - Analysis and Fix

## Problem Summary
PETSc interpolation (`rbf=False`) returns ~1e-19 instead of correct values after solver.solve(), while RBF interpolation (`rbf=True`) works correctly.

## Root Cause
All solvers directly assign to `variable.vec.array[:]` within `mesh.access()` context, but:
1. Direct `vec.array` assignment bypasses the callback system
2. The new `mesh.access()` dummy context manager doesn't set `mesh._stale_lvec = True` on exit
3. Therefore `mesh.lvec` never gets updated with solver results
4. PETSc interpolation uses stale (zero) `mesh.lvec` values

## Affected Solver Classes and Locations
File: `/Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/src/underworld3/cython/petsc_generic_snes_solvers.pyx`

### 1. SNES_Scalar (Poisson, AdvDiff, etc.)
**Line 1026:**
```python
with self.mesh.access(self.u,):
    self.dm.globalToLocal(gvec, lvec)
    ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
    self.u.vec.array[:] = lvec.array[:]  # ← Needs flag update
```

### 2. SNES_Vector  
**Line 1712:**
```python
with self.mesh.access(self.u):
    self.dm.globalToLocal(gvec, lvec)
    # ...
    ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
    self.u.vec.array[:] = lvec.array[:]  # ← Needs flag update
```

### 3. SNES_Stokes_SaddlePt (Stokes solver)
**Lines 2924-2926:**
```python
with self.mesh.access(self.Unknowns.p, self.Unknowns.u):
     for name, var in self.fields.items():
         if name=='velocity':
             var.vec.array[:] = clvec.getSubVector(velocity_is).array[:]  # ← Needs flag update
         elif name=='pressure':
             var.vec.array[:] = clvec.getSubVector(pressure_is).array[:]  # ← Needs flag update
```

## Recommended Fix

Add `self.mesh._stale_lvec = True` immediately after the `vec.array` assignments in each solver.

### Example for SNES_Scalar (line 1026):
```python
with self.mesh.access(self.u,):
    self.dm.globalToLocal(gvec, lvec)
    ierr = DMPlexSNESComputeBoundaryFEM(dm.dm, <void*>clvec.vec, NULL); CHKERRQ(ierr)
    self.u.vec.array[:] = lvec.array[:]
    self.mesh._stale_lvec = True  # ← ADD THIS LINE
```

### Example for SNES_Stokes_SaddlePt (lines 2924-2926):
```python
with self.mesh.access(self.Unknowns.p, self.Unknowns.u):
     for name, var in self.fields.items():
         if name=='velocity':
             var.vec.array[:] = clvec.getSubVector(velocity_is).array[:]
         elif name=='pressure':
             var.vec.array[:] = clvec.getSubVector(pressure_is).array[:]
     self.mesh._stale_lvec = True  # ← ADD THIS LINE (after loop)
```

## Why This Fix Works

1. **Minimal Change**: Only adds one line per solver, preserving solver stability
2. **Correct Timing**: Flag is set immediately after variable update
3. **Proper Sync**: Next `mesh.update_lvec()` call will rebuild lvec from current `var.vec` values
4. **No Performance Impact**: Flag is a simple boolean assignment

## Alternative Considered (Not Recommended)

Updating `mesh.access()` context manager to always set the flag would affect all code paths and could have unintended side effects. The targeted fix in solvers is safer.

## Testing

After fix, verify with:
```python
# Should return correct values, not ~1e-19
result = uw.function.evaluate(solved_variable.sym[0], points, rbf=False)
```

## Files to Modify

1. `/Users/lmoresi/+Underworld/underworld-pixi-2/underworld3/src/underworld3/cython/petsc_generic_snes_solvers.pyx`
   - Line ~1027 (after SNES_Scalar assignment)
   - Line ~1713 (after SNES_Vector assignment)  
   - Line ~2927 (after SNES_Stokes_SaddlePt loop)