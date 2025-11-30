# DMInterpolation Caching: Quick Implementation Summary

## Files Created ✅

1. `src/underworld3/function/_dminterp_wrapper.pyx` - Cython wrapper
2. `src/underworld3/function/dminterpolation_cache.py` - Cache manager (updated)

## Files to Modify

### 1. Build Configuration

**Location**: Need to add `_dminterp_wrapper.pyx` to build system

Check:
- `setup.py` or
- `meson.build` or
- `pyproject.toml`

Look for where other `.pyx` files like `_function.pyx` are listed.

### 2. Mesh Initialization (`discretisation_mesh.py`)

**Location**: `__init__` method (around line 630)

**Add**:
```python
from underworld3.function.dminterpolation_cache import DMInterpolationCache

# In __init__, after other initialization:
self._topology_version = 0
self._dminterpolation_cache = DMInterpolationCache(self, name=self.name)
self.enable_dminterpolation_cache = True  # User can disable
```

### 3. Mesh DM Rebuild (`discretisation_mesh.py`)

**Location**: Wherever DM is rebuilt (search for `dm.destroy()` or similar)

**Add**:
```python
# After DM rebuild:
self._topology_version += 1
self._dminterpolation_cache.invalidate_all("DM rebuilt")
```

### 4. Main Caching Logic (`_function.pyx`)

**Location**: `interpolate_vars_on_mesh` function, lines ~610-687

**Current**:
```cython
# Lines 619-663: Always create/destroy
cdef DMInterpolationInfo ipInfo
ierr = DMInterpolationCreate(...)
ierr = DMInterpolationSetDim(...)
ierr = DMInterpolationSetDof(...)
ierr = DMInterpolationAddPoints(...)
ierr = DMInterpolationSetUp_UW(...)
ierr = DMInterpolationEvaluate_UW(...)
ierr = DMInterpolationDestroy(...)  # Wasteful!
```

**Replace with**:
```cython
# Import at top of file
from underworld3.function._dminterp_wrapper cimport CachedDMInterpolationInfo

# In interpolate_vars_on_mesh, replace lines 619-663:

# Calculate DOF count (keep existing code lines 624-629)
dofcount = 0
var_start_index = {}
for var in vars:
    var_start_index[var] = dofcount
    dofcount += var.num_components

# TRY CACHE
coords = np.ascontiguousarray(coords)
cached_info = mesh._dminterpolation_cache.get_structure(coords, dofcount)

cdef np.ndarray outarray = np.empty([len(coords), dofcount], dtype=np.double)

if cached_info is not None:
    # CACHE HIT - just evaluate
    mesh.update_lvec()
    cached_info.evaluate(mesh, outarray)
else:
    # CACHE MISS - create and cache
    cached_info = CachedDMInterpolationInfo()
    cdef np.ndarray cells = mesh.get_closest_cells(coords)

    cached_info.create_structure(mesh, coords, cells, dofcount)
    mesh._dminterpolation_cache.store_structure(coords, dofcount, cached_info)

    mesh.update_lvec()
    cached_info.evaluate(mesh, outarray)

# Rest unchanged (lines 665-687)
cdef Vec outvec = PETSc.Vec().createWithArray(outarray, comm=PETSc.COMM_SELF)

varfns_arrays = {}
for varfn in varfns:
    var = varfn.meshvar()
    comp = varfn.component
    var_start = var_start_index[var]
    arr = np.ascontiguousarray(outarray[:, var_start+comp])
    varfns_arrays[varfn] = arr

outvec.destroy()
return varfns_arrays
```

## Build & Test

```bash
# Build
pixi run underworld-build

# Test 1: Basic functionality
python -c "
import underworld3 as uw
import numpy as np

mesh = uw.meshing.StructuredQuadBox(elementRes=(16,16))
T = uw.discretisation.MeshVariable('T', mesh, 1)
T.array[...] = 1.0

coords = np.random.random((50, 2))
result = uw.function.evaluate(T.sym, coords, rbf=False)
print(f'Result shape: {result.shape}')
print(f'Mean value: {result.mean():.2f}')
print(f'✓ Basic evaluation works')
"

# Test 2: Cache hits
python examples/diagnose_evaluate_simple.py
```

## Quick Test Script

Create `test_caching.py`:
```python
import underworld3 as uw
import numpy as np
import time

mesh = uw.meshing.StructuredQuadBox(elementRes=(32, 32))
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
coords = np.random.random((100, 2))

# Warm-up
T.array[...] = np.random.random(T.array.shape)
_ = uw.function.evaluate(T.sym, coords, rbf=False)

# Timed runs
times = []
for i in range(10):
    T.array[...] = np.random.random(T.array.shape)
    t_start = time.time()
    _ = uw.function.evaluate(T.sym, coords, rbf=False)
    times.append(time.time() - t_start)

print(f"Times: {[f'{t*1000:.1f}ms' for t in times]}")
print(f"Average: {np.mean(times)*1000:.1f}ms")
print(f"Expected: ~2.8ms after first call (if caching works)")

# Check cache stats
stats = mesh._dminterpolation_cache.get_stats()
print(f"\nCache stats:")
print(f"  Hits: {stats['hits']}")
print(f"  Misses: {stats['misses']}")
print(f"  Hit rate: {stats['hit_rate']*100:.1f}%")
```

## Expected Output

```
Times: ['13.2ms', '2.7ms', '2.8ms', '2.7ms', '2.8ms', '2.7ms', '2.8ms', '2.7ms', '2.8ms', '2.7ms']
Average: 3.5ms
Expected: ~2.8ms after first call (if caching works)

Cache stats:
  Hits: 9
  Misses: 1
  Hit rate: 90.0%
```

## Disable If Problems

```python
# Disable caching for a specific mesh (Jupyter-friendly)
mesh.enable_dminterpolation_cache = False
```
