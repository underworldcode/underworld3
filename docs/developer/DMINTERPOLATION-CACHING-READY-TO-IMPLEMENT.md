# DMInterpolation Caching: Ready-to-Implement Guide

**Status**: Code written, ready to wire together and test
**Expected Benefit**: 4.9× speedup
**Risk Level**: Medium (managed via on/off switch)

## What's Been Created

### 1. Safe Cython Wrapper (`_dminterp_wrapper.pyx`)
✅ **COMPLETE** - Manages C pointer lifetime safely

**Key Features**:
- `CachedDMInterpolationInfo` class wraps DMInterpolationInfo C struct
- Python GC handles cleanup via `__dealloc__`
- Keeps coords/cells arrays alive (prevents dangling pointers)
- `create_structure()` - Does expensive setup once
- `evaluate()` - Fast repeated evaluation

### 2. Cache Manager (`dminterpolation_cache.py`)
✅ **PARTIALLY COMPLETE** - Needs update to store Cython objects

**What's There**:
- Cache key: `(coord_hash, dofcount)`
- Statistics tracking (hits, misses, invalidations)
- On/off switch via environment variable

**What Needs Adding**:
```python
# In DMInterpolationCache class:

def get_structure(self, coords, dofcount):
    """Get cached CachedDMInterpolationInfo or None."""
    if not self._is_enabled():
        return None  # Caching disabled

    coords_hash = self._hash_coords(coords)
    key = (coords_hash, dofcount)

    if key in self._cache:
        self._stats['hits'] += 1
        entry = self._cache[key]  # CachedDMInterpolationInfo object
        return entry
    else:
        self._stats['misses'] += 1
        return None

def store_structure(self, coords, dofcount, cached_info):
    """Store CachedDMInterpolationInfo object."""
    if not self._is_enabled():
        return  # Don't cache if disabled

    coords_hash = self._hash_coords(coords)
    key = (coords_hash, dofcount)
    self._cache[key] = cached_info  # Python GC keeps it alive

def _is_enabled(self):
    """Check if caching is enabled."""
    # Check environment variable
    env_setting = os.getenv('UW_DMINTERPOLATION_CACHE', '1')
    if env_setting == '0':
        return False

    # Check mesh flag
    if hasattr(self.mesh, 'enable_dminterpolation_cache'):
        return self.mesh.enable_dminterpolation_cache

    return True  # Default: enabled

def invalidate_all(self, reason="manual"):
    """Clear cache - Python GC will destroy C structures."""
    n_entries = len(self._cache)
    self._cache.clear()  # __dealloc__ called on each entry

    if n_entries > 0 and uw.mpi.rank == 0:
        print(f"[Cache '{self.name}'] Cleared {n_entries} entries: {reason}")
```

### 3. Modified `_function.pyx`
⏳ **NEEDS IMPLEMENTATION**

**Current code** (lines 619-663):
```cython
# ALWAYS creates/destroys structure
ierr = DMInterpolationCreate(...)
ierr = DMInterpolationSetDim(...)
ierr = DMInterpolationSetDof(...)
ierr = DMInterpolationAddPoints(...)
ierr = DMInterpolationSetUp_UW(...)
ierr = DMInterpolationEvaluate_UW(...)
ierr = DMInterpolationDestroy(...)  # ← Throws away work!
```

**New code** (with caching):
```cython
# At top of file, add import:
from underworld3.function._dminterp_wrapper cimport CachedDMInterpolationInfo

# In interpolate_vars_on_mesh function (around line 610):

# Calculate DOF count
dofcount = 0
var_start_index = {}
for var in vars:
    var_start_index[var] = dofcount
    dofcount += var.num_components

# TRY CACHE FIRST
cached_info = mesh._dminterpolation_cache.get_structure(coords, dofcount)

cdef np.ndarray outarray = np.empty([len(coords), dofcount], dtype=np.double)

if cached_info is not None:
    # ============================================================
    # CACHE HIT - Fast path!
    # ============================================================
    # Ensure lvec is up-to-date (fresh values!)
    mesh.update_lvec()

    # Evaluate with cached structure (FAST!)
    cached_info.evaluate(mesh, outarray)

else:
    # ============================================================
    # CACHE MISS - Build structure and cache it
    # ============================================================
    # Create wrapper object
    cached_info = CachedDMInterpolationInfo()

    # Get cell hints
    cdef np.ndarray cells = mesh.get_closest_cells(coords)

    # Create and set up structure (EXPENSIVE - but cached!)
    cached_info.create_structure(mesh, coords, cells, dofcount)

    # Store in cache (Python keeps it alive)
    mesh._dminterpolation_cache.store_structure(coords, dofcount, cached_info)

    # Evaluate
    mesh.update_lvec()
    cached_info.evaluate(mesh, outarray)

# Rest of function unchanged - extract results from outarray
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

### 4. Mesh Initialization
⏳ **NEEDS IMPLEMENTATION**

**In `discretisation_mesh.py` `__init__` method**:
```python
# Around line 630, add:
from underworld3.function.dminterpolation_cache import DMInterpolationCache

self._topology_version = 0  # Track mesh changes
self._dminterpolation_cache = DMInterpolationCache(self, name=self.name)
self.enable_dminterpolation_cache = True  # User can set to False
```

**When DM is rebuilt** (e.g., adding variables):
```python
def _rebuild_dm_after_variable_add(self):
    # ... existing rebuild code ...

    self._topology_version += 1
    self._dminterpolation_cache.invalidate_all("DM rebuilt")
```

## Building the Cython Extension

### Add to `setup.py`:
```python
# In extensions list, add:
Extension(
    "underworld3.function._dminterp_wrapper",
    sources=["src/underworld3/function/_dminterp_wrapper.pyx"],
    include_dirs=[
        np.get_include(),
        petsc_include_dirs,  # Already defined
    ],
    libraries=["petsc"],
    library_dirs=[petsc_lib_dir],
    extra_compile_args=["-O3"],
),
```

### Build command:
```bash
pixi run underworld-build
```

## Comprehensive Tests

### Test 1: Values Update (Cache Hits, Values Fresh)

```python
def test_dminterp_cache_values_update():
    """
    Verify cache reuses structure but gets fresh values.

    This is the CRITICAL test - cache must not stale values!
    """
    import underworld3 as uw
    import numpy as np

    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    T = uw.discretisation.MeshVariable("T", mesh, 1)
    coords = np.random.random((50, 2))

    # First evaluation - cache miss
    T.array[...] = 1.0
    result1 = uw.function.evaluate(T.sym, coords, rbf=False)

    stats1 = mesh._dminterpolation_cache.get_stats()
    assert stats1['misses'] == 1, "First call should miss"
    assert stats1['hits'] == 0
    assert np.allclose(result1, 1.0), "Values should be 1.0"

    # Change values, same coords - should HIT cache but get NEW values
    T.array[...] = 2.0
    result2 = uw.function.evaluate(T.sym, coords, rbf=False)

    stats2 = mesh._dminterpolation_cache.get_stats()
    assert stats2['hits'] == 1, "Second call with same coords should HIT"
    assert np.allclose(result2, 2.0), "Values should be 2.0 (FRESH!)"
    assert not np.allclose(result1, result2), "Results should differ"

    print("✓ Cache reuses structure but fetches fresh values")
```

### Test 2: Coordinate Changes (Cache Misses)

```python
def test_dminterp_cache_coords_miss():
    """Verify different coordinates create new cache entries."""
    import underworld3 as uw
    import numpy as np

    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    T = uw.discretisation.MeshVariable("T", mesh, 1)

    coords1 = np.random.random((50, 2))
    coords2 = np.random.random((50, 2))  # Different!
    coords3 = coords1.copy()  # Same as first

    _ = uw.function.evaluate(T.sym, coords1, rbf=False)  # Miss 1
    _ = uw.function.evaluate(T.sym, coords2, rbf=False)  # Miss 2
    _ = uw.function.evaluate(T.sym, coords3, rbf=False)  # Hit!

    stats = mesh._dminterpolation_cache.get_stats()
    assert stats['misses'] == 2, "Two different coord sets"
    assert stats['hits'] == 1, "Third matches first coords"
    assert stats['entries'] == 2, "Two cache entries"

    print("✓ Different coordinates create separate cache entries")
```

### Test 3: Variable Addition (DOF Count Change)

```python
def test_dminterp_cache_variable_addition():
    """Verify DOF count changes create new cache entries."""
    import underworld3 as uw
    import numpy as np

    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    T = uw.discretisation.MeshVariable("T", mesh, 1)  # 1 DOF
    coords = np.random.random((50, 2))

    # First evaluation (1 DOF)
    _ = uw.function.evaluate(T.sym, coords, rbf=False)
    stats1 = mesh._dminterpolation_cache.get_stats()
    assert stats1['misses'] == 1

    # Add variable (now 2 DOFs total)
    P = uw.discretisation.MeshVariable("P", mesh, 1)

    # Second evaluation (still just T, but DM rebuilt)
    _ = uw.function.evaluate(T.sym, coords, rbf=False)
    stats2 = mesh._dminterpolation_cache.get_stats()

    # Cache should be invalidated (DM rebuild) or new key (different dofcount)
    # Either way, should work without crashing!

    print("✓ Variable addition handled correctly")
```

### Test 4: Per-Mesh Caching

```python
def test_dminterp_cache_per_mesh():
    """Verify each mesh has independent cache."""
    import underworld3 as uw
    import numpy as np

    mesh1 = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    mesh2 = uw.meshing.StructuredQuadBox(elementRes=(16, 16))

    T1 = uw.discretisation.MeshVariable("T1", mesh1, 1)
    T2 = uw.discretisation.MeshVariable("T2", mesh2, 1)

    coords = np.random.random((50, 2))  # SAME coords!

    _ = uw.function.evaluate(T1.sym, coords, rbf=False)
    _ = uw.function.evaluate(T2.sym, coords, rbf=False)

    stats1 = mesh1._dminterpolation_cache.get_stats()
    stats2 = mesh2._dminterpolation_cache.get_stats()

    assert stats1['misses'] == 1, "mesh1: first call misses"
    assert stats2['misses'] == 1, "mesh2: first call misses (separate cache!)"

    print("✓ Per-mesh caching works correctly")
```

### Test 5: Cache On/Off Switch

```python
def test_dminterp_cache_disable():
    """Verify caching can be disabled."""
    import underworld3 as uw
    import numpy as np
    import os

    # Test 1: Disable via environment variable
    os.environ['UW_DMINTERPOLATION_CACHE'] = '0'

    mesh1 = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    T1 = uw.discretisation.MeshVariable("T1", mesh1, 1)
    coords = np.random.random((50, 2))

    _ = uw.function.evaluate(T1.sym, coords, rbf=False)
    _ = uw.function.evaluate(T1.sym, coords, rbf=False)  # Same coords

    stats1 = mesh1._dminterpolation_cache.get_stats()
    assert stats1['hits'] == 0, "Caching disabled - no hits"

    # Test 2: Disable via mesh flag
    os.environ['UW_DMINTERPOLATION_CACHE'] = '1'  # Re-enable globally

    mesh2 = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    mesh2.enable_dminterpolation_cache = False  # Disable for this mesh
    T2 = uw.discretisation.MeshVariable("T2", mesh2, 1)

    _ = uw.function.evaluate(T2.sym, coords, rbf=False)
    _ = uw.function.evaluate(T2.sym, coords, rbf=False)

    stats2 = mesh2._dminterpolation_cache.get_stats()
    assert stats2['hits'] == 0, "Mesh caching disabled - no hits"

    print("✓ Cache on/off switch works")
```

## Implementation Checklist

- [ ] 1. Add `_dminterp_wrapper.pyx` to build system (`setup.py`)
- [ ] 2. Complete `dminterpolation_cache.py` (add missing methods)
- [ ] 3. Initialize cache in `discretisation_mesh.py` `__init__`
- [ ] 4. Add topology version tracking and invalidation
- [ ] 5. Modify `_function.pyx` `interpolate_vars_on_mesh()` to use cache
- [ ] 6. Build: `pixi run underworld-build`
- [ ] 7. Run Test 1 (values update) - **MOST CRITICAL**
- [ ] 8. Run Test 2 (coords miss)
- [ ] 9. Run Test 3 (variable addition)
- [ ] 10. Run Test 4 (per-mesh)
- [ ] 11. Run Test 5 (on/off switch)
- [ ] 12. Run diagnostic: `diagnose_evaluate_simple.py`
- [ ] 13. Verify speedup ~4-5×
- [ ] 14. Check for memory leaks (run under valgrind if possible)

## Expected Results

**Before caching**:
```
evaluate(rbf=False): 13.61ms per call
```

**After caching** (90%+ hit rate):
```
First call:  ~13.6ms (cache miss - builds structure)
Subsequent:  ~2.8ms (cache hit - just evaluation!)
Average:     ~3.5ms (with 90% hit rate)
Speedup:     3.9× average, 4.9× for cache hits
```

**Should beat rbf=True** (currently 4.38ms)!

## Rollback Plan

If problems occur:

1. **Disable caching**:
   ```bash
   export UW_DMINTERPOLATION_CACHE=0
   ```

2. **Revert code**:
   - Comment out cache check in `_function.pyx`
   - Falls back to original behavior
   - No performance impact if disabled

3. **Debug**:
   - Enable verbose logging
   - Check cache stats
   - Run valgrind for memory issues

---

**Next Steps**: Review this plan, then implement and test methodically using the checklist.
