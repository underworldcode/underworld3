# DMInterpolation Caching Implementation Plan

**Status**: Ready to implement
**Risk**: Medium (Cython C pointer management)
**Expected Benefit**: 4.9× speedup for repeated evaluations

## Implementation Strategy

### Phase 1: Minimal Caching (SAFE)

**Approach**: Cache at Python level, minimal Cython changes

Instead of caching the C struct directly (risky), cache the expensive computation results:
- Cache: Cell hints + setup parameters
- Reuse: Skip get_closest_cells() call
- Rebuild: DMInterpolationInfo each time, but with cached cells

**Expected speedup**: ~1.1× (minimal, but SAFE)

**Why this first**:
- No C pointer management
- Easy to test and validate
- Low risk of memory bugs
- Proves the caching framework works

### Phase 2: Structure Caching (OPTIMAL)

**Approach**: Cache the full DMInterpolationInfo structure

**Key challenge**: DMInterpolationInfo is a C struct that must stay alive

**Solution**:
1. Wrap DMInterpolationInfo in a Python-managed object
2. Create Cython extension class to safely hold the pointer
3. Implement proper cleanup on cache invalidation

**Expected speedup**: ~4.9× (full potential)

## Phase 1 Implementation (THIS SESSION)

### 1. Initialize Cache on Mesh

```python
# In discretisation_mesh.py __init__ (around line 630):
from underworld3.function.dminterpolation_cache import DMInterpolationCache
self._dminterpolation_cache = DMInterpolationCache(self, name=self.name)
```

### 2. Modify _function.pyx: Check Cache Before get_closest_cells

```cython
# In interpolate_vars_on_mesh (line ~650):

# BEFORE (current):
cdef np.ndarray cells = mesh.get_closest_cells(coords)

# AFTER (with simple caching):
# Try cache first
cached_cells = mesh._dminterpolation_cache.get_cells(coords, dofcount)
cdef np.ndarray cells
if cached_cells is not None:
    cells = cached_cells  # Cache hit!
else:
    cells = mesh.get_closest_cells(coords)  # Cache miss - compute
    mesh._dminterpolation_cache.store_cells(coords, dofcount, cells.copy())
```

### 3. Add cells caching methods to DMInterpolationCache

```python
class DMInterpolationCache:
    def get_cells(self, coords, dofcount):
        """Get cached cell hints or None."""
        coords_hash = self._hash_coords(coords)
        key = (coords_hash, dofcount)

        if key in self._cache:
            self._stats['hits'] += 1
            return self._cache[key]['cells']  # numpy array
        else:
            self._stats['misses'] += 1
            return None

    def store_cells(self, coords, dofcount, cells):
        """Store cell hints."""
        coords_hash = self._hash_coords(coords)
        key = (coords_hash, dofcount)
        self._cache[key] = {'cells': cells.copy()}
```

### 4. Add Topology Version Tracking

```python
# In discretisation_mesh.py:
def __init__(self, ...):
    # ...
    self._topology_version = 0
    self._dminterpolation_cache = DMInterpolationCache(self)

def _rebuild_dm_after_variable_add(self):
    # ... existing rebuild code ...
    self._topology_version += 1  # Trigger cache invalidation
    self._dminterpolation_cache.invalidate_all("DM rebuilt")
```

### 5. Test with Diagnostic

```bash
pixi run -e default python examples/diagnose_evaluate_simple.py
```

**Expected result**:
- Time: ~13.5ms → ~13.4ms (minimal improvement)
- But cache hits should be 90%+
- Proves caching framework works

## Phase 2 Implementation (NEXT SESSION - After Phase 1 Validated)

### Challenge: Safe C Pointer Management

**Problem**: DMInterpolationInfo is created at line 621, destroyed at line 663. To cache it:
1. Don't destroy cached structures
2. Keep coords/cells arrays alive (structure references them)
3. Properly clean up on invalidation

**Solution**: Cython extension class

```cython
# New file: src/underworld3/function/_dminterp_cache.pxd

cdef extern from "petsc.h":
    ctypedef struct DMInterpolationInfo:
        pass

cdef class CachedDMInterpolationInfo:
    """
    Python-managed wrapper for DMInterpolationInfo C struct.

    Ensures proper cleanup via Python reference counting.
    """
    cdef DMInterpolationInfo _ipInfo
    cdef public object coords  # Keep alive
    cdef public object cells   # Keep alive
    cdef public int dofcount
    cdef public bint is_valid

    def __cinit__(self):
        self.is_valid = False

    def __dealloc__(self):
        if self.is_valid:
            DMInterpolationDestroy(&self._ipInfo)
            self.is_valid = False
```

**Usage in _function.pyx**:

```cython
# Check cache
cached_info = mesh._dminterpolation_cache.get_structure(coords, dofcount)

if cached_info is not None:
    # CACHE HIT - reuse structure
    ipInfo = cached_info._ipInfo
    # Skip Create/SetUp, jump to Evaluate
else:
    # CACHE MISS - create new structure
    cached_info = CachedDMInterpolationInfo()
    ierr = DMInterpolationCreate(MPI_COMM_SELF, &cached_info._ipInfo)
    # ... SetDim, SetDof, AddPoints, SetUp ...
    cached_info.coords = coords.copy()  # Keep alive!
    cached_info.cells = cells.copy()
    cached_info.dofcount = dofcount
    cached_info.is_valid = True

    # Store in cache (Python ref keeps it alive)
    mesh._dminterpolation_cache.store_structure(coords, dofcount, cached_info)

    ipInfo = cached_info._ipInfo

# EVALUATE (same for both cache hit and miss)
ierr = DMInterpolationEvaluate_UW(ipInfo, dm.dm, pyfieldvec.vec, outvec.vec)

# DON'T DESTROY if cached (Python __dealloc__ will handle it)
```

## Testing Strategy

### Test 1: Value Changes (cache should HIT, values fresh)

```python
def test_cache_values_update():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    T = uw.discretisation.MeshVariable("T", mesh, 1)
    coords = np.random.random((50, 2))

    # First evaluation
    T.array[...] = 1.0
    result1 = uw.function.evaluate(T.sym, coords, rbf=False)
    assert np.allclose(result1, 1.0)

    # Change values, same coords
    T.array[...] = 2.0
    result2 = uw.function.evaluate(T.sym, coords, rbf=False)

    # Check cache hit
    stats = mesh._dminterpolation_cache.get_stats()
    assert stats['hits'] == 1, "Should reuse cache"

    # Check values updated
    assert np.allclose(result2, 2.0), "Values should be fresh!"
    assert not np.allclose(result1, result2), "Results should differ"
```

### Test 2: Coordinate Changes (cache should MISS)

```python
def test_cache_coords_miss():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    T = uw.discretisation.MeshVariable("T", mesh, 1)

    coords1 = np.random.random((50, 2))
    coords2 = np.random.random((50, 2))  # Different!

    _ = uw.function.evaluate(T.sym, coords1, rbf=False)
    _ = uw.function.evaluate(T.sym, coords2, rbf=False)

    stats = mesh._dminterpolation_cache.get_stats()
    assert stats['misses'] == 2, "Different coords should miss"
    assert stats['hits'] == 0
```

### Test 3: Variable Addition (cache should invalidate or new key)

```python
def test_cache_variable_addition():
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    T = uw.discretisation.MeshVariable("T", mesh, 1)
    coords = np.random.random((50, 2))

    # First evaluation (1 variable, 1 DOF)
    _ = uw.function.evaluate(T.sym, coords, rbf=False)
    stats1 = mesh._dminterpolation_cache.get_stats()

    # Add variable (changes DOF count)
    P = uw.discretisation.MeshVariable("P", mesh, 1)

    # Second evaluation (2 variables, 2 DOFs)
    _ = uw.function.evaluate(T.sym, coords, rbf=False)
    stats2 = mesh._dminterpolation_cache.get_stats()

    # Cache should either:
    # Option 1: Be invalidated (entries cleared)
    # Option 2: New key (coord_hash, new_dofcount)

    # Either way, should NOT reuse old structure
    # (we'll check this by ensuring no crash!)
```

### Test 4: Different Meshes (separate caches)

```python
def test_cache_per_mesh():
    mesh1 = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    mesh2 = uw.meshing.StructuredQuadBox(elementRes=(16, 16))  # Same size, different mesh!

    T1 = uw.discretisation.MeshVariable("T1", mesh1, 1)
    T2 = uw.discretisation.MeshVariable("T2", mesh2, 1)

    coords = np.random.random((50, 2))  # Same coords!

    _ = uw.function.evaluate(T1.sym, coords, rbf=False)
    _ = uw.function.evaluate(T2.sym, coords, rbf=False)

    # Each mesh should have independent cache
    stats1 = mesh1._dminterpolation_cache.get_stats()
    stats2 = mesh2._dminterpolation_cache.get_stats()

    assert stats1['misses'] == 1, "mesh1: first eval should miss"
    assert stats2['misses'] == 1, "mesh2: first eval should miss"
```

## Risk Mitigation

### Phase 1 (Low Risk):
- ✅ Pure Python caching
- ✅ No C pointer management
- ✅ Minimal Cython changes
- ✅ Easy to disable if problems

### Phase 2 (Medium Risk):
- ⚠️ C pointer lifecycle management
- ⚠️ Memory leaks if cleanup fails
- ⚠️ Segfaults if pointer invalidated

**Mitigation**:
1. Implement Phase 1 first, validate thoroughly
2. Use Cython extension class for safety (Python GC manages cleanup)
3. Add extensive logging during development
4. Run under valgrind to detect leaks
5. Keep fallback to Phase 1 if issues arise

## Success Criteria

**Phase 1**:
- ✅ Cache hit rate >90% for typical usage
- ✅ All tests pass
- ✅ No crashes or memory issues
- ✅ Minimal speedup (~1.1×) but proves framework

**Phase 2**:
- ✅ Speedup ~4-5× for cached evaluations
- ✅ Performance matches or beats rbf=True (~4.4ms)
- ✅ All tests pass
- ✅ No memory leaks (valgrind clean)
- ✅ Cache statistics show high hit rate

---

**Decision**: Implement Phase 1 this session, Phase 2 after validation.
