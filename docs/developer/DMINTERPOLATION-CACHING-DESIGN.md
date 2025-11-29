# DMInterpolation Caching Design for evaluate() Performance

**Date**: 2025-11-16
**Problem**: `evaluate(rbf=False)` creates/destroys DMInterpolation structures on every call (4,446 Index Sets for 11 calls!)
**Goal**: Design caching system that properly handles lazy evaluation

## Root Cause Analysis

### Current Behavior (SLOW)
```python
# EVERY evaluate() call does this:
DMInterpolationCreate()      # Create structure
DMInterpolationSetUp()        # Build spatial index (404 Index Sets!)
DMInterpolationEvaluate()     # Interpolate values
DMInterpolationDestroy()      # ← THROWS AWAY ALL SETUP WORK!
```

**Performance**: 12.7ms per call (vs 4.3ms for rbf=True)

### Why Previous Caching Failed

**Old approach** (lines 589-607 in `_function.pyx`):
- Cached **interpolated results** (variable values at coordinates)
- Invalidation trigger: "whenever variables might have changed"
- Problem: With lazy evaluation + direct array access, couldn't detect when to invalidate

```python
# Old caching (DISABLED at line 599):
if coord_hash == mesh._evaluation_hash:
    return mesh._evaluation_interpolated_results  # ← Cached RESULTS
```

**Why it failed**:
- Results become stale when variable values change
- Relied on `mesh.access()` to clear cache → doesn't work with direct array access
- Comment (lines 595-597): "This is not captured by a simple coordinate hash"

## Key Insight: Cache Structure, Not Results!

`★ CRITICAL DISTINCTION ★`

**DMInterpolation structure** depends on:
- ✅ Mesh topology (constant during run)
- ✅ Coordinate locations (changes when evaluating at different points)

**DMInterpolation structure** does NOT depend on:
- ❌ Variable values (these are just interpolated, structure stays same!)

Therefore: **Cache the setup, re-evaluate as needed**

## Proposed Design

### Architecture

```python
class DMInterpolationCache:
    """
    Caches DMInterpolation structures (not results!) for fast repeated evaluate() calls.

    Key insight: The expensive part is DMInterpolationSetUp (spatial indexing),
    not DMInterpolationEvaluate (actual interpolation). Structure only depends
    on mesh topology + coordinate locations, NOT on variable values.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self._cache = {}  # {coord_hash: (ipInfo_handle, coords_array, cell_hints)}
        self._mesh_version = mesh._topology_version  # For detecting mesh changes

    def get_or_create(self, coords):
        """
        Get cached DMInterpolation structure or create new one.

        Returns
        -------
        ipInfo_handle : int
            Opaque handle to cached DMInterpolationInfo structure
        coord_hash : int
            Hash of coordinates (for verification)
        """
        # Compute coordinate hash
        coord_hash = self._hash_coords(coords)

        # Check mesh version (topology changes invalidate ALL caches)
        if self._mesh_version != self.mesh._topology_version:
            self.clear_all()
            self._mesh_version = self.mesh._topology_version

        # Return cached structure if available
        if coord_hash in self._cache:
            return self._cache[coord_hash]

        # Create new DMInterpolation structure
        ipInfo_handle = self._create_interpolation_structure(coords)
        self._cache[coord_hash] = ipInfo_handle

        return ipInfo_handle

    def evaluate_with_cached_structure(self, ipInfo_handle, varfns):
        """
        Evaluate variables using cached DMInterpolation structure.

        This is the FAST path - no setup overhead!
        """
        # DMInterpolationEvaluate only - no Create/SetUp/Destroy!
        pass

    def invalidate_coords(self, coords):
        """Remove cached structure for specific coordinates."""
        coord_hash = self._hash_coords(coords)
        if coord_hash in self._cache:
            self._destroy_interpolation_structure(self._cache[coord_hash])
            del self._cache[coord_hash]

    def clear_all(self):
        """Clear all cached structures (e.g., after mesh topology change)."""
        for ipInfo_handle in self._cache.values():
            self._destroy_interpolation_structure(ipInfo_handle)
        self._cache.clear()
```

### Cache Key Strategy

**Hash coordinates** using xxhash (fast, already in codebase):

```python
def _hash_coords(self, coords):
    """
    Fast coordinate hashing for cache lookups.

    Uses xxhash (already used in old system) for speed.
    Handles numpy array properly with contiguous copy.
    """
    import xxhash
    xxh = xxhash.xxh64()
    xxh.update(np.ascontiguousarray(coords))
    return xxh.intdigest()
```

**Why this works**:
- Different coordinate sets → different hash → different cache entry
- Same coordinates → same hash → reuse structure
- Variable values change → hash unchanged → cache HIT! ✓

### Cache Invalidation Triggers

**INVALIDATE when**:
1. **Mesh topology changes** (rare):
   - Remeshing, refinement, coarsening
   - Clear ALL cache entries
   - Detect via `mesh._topology_version` counter

2. **Coordinates change** (common):
   - Evaluating at different point set
   - Only invalidate entry for old coordinates
   - Automatic via hash mismatch

**DO NOT INVALIDATE when**:
- ✅ Variable values change (structure independent of values!)
- ✅ Time stepping (coordinates same, values different)
- ✅ Solver iterations (coordinates same, values changing)

### Integration with Mesh Object

Store cache on mesh (since DMInterpolation is per-mesh):

```python
# In discretisation_mesh.py __init__:
self._dminterpolation_cache = DMInterpolationCache(self)
self._topology_version = 0  # Increment on topology changes
```

**Mesh topology version**:
- Increment when: remeshing, refinement, variable add/remove (DM rebuild)
- Used by cache to detect invalidation need
- Simple integer counter, no complex tracking

### Memory Management

**Lifecycle**:
- **Create**: On first evaluate() with new coordinate set
- **Reuse**: On subsequent evaluate() calls with same coordinates
- **Destroy**:
  - When coordinates no longer needed (manual or LRU eviction)
  - When mesh destroyed (cleanup all structures)
  - When mesh topology changes (rebuild required)

**LRU eviction** (optional, for memory-constrained scenarios):
```python
from collections import OrderedDict

class DMInterpolationCache:
    def __init__(self, mesh, max_entries=10):
        self._cache = OrderedDict()  # LRU tracking
        self._max_entries = max_entries

    def _evict_oldest(self):
        """Remove least-recently-used entry when cache full."""
        if len(self._cache) >= self._max_entries:
            oldest_key, oldest_handle = self._cache.popitem(last=False)
            self._destroy_interpolation_structure(oldest_handle)
```

## Implementation Plan

### Phase 1: Core Caching (Immediate)

1. **Create DMInterpolationCache class** in `_function.pyx`
   - Hash-based coordinate lookup
   - Store ipInfo handles (Cython pointers wrapped safely)
   - Mesh version tracking

2. **Modify petsc_interpolate()** (lines 619-663)
   ```python
   # OLD (current):
   DMInterpolationCreate()
   DMInterpolationSetUp()
   DMInterpolationEvaluate()
   DMInterpolationDestroy()  # ← Every call!

   # NEW (cached):
   ipInfo = mesh._dminterpolation_cache.get_or_create(coords)
   DMInterpolationEvaluate(ipInfo)  # ← Only this on cache hit!
   # No destroy - cache owns lifecycle
   ```

3. **Add mesh topology version**
   - `discretisation_mesh.py`: Add `self._topology_version = 0`
   - Increment when: DM rebuilt (e.g., adding variables)

4. **Test with diagnostic script**
   - Verify Index Set count drops from 4,446 → ~400 (10x reduction)
   - Measure timing: should match or beat rbf=True performance

### Phase 2: Robustness (Follow-up)

5. **LRU eviction** for memory management (optional)
6. **Explicit cache control** API:
   ```python
   mesh.clear_evaluation_cache()  # Manual invalidation
   mesh.warm_evaluation_cache(coords)  # Pre-populate
   ```
7. **Statistics/monitoring**:
   ```python
   mesh.evaluation_cache_stats()  # Hit rate, memory usage
   ```

### Phase 3: Advanced Features (Future)

8. **Approximate coordinate matching** (fuzzy cache hits)
   - Tolerate small coordinate perturbations
   - Use spatial binning instead of exact hash

9. **Parallel considerations**
   - Per-rank caching (current MPI_COMM_SELF model)
   - Or global coordination for shared coordinates

## Expected Performance Improvement

**Before** (current, rbf=False):
- Index Sets created: 4,446 (for 11 calls) = 404 per call
- Time per call: 12.7ms
- Cache hit rate: 0% (no caching)

**After** (with caching):
- Index Sets created: ~400 (for 11 calls) = 36 first call, 0 subsequent
- Time per call: ~4-5ms (similar to rbf=True)
- Cache hit rate: 90%+ (typical workflow)

**Speedup**: **3× faster** for repeated evaluations at same coordinates

**Use cases that benefit**:
- ✅ Time-stepping visualization (same viz points, changing field values)
- ✅ Particle tracking (particles at fixed snapshot times)
- ✅ Convergence studies (probe values during iterations)
- ✅ Interactive visualization (re-render while solver runs)

## Compatibility with Lazy Evaluation

**Why this design works with lazy evaluation**:

1. **No value dependency**: Cache doesn't care about variable values
2. **No access tracking needed**: Don't need to detect when variables change
3. **Coordinate-driven**: Only coordinates matter for cache key
4. **Explicit evaluation**: Each evaluate() call gets fresh values from PETSc

**Contrast with old system**:
- Old: Cached results → needed to know when values changed → broke with lazy eval
- New: Cache structure → values irrelevant → works regardless of update pattern

## Testing Strategy

### Unit Tests

```python
def test_dminterpolation_cache_hit():
    """Verify cache reuse for same coordinates."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    T = uw.discretisation.MeshVariable("T", mesh, 1)
    coords = np.random.random((100, 2))

    # First call - cache miss
    result1 = uw.function.evaluate(T.sym, coords, rbf=False)
    cache_stats1 = mesh.evaluation_cache_stats()

    # Change values, same coords - should HIT cache
    T.array[...] = np.random.random(T.array.shape)
    result2 = uw.function.evaluate(T.sym, coords, rbf=False)
    cache_stats2 = mesh.evaluation_cache_stats()

    assert cache_stats2['hits'] == 1  # ← Cache hit!
    assert result1.shape == result2.shape
    assert not np.allclose(result1, result2)  # Values changed

def test_dminterpolation_cache_miss():
    """Verify cache invalidation for different coordinates."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    T = uw.discretisation.MeshVariable("T", mesh, 1)

    coords1 = np.random.random((100, 2))
    coords2 = np.random.random((100, 2))  # Different!

    result1 = uw.function.evaluate(T.sym, coords1, rbf=False)
    result2 = uw.function.evaluate(T.sym, coords2, rbf=False)

    cache_stats = mesh.evaluation_cache_stats()
    assert cache_stats['misses'] == 2  # Both missed (different coords)

def test_dminterpolation_topology_invalidation():
    """Verify cache clears on mesh topology change."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(16, 16))
    T = uw.discretisation.MeshVariable("T", mesh, 1)
    coords = np.random.random((100, 2))

    # Populate cache
    _ = uw.function.evaluate(T.sym, coords, rbf=False)
    assert len(mesh._dminterpolation_cache._cache) == 1

    # Add new variable (triggers DM rebuild)
    P = uw.discretisation.MeshVariable("P", mesh, 1)

    # Cache should be empty
    assert len(mesh._dminterpolation_cache._cache) == 0
```

### Performance Benchmarks

```python
def benchmark_evaluate_caching():
    """Measure speedup from caching."""
    mesh = uw.meshing.StructuredQuadBox(elementRes=(32, 32))
    T = uw.discretisation.MeshVariable("T", mesh, 1)
    T.array[...] = np.random.random(T.array.shape)
    coords = np.random.random((100, 2))

    # Warm-up (populate cache)
    _ = uw.function.evaluate(T.sym, coords, rbf=False)

    # Benchmark cached evaluations
    n_calls = 100
    start = time.time()
    for i in range(n_calls):
        T.array[...] = np.random.random(T.array.shape)  # Change values
        _ = uw.function.evaluate(T.sym, coords, rbf=False)
    elapsed = time.time() - start

    print(f"Cached evaluate: {elapsed/n_calls*1000:.2f}ms per call")
    print(f"Expected: ~4-5ms (similar to rbf=True)")
```

## Questions for Code Review

1. **Memory management**: Should we implement LRU eviction immediately or later?
   - Pro: Prevents unbounded cache growth
   - Con: Adds complexity, may evict frequently-used entries

2. **Cache scope**: Per-mesh caching (proposed) vs global cache?
   - Current: Each mesh has own cache
   - Alternative: Global cache keyed by (mesh_id, coord_hash)

3. **Coordinate comparison tolerance**: Exact hash or fuzzy matching?
   - Exact: Simple, but misses near-duplicate coordinates
   - Fuzzy: More cache hits, but complex implementation

4. **API for cache control**: Do users need explicit cache management?
   - Current: Fully automatic
   - Optional: `mesh.clear_evaluation_cache()`, `mesh.warm_cache(coords)`

5. **Parallel implications**: Does MPI_COMM_SELF model need rethinking?
   - Current: Each rank has independent cache
   - Future: Could share structures across ranks for same coordinates?

---

**Next Steps**:
1. Review this design
2. Implement Phase 1 (core caching)
3. Validate with diagnostic script (Index Set count, timing)
4. Add tests from testing strategy
5. Document user-facing behavior changes (if any)
