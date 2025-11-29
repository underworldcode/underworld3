# Instrumented Caching: Self-Measuring Optimization

**Date**: 2025-11-16
**Philosophy**: Measure first, cache second, keep measuring!

## Core Principle

> **"Optimization without measurement is guesswork"**

Every caching system should:
1. **Track its own effectiveness** (hit rate, time saved)
2. **Identify when it breaks** (invalidation events and why)
3. **Justify its existence** (prove it's actually helping)

## Design Pattern: Instrumented Cache

### Basic Structure

```python
class InstrumentedCache:
    """
    Cache that tracks and reports its own performance metrics.

    Metrics tracked:
    - Hit/miss counts and rates
    - Time saved by cache hits
    - Invalidation events (why and when)
    - Memory usage
    - Cache effectiveness over time
    """

    def __init__(self, name, cache_fn, key_fn, invalidate_fn=None):
        self.name = name
        self._cache = {}
        self._cache_fn = cache_fn        # Function to compute value
        self._key_fn = key_fn              # Function to compute cache key
        self._invalidate_fn = invalidate_fn  # Optional invalidation check

        # Instrumentation
        self._stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'time_saved': 0.0,  # Cumulative time saved by hits
            'time_computing': 0.0,  # Time spent computing on misses
            'last_reset': time.time(),
        }

    def get(self, *args, **kwargs):
        """
        Get cached value or compute if missing.

        Returns value AND updates statistics.
        """
        # Compute cache key
        key = self._key_fn(*args, **kwargs)

        # Check for invalidation
        if self._invalidate_fn and self._invalidate_fn():
            self._invalidate_all("invalidation function returned True")

        # Try cache
        if key in self._cache:
            # CACHE HIT
            self._stats['hits'] += 1

            # Estimate time saved (re-running cache_fn would take this long)
            # We can measure this by occasionally re-computing and comparing
            t_start = time.time()
            cached_value = self._cache[key]
            t_cached = time.time() - t_start  # Nearly zero!

            # Use historical average for time_saved estimate
            if self._stats['misses'] > 0:
                avg_compute_time = self._stats['time_computing'] / self._stats['misses']
                self._stats['time_saved'] += avg_compute_time

            return cached_value

        else:
            # CACHE MISS - compute value
            self._stats['misses'] += 1

            t_start = time.time()
            value = self._cache_fn(*args, **kwargs)
            t_compute = time.time() - t_start

            self._stats['time_computing'] += t_compute

            # Store in cache
            self._cache[key] = value

            return value

    def _invalidate_all(self, reason):
        """Clear cache and record why."""
        self._stats['invalidations'] += 1
        self._cache.clear()

        if uw.mpi.rank == 0:
            print(f"Cache '{self.name}' invalidated: {reason}")

    def invalidate_key(self, key, reason="manual"):
        """Invalidate specific cache entry."""
        if key in self._cache:
            del self._cache[key]
            self._stats['invalidations'] += 1

            if uw.mpi.rank == 0:
                print(f"Cache '{self.name}' key invalidated: {reason}")

    def get_stats(self):
        """
        Get cache performance statistics.

        Returns
        -------
        dict with:
            - hit_rate: % of requests served from cache
            - miss_rate: % of requests that required computation
            - time_saved: Total time saved by cache hits (estimated)
            - avg_compute_time: Average time to compute on miss
            - speedup: Effective speedup from caching
            - efficiency: Time saved vs time spent computing
        """
        total_requests = self._stats['hits'] + self._stats['misses']

        if total_requests == 0:
            return {
                'hit_rate': 0.0,
                'miss_rate': 0.0,
                'time_saved': 0.0,
                'avg_compute_time': 0.0,
                'speedup': 1.0,
                'efficiency': 0.0,
            }

        hit_rate = self._stats['hits'] / total_requests
        miss_rate = self._stats['misses'] / total_requests

        avg_compute_time = (
            self._stats['time_computing'] / self._stats['misses']
            if self._stats['misses'] > 0
            else 0.0
        )

        # Speedup: What would total time be without cache vs with cache?
        time_without_cache = total_requests * avg_compute_time
        time_with_cache = self._stats['time_computing']  # Only misses computed
        speedup = time_without_cache / time_with_cache if time_with_cache > 0 else 1.0

        # Efficiency: How much time saved relative to time spent?
        efficiency = (
            self._stats['time_saved'] / self._stats['time_computing']
            if self._stats['time_computing'] > 0
            else 0.0
        )

        return {
            'requests': total_requests,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'invalidations': self._stats['invalidations'],
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,
            'time_saved': self._stats['time_saved'],
            'time_computing': self._stats['time_computing'],
            'avg_compute_time': avg_compute_time,
            'speedup': speedup,
            'efficiency': efficiency,
            'memory_bytes': sum(sys.getsizeof(v) for v in self._cache.values()),
        }

    def print_stats(self):
        """Pretty-print cache statistics."""
        stats = self.get_stats()

        print(f"\n{'=' * 60}")
        print(f"Cache Statistics: {self.name}")
        print(f"{'=' * 60}")
        print(f"Requests:        {stats['requests']:>10,}")
        print(f"  Hits:          {stats['hits']:>10,}  ({stats['hit_rate']*100:>5.1f}%)")
        print(f"  Misses:        {stats['misses']:>10,}  ({stats['miss_rate']*100:>5.1f}%)")
        print(f"  Invalidations: {stats['invalidations']:>10,}")
        print()
        print(f"Performance:")
        print(f"  Time saved:    {stats['time_saved']:>10.3f}s")
        print(f"  Time computing:{stats['time_computing']:>10.3f}s")
        print(f"  Speedup:       {stats['speedup']:>10.2f}x")
        print(f"  Efficiency:    {stats['efficiency']:>10.1f}x (time saved / time computing)")
        print()
        print(f"Memory:          {stats['memory_bytes'] / 1024:>10.1f} KB")
        print(f"{'=' * 60}\n")

    def reset_stats(self):
        """Reset statistics (but keep cached values)."""
        self._stats = {
            'hits': 0,
            'misses': 0,
            'invalidations': 0,
            'time_saved': 0.0,
            'time_computing': 0.0,
            'last_reset': time.time(),
        }
```

## Application to evaluate() Caching

### Option 1: Cache Cell Hints

```python
def create_cell_hints_cache(mesh):
    """Create instrumented cache for get_closest_cells() results."""

    def compute_cells(coords):
        """Expensive: KDTree query."""
        return mesh._original_get_closest_cells(coords)

    def hash_coords(coords):
        """Cache key: hash of coordinates."""
        import xxhash
        xxh = xxhash.xxh64()
        xxh.update(np.ascontiguousarray(coords))
        return xxh.intdigest()

    def check_invalidation():
        """Invalidate if mesh topology changed."""
        if not hasattr(mesh, '_topology_version_at_cache_create'):
            mesh._topology_version_at_cache_create = mesh._topology_version
            return False

        if mesh._topology_version != mesh._topology_version_at_cache_create:
            mesh._topology_version_at_cache_create = mesh._topology_version
            return True

        return False

    return InstrumentedCache(
        name=f"cell_hints_{mesh.name}",
        cache_fn=compute_cells,
        key_fn=hash_coords,
        invalidate_fn=check_invalidation,
    )
```

### Option 2: Cache Entire DMInterpolation Structure

```python
class DMInterpolationCache(InstrumentedCache):
    """
    Specialized cache for DMInterpolation structures.

    Key insight from user: Variable structure rarely changes!
    So we CAN cache the full setup if we track DOF count.
    """

    def __init__(self, mesh):
        self.mesh = mesh

        def compute_structure(coords, dofcount):
            """Create and setup DMInterpolation structure."""
            # This is the expensive part we want to cache!
            ipInfo = self._create_dm_interpolation(coords, dofcount)
            return ipInfo

        def make_key(coords, dofcount):
            """Key: (coord_hash, dofcount)."""
            import xxhash
            xxh = xxhash.xxh64()
            xxh.update(np.ascontiguousarray(coords))
            coord_hash = xxh.intdigest()
            return (coord_hash, dofcount)

        def check_invalidation():
            """Invalidate if mesh topology changed (rare!)."""
            return mesh._topology_version != self._mesh_version

        super().__init__(
            name=f"dminterp_{mesh.name}",
            cache_fn=compute_structure,
            key_fn=make_key,
            invalidate_fn=check_invalidation,
        )

        self._mesh_version = mesh._topology_version

    def _create_dm_interpolation(self, coords, dofcount):
        """
        Create DMInterpolation structure (what we're caching).

        This does DMInterpolationCreate + SetUp but NOT Evaluate.
        """
        # Implementation details...
        pass

    def evaluate_with_cached_structure(self, ipInfo, lvec, outvec):
        """
        Evaluate using cached structure.

        This is the FAST path - just DMInterpolationEvaluate!
        """
        # Implementation details...
        pass
```

## Integration with PETSc Logging

Add custom PETSc events that are visible in `uw.timing.print_petsc_log()`:

```python
class InstrumentedCache:
    def __init__(self, ...):
        # ... existing code ...

        # Create PETSc events for cache operations
        from petsc4py import PETSc
        self._event_hit = PETSc.Log.Event(f"Cache_{name}_Hit")
        self._event_miss = PETSc.Log.Event(f"Cache_{name}_Miss")
        self._event_compute = PETSc.Log.Event(f"Cache_{name}_Compute")

    def get(self, *args, **kwargs):
        key = self._key_fn(*args, **kwargs)

        if key in self._cache:
            # CACHE HIT
            self._event_hit.begin()
            self._stats['hits'] += 1
            value = self._cache[key]
            self._event_hit.end()
            return value

        else:
            # CACHE MISS
            self._event_miss.begin()
            self._stats['misses'] += 1

            self._event_compute.begin()
            t_start = time.time()
            value = self._cache_fn(*args, **kwargs)
            t_compute = time.time() - t_start
            self._event_compute.end()

            self._stats['time_computing'] += t_compute
            self._cache[key] = value

            self._event_miss.end()
            return value
```

**Result**: Cache operations now appear in PETSc log alongside DMInterpolation operations!

```
Event                Count      Time (sec)
Cache_cell_hints_Hit    9    0.000001  <-- FAST! (9 hits)
Cache_cell_hints_Miss   1    0.002134  <-- First call (1 miss)
Cache_cell_hints_Compute 1   0.002100  <-- Actual computation time
DMInterpolationSetUp   10    0.088000  <-- Still happens (uses cached cells)
```

## Usage Example

```python
# In discretisation_mesh.py __init__:
from underworld3.utilities.instrumented_cache import create_cell_hints_cache

self._cell_hints_cache = create_cell_hints_cache(self)
self._topology_version = 0

# In mesh methods that change topology:
def _rebuild_dm(self):
    # ... rebuild DM ...
    self._topology_version += 1  # Triggers cache invalidation

# In get_closest_cells:
def get_closest_cells(self, coords):
    return self._cell_hints_cache.get(coords)

# At end of script/notebook:
mesh._cell_hints_cache.print_stats()
```

**Output**:
```
============================================================
Cache Statistics: cell_hints_main_mesh
============================================================
Requests:               100
  Hits:                  99  ( 99.0%)
  Misses:                 1  (  1.0%)
  Invalidations:          0

Performance:
  Time saved:          0.207s
  Time computing:      0.002s
  Speedup:            50.00x
  Efficiency:        103.5x (time saved / time computing)

Memory:               0.4 KB
============================================================
```

## Systematic Profiling Workflow

### 1. Baseline (No Caching)

```bash
pixi run -e default python diagnose_evaluate_detailed_breakdown.py > baseline.txt
```

**Analyze**:
- Which component takes most time?
- How many Index Sets created?
- Where are the PETSc operations?

### 2. Implement Caching (with instrumentation)

Add `InstrumentedCache` to bottleneck identified in step 1.

### 3. Measure Improvement

```bash
pixi run -e default python diagnose_evaluate_detailed_breakdown.py > with_cache.txt
diff baseline.txt with_cache.txt
```

**Validate**:
- Did time decrease?
- Is cache hit rate high?
- Are Index Sets reduced?
- Is speedup as expected?

### 4. Monitor in Production

Add cache stats to solver output:

```python
# At end of time-stepping loop:
if time_step % 10 == 0:
    mesh.evaluation_cache.print_stats()
    mesh.evaluation_cache.reset_stats()  # Reset for next 10 steps
```

## Benefits of Instrumented Caching

1. **Self-Validating**: Cache proves it's helping with metrics
2. **Debuggable**: Can see when/why invalidations happen
3. **Tunable**: Adjust cache size/strategy based on measured hit rates
4. **Integrated**: PETSc events make cache visible in standard profiling
5. **Educational**: Users learn what's expensive by seeing cache stats

## Questions Answered by Instrumentation

Before implementing ANY caching, the breakdown diagnostic tells us:

1. **Where is the time?**
   - Is get_closest_cells() the bottleneck?
   - Or is DMInterpolationSetUp worse?
   - How much is Python overhead?

2. **How often would cache hit?**
   - Same coordinates every call? → High hit rate
   - Random coordinates? → Low hit rate, don't cache!

3. **What's already cached?**
   - KDTree build happens once (already cached!)
   - What else is being recomputed unnecessarily?

4. **Is caching even worth it?**
   - If bottleneck is 1ms, don't bother
   - If bottleneck is 100ms and reused, CACHE IT!

---

**Philosophy**: Measure → Cache → Measure → Iterate

This way we're not guessing about performance, we're **knowing**.
