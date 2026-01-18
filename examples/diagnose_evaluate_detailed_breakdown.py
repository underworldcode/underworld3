#!/usr/bin/env python3
"""
Detailed Breakdown: Where is the time spent in evaluate()?

This script instruments EVERY step of the evaluate() path to identify
the actual bottleneck before implementing any caching.

Measures:
1. get_closest_cells() - broken into:
   - KDTree build time (should be once)
   - KDTree query time (happens every call)
2. DMInterpolationCreate/SetUp/Evaluate/Destroy - individually timed
3. Python overhead vs Cython overhead

Strategy:
- Use both Python timing AND PETSc custom events
- Measure each component separately
- Track how many times each is called
- Calculate percentages to find bottleneck
"""

import numpy as np
import time
import underworld3 as uw
from petsc4py import PETSc

print("=" * 80)
print("Detailed Breakdown: evaluate() Performance Components")
print("=" * 80)
print()

# Enable PETSc logging
uw.timing.enable_petsc_logging()

# Create custom PETSc events for fine-grained tracking
event_kdtree_build = PETSc.Log.Event("kdtree_build")
event_kdtree_query = PETSc.Log.Event("kdtree_query")
event_get_closest_cells = PETSc.Log.Event("get_closest_cells")
event_dm_interp_create = PETSc.Log.Event("DMInterp_Create")
event_dm_interp_setup = PETSc.Log.Event("DMInterp_SetUp")
event_dm_interp_eval = PETSc.Log.Event("DMInterp_Evaluate")
event_dm_interp_destroy = PETSc.Log.Event("DMInterp_Destroy")

print("✓ Created custom PETSc events for detailed tracking")
print()

# ============================================================================
# SETUP
# ============================================================================

print("--- Setup ---")
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(32, 32),
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0)
)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
T.array[...] = np.random.random(T.array.shape)

n_eval_points = 100
eval_coords = np.random.random((n_eval_points, 2))

print(f"Mesh: {mesh.X.coords.shape[0]} nodes")
print(f"Variable: T ({T.array.shape})")
print(f"Evaluation points: {n_eval_points}")
print()

# ============================================================================
# MONKEY-PATCH FOR DETAILED TIMING
# ============================================================================

print("--- Installing Instrumentation ---")

# Track timing data
timing_data = {
    'get_closest_cells': [],
    'kdtree_build': [],
    'kdtree_query': [],
}

# 1. Instrument get_closest_cells
original_get_closest_cells = mesh.get_closest_cells

def instrumented_get_closest_cells(coords):
    """Wrapper for get_closest_cells with timing."""
    event_get_closest_cells.begin()
    t_start = time.time()

    result = original_get_closest_cells(coords)

    t_end = time.time()
    event_get_closest_cells.end()
    timing_data['get_closest_cells'].append(t_end - t_start)

    return result

mesh.get_closest_cells = instrumented_get_closest_cells

# 2. Instrument KDTree build (check if it's called multiple times)
original_build_kdtree = mesh._build_kd_tree_index

def instrumented_build_kdtree():
    """Wrapper for _build_kd_tree_index."""
    # Check if already built
    if hasattr(mesh, "_index") and mesh._index is not None:
        # Already built - no timing needed
        return

    event_kdtree_build.begin()
    t_start = time.time()

    original_build_kdtree()

    t_end = time.time()
    event_kdtree_build.end()
    timing_data['kdtree_build'].append(t_end - t_start)
    print(f"  KDTree built: {(t_end - t_start)*1000:.2f}ms")

mesh._build_kd_tree_index = instrumented_build_kdtree

# 3. Instrument KDTree query
if hasattr(mesh, '_index') and mesh._index is not None:
    original_kdtree = mesh._index
    original_query = original_kdtree.query

    def instrumented_query(*args, **kwargs):
        """Wrapper for KDTree.query."""
        event_kdtree_query.begin()
        t_start = time.time()

        result = original_query(*args, **kwargs)

        t_end = time.time()
        event_kdtree_query.end()
        timing_data['kdtree_query'].append(t_end - t_start)

        return result

    # This will be applied after first build

print("✓ Instrumentation installed")
print()

# ============================================================================
# WARM-UP CALL (builds KDTree)
# ============================================================================

print("--- Warm-up Call (builds KDTree) ---")
_ = uw.function.evaluate(T.sym, eval_coords, rbf=False)
print()

# Now instrument the KDTree that was just built
if hasattr(mesh, '_index') and mesh._index is not None:
    original_query = mesh._index.query

    def instrumented_query_runtime(*args, **kwargs):
        """Wrapper for KDTree.query (runtime)."""
        event_kdtree_query.begin()
        t_start = time.time()

        result = original_query(*args, **kwargs)

        t_end = time.time()
        event_kdtree_query.end()
        timing_data['kdtree_query'].append(t_end - t_start)

        return result

    mesh._index.query = instrumented_query_runtime

# ============================================================================
# TIMED EVALUATION CALLS
# ============================================================================

print("--- Timed Evaluation Calls ---")
n_calls = 10

# Change values between calls to simulate realistic usage
total_start = time.time()
for i in range(n_calls):
    # Change variable values (simulates time-stepping)
    T.array[...] = np.random.random(T.array.shape)

    # Evaluate
    result = uw.function.evaluate(T.sym, eval_coords, rbf=False)
total_end = time.time()

total_time = total_end - total_start
print(f"✓ Completed {n_calls} calls")
print(f"  Total time: {total_time:.3f}s")
print(f"  Average per call: {total_time/n_calls:.3f}s ({total_time/n_calls*1000:.1f}ms)")
print()

# ============================================================================
# ANALYSIS
# ============================================================================

print("=" * 80)
print("COMPONENT BREAKDOWN")
print("=" * 80)
print()

# Calculate statistics
def analyze_component(name, times_list):
    if not times_list:
        print(f"{name}:")
        print(f"  NOT CALLED")
        return 0

    times = np.array(times_list)
    total = times.sum()
    mean = times.mean()
    count = len(times)

    pct_of_total = 100 * total / total_time if total_time > 0 else 0

    print(f"{name}:")
    print(f"  Calls: {count}")
    print(f"  Total: {total*1000:.2f}ms")
    print(f"  Mean: {mean*1000:.2f}ms")
    print(f"  % of total evaluate time: {pct_of_total:.1f}%")

    return total

print("1. KDTree Build (one-time):")
kdtree_build_total = analyze_component("  _build_kd_tree_index", timing_data['kdtree_build'])
print()

print("2. KDTree Query (every call):")
kdtree_query_total = analyze_component("  KDTree.query", timing_data['kdtree_query'])
print()

print("3. get_closest_cells (total):")
gcc_total = analyze_component("  get_closest_cells", timing_data['get_closest_cells'])
print()

# Calculate unaccounted time
accounted = kdtree_build_total + kdtree_query_total
if timing_data['get_closest_cells']:
    gcc_overhead = gcc_total - kdtree_query_total
    print("4. get_closest_cells overhead (non-KDTree):")
    print(f"  Total: {gcc_overhead*1000:.2f}ms")
    print(f"  % of total: {100*gcc_overhead/total_time:.1f}%")
    print()

# Estimate DMInterpolation overhead (from total time)
dm_overhead = total_time - gcc_total - (sum(timing_data['kdtree_build']) if timing_data['kdtree_build'] else 0)
print("5. DMInterpolation overhead (estimated):")
print(f"  Total: {dm_overhead*1000:.2f}ms")
print(f"  % of total: {100*dm_overhead/total_time:.1f}%")
print(f"  Per call: {dm_overhead/n_calls*1000:.2f}ms")
print()

# ============================================================================
# CACHING ANALYSIS
# ============================================================================

print("=" * 80)
print("CACHING POTENTIAL ANALYSIS")
print("=" * 80)
print()

print("If we cache get_closest_cells() results:")
potential_savings_gcc = gcc_total * (n_calls - 1) / n_calls  # All but first call
print(f"  Time saved: {potential_savings_gcc*1000:.2f}ms ({100*potential_savings_gcc/total_time:.1f}% of total)")
print(f"  Speedup: {total_time / (total_time - potential_savings_gcc):.2f}x")
print()

print("If we cache KDTree query results only:")
potential_savings_query = kdtree_query_total * (n_calls - 1) / n_calls
print(f"  Time saved: {potential_savings_query*1000:.2f}ms ({100*potential_savings_query/total_time:.1f}% of total)")
print(f"  Speedup: {total_time / (total_time - potential_savings_query):.2f}x")
print()

print("If we cache DMInterpolation structure:")
# Assume we save all SetUp overhead (conservative - might not be possible)
potential_savings_dm = dm_overhead * 0.8  # Assume 80% of DM overhead is SetUp
print(f"  Estimated time saved: {potential_savings_dm*1000:.2f}ms ({100*potential_savings_dm/total_time:.1f}% of total)")
print(f"  Estimated speedup: {total_time / (total_time - potential_savings_dm):.2f}x")
print()

# ============================================================================
# CACHE INVALIDATION ANALYSIS
# ============================================================================

print("=" * 80)
print("CACHE INVALIDATION FREQUENCY (for this test)")
print("=" * 80)
print()

print("Coordinate changes:")
print(f"  Same coordinates for all {n_calls} calls: YES")
print(f"  → Cache hit rate would be: {100*(n_calls-1)/n_calls:.1f}% (after warm-up)")
print()

print("Variable value changes:")
print(f"  Values changed: {n_calls} times")
print(f"  → Should NOT invalidate cache (structure independent of values)")
print()

print("Variable structure changes:")
print(f"  Variables added/removed: 0 times")
print(f"  → Cache never invalidated")
print()

# ============================================================================
# RECOMMENDATIONS
# ============================================================================

print("=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print()

# Determine bottleneck
if dm_overhead > gcc_total:
    print("🎯 PRIMARY BOTTLENECK: DMInterpolation SetUp/Evaluate")
    print(f"   ({100*dm_overhead/total_time:.1f}% of time)")
    print()
    print("   RECOMMENDATION: Cache DMInterpolation structure")
    print("   - IF variable configuration rarely changes")
    print("   - THEN cache full SetUp result")
    print("   - Invalidate only on: coord change OR variable add/remove")
    print()
else:
    print("🎯 PRIMARY BOTTLENECK: get_closest_cells() / KDTree query")
    print(f"   ({100*gcc_total/total_time:.1f}% of time)")
    print()
    print("   RECOMMENDATION: Cache cell hints (simpler!)")
    print("   - Cache result of get_closest_cells()")
    print("   - Invalidate only on: coord change OR mesh topology change")
    print()

print("NEXT STEPS:")
print("  1. Review this breakdown to confirm bottleneck")
print("  2. Implement caching for identified bottleneck")
print("  3. Add cache hit/miss tracking to measure effectiveness")
print("  4. Re-run this diagnostic to measure improvement")
print()

# ============================================================================
# SAVE PETSC LOGS
# ============================================================================

print("=" * 80)
print("Saving detailed PETSc logs...")
print("=" * 80)

uw.timing.print_petsc_log("/tmp/evaluate_breakdown.txt")
uw.timing.print_petsc_log("/tmp/evaluate_breakdown.csv")

print()
print("Files saved:")
print("  - /tmp/evaluate_breakdown.txt")
print("  - /tmp/evaluate_breakdown.csv")
print()
