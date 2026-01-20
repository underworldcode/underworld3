#!/usr/bin/env python3
"""
Detailed evaluate() Performance Diagnosis with Python-Level Timing

This script adds Python timing + custom PETSc events to isolate exactly
where the 4,446 Index Sets are being created.

Strategy:
1. Use PETSc custom events to track specific code sections
2. Add Python timing around suspected bottlenecks
3. Focus on get_closest_cells() which likely creates Index Sets
"""

import numpy as np
import time
import underworld3 as uw
from petsc4py import PETSc

print("=" * 80)
print("Detailed Diagnostic: evaluate() Performance with Python Timing")
print("=" * 80)
print()

# Enable PETSc logging
print("--- Enabling PETSc Logging + Custom Events ---")
uw.timing.enable_petsc_logging()

# Create custom PETSc events to track specific sections
event_get_closest_cells = PETSc.Log.Event("get_closest_cells")
event_dm_interpolation_setup = PETSc.Log.Event("DMInterpolation_setup")
event_dm_interpolation_eval = PETSc.Log.Event("DMInterpolation_eval")
event_python_overhead = PETSc.Log.Event("python_overhead")

print("✓ Created custom PETSc events:")
print("  - get_closest_cells")
print("  - DMInterpolation_setup")
print("  - DMInterpolation_eval")
print("  - python_overhead")
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
# INSTRUMENTED EVALUATION - Track get_closest_cells()
# ============================================================================

print("--- Instrumented evaluate() Test ---")
print("Timing get_closest_cells() which likely creates Index Sets")
print()

# Monkey-patch mesh.get_closest_cells to add timing
original_get_closest_cells = mesh.get_closest_cells

def instrumented_get_closest_cells(coords):
    """Wrapper to track get_closest_cells performance"""
    event_get_closest_cells.begin()
    t_start = time.time()

    result = original_get_closest_cells(coords)

    t_end = time.time()
    event_get_closest_cells.end()

    # Store timing data
    if not hasattr(instrumented_get_closest_cells, 'times'):
        instrumented_get_closest_cells.times = []
    instrumented_get_closest_cells.times.append(t_end - t_start)

    return result

# Apply monkey-patch
mesh.get_closest_cells = instrumented_get_closest_cells

# Warm-up call
_ = uw.function.evaluate(T.sym, eval_coords, rbf=False)

# Reset timing data
instrumented_get_closest_cells.times = []

# Run multiple calls
n_calls = 10
print(f"Running {n_calls} evaluate() calls with instrumentation...")

total_start = time.time()
for i in range(n_calls):
    event_python_overhead.begin()
    result = uw.function.evaluate(T.sym, eval_coords, rbf=False)
    event_python_overhead.end()
total_end = time.time()

print(f"✓ Completed {n_calls} calls")
print(f"  Total time: {(total_end - total_start):.3f}s")
print(f"  Average: {(total_end - total_start)/n_calls:.3f}s per call")
print()

# Analyze get_closest_cells timing
if instrumented_get_closest_cells.times:
    gcc_times = instrumented_get_closest_cells.times
    print(f"--- get_closest_cells() Analysis ---")
    print(f"  Calls: {len(gcc_times)}")
    print(f"  Total time: {sum(gcc_times):.3f}s")
    print(f"  Average: {np.mean(gcc_times):.4f}s per call")
    print(f"  Min/Max: {np.min(gcc_times):.4f}s / {np.max(gcc_times):.4f}s")
    print(f"  % of total: {100*sum(gcc_times)/(total_end-total_start):.1f}%")
    print()

# Restore original method
mesh.get_closest_cells = original_get_closest_cells

# ============================================================================
# DETAILED COMPONENT BREAKDOWN
# ============================================================================

print("--- Component Breakdown Test ---")
print("Testing individual operations in isolation")
print()

# Test 1: Just get_closest_cells
print("1. get_closest_cells() alone:")
t_start = time.time()
for i in range(n_calls):
    _ = mesh.get_closest_cells(eval_coords)
t_end = time.time()
print(f"   {n_calls} calls: {(t_end-t_start):.3f}s total, {(t_end-t_start)/n_calls:.4f}s avg")
print()

# Test 2: points_in_domain (used in evaluate_nd)
print("2. points_in_domain() (used in evaluate):")
t_start = time.time()
for i in range(n_calls):
    _ = mesh.points_in_domain(eval_coords, strict_validation=False)
t_end = time.time()
print(f"   {n_calls} calls: {(t_end-t_start):.3f}s total, {(t_end-t_start)/n_calls:.4f}s avg")
print()

# Test 3: RBF evaluate for comparison
print("3. RBF evaluate (for comparison):")
t_start = time.time()
for i in range(n_calls):
    _ = uw.function.evaluate(T.sym, eval_coords, rbf=True)
t_end = time.time()
print(f"   {n_calls} calls: {(t_end-t_start):.3f}s total, {(t_end-t_start)/n_calls:.4f}s avg")
print()

# ============================================================================
# PRINT PETSC LOGS
# ============================================================================

print("=" * 80)
print("PETSc LOGGING RESULTS")
print("=" * 80)
print()
print("Look for:")
print("  1. get_closest_cells event - How much time here?")
print("  2. python_overhead event - Captures evaluate() wrapper time")
print("  3. Index Set counts - Should see the 4,446 creations")
print()

uw.timing.print_petsc_log()

# Save detailed logs
uw.timing.print_petsc_log("/tmp/evaluate_detailed.txt")
uw.timing.print_petsc_log("/tmp/evaluate_detailed.csv")

print()
print("=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print()
print("Key Findings:")
print()
print("1. **get_closest_cells()** timing:")
if instrumented_get_closest_cells.times:
    print(f"   - Called {len(gcc_times)} times for {n_calls} evaluate() calls")
    print(f"   - Average: {np.mean(gcc_times)*1000:.2f}ms per call")
    print(f"   - Total: {sum(gcc_times):.3f}s ({100*sum(gcc_times)/(total_end-total_start):.1f}% of evaluate time)")
else:
    print("   - No timing data collected")
print()

print("2. **Index Set creation**:")
print("   - Check PETSc log above for Index Set object counts")
print("   - rbf=False should show ~4,446 Index Sets for 11 calls")
print("   - This suggests get_closest_cells() or DMInterpolationSetUp is rebuilding structures")
print()

print("3. **Optimization opportunities**:")
print("   - If get_closest_cells() is slow → cache KDTree or cell hints")
print("   - If DMInterpolationSetUp is slow → reuse interpolation structure")
print("   - If Index Sets are excessive → investigate why not reused")
print()

print("Files saved:")
print("  - /tmp/evaluate_detailed.txt")
print("  - /tmp/evaluate_detailed.csv")
print()
