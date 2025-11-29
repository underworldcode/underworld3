#!/usr/bin/env python3
"""
Simple Performance Breakdown for evaluate()

Measures the major components without complex instrumentation.
"""

import numpy as np
import time
import underworld3 as uw

print("=" * 80)
print("Simple Performance Breakdown: evaluate()")
print("=" * 80)
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
# TIME INDIVIDUAL COMPONENTS
# ============================================================================

print("--- Timing Components ---")
print()

# 1. get_closest_cells (includes KDTree build on first call)
print("1. get_closest_cells():")
t_start = time.time()
cells_first = mesh.get_closest_cells(eval_coords)
t_first = time.time() - t_start
print(f"   First call (builds KDTree): {t_first*1000:.2f}ms")

t_start = time.time()
for i in range(10):
    cells = mesh.get_closest_cells(eval_coords)
t_avg = (time.time() - t_start) / 10
print(f"   Avg subsequent calls: {t_avg*1000:.2f}ms")
print()

# 2. Full evaluate (rbf=False)
print("2. evaluate(rbf=False):")
# Warm-up
_ = uw.function.evaluate(T.sym, eval_coords, rbf=False)

# Timed calls
n_calls = 10
times = []
for i in range(n_calls):
    T.array[...] = np.random.random(T.array.shape)  # Change values
    t_start = time.time()
    _ = uw.function.evaluate(T.sym, eval_coords, rbf=False)
    times.append(time.time() - t_start)

print(f"   Avg time: {np.mean(times)*1000:.2f}ms")
print(f"   Min/Max: {np.min(times)*1000:.2f}ms / {np.max(times)*1000:.2f}ms")
print()

# 3. Full evaluate (rbf=True) for comparison
print("3. evaluate(rbf=True) for comparison:")
_ = uw.function.evaluate(T.sym, eval_coords, rbf=True)  # Warm-up

times_rbf = []
for i in range(n_calls):
    T.array[...] = np.random.random(T.array.shape)
    t_start = time.time()
    _ = uw.function.evaluate(T.sym, eval_coords, rbf=True)
    times_rbf.append(time.time() - t_start)

print(f"   Avg time: {np.mean(times_rbf)*1000:.2f}ms")
print(f"   Min/Max: {np.min(times_rbf)*1000:.2f}ms / {np.max(times_rbf)*1000:.2f}ms")
print()

# ============================================================================
# ANALYSIS
# ============================================================================

print("=" * 80)
print("ANALYSIS")
print("=" * 80)
print()

rbf_false_time = np.mean(times)
rbf_true_time = np.mean(times_rbf)
gcc_overhead = t_avg

print("Breakdown for rbf=False:")
print(f"  get_closest_cells: {gcc_overhead*1000:.2f}ms ({100*gcc_overhead/rbf_false_time:.1f}%)")
print(f"  DMInterpolation:   {(rbf_false_time-gcc_overhead)*1000:.2f}ms ({100*(rbf_false_time-gcc_overhead)/rbf_false_time:.1f}%)")
print(f"  Total:             {rbf_false_time*1000:.2f}ms")
print()

print("Comparison:")
print(f"  rbf=False: {rbf_false_time*1000:.2f}ms")
print(f"  rbf=True:  {rbf_true_time*1000:.2f}ms")
print(f"  Ratio:     {rbf_false_time/rbf_true_time:.2f}x (rbf=False is this much slower)")
print()

print("Caching Potential:")
print()
print("  If we cache get_closest_cells() results:")
potential_gcc = gcc_overhead * (n_calls - 1) / n_calls  # All but first
speedup_gcc = rbf_false_time / (rbf_false_time - potential_gcc)
print(f"    Time saved: {potential_gcc*1000:.2f}ms per 10 calls")
print(f"    Speedup: {speedup_gcc:.2f}x")
print(f"    New time: {(rbf_false_time - potential_gcc)*1000:.2f}ms per call")
print()

print("  If we cache DMInterpolation structures:")
dm_overhead = rbf_false_time - gcc_overhead
potential_dm = dm_overhead * 0.8  # Conservative: assume 80% cacheable
speedup_dm = rbf_false_time / (rbf_false_time - potential_dm)
print(f"    Time saved (estimated): {potential_dm*1000:.2f}ms per 10 calls")
print(f"    Speedup (estimated): {speedup_dm:.2f}x")
print(f"    New time (estimated): {(rbf_false_time - potential_dm)*1000:.2f}ms per call")
print()

print("Recommendation:")
if dm_overhead > gcc_overhead * 2:
    print("  → DMInterpolation is the main bottleneck")
    print("  → Consider caching full structure (if variables rarely change)")
    print(f"  → Potential speedup: {speedup_dm:.2f}x")
else:
    print("  → get_closest_cells and DMInterpolation are comparable")
    print("  → Start with simpler cell hints caching")
    print(f"  → Potential speedup: {speedup_gcc:.2f}x")
print()
