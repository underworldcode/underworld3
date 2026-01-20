#!/usr/bin/env python3
"""
Diagnostic Script: Investigate evaluate() Performance Issues

This script uses the new PETSc logging integration to identify performance
bottlenecks in uw.function.evaluate() and global_evaluate().

**Problem**: evaluate() and global_evaluate() are running very slowly for
both rbf=True and rbf=False modes.

**Approach**: Use PETSc logging to see which internal operations are taking time.
"""

import numpy as np
import time
import underworld3 as uw
from petsc4py import PETSc

print("=" * 80)
print("Diagnostic: evaluate() Performance Investigation")
print("=" * 80)
print()

# Enable PETSc logging BEFORE creating any objects
print("--- Enabling PETSc Logging ---")
uw.timing.enable_petsc_logging()
print("✓ PETSc logging enabled")
print()

# Create custom stages for our tests
stage_setup = PETSc.Log.Stage("setup")
stage_evaluate_rbf = PETSc.Log.Stage("evaluate_rbf_false")
stage_evaluate_rbf_true = PETSc.Log.Stage("evaluate_rbf_true")
stage_global_evaluate = PETSc.Log.Stage("global_evaluate")

# ============================================================================
# SETUP: Create mesh and variable
# ============================================================================

print("--- Setup Phase ---")
stage_setup.push()

# Create medium-sized mesh (enough to see performance issues)
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(32, 32),
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0)
)
print(f"Mesh created: {mesh.X.coords.shape[0]} nodes")

# Create a simple scalar variable
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# Set some data
T.array[...] = np.random.random(T.array.shape)
print(f"Variable created: T.array.shape = {T.array.shape}")

# Create evaluation points (fewer than mesh nodes)
n_eval_points = 100
eval_coords = np.random.random((n_eval_points, 2))
print(f"Evaluation points: {n_eval_points}")

stage_setup.pop()
print()

# ============================================================================
# TEST 1: evaluate() with rbf=False (default interpolation)
# ============================================================================

print("--- Test 1: evaluate() with rbf=False ---")
stage_evaluate_rbf.push()

# Warm-up call (first call often slower due to setup)
_ = uw.function.evaluate(T.sym, eval_coords, rbf=False)

# Timed calls
n_calls = 10
start_time = time.time()
for i in range(n_calls):
    result = uw.function.evaluate(T.sym, eval_coords, rbf=False)
elapsed = time.time() - start_time

print(f"Completed {n_calls} evaluate() calls (rbf=False)")
print(f"Total time: {elapsed:.3f}s")
print(f"Average per call: {elapsed/n_calls:.3f}s")
print(f"Result shape: {result.shape}")

stage_evaluate_rbf.pop()
print()

# ============================================================================
# TEST 2: evaluate() with rbf=True (radial basis function)
# ============================================================================

print("--- Test 2: evaluate() with rbf=True ---")
stage_evaluate_rbf_true.push()

# Warm-up
_ = uw.function.evaluate(T.sym, eval_coords, rbf=True)

# Timed calls
start_time = time.time()
for i in range(n_calls):
    result_rbf = uw.function.evaluate(T.sym, eval_coords, rbf=True)
elapsed_rbf = time.time() - start_time

print(f"Completed {n_calls} evaluate() calls (rbf=True)")
print(f"Total time: {elapsed_rbf:.3f}s")
print(f"Average per call: {elapsed_rbf/n_calls:.3f}s")
print(f"Result shape: {result_rbf.shape}")

stage_evaluate_rbf_true.pop()
print()

# ============================================================================
# TEST 3: global_evaluate()
# ============================================================================
# Note: Skipping global_evaluate() due to API issue with coords parameter
# Focus on the more commonly used evaluate() function

print("--- Test 3: Skipping global_evaluate() ---")
print("(global_evaluate has API issue - focusing on evaluate() instead)")
print()
elapsed_global = 0
n_global_calls = 0

# ============================================================================
# RESULTS: Print PETSc Logging Output
# ============================================================================

print("=" * 80)
print("PETSc PERFORMANCE ANALYSIS")
print("=" * 80)
print()
print("This shows WHERE the time is being spent inside evaluate():")
print()

# Print to console
uw.timing.print_petsc_log()

# Also save to files for detailed analysis
print()
print("=" * 80)
print("SAVING DETAILED LOGS")
print("=" * 80)
print()
uw.timing.print_petsc_log("/tmp/evaluate_performance.txt")
uw.timing.print_petsc_log("/tmp/evaluate_performance.csv")

print()
print("=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print()
print("Key questions to answer from the PETSc log:")
print()
print("1. **DMInterpolation operations**: How much time in DMInterpolationSetUp/Evaluate?")
print("   - DMInterpolationSetUp: Building interpolation structure")
print("   - DMInterpolationEvaluate: Actual interpolation")
print()
print("2. **KDTree operations**: Any time in spatial indexing?")
print("   - Look for custom events or DMPlex operations")
print()
print("3. **Memory operations**: VecCreate, VecCopy, VecDestroy?")
print("   - Excessive allocations could indicate inefficiency")
print()
print("4. **Per-stage breakdown**:")
print(f"   - evaluate_rbf_false: {n_calls} calls, {elapsed:.3f}s total")
print(f"   - evaluate_rbf_true: {n_calls} calls, {elapsed_rbf:.3f}s total")
print(f"   - global_evaluate: {n_global_calls} calls, {elapsed_global:.3f}s total")
print()
print("5. **Call counts**: Are operations being repeated unnecessarily?")
print("   - High call counts relative to number of evaluate() calls")
print("   - suggests setup/teardown happening every time")
print()
print("Look for operations in the PETSc log above with:")
print("- High time percentage (>10%)")
print("- High call counts (>100 for our test)")
print("- DMInterpolation, DMPlex, Vec, or custom UW operations")
print()
print("=" * 80)
print()
print("Files saved for analysis:")
print("- /tmp/evaluate_performance.txt (human-readable)")
print("- /tmp/evaluate_performance.csv (spreadsheet import)")
print()
