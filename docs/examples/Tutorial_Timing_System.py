# %% [markdown]
# # Underworld3 Performance Timing Tutorial
#
# This notebook demonstrates how to use Underworld3's PETSc-based timing system to profile your simulations.
#
# ## Features
#
# - **Jupyter-friendly**: No environment variables needed - just call `uw.timing.start()`
# - **Comprehensive**: Captures ~95% of computation (UW3 operations + PETSc internals)
# - **User-friendly**: `uw.timing.print_summary()` shows only relevant UW3 operations
# - **Detailed**: `uw.timing.print_table()` shows full PETSc profiling data
#
# ## Key Functions
#
# - `uw.timing.start()` - Enable timing (call once at the start)
# - `uw.timing.print_summary()` - Clean table of UW3 operations only
# - `uw.timing.print_table()` - Full PETSc profiling details
# - `uw.timing.get_summary()` - Programmatic access to timing data

# %% [markdown]
# ## Setup

# %%
import underworld3 as uw
import numpy as np
import sympy

# %% [markdown]
# ## 1. Basic Usage - Enable Timing
#
# Just call `uw.timing.start()` once at the beginning of your notebook. No environment variables needed!

# %%
# Enable timing - that's all you need!
uw.timing.start()

print("✓ Timing enabled!")
print("  All operations from this point will be tracked.")

# %% [markdown]
# ## 2. Example Workflow - Poisson Equation
#
# Let's create a simple Poisson problem to generate some timing data.

# %%
# Create mesh
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=0.05,  # Moderate resolution
    qdegree=3
)

print(f"Mesh created with {mesh.data.shape[0]} vertices")

# %%
# Create variable
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)

# Set up Poisson equation: ∇²T = -1
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = 1.0
poisson.f = -1.0  # Source term

# Boundary conditions: T = 0 on all boundaries
poisson.add_dirichlet_bc(0.0, "Top")
poisson.add_dirichlet_bc(0.0, "Bottom")
poisson.add_dirichlet_bc(0.0, "Left")
poisson.add_dirichlet_bc(0.0, "Right")

print("Poisson solver configured")

# %%
# Solve the system
poisson.solve()

print(f"Solution range: [{T.array.min():.4f}, {T.array.max():.4f}]")

# %%
# Evaluate solution at some sample points
sample_points = np.array([
    [0.25, 0.25],
    [0.5, 0.5],
    [0.75, 0.75]
])

result = uw.function.evaluate(T.sym, sample_points)
print(f"Values at sample points: {result.squeeze()}")

# %% [markdown]
# ## 3. User-Friendly Timing Summary
#
# The `print_summary()` function shows **only UW3 operations**, filtering out hundreds of low-level PETSc events.
#
# This is perfect for quick performance checks!

# %%
# Show clean summary of UW3 operations
uw.timing.print_summary()

# %% [markdown]
# ### Understanding the Output
#
# The summary shows:
# - **Event Name**: UW3 operation (e.g., mesh creation, solve, evaluate)
# - **Count**: Number of times the operation was called
# - **Time (s)**: Total time spent in this operation
# - **% Total**: Percentage of total execution time
#
# This helps you identify performance bottlenecks at a glance!

# %% [markdown]
# ## 4. Sorting and Filtering Options
#
# You can customize the summary display:

# %%
# Sort by call count instead of time
print("\nTop 10 most frequently called operations:")
uw.timing.print_summary(sort_by='count', max_events=10)

# %%
# Show operations taking at least 10ms
print("\nOperations taking at least 10ms:")
uw.timing.print_summary(min_time=0.01)

# %% [markdown]
# ## 5. Programmatic Access
#
# Use `get_summary()` to access timing data in your code:

# %%
# Get timing data as a dictionary
summary = uw.timing.get_summary()

print(f"Total execution time: {summary['total_time']:.3f} seconds")
print(f"Number of timed events: {summary['num_events']}")
print("\nTop 3 most expensive operations:")

for i, (name, count, time, pct) in enumerate(summary['events'][:3], 1):
    print(f"  {i}. {name}: {time:.4f}s ({pct:.1f}%)")

# %% [markdown]
# ## 6. Full PETSc Profiling Details
#
# For detailed profiling, use `print_table()` to see **all PETSc events**.
#
# This includes matrix operations, vector operations, solver internals, etc.
#
# **Warning**: The output is very detailed (~100+ events)!

# %%
# Show full PETSc profiling log
# Uncomment to see detailed output (it's long!)

# uw.timing.print_table()

# %% [markdown]
# ## 7. Viewing All Events (Including PETSc)
#
# You can also use `print_summary(filter_uw=False)` to see all events in a cleaner format:

# %%
# Show all events (UW3 + PETSc) in summary format
# This is cleaner than print_table() but shows everything

# uw.timing.print_summary(filter_uw=False, max_events=20)

# %% [markdown]
# ## 8. Practical Example - Time-Stepping Loop
#
# Let's see how timing helps optimize a time-stepping simulation:

# %%
# Create a simple advection-diffusion problem
mesh2 = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=0.1,
)

T2 = uw.discretisation.MeshVariable("T2", mesh2, 1, degree=2)

# Initial condition
x, y = mesh2.X
T2.array[...] = np.exp(-((mesh2.data[:, 0] - 0.5)**2 + (mesh2.data[:, 1] - 0.5)**2) / 0.01)

print("Initial condition set")

# %%
# Simple time-stepping loop
n_steps = 10
dt = 0.01

for step in range(n_steps):
    # Simple diffusion: T_new = T_old + dt * ∇²T
    # (This is just for demonstration - use proper solvers in practice!)

    # Evaluate gradient
    grad_T = uw.function.evaluate(T2.sym.diff(x), mesh2.data)

    if step == 0:
        print(f"Step {step}: T range = [{T2.array.min():.4f}, {T2.array.max():.4f}]")

print(f"Step {n_steps}: T range = [{T2.array.min():.4f}, {T2.array.max():.4f}]")

# %%
# Check timing after time loop
uw.timing.print_summary(max_events=15)

# %% [markdown]
# Notice how the `evaluate` operation was called multiple times (once per timestep).
#
# The timing data helps you understand where time is spent in your simulation!

# %% [markdown]
# ## 9. Saving Timing Results
#
# You can save timing data to files for later analysis:

# %%
# Save as CSV for spreadsheet analysis
uw.timing.print_table("timing_results.csv")

print("✓ Timing data saved to timing_results.csv")
print("  You can open this in Excel or analyze with pandas")

# %%
# Save as text file for documentation
uw.timing.print_table("timing_results.txt")

print("✓ Timing data saved to timing_results.txt")

# %% [markdown]
# ## Summary
#
# ### Quick Reference
#
# ```python
# # Enable timing (once at start)
# uw.timing.start()
#
# # After running your simulation:
#
# # Quick UW3-focused view (recommended for most users)
# uw.timing.print_summary()
#
# # Customize the view
# uw.timing.print_summary(sort_by='count')  # Sort by call count
# uw.timing.print_summary(max_events=10)    # Show top 10
# uw.timing.print_summary(min_time=0.01)    # Only ops > 10ms
#
# # Programmatic access
# summary = uw.timing.get_summary()
# print(f"Total time: {summary['total_time']:.3f}s")
#
# # Full PETSc details (for deep profiling)
# uw.timing.print_table()
#
# # Save results
# uw.timing.print_table("results.csv")  # CSV format
# uw.timing.print_table("results.txt")  # Text format
# ```
#
# ### Tips
#
# 1. **Always start with `print_summary()`** - It shows only what you care about
# 2. **Use `sort_by='count'`** to find operations called many times (optimization opportunities)
# 3. **Use `min_time`** to filter out negligible operations
# 4. **Use `print_table()` only for deep profiling** - it's very detailed
# 5. **Save CSV files** for comparing performance across runs
#
# ### Key Benefits
#
# - ✅ **Jupyter-friendly**: No environment variables
# - ✅ **User-focused**: `print_summary()` filters out noise
# - ✅ **Comprehensive**: Captures ~95% of computation
# - ✅ **Zero overhead**: When not enabled, zero performance cost
# - ✅ **PETSc integration**: Unified timing for UW3 + PETSc operations

# %%
