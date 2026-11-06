---
title: "Parallel-Safe Scripting"
---

# Parallel-Safe Scripting in Underworld3

## Overview

Underworld3 uses PETSc for parallel operations, which means **you rarely need to use MPI directly**. PETSc handles domain decomposition, ghost cells, and synchronization automatically. However, understanding parallel safety is crucial for writing correct scripts that work on multiple processors.

## Key Principle: PETSc Handles Parallelism

**Critical Understanding**: You don't write MPI code - you write parallel-safe UW3 code.

- All mesh operations are inherently parallel via PETSc
- Solvers automatically distribute work across processors
- Use UW3 API functions, not direct MPI calls

The main use of `uw.mpi.rank` is for conditional output/visualization.

## MPI + Thread Pools (Oversubscription)

When running with MPI, each rank can also spawn BLAS/OpenMP worker threads.
If this is not controlled, total runnable threads can explode and performance
can degrade severely.

Example: `mpirun -np 8` with OpenBLAS default `10` threads can create up to
`80` compute threads, often slower than expected.

### Default Underworld3 Policy

Underworld3 now applies MPI-safe defaults (thread pool size `1`) unless users
explicitly set their own values:

- `OMP_NUM_THREADS`
- `OPENBLAS_NUM_THREADS`
- `MKL_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`
- `NUMEXPR_NUM_THREADS`

This happens in two places:

1. `./uw` launcher: sets defaults before Python starts.
2. `underworld3` import path: applies the same defaults for MPI runs if unset.

### Runtime Warning

If running with MPI and any of the thread variables above are explicitly set
to values greater than `1`, Underworld3 prints a rank-0 warning about possible
oversubscription.

### User Controls

- Disable automatic thread caps:

```bash
export UW_DISABLE_THREAD_CAPS=1
```

- Suppress warning (keep your explicit thread settings):

```bash
export UW_SUPPRESS_THREAD_WARNING=1
```

### Recommended Practice

For most MPI benchmark and production jobs, keep `1` thread per rank unless
you are intentionally tuning hybrid MPI+threads.

## Parallel-Safe Output

### The Problem with Rank Conditionals

This common pattern is **dangerous**:

```python
# DANGEROUS - can hang in parallel!
if uw.mpi.rank == 0:
    stats = var.stats()  # Collective operation - rank 0 waits forever for the others
    print(f"Stats: {stats}")
```

**Why it hangs**: `var.stats()` is a **collective operation** - ALL ranks must call it. If only rank 0 calls it, rank 0 waits forever for the other ranks, which never call it.

### The Solution: Parallel Print

Use `uw.pprint()` for rank-specific output:

```python
# Safe - all ranks execute stats(), only rank 0 prints
uw.pprint(f"Stats: {var.stats()}")

# Debug - print from multiple ranks
uw.pprint(f"Local max: {var.data.max()}", proc=slice(0, 4))
```

## Rank Selection Syntax

`uw.pprint()` supports flexible rank selection:

### Basic Selection

```python
# Single rank
uw.pprint("Only rank 0")

# All ranks
uw.pprint("Everyone prints this", proc=None)

# Range of ranks (Python slice)
uw.pprint("Ranks 0-3", proc=slice(0, 4))
uw.pprint("Ranks 2, 4, 6", proc=slice(2, 8, 2))

# Specific ranks (list/tuple)
uw.pprint("Ranks 0, 3, and 7", proc=[0, 3, 7])
```

### Named Patterns

```python
uw.pprint("All ranks", proc='all')
uw.pprint("Rank 0 only", proc='first')
uw.pprint("Highest rank only", proc='last')
uw.pprint("Even-numbered ranks", proc='even')
uw.pprint("Odd-numbered ranks", proc='odd')
```

### Advanced Selection

```python
# Percentage of ranks
uw.pprint("First 10% of ranks", proc='10%')

# Function-based
uw.pprint("Every third rank", proc=lambda r: r % 3 == 0)

# NumPy arrays
import numpy as np
mask = np.array([True, False, True, False])
uw.pprint("Using boolean mask", proc=mask)
```

## Selective Execution Context

For code that should **only execute on certain ranks** (not just print), use `selective_ranks()`. Entering the block is collective: every rank runs the `with` body, and the context manager yields `True` on the selected ranks and `False` on the others. The `if` on that value is what restricts the code:

```python
# Visualization - only rank 0 executes
with uw.selective_ranks(0) as should_execute:
    if should_execute:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(x, temperature.data[:, 0])
        plt.savefig("temp_profile.png")

# Multiple ranks
with uw.selective_ranks(slice(0, 4)) as should_execute:
    if should_execute:
        # Only ranks 0-3 execute this block
        process_local_partition()
```

````{admonition} The if check is required
:class: warning

The `with` body runs on every rank. Without the `if should_execute:` check, every rank executes the code. A file written that way is written by every rank at once and can be corrupted. The correct form:

```python
with uw.selective_ranks(0) as should_execute:
    if should_execute:
        # Your rank-specific code here
        pass
```
````

### Collective Operation Safety

````{admonition} Avoid collective operations in selective blocks
:class: warning

**Never call collective operations inside the selected part of a `selective_ranks()` block.** Only the selected ranks reach them, so the others never join:

```python
# WRONG - only rank 0 calls the collective
with uw.selective_ranks(0) as should_execute:
    if should_execute:
        stats = var.stats()  # raises CollectiveOperationError

# RIGHT - Use pprint instead
uw.pprint(f"Stats: {var.stats()}")  # All ranks execute stats(), only rank 0 prints
```
````

Methods marked as collective with `uw.collective_operation` (among them `var.stats()`, `solver.solve()`, `swarm.migrate()` and the swarm `global_*` reductions) raise `CollectiveOperationError` on each rank that reaches them inside a `selective_ranks()` block that does not select every rank. The other ranks continue, so they can still stall at their next collective; the error names the function so the cause is found. Unmarked collectives are not detected and hang: raw PETSc or MPI calls, and parallel HDF5 writes such as `mesh.write()`, `mesh.write_timestep()` and `swarm.save()`.

## Understanding Collective Operations

**Collective operations** require all MPI ranks to participate:

### Common Collective Operations

```python
# Solver operations
stokes.solve()           # All ranks must call
var.stats()              # All ranks must call
mesh.write("file.h5")    # Collective I/O (parallel HDF5)

# Data operations  
var.rbf_interpolate()    # All ranks participate
swarm.migrate()          # Redistributes particles
```

### Local Operations (Safe)

```python
# These only access local data
var.data[...]            # Local partition only
var.data.max()           # Local maximum
print(var.data.shape)    # Local shape
```

## Practical Patterns

### Pattern 1: Time-Stepping Output

```python
# Time-stepping loop with progress output
for step in range(nsteps):
    dt = stokes.estimate_dt()
    
    # All ranks execute solver, only rank 0 prints progress
    stokes.solve()
    uw.pprint(f"Step {step}: dt={dt:.3e}, time={time:.3e}")
    
    # Update fields...
    time += dt
```

### Pattern 2: Convergence Monitoring

```python
# Monitor solver convergence
solver.solve()

# Get global statistics (collective), print on rank 0
velocity_stats = velocity.stats()
pressure_stats = pressure.stats()

uw.pprint(f"Velocity max: {velocity_stats['max']:.6e}")
uw.pprint(f"Pressure range: [{pressure_stats['min']:.6e}, {pressure_stats['max']:.6e}]")
```

### Pattern 3: Debugging Parallel Decomposition

```python
# Check local data on multiple ranks
uw.pprint(f"Rank {uw.mpi.rank}: Local elements = {mesh.dm.getLocalSize()}", proc=slice(0, 4))
uw.pprint(f"Rank {uw.mpi.rank}: Partition shape = {var.data.shape}", proc='all')

# Check first and last rank only
uw.pprint(f"Rank {uw.mpi.rank}: Boundary points = {boundary_count}", proc=[0, uw.mpi.size-1])

# Custom selection - every 4th rank
uw.pprint(f"Rank {uw.mpi.rank}: Memory usage = {psutil.Process().memory_info().rss / 1e9:.2f} GB", proc=lambda r: r % 4 == 0)
```

### Pattern 4: Visualization and I/O

```python
# Only rank 0 does visualization
with uw.selective_ranks(0) as should_execute:
    if should_execute:
        import pyvista as pv
        
        plotter = pv.Plotter(off_screen=True)
        plotter.add_mesh(mesh.pyvista_mesh, scalars=temperature.data)
        plotter.camera_position = 'xy'
        plotter.show(screenshot=f"temp_{step:04d}.png")
```

### Pattern 5: Mesh Information Display

```python
# Display mesh statistics (from mesh.view())
uw.pprint(f"\nMesh # {mesh.instance}: {mesh.name}\n")
uw.pprint(f"Number of cells: {num_cells}\n")

if len(mesh.vars) > 0:
    uw.pprint(f"| Variable Name       | component | degree |     type        |")
    uw.pprint(f"| ---------------------------------------------------------- |")
    for vname in mesh.vars.keys():
        v = mesh.vars[vname]
        uw.pprint(f"| {v.clean_name:<20}|{v.num_components:^10} |{v.degree:^7} | {v.vtype.name:^15} |")
    uw.pprint(f"| ---------------------------------------------------------- |")
else:
    uw.pprint(f"No variables are defined on the mesh\n")
```

### Pattern 6: Collective I/O, Serial Bookkeeping

Checkpoint writes are collective (parallel HDF5), so every rank calls them. Only the serial bookkeeping around them belongs to one rank:

```python
# Every rank writes its part of the checkpoint
mesh.write_timestep("model", index=step, meshVars=[velocity, pressure], outputPath="output")
swarm.save(f"output/particles_{step}.h5")

# Only rank 0 appends to the run log
with uw.selective_ranks(0) as should_execute:
    if should_execute:
        with open("output/run.log", "a") as log:
            log.write(f"step {step} written\n")
```

## Testing Parallel Safety

Always test scripts with multiple processors:

```bash
# Test with 2 processors
mpirun -np 2 python my_script.py

# Test with 4 processors
mpirun -np 4 python my_script.py
```

If your script hangs, check for:
1. Collective operations inside rank conditionals
2. Rank-specific execution of collective operations
3. Mismatched barriers or synchronization

## Common Pitfalls

### Pitfall 1: Collective in Conditional

```python
# WRONG - hangs
if uw.mpi.rank == 0:
    result = var.stats()  # Other ranks wait forever

# RIGHT - all execute, selective output
uw.pprint(f"Result: {var.stats()}")
```

### Pitfall 2: Assuming Global Data

```python
# WRONG - data is local to each rank
total_points = var.data.shape[0]  # Only local points!

# RIGHT - use collective operation
total_points = var.stats()['count']  # Global count
```

### Pitfall 3: Serial Libraries in Parallel

```python
# WRONG - matplotlib on all ranks (crashes or conflicts)
import matplotlib.pyplot as plt
plt.plot(x, y)

# RIGHT - only rank 0
with uw.selective_ranks(0) as should_execute:
    if should_execute:
        import matplotlib.pyplot as plt
        plt.plot(x, y)
```

## Migration from Old Patterns

If you have existing code using rank conditionals, here's how to migrate:

### Old Pattern → New Pattern

```python
# OLD: Rank conditional with print
if uw.mpi.rank == 0:
    print(f"Iteration {step}, Time: {time}")

# NEW: Use pprint
uw.pprint(f"Iteration {step}, Time: {time}")
```

```python
# OLD: Collective operation in conditional (DANGEROUS!)
if uw.mpi.rank == 0:
    max_value = var.stats()['max']
    print(f"Max: {max_value}")

# NEW: All ranks execute, selected ranks print
uw.pprint(f"Max: {var.stats()['max']}")
```

```python
# OLD: Rank conditional for visualization
if uw.mpi.rank == 0:
    import pyvista as pv
    plotter = pv.Plotter()
    plotter.add_mesh(mesh.pyvista_mesh)
    plotter.show()

# NEW: Use selective_ranks
with uw.selective_ranks(0) as should_execute:
    if should_execute:
        import pyvista as pv
        plotter = pv.Plotter()
        plotter.add_mesh(mesh.pyvista_mesh)
        plotter.show()
```

```{admonition} Why the change?
:class: note

The new patterns are safer because:

1. **`uw.pprint()`** ensures all ranks evaluate arguments (preventing collective operation deadlocks)
2. **`selective_ranks()`** makes it explicit which code is rank-specific
3. Code is more readable and intention is clear
4. Marked collectives called inside a selective block are reported (`CollectiveOperationError`)
```

## When to Use What

| Task | Use | Example |
|------|-----|---------|
| Print from specific ranks | `uw.pprint(..., proc=ranks)` | `uw.pprint("Done")` |
| Serial code (viz, I/O) | `with uw.selective_ranks(ranks) as should_execute:` then `if should_execute:` | Matplotlib, file writes |
| Collective with output | `uw.pprint()` with collective args | `uw.pprint(var.stats())` |
| Debug multiple ranks | `uw.pprint(..., proc='all')` | Check local data |

## Best Practices

1. **Never use direct MPI** unless absolutely necessary
2. **Use `uw.pprint()`** instead of `if uw.mpi.rank == 0: print()`
3. **Wrap serial libraries** in `selective_ranks(0)` and check the flag it yields
4. **Test with multiple processors** early and often
5. **Check error messages** - they tell you exactly what's wrong

## Advanced: Rank Groups

For complex patterns, define rank groups:

```python
class RankGroups:
    def __init__(self):
        size = uw.mpi.size
        self.io_rank = 0
        self.compute_ranks = list(range(1, size))
        self.sample_ranks = list(range(0, size, size // 4))

groups = RankGroups()

# Use throughout script
uw.pprint("Saving data...", proc=groups.io_rank)
with uw.selective_ranks(groups.compute_ranks) as should_execute:
    if should_execute:
        heavy_computation()
```

## Related Documentation

- [Performance Optimization](performance.md) - Profiling parallel code
- [Troubleshooting](troubleshooting.md) - Debugging parallel issues
- [Developer: PETSc Integration](../developer/subsystems/petsc-integration.md) - How PETSc manages parallelism

## Quick Reference

### API Summary

**`uw.pprint(*args, proc=0, prefix=None, **kwargs)`**
- Print from selected ranks
- All ranks evaluate arguments (safe for collective operations)
- Optional rank prefix (default: `[0]`)

**`uw.selective_ranks(ranks)`**
- Context manager for rank-specific execution; every rank enters the block
- Yields `True` on the selected ranks, `False` on the others
- The `if should_execute:` check is what restricts execution

### Rank Selection Quick Reference

| Syntax | Selects | Example |
|--------|---------|---------|
| `0` | Single rank | `uw.pprint("message")` |
| `slice(0, 4)` | Range of ranks | `uw.pprint(..., proc=slice(0, 4))` |
| `[0, 3, 7]` | Specific ranks | `uw.pprint(..., proc=[0, 3, 7])` |
| `'all'` or `None` | All ranks | `uw.pprint(..., proc='all')` |
| `'first'` | Rank 0 | `uw.pprint(..., proc='first')` |
| `'last'` | Highest rank | `uw.pprint(..., proc='last')` |
| `'even'` | Even ranks | `uw.pprint(..., proc='even')` |
| `'odd'` | Odd ranks | `uw.pprint(..., proc='odd')` |
| `'10%'` | First 10% of ranks | `uw.pprint(..., proc='10%')` |
| `lambda r: ...` | Custom function | `uw.pprint(..., proc=lambda r: r % 3 == 0)` |

### Common Collective Operations

These operations require **ALL ranks** to participate:

| Operation | Type | Example |
|-----------|------|---------|
| `var.stats()` | Statistics | Global min/max/mean |
| `solver.solve()` | Solver | All PETSc solvers |
| `mesh.write()`, `mesh.write_timestep()`, `swarm.save()` | I/O | Parallel HDF5 write |
| `var.rbf_interpolate()` | Interpolation | Radial basis functions |
| `swarm.migrate()` | Redistribution | Particle migration |

### Migration Checklist

- [ ] Replace `if uw.mpi.rank == 0: print(...)` with `uw.pprint(...)`
- [ ] Replace `if uw.mpi.rank == 0:` for visualization with `with uw.selective_ranks(0) as should_execute:` and `if should_execute:`
- [ ] Ensure collective operations are called on ALL ranks
- [ ] Test with `mpirun -np 2` and `mpirun -np 4`
- [ ] Check for deadlocks (script hangs = collective operation issue)

## Timing Output at Extreme Scale

`uw.timing.print_table()` ultimately calls PETSc's `PetscLogView`. At very
high CPU counts (≳1000 ranks), the **ASCII output path** can hang —
typically appearing as a job that completes its computation cleanly but
never exits. The CSV write path uses a different, less collective-heavy
strategy and avoids the issue:

```python
# Default — fine at small scale, can hang at ≳1000 ranks
uw.timing.print_table()
uw.timing.print_table("results.txt")

# Safe at any scale — recommended for HPC runs
uw.timing.print_table("results.csv")
```

The behaviour is in PETSc, not Underworld; choosing CSV at scale is the
recommended workaround. (Issue #134.)

## Summary

**Key Takeaways:**

1. **PETSc handles parallelism** - you write parallel-safe UW3 code
2. **Use `uw.pprint(..., proc=ranks)`** for output on specific ranks
3. **Use `with uw.selective_ranks(ranks) as should_execute:` and `if should_execute:`** for serial operations
4. **Collective operations must run on ALL ranks** - never inside rank conditionals
5. **Test with `mpirun -np N`** to catch issues early
6. **At ≳1000 ranks, write timing output as `.csv`** to avoid `PetscLogView` hangs

The parallel safety system makes parallel programming in Underworld3 safer and more intuitive - collective operations are evaluated on all ranks automatically, preventing common deadlock scenarios!
