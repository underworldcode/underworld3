# Parallel Safety Analysis for Underworld3

## Problem Statement

User code frequently uses `if uw.mpi.rank == 0:` blocks for:
- Printing output
- Visualization (matplotlib, pyvista)
- File operations

**The Danger**: If a collective operation appears inside a rank-conditional block, other ranks hang waiting for coordination that never comes.

## Common Problematic Patterns

### 1. Collective Operations Inside Rank Conditionals

```python
# DANGEROUS - collective operation only on rank 0
if uw.mpi.rank == 0:
    stats = mesh_var.stats()  # Collective! Other ranks hang
    print(f"Stats: {stats}")
```

### 2. Mesh/Geometry Operations

```python
# FOUND IN EXAMPLES - gmsh operations
if uw.mpi.rank == 0:
    gmsh.write("mesh.msh")  # May be collective
```

### 3. Variable Data Access (Actually Safe)

```python
# SAFE - .data accesses local partition
if uw.mpi.rank == 0:
    print(f"First 10 values: {var.data[0:10]}")  # OK - local data only
```

## Analysis Results

### Files Checked
- 77 Python files with `if uw.mpi.rank == 0:` blocks
- Examples, tests, and core source code

### Issues Found

**Collective operations in rank conditionals:**
1. **gmsh.write()** - Found in 8 example files
   - Mesh generation operations may be collective
   - Examples: Ex_Stokes_Ellipse_Cartesian.py, Ex_Compression_Example.py

2. **File operations** - Variable `.write()` methods
   - Some HDF5 operations are collective

### Safe Patterns (No Issues)

Most code is actually **safe**:
- Simple print statements
- Local data access (`.data[...]`)
- Visualization setup (matplotlib imports)
- File writes that are properly guarded

## User Experience Issues

### Problem for Notebook Users

Users write natural serial code in notebooks:

```python
# In a notebook - looks fine in serial
import matplotlib.pyplot as plt

# ... run solver ...

# Visualize results
plt.figure()
plt.plot(x_coords, temperature.data[:, 0])  # Uses .data - collective?
plt.show()
```

**In parallel**: If `.data` property or any evaluation triggers collective operations, this breaks.

### Current Tutorial Notebooks

**Good news**: Beginner tutorials (`docs/beginner/tutorials/`) contain **zero** `if uw.mpi.rank == 0:` conditionals.

They appear to be written for serial execution, which is appropriate for learning.

## Proposed Solutions

### Option 1: Parallel-Safe Output Context Manager

```python
class ParallelSafeOutput:
    """Execute code on all ranks but suppress output on rank != 0"""
    
    def __enter__(self):
        if uw.mpi.rank != 0:
            self._stdout = sys.stdout
            sys.stdout = io.StringIO()  # Redirect to nowhere
        return self
    
    def __exit__(self, *args):
        if uw.mpi.rank != 0:
            sys.stdout = self._stdout

# Usage
with uw.parallel_print():
    # This runs on ALL ranks (safe for collective ops)
    # But only rank 0 sees output
    stats = mesh_var.stats()  # Collective - all ranks participate
    print(f"Stats: {stats}")   # Only rank 0 prints
```

### Option 2: Smart Printing Function

```python
def print0(*args, **kwargs):
    """Print only on rank 0, but always safe for collective args"""
    # Evaluate all arguments on all ranks (collective ops happen)
    if uw.mpi.rank == 0:
        print(*args, **kwargs)

# Usage
print0(f"Stats: {mesh_var.stats()}")  # Safe - stats() runs on all ranks
```

### Option 3: Lazy Evaluation Wrapper

```python
def rank0_eval(func):
    """Decorator that makes collective operations safe in rank conditionals"""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)  # Execute on all ranks
        return result if uw.mpi.rank == 0 else None
    return wrapper

# Apply to collective methods
MeshVariable.stats = rank0_eval(MeshVariable.stats)
```

## Recommendations

### For Normal Users (Notebooks/Scripts)

1. **Add to UW3 API:**
   ```python
   uw.print0()  # Print on rank 0
   uw.parallel_print()  # Context manager for safe output
   ```

2. **Documentation pattern:**
   ```python
   # Recommended pattern for parallel-safe visualization
   with uw.parallel_print():
       # All ranks execute, only rank 0 shows output
       fig, ax = plt.subplots()
       ax.plot(coords, var.data[:, 0])
       plt.savefig("output.png")
   ```

3. **Tutorial updates:**
   - Add "Parallel Safety" section to advanced docs
   - Show parallel-safe alternatives to common patterns
   - Explain collective vs local operations

### For Advanced Developers

1. **Audit collective operations:**
   - Mark methods that are collective in docstrings
   - Consider adding `@collective_operation` decorator for documentation

2. **Testing:**
   - Run all tutorials with `mpirun -np 2` to catch issues
   - Add parallel tests for common user patterns

### Quick Win: Identify Collective Operations

Add documentation to identify which operations are collective:

```python
class MeshVariable:
    def stats(self):
        """
        Calculate statistics.
        
        .. collective::
            This operation requires all MPI ranks to participate.
        """
        pass
```

## Next Steps

1. Implement `uw.print0()` and `uw.parallel_print()` utilities
2. Update advanced documentation with parallel safety guide  
3. Audit and document collective operations in API
4. Test tutorials in parallel (mpirun -np 2/4)
5. Create examples showing parallel-safe visualization patterns