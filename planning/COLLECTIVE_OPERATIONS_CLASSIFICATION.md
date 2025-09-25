# Collective Operations Classification for Underworld3

## Key Insight from Working Code

**The codebase currently works well in parallel** - this tells us which operations are actually collective vs local.

## Definite Collective Operations

### Solver Operations (Always Collective)
```python
# All solve operations require all ranks
stokes.solve()           # ✓ Collective
poisson.solve()          # ✓ Collective
advdiff.solve()          # ✓ Collective

# Any solver method
solver.update()          # ✓ Collective
solver.reset()           # ✓ Collective
```

### Linear Algebra Operations (PETSc Collective)
```python
# Dot products - collective
vec.dot(other)           # ✓ Collective
vec.norm()               # ✓ Collective
vec.sum()                # ✓ Collective

# Matrix operations - collective
mat.mult(vec, result)    # ✓ Collective
mat.multTranspose()      # ✓ Collective

# DM operations
dm.localToGlobal()       # ✓ Collective
dm.globalToLocal()       # ✓ Collective
```

### Statistical/Reduction Operations
```python
# Anything computing global statistics
var.stats()              # ✓ Collective (computes global min/max/mean)
var.rbf_interpolate()    # ✓ Collective (RBF across all data)

# These require all ranks to contribute
mesh.integrate()         # ✓ Collective
swarm.populate()         # ✓ Collective (if redistributing)
```

### Data Modification with Sync
```python
# Anything changing distributed data requiring sync
var.data[...] = values   # ✓ Collective (triggers PETSc sync)
mesh.update_lvec()       # ✓ Collective (sync local/global vectors)
swarm.migrate()          # ✓ Collective (redistribute particles)

# File I/O (parallel HDF5)
mesh.save("file.h5")     # ✓ Collective
var.save("file.h5")      # ✓ Collective
mesh.load("file.h5")     # ✓ Collective
```

## Definite Local Operations

### Data Access (Read-Only)
```python
# Accessing local partition - NOT collective
var.data                 # ✗ NOT collective (local view)
var.data.shape           # ✗ NOT collective (local shape)
var.data.max()           # ✗ NOT collective (local max)
var.data.min()           # ✗ NOT collective (local min)
var.data[0:10]           # ✗ NOT collective (local slice)

# Array properties
var.data.dtype           # ✗ NOT collective
var.data.size            # ✗ NOT collective (local size)
```

### Output Operations
```python
# Printing - NOT collective
print(...)               # ✗ NOT collective
sys.stdout.write()       # ✗ NOT collective

# Visualization (when properly guarded)
plt.plot()               # ✗ NOT collective (serial library)
plt.savefig()            # ✗ NOT collective
```

### Local Computations
```python
# Operations on local data only
numpy.max(var.data)      # ✗ NOT collective (operates on local partition)
var.data * 2             # ✗ NOT collective (local array operation)
var.data + other.data    # ✗ NOT collective (local arrays)
```

## Context-Dependent Operations

### PETSc Vector Operations
**Rule**: If it accesses the PETSc Vec object directly, it's probably collective.

```python
# Collective - global operations
var.vec.norm()           # ✓ Collective
var.vec.dot()            # ✓ Collective
var.vec.max()            # ✓ Collective (global max)

# Local - just accessing local data
var.vec.array            # ✗ NOT collective (local array view)
var.vec.getArray()       # ✗ NOT collective
```

### Swarm Operations
```python
# Collective - redistribution
swarm.migrate()          # ✓ Collective
swarm.add_particles()    # ✓ Collective (if triggers migration)

# Local - accessing local particles
swarm.data               # ✗ NOT collective (local particles)
swarm.particle_count     # ✗ NOT collective (local count)

# Collective - global count
swarm.get_global_count() # ✓ Collective
```

## How to Use PETSc Documentation

PETSc documents collective operations. Map to UW3:

### PETSc → UW3 Mapping

```python
# PETSc Collective → UW3 Method
VecNorm()           → var.vec.norm()          # ✓ Collective
VecDot()            → var.vec.dot()           # ✓ Collective
VecMax()            → var.vec.max()           # ✓ Collective
MatMult()           → mat.mult()              # ✓ Collective
KSPSolve()          → solver.solve()          # ✓ Collective
DMLocalToGlobal()   → mesh.dm.localToGlobal() # ✓ Collective
DMGlobalToLocal()   → mesh.dm.globalToLocal() # ✓ Collective

# PETSc Local → UW3 Method  
VecGetArray()       → var.vec.array           # ✗ NOT collective
VecGetLocalSize()   → var.vec.getLocalSize()  # ✗ NOT collective
```

**Reference**: Check PETSc function documentation - if marked "Collective", it's collective in UW3 too.

## The Sync Rule

**Key Principle**: Data access is NOT collective, data *modification* IS collective.

```python
# Reading - NOT collective
value = var.data[10]     # ✗ Local read

# Writing - COLLECTIVE (triggers sync)
var.data[10] = value     # ✓ Collective (PETSc sync required)

# Why? Writing requires:
# 1. Update local values
# 2. Sync to global vector
# 3. Update ghost values on other ranks
```

### NDArray_With_Callback Behavior

```python
# The callback makes writes collective
var.data[...] = values   # Triggers pack_to_petsc() → collective

# Reads are local
values = var.data[...]   # Just reads local partition → not collective
```

## Building the Registry

### Auto-Mark from PETSc Patterns

```python
import petsc4py.PETSc as PETSc

# Known PETSc collective operations (from docs)
PETSC_COLLECTIVE_OPS = {
    # Vec operations
    'norm', 'dot', 'tdot', 'max', 'min', 'sum',
    'maxpy', 'aypx', 'axpy', 'axpby',
    
    # Mat operations  
    'mult', 'multAdd', 'multTranspose', 'solve',
    
    # DM operations
    'localToGlobal', 'globalToLocal',
    'createGlobalVec', 'createLocalVec',
    
    # KSP/SNES operations
    'solve', 'setUp', 'setFromOptions',
    
    # I/O operations
    'view', 'load',
}

# Map to UW3
UW3_COLLECTIVE_OPS = {
    # From PETSc wrapping
    'MeshVariable.stats',      # Uses vec.max(), vec.min() 
    'MeshVariable.norm',       # Uses vec.norm()
    'Solver.solve',            # Uses KSP.solve() or SNES.solve()
    
    # From explicit implementation
    'Swarm.migrate',           # Redistributes particles
    'Swarm.rbf_interpolate',   # RBF across all data
    
    # From sync requirements
    'MeshVariable.data.__setitem__',  # Writing triggers sync
}
```

### Check Against Working Code

```python
def validate_classification():
    """Validate using known-working code patterns."""
    
    # Scan for operations in rank conditionals (should NOT be collective)
    found_in_conditionals = scan_rank_conditionals()
    
    # Check for conflicts
    conflicts = UW3_COLLECTIVE_OPS & found_in_conditionals
    
    if conflicts:
        print("⚠️  These are marked collective but found in rank conditionals:")
        for op in conflicts:
            print(f"  - {op}")
            print(f"    Either: Code is buggy OR classification is wrong")
    
    # Operations in conditionals are definitely NOT collective
    definitely_not_collective = found_in_conditionals - UW3_COLLECTIVE_OPS
    
    print("✓ Confirmed NOT collective (found in rank conditionals):")
    for op in definitely_not_collective:
        print(f"  - {op}")
```

## Practical Guidelines for Developers

### When Adding New UW3 Methods

**Ask these questions:**

1. **Does it call PETSc collective operations?** → Mark collective
2. **Does it compute global statistics?** → Mark collective  
3. **Does it modify distributed data?** → Mark collective
4. **Does it only read local data?** → NOT collective

### Examples

```python
# New method - global maximum
def global_max(self):
    return self.vec.max()  # PETSc Vec.max() is collective
    # → Mark @collective_operation

# New method - local maximum
def local_max(self):
    return self.data.max()  # NumPy on local array
    # → NOT collective

# New method - update with sync
def update_values(self, values):
    self.data[...] = values  # Triggers pack → collective
    # → Mark @collective_operation

# New method - query property
def get_local_size(self):
    return self.data.shape[0]  # Just local info
    # → NOT collective
```

## Testing Strategy

### Parallel Test for Collective Classification

```python
def test_collective_classification():
    """Test operations to verify collective vs local."""
    
    # Test 1: Operations that SHOULD work on single rank
    with uw.selective_ranks(0):
        # These should NOT raise errors (not collective)
        shape = var.data.shape
        local_max = var.data.max()
        print(f"Local: {local_max}")
    
    # Test 2: Operations that SHOULD fail on single rank
    try:
        with uw.selective_ranks(0):
            stats = var.stats()  # Should raise CollectiveOperationError
        assert False, "Should have raised error"
    except CollectiveOperationError:
        pass  # Expected
    
    # Test 3: Run with mpirun -np 2
    # If it hangs, operation was collective but not marked
```

### Add to CI/CD

```bash
# Test parallel safety with 2 and 4 processes
mpirun -np 2 pytest tests/test_collective_ops.py
mpirun -np 4 pytest tests/test_collective_ops.py
```

## Summary Table

| Operation Type | Collective? | Example |
|---------------|-------------|---------|
| Solver.solve() | ✓ Yes | `stokes.solve()` |
| Vec dot/norm | ✓ Yes | `vec.norm()`, `vec.dot()` |
| Global stats | ✓ Yes | `var.stats()` |
| Data write | ✓ Yes | `var.data[...] = x` |
| Migration | ✓ Yes | `swarm.migrate()` |
| File I/O | ✓ Yes | `mesh.save()` |
| Data read | ✗ No | `var.data[...]` |
| Local stats | ✗ No | `var.data.max()` |
| Print/output | ✗ No | `print()` |
| Visualization | ✗ No | `plt.plot()` |

## Implementation Priority

1. **Mark obvious collective operations** with `@collective_operation`
   - All solve methods
   - All PETSc Vec collective ops
   - All data modification methods

2. **Validate against working code** 
   - Scan rank conditionals
   - Identify conflicts
   - Fix bugs or reclassify

3. **Build comprehensive registry**
   - From PETSc documentation
   - From code analysis
   - From testing

4. **Document in docstrings**
   - Clear "Collective" vs "Local" markers
   - Link to PETSc docs where relevant