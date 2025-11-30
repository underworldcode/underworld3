# Parallel Safety System Design

## Architecture Overview

A coordinated system of decorators and context managers that makes parallel-safe code natural to write.

## Core Components

### 1. MPI Context Manager (`rank_conditional`)

```python
# In src/underworld3/mpi.py

class MPIContext:
    def __init__(self):
        self._rank = MPI.COMM_WORLD.Get_rank()
        self._size = MPI.COMM_WORLD.Get_size()
        self._conditional_rank = None  # Track which rank is "active"
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def size(self):
        return self._size
    
    @contextmanager
    def rank_conditional(self, rank=0):
        """
        Context for rank-specific code that safely handles collective operations.
        
        Usage:
            with uw.mpi.rank_conditional(0):
                # Code here runs on ALL ranks (collective ops safe)
                # But only rank 0 sees output/returns values
                stats = var.stats()  # All ranks participate
                print(stats)         # Only rank 0 prints
        """
        old_conditional = self._conditional_rank
        self._conditional_rank = rank
        try:
            yield (self._rank == rank)
        finally:
            self._conditional_rank = old_conditional
    
    @property
    def is_conditional(self):
        """Check if currently inside a rank conditional"""
        return self._conditional_rank is not None
    
    @property
    def is_active_rank(self):
        """Check if current rank is the active one in conditional"""
        if self._conditional_rank is None:
            return True  # No conditional, all ranks active
        return self._rank == self._conditional_rank

# Global instance
mpi = MPIContext()
```

### 2. Collective Operation Decorator

```python
# In src/underworld3/utilities/parallel.py

def collective_operation(func):
    """
    Mark and validate collective operations.
    
    - Adds documentation warning
    - In debug mode: warns if called in old-style rank conditional
    - Always executes on all ranks, returns None on non-active ranks
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Check for dangerous old-style pattern
        if uw.mpi.is_conditional and not hasattr(wrapper, '_conditional_safe'):
            # We're in new-style rank_conditional - safe!
            result = func(self, *args, **kwargs)
            # Suppress return on non-active ranks
            return result if uw.mpi.is_active_rank else None
        else:
            # Normal execution
            return func(self, *args, **kwargs)
    
    # Mark as collective for documentation
    wrapper._is_collective = True
    
    # Update docstring
    if func.__doc__:
        wrapper.__doc__ = func.__doc__ + """
        
        .. warning::
            **Collective Operation**: All MPI ranks must execute this method.
            Use ``with uw.mpi.rank_conditional():`` for safe rank-specific code.
        """
    
    return wrapper
```

### 3. Print Functions Using Context

```python
# In src/underworld3/utilities/parallel.py

def print0(*args, **kwargs):
    """
    Print only on rank 0, safe for collective arguments.
    
    This function evaluates all arguments on ALL ranks (so collective
    operations work correctly), but only prints on rank 0.
    
    Examples:
        # Safe - stats() runs on all ranks
        uw.print0(f"Max value: {var.stats()}")
        
        # Equivalent to:
        with uw.mpi.rank_conditional(0):
            print(f"Max value: {var.stats()}")
    """
    with uw.mpi.rank_conditional(0) as is_active:
        if is_active:
            print(*args, **kwargs)

def printf(*args, **kwargs):
    """
    Print with automatic rank prefix for debugging.
    
    Usage:
        uw.printf("Processing element", elem_id)
        # Rank 0: [Rank 0] Processing element 42
        # Rank 1: [Rank 1] Processing element 13
    """
    rank = uw.mpi.rank
    print(f"[Rank {rank}]", *args, **kwargs)
```

### 4. Serial-Only Context Manager

```python
# In src/underworld3/utilities/parallel.py

@contextmanager
def serial_only(rank=0):
    """
    Context for operations that MUST be serial (gmsh, matplotlib, pyvista).
    
    - Only executes on specified rank
    - Other ranks skip entirely
    - Use for operations that can't be made collective
    
    Usage:
        with uw.serial_only(0):
            # Only rank 0 executes this block
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(x, y)
            plt.savefig("plot.png")
        # All ranks sync here
    """
    if uw.mpi.rank == rank:
        yield True
    else:
        yield False
    
    # Barrier to ensure rank 0 finishes before others continue
    uw.mpi.barrier()

# Convenience aliases
rank0_only = partial(serial_only, rank=0)
```

### 5. Visualization Wrappers

```python
# In src/underworld3/utilities/parallel.py

class SerialVisualization:
    """Wrapper to ensure visualization is always serial."""
    
    @staticmethod
    @contextmanager
    def matplotlib():
        """Safe matplotlib context"""
        with uw.serial_only(0) as is_active:
            if is_active:
                import matplotlib.pyplot as plt
                yield plt
            else:
                yield None
    
    @staticmethod  
    @contextmanager
    def pyvista():
        """Safe pyvista context"""
        with uw.serial_only(0) as is_active:
            if is_active:
                import pyvista as pv
                yield pv
            else:
                yield None
    
    @staticmethod
    @contextmanager
    def gmsh():
        """Safe gmsh context"""
        with uw.serial_only(0) as is_active:
            if is_active:
                import gmsh
                yield gmsh
            else:
                yield None

# Add to uw namespace
visualization = SerialVisualization()
```

## Usage Examples

### Example 1: Collective Operation with Output

```python
# OLD (dangerous in parallel)
if uw.mpi.rank == 0:
    stats = var.stats()  # HANGS - other ranks wait forever
    print(f"Stats: {stats}")

# NEW (safe with context)
with uw.mpi.rank_conditional(0):
    stats = var.stats()  # All ranks execute
    print(f"Stats: {stats}")  # Only rank 0 prints

# NEW (shortcut)
uw.print0(f"Stats: {var.stats()}")  # Even simpler!
```

### Example 2: Visualization (Serial Only)

```python
# Matplotlib - serial only
with uw.visualization.matplotlib() as plt:
    if plt:  # Only rank 0
        fig, ax = plt.subplots()
        ax.plot(coords, temperature.data[:, 0])
        plt.savefig("temp.png")
# All ranks sync after visualization

# Pyvista - serial only  
with uw.visualization.pyvista() as pv:
    if pv:  # Only rank 0
        plotter = pv.Plotter()
        plotter.add_mesh(mesh.pyvista_mesh)
        plotter.show()
```

### Example 3: Mixed Operations

```python
# Timestep loop with mixed operations
for step in range(n_steps):
    # Solve - collective
    stokes.solve()
    
    # Stats - collective, print on rank 0
    uw.print0(f"Step {step}, max velocity: {v.stats()}")
    
    # Checkpoint - collective
    if step % 10 == 0:
        mesh.save(f"checkpoint_{step}.h5")
    
    # Visualization - serial only on rank 0
    if step % 5 == 0:
        with uw.visualization.matplotlib() as plt:
            if plt:
                plot_velocity_field(step)
```

### Example 4: Gmsh Mesh Generation

```python
# Mesh generation - serial only
with uw.visualization.gmsh() as gmsh:
    if gmsh:
        gmsh.initialize()
        # ... build geometry ...
        gmsh.write("mesh.msh")
        gmsh.finalize()

# Load on all ranks - collective
mesh = uw.discretisation.Mesh("mesh.msh")
```

## API Summary

```python
# Context managers
with uw.mpi.rank_conditional(0):      # Safe for collective ops
with uw.serial_only(0):                # Serial only, with barrier
with uw.visualization.matplotlib():    # Safe matplotlib
with uw.visualization.pyvista():       # Safe pyvista
with uw.visualization.gmsh():          # Safe gmsh

# Print functions
uw.print0("message")                   # Print on rank 0, safe for collective args
uw.printf("debug")                     # Print with rank prefix

# Decorator
@collective_operation                  # Mark collective methods
def stats(self): ...

# Properties
uw.mpi.is_conditional                  # Inside rank_conditional?
uw.mpi.is_active_rank                  # Current rank is active?
```

## Implementation Plan

### Phase 1: Core Infrastructure
1. Add `MPIContext` class with `rank_conditional()` to `mpi.py`
2. Add `collective_operation` decorator to `utilities/parallel.py`
3. Add `print0()` and `printf()` functions

### Phase 2: Serial Wrappers
4. Add `serial_only()` context manager
5. Add `SerialVisualization` class with matplotlib/pyvista/gmsh wrappers

### Phase 3: Apply to Codebase
6. Add `@collective_operation` to collective methods:
   - `MeshVariable.stats()`
   - `MeshVariable.rbf_interpolate()`
   - `Mesh.save()` / `Mesh.load()`
   - Solver methods
7. Update examples to use new patterns
8. Add tests for parallel safety

### Phase 4: Documentation
9. Update advanced docs with parallel safety guide
10. Add examples showing each pattern
11. Migration guide for updating old code

## Benefits

1. **Safety**: Collective operations work correctly even in rank-specific code
2. **Clarity**: Code clearly shows what's collective vs serial
3. **Simplicity**: `uw.print0()` is easier than manual rank checks
4. **Correctness**: Visualization guaranteed to be serial with automatic barriers
5. **Documentation**: `@collective_operation` makes collective methods obvious

## Migration Path

### Old Code
```python
if uw.mpi.rank == 0:
    stats = var.stats()
    print(stats)
    plt.plot(x, y)
```

### New Code (works correctly in parallel)
```python
# Method 1: Context manager
with uw.mpi.rank_conditional(0):
    stats = var.stats()
    print(stats)

# Method 2: Shortcut
uw.print0(f"Stats: {var.stats()}")

# Visualization - separate serial block
with uw.visualization.matplotlib() as plt:
    if plt:
        plt.plot(x, y)
```