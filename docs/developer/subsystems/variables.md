---
title: "Variable System"
---

# Variable System Documentation

```{note} Documentation In Progress
This section is under development. Variable system documentation is being migrated from existing docstrings and implementation notes.
```

## Overview

The variable system in Underworld3 provides field storage and manipulation for both mesh-based and particle-based data.

## Components

- **MeshVariable**: Fields defined on mesh nodes or elements
- **SwarmVariable**: Fields defined on particle swarms
- **Proxy Variables**: Mesh representations of swarm data via RBF interpolation

## Global Statistics Methods

### UnitAwareArray Global Operations

For `UnitAwareArray` objects (such as `mesh.X.coords`), a suite of global statistics methods is available that perform MPI-aware reductions across all ranks while preserving units:

#### Available Methods

| Method | Description | Units | Example |
|--------|-------------|-------|---------|
| `global_max()` | Maximum value across all ranks | Original units | `coords[:, 1].global_max()` |
| `global_min()` | Minimum value across all ranks | Original units | `coords[:, 1].global_min()` |
| `global_sum()` | Sum across all ranks | Original units | `coords[:, 1].global_sum()` |
| `global_mean()` | Mean across all ranks | Original units | `coords[:, 1].global_mean()` |
| `global_var()` | Variance across all ranks | Units squared | `coords[:, 1].global_var()` |
| `global_std()` | Standard deviation across all ranks | Original units | `coords[:, 1].global_std()` |
| `global_norm()` | L2 norm across all ranks | Original units | `coords[:, 1].global_norm()` |
| `global_size()` | Total element count across all ranks | Dimensionless (int) | `coords.global_size()` |
| `global_rms()` | Root mean square across all ranks | Original units | `coords[:, 1].global_rms()` |

#### Usage Example

```python
import underworld3 as uw

mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1000.0, 2900.0),
    cellSize=250.0,
    units="km",
)

# Get y-coordinates (UnitAwareArray)
y_coords = mesh.X.coords[:, 1]

# Local operations (different on each rank)
local_max = y_coords.max()  # Only this rank's data

# Global operations (same on all ranks)
global_max = y_coords.global_max()  # Maximum across all ranks
global_rms = y_coords.global_rms()  # RMS across all ranks

print(f"Global maximum: {global_max}")  # Same value on all ranks
print(f"Global RMS: {global_rms}")      # Same value on all ranks
```

#### Relationships Between Methods

```python
# RMS is defined as:
rms = global_norm() / sqrt(global_size())

# These are equivalent:
rms1 = coords.global_rms()
rms2 = coords.global_norm() / np.sqrt(coords.global_size())
```

### ```{warning} Important Caveats
**These statistics operate on nodal/element values, NOT domain-integrated quantities.**

For arrays associated with mesh variables:

1. **Non-uniform element sizes**: If mesh elements have varying sizes, `global_mean()` is NOT equivalent to the domain-averaged value. Element-wise statistics weight all nodes equally, regardless of the volume they represent.

2. **Boundary treatment**: Boundary nodes may be weighted differently than interior nodes depending on the finite element formulation.

3. **True domain averages require integration**:
   ```python
   # Element-wise mean (NOT domain-integrated)
   nodal_mean = temperature.array.global_mean()  # ❌ Not physical average

   # True domain-averaged temperature (requires integration)
   domain_avg = uw.maths.Integral(temperature, mesh).evaluate() / mesh.volume  # ✅ Correct
   ```

4. **When element-wise statistics are appropriate**:
   - Checking data ranges for validation/debugging
   - Computing norms for convergence testing
   - Monitoring min/max values for CFL conditions
   - Quick diagnostics during development

5. **When domain integration is required**:
   - Computing physical averages (mean temperature, velocity, etc.)
   - Evaluating bulk properties (total kinetic energy, etc.)
   - Post-processing for publication-quality results
   - Comparing with analytical solutions

**TODO**: Ensure `uw.maths.Integral` correctly handles units in all cases.
```

### MeshVariable Statistics (PETSc-based)

`MeshVariable` objects have their own statistics methods that work through PETSc vectors:

```python
temperature = uw.discretisation.MeshVariable("T", mesh, 1, units="K")

# These use PETSc global reductions (MPI-aware)
t_max = temperature.max()   # Tuple for multi-component
t_min = temperature.min()
t_mean = temperature.mean()
t_sum = temperature.sum()
```

**Note**: MeshVariable statistics methods return tuples for multi-component variables and use PETSc's internal MPI operations. They are **NOT** the same as the UnitAwareArray global methods.

### SwarmVariable Statistics (Selective Global Operations)

`SwarmVariable` objects provide **limited** global statistics methods suitable for particle-based data:

#### Available Methods

| Method | Description | Units | Example |
|--------|-------------|-------|---------|
| `global_max()` | Maximum value across all ranks | Original units | `temp_swarm.global_max()` |
| `global_min()` | Minimum value across all ranks | Original units | `temp_swarm.global_min()` |
| `global_sum()` | Sum across all ranks | Original units | `temp_swarm.global_sum()` |
| `global_norm()` | L2 norm across all ranks | Original units | `temp_swarm.global_norm()` |
| `global_size()` | Total particle count across all ranks | Dimensionless (int) | `temp_swarm.global_size()` |

#### Usage Example

```python
import underworld3 as uw

# Create swarm and variable
swarm = uw.swarm.Swarm(mesh)
temperature = swarm.add_variable("T", 1, units="K")

# Populate and initialize
swarm.populate(fill_param=3)
with uw.synchronised_array_update():
    temperature.data[:, 0] = 300.0 + np.random.randn(swarm.local_size) * 10.0

# Global operations (same result on all ranks)
max_temp = temperature.global_max()
min_temp = temperature.global_min()
total_particles = temperature.global_size()

print(f"Temperature range: {min_temp} to {max_temp}")
print(f"Total particles: {total_particles}")
```

#### Important: Why No global_mean() or global_rms()?

**SwarmVariable does NOT provide `global_mean()`, `global_rms()`, `global_var()`, or `global_std()`** because these operations are **physically meaningless** for particle data.

**Reason**: Particles are **non-uniformly distributed** in the domain due to:
- Initial population strategy (controlled by `fill_param`)
- Advection concentrating particles in flow regions
- Material boundaries and compositional interfaces
- Migration and redistribution patterns

**Consequence**: A simple mean across particle values does NOT represent a domain-averaged quantity. Regions with more particles would dominate the mean, regardless of their physical volume.

#### Computing True Domain Averages from Swarm Data

To obtain physically meaningful domain-averaged values, use the **proxy mesh variable** with proper domain integration:

```python
# Create swarm variable with mesh proxy
temperature = swarm.add_variable("T", 1, proxy_degree=1, units="K")

# The proxy creates a mesh field representation via RBF interpolation
# Use domain integration on the proxy field:
volume = uw.maths.Integral(1.0, mesh).evaluate()
temp_integral = uw.maths.Integral(temperature.sym, mesh).evaluate()
domain_avg_temp = temp_integral / volume

# This gives the true volume-weighted domain average
print(f"Domain-averaged temperature: {domain_avg_temp}")
```

**Key Distinction**:
- ❌ `temperature.global_mean()` - **NOT PROVIDED** (would be misleading)
- ✅ `uw.maths.Integral(temperature.sym, mesh).evaluate() / mesh.volume` - Correct domain average

## Related Documentation

- [Data Access Patterns](data-access.md) - How to read and write variable data
- [NDArray System](../UW3_Developers_NDArrays.md) - Array interface details
- [Swarm System](swarm-system.md) - Particle-based variables
- [Units System](../../beginner/tutorials/12-Units_System.ipynb) - Working with physical units
- [Parallel Computing](../../advanced/parallel-computing.md) - MPI patterns and best practices