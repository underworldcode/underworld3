# Swarm Integration Statistics: Accurate Spatial Statistics for Non-Uniform Particle Distributions

## Overview

When computing statistics (mean, standard deviation) from swarm particles, there is a critical distinction between:

1. **Particle-weighted statistics** (simple arithmetic): `swarm_var.array.mean()`, `swarm_var.array.std()`
2. **Space-weighted statistics** (integration-based): Using proxy variables with `uw.maths.Integral()`

This document explains when and how to use each approach, and the underlying physics.

## The Problem: Non-Uniform Particle Distributions

### Why Simple Arithmetic Statistics Are Insufficient

Swarm particles are typically non-uniformly distributed in space. When you compute a simple arithmetic mean:

```python
mean_value = swarm_var.array.mean()  # Simple arithmetic mean
```

This computes:
$$\bar{f} = \frac{1}{N} \sum_{i=1}^{N} f_i$$

where $f_i$ are particle values. This is **particle-weighted** - regions with more particles have more influence on the result.

**Example**: If you cluster 100 particles in the left half of the domain and only a few in the right half, the mean will be biased toward left-side values, even though the domain is evenly divided.

### What Users Actually Want

For most geoscience applications, you want the **spatial mean** - the true average value across the entire domain:

$$\bar{f}_{spatial} = \frac{\int_{\Omega} f(x) dV}{\int_{\Omega} dV}$$

This equally weights all spatial regions, regardless of particle density.

## The Solution: Proxy Variables with Integration

### How It Works

Underworld3 automatically creates **proxy mesh variables** when you create swarm variables with `proxy_degree > 0`:

```python
swarm = uw.swarm.Swarm(mesh)
swarm_var = uw.swarm.SwarmVariable(
    "temperature",
    swarm,
    1,
    proxy_degree=2  # Creates RBF-interpolated mesh proxy
)
swarm.populate(fill_param=3)
```

The proxy variable:
1. **Interpolates** particle values to mesh nodes using Radial Basis Functions (RBF)
2. Respects mesh structure (only uses local particles)
3. Can be **integrated** to compute spatial statistics

### Computing Spatial Mean

```python
# Step 1: Ensure proxy is up to date by accessing .sym
proxy = swarm_var.sym

# Step 2: Integrate proxy over domain
I_vol = uw.maths.Integral(mesh, fn=1.0)
I_f = uw.maths.Integral(mesh, fn=swarm_var.sym[0])

# Step 3: Compute spatial mean
volume = I_vol.evaluate()
spatial_mean = I_f.evaluate() / volume
```

### Computing Spatial Standard Deviation

```python
# Compute spatial variance: E[f²] - (E[f])²
I_vol = uw.maths.Integral(mesh, fn=1.0)
I_f = uw.maths.Integral(mesh, fn=swarm_var.sym[0])
I_f2 = uw.maths.Integral(mesh, fn=swarm_var.sym[0]**2)

volume = I_vol.evaluate()
mean_f = I_f.evaluate() / volume
mean_f2 = I_f2.evaluate() / volume

variance = mean_f2 - mean_f**2
spatial_std = np.sqrt(max(variance, 0.0))
```

## Complete Example

```python
import underworld3 as uw
import numpy as np

# Create mesh and swarm
mesh = uw.meshing.StructuredQuadBox(
    minCoords=(-1.0, -1.0),
    maxCoords=(1.0, 1.0),
    elementRes=(10, 10)
)
swarm = uw.swarm.Swarm(mesh)

# Create swarm variable with proxy BEFORE populating
T = uw.swarm.SwarmVariable(
    "Temperature",
    swarm,
    1,
    proxy_degree=2,  # RBF interpolation degree
    proxy_continuous=True
)

# Populate swarm
swarm.populate(fill_param=4)

# Set particle temperatures
T.data[:, 0] = 273.0 + 100.0 * (1.0 - (x**2 + y**2))

# ===== APPROACH 1: Particle-weighted statistics =====
particle_mean = T.array.mean()
particle_std = T.array.std()
print(f"Particle-weighted mean: {particle_mean:.2f} K")
print(f"Particle-weighted std:  {particle_std:.2f} K")

# ===== APPROACH 2: Space-weighted (integration) statistics =====
I_vol = uw.maths.Integral(mesh, fn=1.0)
I_T = uw.maths.Integral(mesh, fn=T.sym[0])
I_T2 = uw.maths.Integral(mesh, fn=T.sym[0]**2)

volume = I_vol.evaluate()
spatial_mean = I_T.evaluate() / volume
mean_T2 = I_T2.evaluate() / volume
spatial_variance = mean_T2 - spatial_mean**2
spatial_std = np.sqrt(max(spatial_variance, 0.0))

print(f"Spatial mean: {spatial_mean:.2f} K")
print(f"Spatial std:  {spatial_std:.2f} K")
```

## Proxy Degree Selection

Choose `proxy_degree` based on your needs:

| Degree | Smoothness | Speed | Accuracy | Best For |
|--------|-----------|-------|----------|----------|
| 0 | Piecewise constant | ⭐⭐⭐ Fast | Poor | Quick visualization |
| 1 | Linear | ⭐⭐ Medium | Good | Most use cases |
| 2 | Quadratic | ⭐ Slower | Very Good | Integration, derivatives |
| 3 | Cubic | ⭐ Slow | Excellent | High-precision derivatives |

**Recommendation**: Use `proxy_degree=2` for integration and statistics computations.

## Caveats and Limitations

### ⚠️ Swarm Variables Must Be Created Before Population

**CRITICAL**: You MUST create `SwarmVariable` before calling `swarm.populate()` or `swarm.add_particles_with_coordinates()`:

```python
# ✅ CORRECT
swarm = uw.swarm.Swarm(mesh)
var = uw.swarm.SwarmVariable("scalar", swarm, 1)  # Create FIRST
swarm.populate(fill_param=3)                        # Then populate

# ❌ WRONG - This will raise RuntimeError
swarm = uw.swarm.Swarm(mesh)
swarm.populate(fill_param=3)                        # Never do this first!
var = uw.swarm.SwarmVariable("scalar", swarm, 1)   # This will fail
```

### RBF Interpolation Artifacts

The RBF proxy variable may have interpolation artifacts:

1. **Edge effects**: Values near domain boundaries may be less accurate
2. **Extrapolation**: RBF doesn't extrapolate well outside particle cloud
3. **k-NN dependence**: Accuracy depends on number of nearest neighbors used

**Mitigation**:
- Ensure particles are distributed throughout domain
- Use `proxy_continuous=True` for smoother interpolation
- Increase `proxy_degree` for more accurate interpolation

### Integration Accuracy

`uw.maths.Integral()` uses Gauss-Legendre quadrature. Accuracy depends on:

1. **Mesh resolution**: Finer elements → higher accuracy
2. **Quadrature order**: Default is usually sufficient
3. **Proxy smoothness**: Higher `proxy_degree` → better approximation

For convergence studies, refine the mesh and increase `proxy_degree`.

### Performance Considerations

- RBF interpolation is **O(N log N)** where N = number of particles
- Integration scales with mesh resolution
- Accessing `.sym` triggers automatic proxy update (lazy evaluation)

For large swarms (>1M particles), integration-based statistics can be expensive.

## When to Use Each Approach

### Use Particle-Weighted Statistics When:

- ✅ You want particle-based statistics (e.g., outlier detection)
- ✅ You need fast computation
- ✅ You're analyzing particle population properties (density analysis)
- ✅ Particle distribution is intentionally non-uniform

### Use Integration-Based Statistics When:

- ✅ You want true spatial statistics
- ✅ You're comparing with mesh-based variables
- ✅ You need high accuracy
- ✅ Particles are meant to sample a continuous field
- ✅ You're computing derived quantities (gradients, divergences)

## API Reference

### Creating Swarm Variables with Proxy

```python
var = uw.swarm.SwarmVariable(
    name="variable_name",
    swarm=swarm_object,
    num_components=1,           # Scalar: 1, Vector: dim, Tensor: dim²
    proxy_degree=2,              # RBF interpolation degree (0-3)
    proxy_continuous=True,       # Continuous interpolation
    dtype=float64                # Data type
)
```

### Accessing Proxy for Integration

```python
# Access symbolic proxy (triggers lazy update if needed)
proxy = var.sym           # Full proxy variable
component = var.sym[0]    # Access specific component
derivative = var.sym[0].diff(mesh.N.x)  # Compute derivatives
```

### Integration Operations

```python
# Simple integration
integral = uw.maths.Integral(mesh, fn=var.sym[0]).evaluate()

# Weighted integration
weighted = uw.maths.Integral(mesh, fn=weight * var.sym[0]).evaluate()

# Vector operations
magnitude = uw.maths.Integral(mesh, fn=var.sym.norm()).evaluate()
```

## Testing and Validation

A comprehensive test suite validates integration with swarms:

**File**: `tests/test_0852_swarm_integration_statistics.py`

Tests cover:
- ✅ Uniform vs clustered particle distributions
- ✅ Arithmetic vs integration-based mean/std
- ✅ RBF interpolation accuracy
- ✅ Proxy variable creation and updates
- ✅ Constant field preservation
- ✅ Standard deviation computation

Run tests:
```bash
pixi run -e default pytest tests/test_0852_swarm_integration_statistics.py -v
```

## See Also

- **Proxy Variables**: See `swarm.py:651-735` for implementation
- **RBF Interpolation**: See `swarm.py:1025-1062` for `rbf_interpolate()` method
- **Integration**: See `maths/__init__.py` for `Integral` class
- **Examples**: `docs/examples/utilities/advanced/Ex_Integrals_on_Meshes.py`

## Future Improvements

### Currently Unimplemented

1. **Weighted RBF interpolation**: Using particle masses as weights
2. **Error estimation**: Quantifying proxy approximation error
3. **Adaptive proxy degree**: Automatic degree selection based on particle density
4. **GPU acceleration**: CUDA support for large swarms

### Planned Features

- Integration with uncertainty quantification framework
- Anisotropic RBF kernels for stretched domains
- Domain-aware RBF (respects boundaries)
- Streaming integration for very large swarms

## References

- Meshless methods and RBF interpolation: Wendland, H. (2005)
- Gauss-Legendre quadrature: Abramowitz & Stegun
- PETSc integration: Collective operations in Integral class

---

**Last Updated**: 2025-10-25
**Status**: Complete and tested (7/7 tests passing)
**Related Issues**: TODO - test how integration works for swarmVariables ✅ COMPLETE
