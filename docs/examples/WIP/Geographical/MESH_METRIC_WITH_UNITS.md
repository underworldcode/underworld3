# Mesh Metric with Units - Understanding the Issue

## The Problem

In your original workflow (line 196-207), you have:

```python
with cs_mesh.access(H):
    H.data[:, 0] = uw.function.evalf(
        sympy.Piecewise(
            (mesh_adapation_parameter, fault_distance.sym[0] < cs_mesh.get_min_radius() * 33),
            (mesh_adapation_parameter, ((1 - r) * 6370) < 1),
            (100, True),
        ),
        H.coords,
    )
```

Where:
- `mesh_adapation_parameter = 6.6e6 * (mesh_k_elts/100)` (typical value: 6.6e6)
- This is used for mesh adaptation via `uw.adaptivity.mesh_adapt_meshVar(cs_mesh, H, Metric)`

**The question**: What are the units of `H`? How does it work with a GEOGRAPHIC mesh that has natural units (km)?

---

## Understanding Mesh Adaptation Metric

### What is the Metric `H`?

The metric `H` is a **target element size field**. It specifies the desired element size at each point in the mesh. The mesh adaptation algorithm tries to create elements with sizes close to this target.

### Original Spherical Mesh (Dimensionless)

In your original `RegionalSphericalBox`:
- Coordinates are **normalized**: radius goes from ~0.94 to 1.0
- Distances are **dimensionless**
- `H` values like `6.6e6` are **dimensionless** "relative size" parameters
- The actual meaning is: `H` ∝ 1/desired_element_density

### Geographic Mesh (With Units)

With `RegionalGeographicBox`:
- Coordinates have **natural units**: km for depth, degrees for lon/lat
- Cartesian coordinates are in **km**
- The mesh metric should be in **physical units**

---

## Solution Approaches

### Approach 1: Dimensionless Metric (Current Pattern)

**Keep using dimensionless values** - the adaptation algorithm doesn't care about units, only relative sizes.

```python
# Create metric field (dimensionless)
H = uw.discretisation.MeshVariable("H", mesh, 1)
Metric = uw.discretisation.MeshVariable("M", mesh, 1, degree=1)

# Fault distance in km
fault_distance_km = ...  # Computed as before

# Set metric based on conditions
# Use dimensionless values like before
mesh_adaptation_parameter = 6.6e6

import underworld3.function
H_values = underworld3.function.evaluate(
    sympy.Piecewise(
        (mesh_adaptation_parameter, fault_distance_km < 33),  # Near faults
        (mesh_adaptation_parameter, depth < 1),               # Near surface
        (100, True),                                          # Elsewhere
    ),
    H.coords,
)
H.array[:, 0] = H_values.reshape(-1, 1)
```

**Why this works**: The adaptation algorithm uses relative values. Whether your coordinates are in km or normalized doesn't matter - only the ratio of metric values matters.

---

### Approach 2: Physical Units for Metric

**Interpret `H` as a physical length scale** (in km).

```python
# Metric in km - target element size
H = uw.discretisation.MeshVariable("H", mesh, 1)

# Typical crustal element size: 10 km
# Near faults: 2 km (finer)
# Away from faults: 20 km (coarser)

H_values = underworld3.function.evaluate(
    sympy.Piecewise(
        (2.0, fault_distance_km < 33),    # 2 km elements near faults
        (2.0, depth < 1),                  # 2 km elements near surface
        (20.0, True),                      # 20 km elements elsewhere
    ),
    H.coords,
)
H.array[:, 0] = H_values.reshape(-1, 1)
```

**Advantage**: Clear physical meaning - you specify actual target element sizes.

**Question**: Does `mesh_adapt_meshVar` expect physical or dimensionless metrics?

---

### Approach 3: Hybrid (Recommended)

**Use physical reasoning but dimensionless values** - calibrate based on your target element count.

The relationship between metric value and element count is roughly:
```
N_elements ∝ (Domain_Volume / H_average^3)
```

For your domain:
- Volume ~ 5° × 5° × 400 km ≈ (550 km)² × 400 km ≈ 1.2e8 km³
- Target: 100k elements → H_average ≈ (1.2e8 / 1e5)^(1/3) ≈ 11 km

So you could use:
```python
# Physical interpretation
H_coarse = 20.0   # 20 km elements (coarse)
H_fine = 5.0      # 5 km elements (fine)

# Near faults and surface: fine mesh
# Elsewhere: coarse mesh
```

But if the algorithm expects dimensionless, convert:
```python
# Characteristic length scale of domain
L_char = 400.0  # km (depth extent)

# Dimensionless metric
H_coarse_dimensionless = (L_char / H_coarse)**3 * 1e6  # Empirical scaling
H_fine_dimensionless = (L_char / H_fine)**3 * 1e6
```

---

## Testing Strategy

### Step 1: Test with Simple Values

```python
# Try simple constant values first
H.array[:] = 1e6  # Uniform coarse
# Adapt and check element count

H.array[:] = 1e7  # Uniform fine
# Adapt and check element count

# This tells you the relationship between H and element count
```

### Step 2: Test with Spatially Varying

```python
# Make half the mesh fine, half coarse
H.array[depth < 200] = 1e7  # Fine in upper half
H.array[depth >= 200] = 1e6  # Coarse in lower half
# Check that adaptation creates finer mesh in upper half
```

### Step 3: Calibrate for Target Count

Based on Steps 1-2, determine:
- What H value gives you 50k elements?
- What H value gives you 100k elements?
- What H value gives you 200k elements?

Then use those calibrated values.

---

## Recommended Workflow for Your Case

### Option A: Keep Dimensionless (Safest)

Use the same values as before, just update the coordinate access:

```python
# Symbolic coordinates
λ_lon, λ_lat, λ_d = mesh.CoordinateSystem.geo[:]

# Fault distance (in km - but still use absolute values)
fault_distance_km = fault_distance.sym[0]

# Mesh metric - use same dimensionless values as before
mesh_adaptation_parameter = 6.6e6 * (mesh_k_elts / 100)

H_expr = sympy.Piecewise(
    (mesh_adaptation_parameter, fault_distance_km < 33),  # km, but dimensionless comparison
    (mesh_adaptation_parameter, λ_d < 1),                  # depth < 1 km
    (100, True),
)

H_values = underworld3.function.evaluate(H_expr, H.coords)
H.array[:, 0] = H_values.reshape(-1, 1)
```

**This should work because**:
- The metric is still dimensionless (6.6e6 vs 100)
- The conditions use physical coordinates (fault_distance < 33 km, depth < 1 km)
- The adaptation algorithm only cares about relative H values

### Option B: Test Physical Interpretation

Try using physical element sizes:

```python
# Element size in km
H_fine = 5.0    # Near faults/surface
H_coarse = 20.0  # Elsewhere

H_expr = sympy.Piecewise(
    (H_fine, fault_distance_km < 33),
    (H_fine, λ_d < 1),
    (H_coarse, True),
)
```

Then check:
1. Does adaptation work?
2. How many elements do you get?
3. Is the mesh refined where expected?

If element count is wrong, scale by a constant:
```python
scale_factor = 1000  # Empirically determined
H_fine_scaled = H_fine * scale_factor
H_coarse_scaled = H_coarse * scale_factor
```

---

## Key Insight

**The mesh adaptation algorithm likely doesn't know about units** - it just sees numbers. Whether you use:
- Dimensionless values (6.6e6 vs 100)
- Physical values (5 km vs 20 km)
- Scaled physical values (5000 vs 20000)

**Only the relative ratio matters**: H_fine / H_coarse

The absolute scale determines the total element count, which you calibrate empirically.

---

## Next Step: Experiment!

Create a simple test:

```python
# Test 1: Uniform metric
H.array[:] = 1e6
meshA1 = uw.adaptivity.mesh_adapt_meshVar(mesh, H, Metric)[1]
print(f"Uniform H=1e6: {meshA1.data.shape[0]} nodes")

# Test 2: Spatially varying
H.array[depth < 200] = 1e7
H.array[depth >= 200] = 1e6
meshA2 = uw.adaptivity.mesh_adapt_meshVar(mesh, H, Metric)[1]
print(f"Varying H: {meshA2.data.shape[0]} nodes")

# Test 3: Physical interpretation
H.array[:] = 10.0  # 10 km elements
meshA3 = uw.adaptivity.mesh_adapt_meshVar(mesh, H, Metric)[1]
print(f"Physical H=10km: {meshA3.data.shape[0]} nodes")
```

This will tell us how the metric field works with the geographic mesh!

---

## My Recommendation

**Start with Option A** (keep dimensionless values), because:
1. It's what worked before
2. Geographic coordinates don't change the adaptation algorithm
3. The conditions (fault_distance < 33 km) now have clear physical meaning
4. You can refine later if needed

The key change from your original code is just:
- Use `λ_d` instead of `((1 - r) * 6370)`
- Use `fault_distance_km` with physical threshold
- Everything else stays the same!
