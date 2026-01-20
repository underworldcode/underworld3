# %% [markdown]
"""
# 📊 Geometric Sampling System Demonstration

**PURPOSE:** Demonstrate the complete geometric sampling system with global_evaluate integration

## Description

This example shows how to use the new geometric sampling methods that provide:
1. **Cartesian coordinates** for `global_evaluate()` calls
2. **Natural coordinates** for intuitive plotting and analysis
3. **Coordinate system-aware profiles** (radial, tangential, meridional, etc.)
4. **Generic line sampling** along arbitrary directions

**Key Features:**
- Automatic coordinate conversion between Cartesian and natural systems
- Profile sampling optimized for each coordinate system type
- Direct integration with `global_evaluate()` for field analysis
- Plotting-friendly natural coordinates for visualization

**Mathematical Foundation:**

The sampling system provides two coordinate arrays:
- **Cartesian coordinates**: Used by `global_evaluate()` for field evaluation
- **Natural coordinates**: Used for plotting and analysis (r,θ), (r,θ,φ), etc.

This dual approach ensures computational efficiency while maintaining intuitive interpretation.
"""

# %% [markdown]
"""
## Setup and Imports
"""

# %%
import underworld3 as uw
import numpy as np
import sympy

# %% [markdown]
"""
## Cartesian Coordinate System Sampling

In Cartesian systems, natural coordinates equal Cartesian coordinates,
making sampling straightforward for rectangular domains.
"""

# %%
print("=== CARTESIAN SAMPLING ===")

# Create structured box mesh
box_mesh = uw.meshing.StructuredQuadBox(
    elementRes=(8, 8),
    minCoords=(0.0, 0.0), 
    maxCoords=(2.0, 1.0)
)

cs_cart = box_mesh.CoordinateSystem
print(f"Coordinate system: {cs_cart.coordinate_type}")
print(f"Available profile types: horizontal, vertical, diagonal")
print()

# 1. Horizontal profile sampling
horizontal_sample = cs_cart.create_profile_sample(
    'horizontal',
    y_position=0.3,
    x_range=(0.2, 1.8), 
    num_points=10
)

print("Horizontal profile sample:")
print(f"  Shape: {horizontal_sample['cartesian_coords'].shape}")
print(f"  Cartesian coords (first 3): {horizontal_sample['cartesian_coords'][:3]}")
print(f"  Natural coords (first 3): {horizontal_sample['natural_coords'][:3]}")
print(f"  Parameters (x-values): {horizontal_sample['parameters'][:3]}")
print()

# 2. Vertical profile sampling  
vertical_sample = cs_cart.create_profile_sample(
    'vertical',
    x_position=1.0,
    y_range=(0.1, 0.9),
    num_points=8
)

print("Vertical profile sample:")
print(f"  Shape: {vertical_sample['cartesian_coords'].shape}")
print(f"  Y-range: {vertical_sample['parameters'][0]:.2f} to {vertical_sample['parameters'][-1]:.2f}")
print()

# 3. Diagonal profile sampling
diagonal_sample = cs_cart.create_profile_sample(
    'diagonal',
    start_point=[0.2, 0.2],
    end_point=[1.8, 0.8],
    num_points=6
)

print("Diagonal profile sample:")
print(f"  Start: {diagonal_sample['cartesian_coords'][0]}")
print(f"  End: {diagonal_sample['cartesian_coords'][-1]}")

# %% [markdown]
"""
## Cylindrical Coordinate System Sampling

Cylindrical systems provide radial and tangential sampling patterns,
with automatic conversion between (x,y) and (r,θ) representations.
"""

# %%
print("\n=== CYLINDRICAL SAMPLING ===")

# Create annulus mesh
annulus_mesh = uw.meshing.Annulus(
    radiusOuter=1.0,
    radiusInner=0.4,
    cellSize=0.1
)

cs_cyl = annulus_mesh.CoordinateSystem
print(f"Coordinate system: {cs_cyl.coordinate_type}")
print(f"Available profile types: radial, tangential, vertical")
print()

# 1. Radial profile sampling
radial_sample = cs_cyl.create_profile_sample(
    'radial',
    theta=np.pi/3,  # 60 degrees
    r_range=(0.5, 0.95),
    num_points=8
)

print("Radial profile sample:")
print(f"  Shape: {radial_sample['cartesian_coords'].shape}")
print(f"  Cartesian coords (first 3): {radial_sample['cartesian_coords'][:3]}")
print(f"  Natural coords (r, θ) (first 3): {radial_sample['natural_coords'][:3]}")
print(f"  Parameters (r-values): {radial_sample['parameters'][:3]}")
print()

# 2. Tangential (circular arc) profile sampling
tangential_sample = cs_cyl.create_profile_sample(
    'tangential',
    radius=0.7,
    theta_range=(0, np.pi),  # Semi-circle
    num_points=12
)

print("Tangential profile sample:")
print(f"  Shape: {tangential_sample['cartesian_coords'].shape}")
print(f"  First Cartesian point: {tangential_sample['cartesian_coords'][0]}")
print(f"  Last Cartesian point: {tangential_sample['cartesian_coords'][-1]}")
print(f"  Natural coords - constant radius: {np.unique(tangential_sample['natural_coords'][:, 0])}")
print(f"  Theta range: {tangential_sample['parameters'][0]:.2f} to {tangential_sample['parameters'][-1]:.2f}")
print()

# 3. Vertical profile in cylindrical system
vertical_cyl_sample = cs_cyl.create_profile_sample(
    'vertical',
    x_position=0.2,
    y_range=(-0.8, 0.8),
    num_points=10
)

print("Vertical profile in cylindrical system:")
print(f"  Cartesian coords (y varies): {vertical_cyl_sample['cartesian_coords'][:3]}")
print(f"  Converted to natural (r, θ): {vertical_cyl_sample['natural_coords'][:3]}")

# %% [markdown]
"""
## Spherical Coordinate System Sampling  

Spherical systems provide radial, meridional, and azimuthal sampling,
with conversion between (x,y,z) and (r,θ,φ) coordinate systems.
"""

# %%
print("\n=== SPHERICAL SAMPLING ===")

# Create spherical mesh
sphere_mesh = uw.meshing.CubedSphere(
    radiusOuter=1.0,
    radiusInner=0.6,
    numElements=4
)

cs_sph = sphere_mesh.CoordinateSystem
print(f"Coordinate system: {cs_sph.coordinate_type}")
print(f"Available profile types: radial, meridional, azimuthal")
print()

# 1. Radial profile sampling
radial_sph_sample = cs_sph.create_profile_sample(
    'radial',
    theta=np.pi/2,  # Equatorial plane
    phi=np.pi/4,    # 45 degrees longitude
    r_range=(0.65, 0.95),
    num_points=6
)

print("Radial profile in spherical system:")
print(f"  Shape: {radial_sph_sample['cartesian_coords'].shape}")
print(f"  Cartesian coords: {radial_sph_sample['cartesian_coords'][:3]}")
print(f"  Natural coords (r, θ, φ): {radial_sph_sample['natural_coords'][:3]}")
print(f"  Radial parameters: {radial_sph_sample['parameters'][:3]}")
print()

# 2. Meridional profile sampling (longitude line)
meridional_sample = cs_sph.create_profile_sample(
    'meridional',
    radius=0.8,
    phi=0.0,  # Prime meridian
    theta_range=(np.pi/4, 3*np.pi/4),  # From 45° to 135° colatitude
    num_points=8
)

print("Meridional profile sampling:")
print(f"  Shape: {meridional_sample['cartesian_coords'].shape}")
print(f"  Natural coords - constant radius: {np.unique(meridional_sample['natural_coords'][:, 0])}")
print(f"  Natural coords - constant phi: {np.unique(meridional_sample['natural_coords'][:, 2])}")
print(f"  Theta range: {meridional_sample['parameters'][0]:.2f} to {meridional_sample['parameters'][-1]:.2f}")
print()

# 3. Azimuthal profile sampling (latitude circle)
azimuthal_sample = cs_sph.create_profile_sample(
    'azimuthal',
    radius=0.8,
    theta=np.pi/2,  # Equatorial plane
    phi_range=(0, np.pi),  # Half circle in longitude
    num_points=10
)

print("Azimuthal profile sampling:")
print(f"  Shape: {azimuthal_sample['cartesian_coords'].shape}")
print(f"  First point: {azimuthal_sample['cartesian_coords'][0]}")
print(f"  Last point: {azimuthal_sample['cartesian_coords'][-1]}")
print(f"  Phi range: {azimuthal_sample['parameters'][0]:.2f} to {azimuthal_sample['parameters'][-1]:.2f}")

# %% [markdown]
"""
## Generic Line Sampling

The generic line sampling method works with any coordinate system
and can follow arbitrary directions defined by sympy expressions.
"""

# %%
print("\n=== GENERIC LINE SAMPLING ===")

# Example 1: Line along unit_radial direction in cylindrical system
print("1. Line along radial direction in annulus:")
radial_line = cs_cyl.create_line_sample(
    start_point=[0.3, 0.3],  # Starting point in Cartesian
    direction_vector=cs_cyl.unit_radial,  # Direction from geometric properties
    length=0.4,
    num_points=5
)

print(f"  Cartesian coords: {radial_line['cartesian_coords']}")
print(f"  Natural coords: {radial_line['natural_coords']}")
print(f"  Line parameters: {radial_line['parameters']}")
print()

# Example 2: Line along unit_horizontal direction in Cartesian
print("2. Line along horizontal direction in box:")
horizontal_line = cs_cart.create_line_sample(
    start_point=[0.5, 0.6],
    direction_vector=cs_cart.unit_horizontal,  # [1, 0] vector
    length=1.2,
    num_points=6
)

print(f"  Cartesian coords: {horizontal_line['cartesian_coords']}")
print(f"  Natural coords (same as Cartesian): {horizontal_line['natural_coords']}")

# %% [markdown]
"""
## Integration with global_evaluate()

The key benefit of this sampling system is seamless integration
with `global_evaluate()` for field analysis and visualization.
"""

# %%
print("\n=== GLOBAL_EVALUATE INTEGRATION ===")

# Create a simple field on the box mesh
x, y = box_mesh.CoordinateSystem.X
test_field_expr = x**2 + y**2  # Simple quadratic field

print(f"Test field expression: {test_field_expr}")
print()

# Sample the field along different profiles
profiles_to_test = [
    ("Horizontal", horizontal_sample),
    ("Vertical", vertical_sample),
    ("Diagonal", diagonal_sample)
]

for profile_name, sample_data in profiles_to_test:
    print(f"{profile_name} profile field evaluation:")
    
    # Use cartesian_coords for global_evaluate (this is the key feature!)
    try:
        field_values = uw.function.global_evaluate(
            test_field_expr, 
            sample_data['cartesian_coords']
        )
        
        print(f"  Successfully evaluated field at {len(field_values)} points")
        print(f"  Field value range: {field_values.min():.3f} to {field_values.max():.3f}")
        
        # Show first few values with their coordinates
        for i in range(min(3, len(field_values))):
            cart_coord = sample_data['cartesian_coords'][i]
            nat_coord = sample_data['natural_coords'][i] 
            field_val = field_values[i][0]
            expected_val = cart_coord[0]**2 + cart_coord[1]**2
            
            print(f"    Point {i}: Cartesian{tuple(cart_coord)} -> Field={field_val:.3f} (expected {expected_val:.3f})")
        
    except Exception as e:
        print(f"  ❌ Evaluation failed: {e}")
    
    print()

# %% [markdown]
"""
## Coordinate System Comparison

Compare how the same geometric profile looks in different coordinate systems.
"""

# %%
print("=== COORDINATE SYSTEM COMPARISON ===")

# Create the same radial line in different coordinate representations
print("Radial line from center outward:")

# In Cartesian system (if we had a radial direction)
cart_radial_approx = cs_cart.create_line_sample(
    start_point=[1.0, 0.5],
    direction_vector=cs_cart.unit_horizontal,  # Approximates radial at this point
    length=0.8,
    num_points=5
)

# In cylindrical system (true radial)
cyl_radial = cs_cyl.create_line_sample(
    start_point=[1.0, 0.5],  # Same starting point
    direction_vector=cs_cyl.unit_radial,  # True radial direction
    length=0.8,
    num_points=5
)

print("Cartesian approximation:")
print(f"  Cartesian coords: {cart_radial_approx['cartesian_coords']}")
print("Cylindrical true radial:")
print(f"  Cartesian coords: {cyl_radial['cartesian_coords']}")
print(f"  Natural coords (r, θ): {cyl_radial['natural_coords']}")

# %% [markdown]
"""
## Summary and Benefits

The geometric sampling system provides:

### ✅ **Dual Coordinate Support**
- **Cartesian coordinates**: Direct compatibility with `global_evaluate()`
- **Natural coordinates**: Intuitive interpretation and plotting
- **Automatic conversion**: Seamless transformation between coordinate systems

### ✅ **Coordinate System Awareness**
- **Profile types adapt to mesh**: Radial/tangential for cylindrical, meridional/azimuthal for spherical
- **Geometric properties integration**: Use `unit_radial`, `unit_tangential`, etc. for directions
- **Type safety**: Clear errors for unsupported profile types

### ✅ **Flexible Sampling Options**
- **Generic line sampling**: Follow any sympy-defined direction
- **Profile-specific sampling**: Optimized for common geometric patterns
- **Parameterized control**: Full control over ranges, resolution, positioning

### ✅ **Direct Integration**
- **global_evaluate() compatibility**: Cartesian coordinates work immediately
- **Parallel safety**: Sampling works in parallel contexts
- **Field analysis ready**: Perfect for creating profiles, cross-sections, and data extraction

**Usage Patterns:**
```python
# 1. Quick profiles for visualization
sample = mesh.CoordinateSystem.create_profile_sample('radial', theta=0, num_points=50)
field_data = uw.function.global_evaluate(temperature.sym, sample['cartesian_coords'])
plt.plot(sample['natural_coords'][:, 0], field_data)  # Plot vs radius

# 2. Custom sampling along geometric directions  
sample = mesh.CoordinateSystem.create_line_sample(
    start_point=[0, 0], 
    direction_vector=mesh.CoordinateSystem.unit_vertical,
    length=1.0
)

# 3. Coordinate system comparison
cart_coords = sample['cartesian_coords']  # For computation
nat_coords = sample['natural_coords']     # For interpretation
```

This system bridges the gap between computational efficiency (Cartesian coordinates)
and intuitive analysis (natural coordinates), making field analysis much more accessible.
"""

# %%
print("✅ Geometric Sampling System Demonstration Complete!")
print(f"Demonstrated sampling across {len([box_mesh, annulus_mesh, sphere_mesh])} coordinate systems")
print("Ready for field analysis and visualization workflows!")