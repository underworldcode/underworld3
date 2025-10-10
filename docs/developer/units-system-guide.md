# Units System Guide

## Philosophy

Underworld3 implements a comprehensive units system that allows mixing of different unit systems while maintaining numerical stability and preventing common dimensional analysis errors. The system operates on the principle that users should be able to work with units that are natural to their problem domain (geological time scales, pressures, temperatures) while the computational core uses a consistent internal unit system optimized for numerical precision.

### Core Principles

The units system is built around three fundamental concepts:

1. **Dimensional Consistency**: All operations are checked for dimensional compatibility, preventing errors like adding pressure to temperature.

2. **Seamless Unit Handling**: The system automatically handles unit compatibility between user units and internal computational units without requiring explicit conversion calls.

3. **Numerical Optimization**: Internal computations use a scaled unit system that keeps numerical values near unity, minimizing floating-point precision errors common in geological simulations.

## Implementation Overview

### Technology Stack

Underworld3 uses **Pint** as the primary units library rather than SymPy's units module. This choice provides several advantages:

- **Extensive Unit Database**: Pint includes comprehensive physical units, including geological and geophysical units
- **Flexible Unit Parsing**: Natural string parsing like `"1000 km"` or `"5 cm/year"`
- **Dimensional Analysis**: Automatic dimensional checking and unit arithmetic
- **Performance**: Optimized for numerical computations with NumPy integration

### Integration with SymPy

The units system integrates seamlessly with SymPy expressions through several mechanisms:

- **Expression Unit Detection**: Automatically determines units of symbolic expressions based on constituent variables
- **UWQuantity Objects**: Wrap numerical results with unit information while preserving SymPy compatibility
- **Coordinate Unit Handling**: Physical and model coordinates maintain unit information through symbolic operations

### Coordinate Scaling System

Coordinate handling uses a sophisticated scaling approach:

- **Physical Coordinates**: Real-world coordinates with explicit units (e.g., meters, kilometers)
- **Model Coordinates**: Scaled coordinates with internal unit system optimized for computation
- **Automatic Conversion**: Mesh geometry functions automatically convert between coordinate systems
- **Scale Factor Management**: Internal scale factors maintain the relationship between physical and model units

## Practical Examples

### Example 1: Mantle Convection Setup

This example demonstrates setting up a mantle convection problem with mixed units:

```python
import underworld3 as uw
import numpy as np

# Set up model with geological units
model = uw.Model("mantle_convection")

# Define reference scales using natural geological units
model.set_reference_quantities(
    characteristic_length=2900 * uw.units.km,      # Mantle thickness
    plate_velocity=5 * uw.units.cm / uw.units.year, # Typical plate velocity
    mantle_temperature=1500 * uw.units.kelvin,     # Mantle temperature
    material_density=3300 * uw.units.kg / uw.units.m**3  # Mantle density
)

# Create mesh using model coordinates
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(64, 32),
    minCoords=(0.0, 0.0),
    maxCoords=(2.0, 1.0),  # Model coordinates
    qdegree=2
)

# Define temperature field with units
temperature = uw.discretisation.MeshVariable(
    "temperature", mesh, 1, degree=1,
    units="K"  # Kelvin units
)

# Initialize temperature field using coordinate-based conditions
with uw.synchronised_array_update():
    coords = mesh.data
    x, y = coords[:, 0], coords[:, 1]

    # Linear temperature profile: 300K at surface to 1600K at bottom
    T_surface, T_bottom = 300, 1600
    temperature.array[:, 0, 0] = T_surface + (T_bottom - T_surface) * y
```

### Example 2: Material Property Definition

```python
# Define material properties with appropriate geological units
viscosity = uw.discretisation.MeshVariable(
    "viscosity", mesh, 1, degree=0,
    units=uw.units("Pa.s")  # Pascal-seconds
)

# Set viscosity using geological values
mantle_viscosity = 1e21 * uw.units("Pa.s")  # Typical mantle viscosity
crustal_viscosity = 1e23 * uw.units("Pa.s")  # Crustal viscosity

# ⚠️ Unit checking prevents common errors
try:
    viscosity.array[0, 0, 0] = 300 * uw.units("K")  # Wrong units!
except Exception as e:
    print(f"Unit error caught: {e}")

# ✅ Correct assignment with automatic unit handling
viscosity.array[:, 0, 0] = mantle_viscosity
```

### Example 3: Coordinate Conversion and Geometry

```python
# Define sampling points in physical coordinates
sample_points_physical = np.array([
    [1000, 500],    # 1000 km, 500 km
    [2000, 1000],   # 2000 km, 1000 km
    [500, 250]      # 500 km, 250 km
]) * uw.units("km")

# ✅ Mesh functions automatically handle coordinate conversion
points_in_domain = mesh.points_in_domain(sample_points_physical)
closest_cells = mesh.get_closest_cells(sample_points_physical)

# Evaluate expressions at physical coordinates
x_coord, y_coord = mesh.X  # Physical coordinate symbols
depth_expr = (2900 * uw.units("km")) - y_coord  # Depth calculation

# ✅ Automatic unit propagation through expressions
depths = uw.function.evaluate(depth_expr, sample_points_physical)
print(f"Depths: {depths}")  # Results include units
```

### Example 4: Time Integration with Units

```python
# Define time stepping with geological time scales
total_time = 100 * uw.units("Myr")  # 100 million years
time_step = 0.1 * uw.units("Myr")   # 100,000 year time steps

# Create advection-diffusion system
adv_diff = uw.systems.AdvDiffusionSLCN(
    mesh, temperature,
    velocity_field=velocity,
    diffusivity=1e-6 * uw.units("m**2/s"),  # Thermal diffusivity
    V_fn=velocity
)

# ✅ Time stepping with unit-aware solver
current_time = 0 * uw.units("s")
while current_time < total_time:
    # Solve with automatic unit handling
    adv_diff.solve(time_step)
    current_time += time_step

    # ⚠️ Units preserved through calculations
    print(f"Time: {current_time.to('Myr')}")
```

## Implementation Details

### Variable Unit Metadata

Variables store unit information that persists through operations:

```python
# Create variable with unit metadata
pressure = uw.discretisation.MeshVariable(
    "pressure", mesh, 1, units=uw.units("GPa")
)

# Units are preserved and checked
print(f"Pressure units: {pressure.units}")  # Output: gigapascal

# ✅ Unit-aware mathematical operations
stress_tensor = pressure * uw.function.grad(velocity)  # Units: GPa*s^-1
```

### Expression Unit Detection

The system automatically determines units of complex expressions:

```python
# Physical coordinate expressions have length units
x_phys, y_phys = mesh.X
depth = (2900 * uw.units("km")) - y_phys

# Model coordinate expressions maintain dimensional relationships
x_model, y_model = mesh.N.x, mesh.N.y
normalized_position = x_model / 2.0  # Dimensionless ratio

# ✅ Mixed expressions maintain unit consistency
combined_expr = depth + x_phys * 0.1  # Units: length
```

### Automatic Coordinate Conversion

Mesh geometry functions automatically convert between coordinate systems:

```python
# Define points in different unit systems
points_km = np.array([[1000, 500]]) * uw.units("km")
points_m = np.array([[1000000, 500000]]) * uw.units("m")
points_model = np.array([[1.0, 0.5]])  # Model coordinates

# ✅ All return identical results due to automatic coordinate conversion
result1 = mesh.points_in_domain(points_km)
result2 = mesh.points_in_domain(points_m)
result3 = mesh.points_in_domain(points_model)
assert np.array_equal(result1, result2, result3)
```

### Scale Factor Management

The model automatically manages scale factors for numerical optimization:

```python
# Query current scaling information
scales = model.get_fundamental_scales()
print(f"Length scale: {scales['length']}")     # 2900 kilometer
print(f"Time scale: {scales['time']}")         # ~580 Myr (derived)
print(f"Temperature scale: {scales['temperature']}")  # 1500 kelvin

# ✅ Automatic scaling maintains precision
model_coords = model.to_model_units(points_km)  # Converts to model units
```

## Current Limitations and Future Directions

### Known Limitations

- **Serialization**: Unit metadata is not yet preserved in mesh/variable save/load operations
- **Complex Expressions**: Some deeply nested expressions may not propagate units correctly
- **Performance**: Unit checking adds computational overhead for intensive operations

### Planned Enhancements

- **Serialization Support**: Preserve unit information in file I/O operations
- **Enhanced Expression Analysis**: More sophisticated unit detection for complex symbolic expressions
- **Performance Optimization**: Cached unit analysis for repeated operations

## Error Handling

The units system provides clear error messages for common mistakes:

```python
# ❌ Dimensional incompatibility
try:
    temperature.array[0, 0, 0] = 1000 * uw.units("meter")  # Length assigned to temperature
except Exception as e:
    print(f"Unit error: {e}")

# ❌ Mixed unit operations without conversion
try:
    mixed_expr = (100 * uw.units("kelvin")) + (50 * uw.units("celsius"))
    # Must explicitly convert: uw.units("celsius").to('kelvin')
except Exception as e:
    print(f"Conversion error: {e}")

# ⚠️ Coordinate system mismatches are handled automatically
physical_point = np.array([[2000]]) * uw.units("km")
model_result = mesh.points_in_domain(physical_point)  # Auto-converted
```

This units system provides a robust foundation for geodynamic modeling while maintaining the flexibility to work with natural geological units and scales.