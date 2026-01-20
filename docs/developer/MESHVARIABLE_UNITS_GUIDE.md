# MeshVariable Units Guide

## Quick Reference

### Creating MeshVariables with Units
```python
import underworld3 as uw

# Create mesh
mesh = uw.meshing.StructuredQuadBox(elementRes=(8, 8), minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0))

# Create variables with units
velocity = uw.discretisation.MeshVariable("velocity", mesh, 2, units="m/s")
pressure = uw.discretisation.MeshVariable("pressure", mesh, 1, units="Pa")
temperature = uw.discretisation.MeshVariable("temperature", mesh, 1, units="K")
density = uw.discretisation.MeshVariable("density", mesh, 1, units="kg/m**3")
```

### Accessing Units Information
```python
# Check if variable has units
velocity.has_units        # → True
velocity.units           # → meter / second

# Units are Pint quantities
type(velocity.units)     # → <class 'pint.Quantity'>
```

### Setting Data
```python
# Set data normally - units are tracked conceptually
with uw.synchronised_array_update():
    velocity.array[:, 0, 0] = 0.01    # 1 cm/s in x direction
    velocity.array[:, 0, 1] = 0.0     # 0 in y direction
    pressure.array[:, 0, 0] = 1e9     # 1 GPa
    temperature.array[:, 0, 0] = 1600 # 1600 K
    density.array[:, 0, 0] = 3300     # 3300 kg/m³
```

## Working with Physical Quantities

### Creating Physical Quantities
```python
# Use the units registry
plate_speed = 5 * uw.scaling.units.cm / uw.scaling.units.year
mantle_pressure = 100 * uw.scaling.units.GPa
geological_time = 100e6 * uw.scaling.units.year
mantle_depth = 2900 * uw.scaling.units.km
```

### Unit Conversions
```python
# Convert to different units
plate_speed_ms = plate_speed.to('m/s')          # → 1.58e-09 meter/second
mantle_pressure_Pa = mantle_pressure.to('Pa')  # → 1.0e+11 pascal
time_seconds = geological_time.to('s')         # → 3.16e+15 second
depth_meters = mantle_depth.to('m')           # → 2.9e+06 meter
```

### Non-Dimensionalisation for Solvers
```python
# Convert physical quantities to non-dimensional form
nd_speed = uw.scaling.non_dimensionalise(plate_speed)      # → 5.00e-02
nd_pressure = uw.scaling.non_dimensionalise(mantle_pressure) # → 9.96e+25

# Customize scaling coefficients
coeffs = uw.scaling.get_coefficients()
coeffs["[length]"] = 1000 * uw.scaling.units.km   # Geological length scale
coeffs["[time]"] = 1e6 * uw.scaling.units.year    # Million year time scale
```

## Mathematical Operations

### Direct Mathematical Operations
```python
# MeshVariables work directly in expressions
x, y = mesh.X
divergence = velocity[0].diff(x) + velocity[1].diff(y)
strain_rate = 0.5 * (velocity[0].diff(y) + velocity[1].diff(x))
```

### Component Access
```python
# Access vector components directly
v_x = velocity[0]    # x-component
v_y = velocity[1]    # y-component
```

## Complete Workflow Example

### Realistic Geophysical Setup
```python
import underworld3 as uw
import numpy as np

# 1. Create mesh (mantle section)
mesh = uw.meshing.Annulus(radiusInner=3500e3, radiusOuter=6400e3, cellSize=100e3)

# 2. Create variables with appropriate units
velocity = uw.discretisation.MeshVariable("velocity", mesh, 2, units="m/s")
pressure = uw.discretisation.MeshVariable("pressure", mesh, 1, units="Pa")
temperature = uw.discretisation.MeshVariable("temperature", mesh, 1, units="K")
viscosity = uw.discretisation.MeshVariable("viscosity", mesh, 1, units="Pa*s")

# 3. Set realistic data
with uw.synchronised_array_update():
    coords = mesh.data
    r = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
    theta = np.arctan2(coords[:, 1], coords[:, 0])

    # Plate motion (~5 cm/year)
    omega = 2e-15  # rad/s
    v_theta = omega * r
    velocity.array[:, 0, 0] = -v_theta * np.sin(theta)  # v_x
    velocity.array[:, 0, 1] = v_theta * np.cos(theta)   # v_y

    # Lithostatic pressure
    depth = 6400e3 - r
    pressure.array[:, 0, 0] = 3300 * 9.81 * depth  # ρgh

    # Temperature profile
    temperature.array[:, 0, 0] = 300 + 1300 * (depth / 2900e3)

    # Mantle viscosity
    viscosity.array[:, 0, 0] = 1e21  # Pa⋅s

# 4. Physical quantities for comparison
plate_speed = 5 * uw.scaling.units.cm / uw.scaling.units.year
mantle_visc = 1e21 * uw.scaling.units.Pa * uw.scaling.units.s

print(f"Plate speed: {plate_speed} = {plate_speed.to('m/s')}")
print(f"Max velocity in model: {velocity.array[:, 0, 0].max():.2e} m/s")

# 5. Non-dimensionalisation for solvers
coeffs = uw.scaling.get_coefficients()
coeffs["[length]"] = 2900 * uw.scaling.units.km    # Mantle thickness
coeffs["[time]"] = coeffs["[length]"]**2 / (1e-6 * uw.scaling.units.m**2/uw.scaling.units.s)  # Thermal diffusion time

nd_plate_speed = uw.scaling.non_dimensionalise(plate_speed)
print(f"Non-dimensional plate speed: {nd_plate_speed:.2e}")
```

## Units Registry Access

### Discover Available Units
```python
# View all available units and examples
uw.scaling.units.view()

# View with more details
uw.scaling.units.view(verbose=1)
```

### Common Geophysical Units
```python
# Length
uw.scaling.units.km, uw.scaling.units.m, uw.scaling.units.cm

# Time
uw.scaling.units.year, uw.scaling.units.s, uw.scaling.units.hour

# Velocity
uw.scaling.units.m / uw.scaling.units.s
uw.scaling.units.cm / uw.scaling.units.year

# Pressure
uw.scaling.units.Pa, uw.scaling.units.GPa, uw.scaling.units.bar

# Temperature
uw.scaling.units.K, uw.scaling.units.degC

# Density
uw.scaling.units.kg / uw.scaling.units.m**3

# Viscosity
uw.scaling.units.Pa * uw.scaling.units.s
```

## Key Functions Reference

### Units System Functions
- `uw.scaling.units` - Main Pint units registry
- `uw.scaling.units.view()` - Display available units and examples
- `uw.scaling.get_coefficients()` - Get/modify scaling coefficients
- `uw.scaling.non_dimensionalise(quantity)` - Convert to non-dimensional
- `uw.scaling.dimensionalise(value, units)` - Convert back to dimensional

### MeshVariable Units Methods
- `var.units` - Get units as Pint quantity
- `var.has_units` - Check if variable has units
- `var.set_units(new_units)` - Change units (WARNING: may not convert data)
- `var.check_units_compatibility(other_var)` - Check unit compatibility

## Best Practices

1. **Always specify units** when creating MeshVariables for physical problems
2. **Use consistent unit systems** - either SI or geological units throughout
3. **Non-dimensionalise for solvers** - numerical solvers work best with O(1) values
4. **Track physical meaning** - units help understand what your numbers represent
5. **Use the tutorial** - `docs/beginner/tutorials/12-Units_System.ipynb` has comprehensive examples

## Current Limitations

- Unit conversion methods may have some limitations (warnings may appear)
- Automatic data conversion during unit changes is not fully implemented
- Offset units (like Celsius) have some restrictions
- Units are primarily for tracking and documentation - solver integration is manual

## Documentation Sources

- **Tutorial**: `docs/beginner/tutorials/12-Units_System.ipynb`
- **View method**: `uw.scaling.units.view()`
- **This guide**: Practical examples and workflows
- **Pint documentation**: https://pint.readthedocs.io/