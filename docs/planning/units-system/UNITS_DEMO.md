# Underworld3 Units System - User Guide

This guide demonstrates the improved units system in Underworld3 with clear, discoverable patterns.

## Key Improvements

1. **Direct access**: `uw.units.K` instead of `uw.scaling.units.K`
2. **Better introspection**: Models show units status clearly
3. **Consistent patterns**: `model.view()` and `uw.units.view()` following established conventions
4. **Clear guidance**: Helpful messages when units aren't set up

## Quick Start

### 1. Accessing Units

```python
import underworld3 as uw

# Temperature units
temperature = 1500 * uw.units.K
temp_celsius = 1200 * uw.units.celsius

# Pressure units
pressure = 1e9 * uw.units.Pa
atm_pressure = 1 * uw.units.atm

# Viscosity (compound units)
viscosity = 1e21 * uw.units.Pa * uw.units.s

# Velocity
velocity = 5 * uw.units.cm / uw.units.year
```

### 2. Exploring Available Units

```python
# See all available units and examples
uw.units.view()
```

### 3. Creating a Model with Units

```python
# Create model
model = uw.Model("thermal_convection")

# Initially no units
print(model)
# Output: Model('thermal_convection', meshes=0, 0 variables, 0 swarms, units=not_set)

# Set reference quantities for dimensional analysis
model.set_reference_quantities(
    mantle_temperature=1500*uw.units.K,
    mantle_viscosity=1e21*uw.units.Pa*uw.units.s,
    plate_velocity=5*uw.units.cm/uw.units.year
)

# Now shows units are configured
print(model)
# Output: Model('thermal_convection', meshes=0, 0 variables, 0 swarms, units=set)
```

### 4. Model Introspection

```python
# Comprehensive model information
model.view()
```

This displays:
- Model state and configuration
- Units setup status with specific reference quantities
- Registered components (meshes, variables, swarms)
- Helpful guidance if units aren't configured

### 5. Using Units with Variables

```python
# Create mesh
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0),
    cellSize=0.1
)

# Create variables with units
temperature = uw.discretisation.MeshVariable("T", mesh, 1, units="K")
velocity = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")

# Model now tracks these components
print(model)
# Output: Model('thermal_convection', mesh=UnstructuredSimplexBox, 2 variables, 0 swarms, units=set)
```

## Best Practices

### 1. Always Set Reference Quantities

```python
# Good - provides dimensional analysis and scaling
model.set_reference_quantities(
    mantle_temperature=1500*uw.units.K,
    mantle_viscosity=1e21*uw.units.Pa*uw.units.s
)
```

### 2. Use uw.units for Clarity

```python
# Preferred - clear and discoverable
temperature = 1500 * uw.units.K

# Alternative - more verbose but equivalent
temperature = uw.create_quantity(1500, "K")
```

### 3. Check Model Status

```python
# Quick check
print(model)

# Detailed view
model.view()

# Check programmatically
if not model.get_reference_quantities():
    print("Consider setting reference quantities for dimensional analysis")
```

## Common Units

| Quantity | Units | Example |
|----------|-------|---------|
| Temperature | `uw.units.K`, `uw.units.celsius` | `1500*uw.units.K` |
| Pressure | `uw.units.Pa`, `uw.units.bar`, `uw.units.atm` | `1e9*uw.units.Pa` |
| Viscosity | `uw.units.Pa*uw.units.s` | `1e21*uw.units.Pa*uw.units.s` |
| Length | `uw.units.m`, `uw.units.cm`, `uw.units.km` | `2900*uw.units.km` |
| Time | `uw.units.s`, `uw.units.year`, `uw.units.Ma` | `1*uw.units.Ma` |
| Velocity | `uw.units.m/uw.units.s`, `uw.units.cm/uw.units.year` | `5*uw.units.cm/uw.units.year` |

## Troubleshooting

### Model shows "units=not_set"

```python
# Set reference quantities
model.set_reference_quantities(
    typical_temperature=1500*uw.units.K,
    typical_viscosity=1e21*uw.units.Pa*uw.units.s
)
```

### Can't find specific unit

```python
# Explore available units
uw.units.view()

# Or check what's available
print([attr for attr in dir(uw.units) if not attr.startswith('_')][:20])
```

### Need help with compound units

```python
# Viscosity: pressure × time
viscosity = 1e21 * uw.units.Pa * uw.units.s

# Velocity: length / time
velocity = 5 * uw.units.cm / uw.units.year

# Thermal diffusivity: length² / time
diffusivity = 1e-6 * uw.units.m**2 / uw.units.s
```

This improved system eliminates confusion by:
- Making units easily discoverable (`uw.units`)
- Providing clear status information in model representations
- Following established patterns (`view()` methods)
- Giving helpful guidance when units aren't configured