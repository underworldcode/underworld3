# Units and Statistics Methods Issue

**Date**: 2025-10-23
**Affected Methods**: `.max()`, `.min()`, `.mean()`, `.std()`, and other statistics
**Status**: By design - performance optimization

---

## The Issue

Methods like `.max()`, `.min()` on `MeshVariable` and `SwarmVariable` **do not** return unit-aware quantities. They return plain Python types (floats or tuples).

### Example Problem

```python
v = uw.discretisation.MeshVariable('v', mesh, 2, degree=2, units='m/s')
T = uw.discretisation.MeshVariable('T', mesh, 1, degree=2, units='K')

# These return plain Python types:
v_max = v.max()  # Returns: (vx_max, vy_max) - TUPLE of floats
T_max = T.max()  # Returns: float (not Pint Quantity)

# This FAILS:
v_max.to("cm/year")  #  AttributeError: 'tuple' object has no attribute 'to'
T_max.to("degC")     #  AttributeError: 'float' object has no attribute 'to'
```

---

## Why This Design?

**Performance**: Returning Pint quantities from every statistics operation adds overhead. For large arrays and frequent operations, this becomes significant.

**Consistency**: PETSc and NumPy operations return plain floats/arrays, not unit-aware objects.

---

## The Solution Pattern

### For Scalar Variables (Temperature, Pressure)

```python
# Get plain float from statistics
T_max_nd = T.max()  # Non-dimensional value (plain float)

# Get scaling coefficient
T0_scale = T.scaling_coefficient  # Plain float (e.g., 3000.0 for Kelvin)

# Create Pint quantity for unit conversion
T_max_dimensional = uw.quantity(T_max_nd * T0_scale, "K")

# Now can convert units
T_max_celsius = T_max_dimensional.to("degC")
```

### For Vector Variables (Velocity, Force)

```python
# Get tuple of component maxima
v_max_components = v.max()  # Returns: (vx_max, vy_max)

# Option 1: Get maximum component
v_max_nd = max(v.max())  # Largest component value

# Option 2: Get specific component
v_x_max_nd = v.max()[0]  # X-component maximum
v_y_max_nd = v.max()[1]  # Y-component maximum

# Create dimensional quantity
V0_scale = v.scaling_coefficient  # Plain float
v_max_dimensional = uw.quantity(v_max_nd * V0_scale, "m/s")

# Convert units
v_max_cm_per_year = v_max_dimensional.to("cm/year")
```

### Complete Example (from Notebook 14)

```python
# After solving Stokes flow...

# 1. Get ND statistics (plain floats/tuples)
v_max_nd = max(v.max())  # Maximum velocity component
T_min_nd = T.min()
T_max_nd = T.max()

# 2. Get scaling coefficients
V0 = v.scaling_coefficient  # e.g., 1.58e-9 m/s
T0 = T.scaling_coefficient  # e.g., 3000.0 K

# 3. Create dimensional quantities with units
v_dimensional = uw.quantity(v_max_nd * V0, "m/s")
T_min_dimensional = uw.quantity(T_min_nd * T0, "K")
T_max_dimensional = uw.quantity(T_max_nd * T0, "K")

# 4. Convert to desired units
v_cm_per_year = v_dimensional.to("cm/year")
v_mm_per_year = v_dimensional.to("mm/year")
```

---

## Affected Methods

All statistics methods return plain Python types:

| Method | Scalar Return | Vector Return |
|--------|--------------|---------------|
| `.max()` | `float` | `tuple` of floats |
| `.min()` | `float` | `tuple` of floats |
| `.mean()` | `float` | `tuple` of floats |
| `.std()` | `float` | `tuple` of floats |
| `.stats()` | `dict` of floats | `dict` of tuples |

---

## Best Practices

### 1. Always Wrap for Unit Conversion

```python
# WRONG - will fail
v_converted = v.max().to("cm/year")  # AttributeError

# CORRECT
v_dimensional = uw.quantity(max(v.max()) * v.scaling_coefficient, "m/s")
v_converted = v_dimensional.to("cm/year")
```

### 2. Document Units in Comments

```python
# Get maximum velocity in ND units
v_max_nd = max(v.max())  # Non-dimensional

# Convert to dimensional units (m/s)
v_dimensional = uw.quantity(v_max_nd * V0_scale, "m/s")
```

### 3. Use Variable Units Attribute

```python
# Instead of hardcoding "m/s", use the variable's units:
v_units = v.sym[0].free_symbols  # Extract units from symbolic form
# Or store units separately:
velocity_units = "m/s"
v_dimensional = uw.quantity(v_max_nd * V0_scale, velocity_units)
```

---

## Alternative: Direct Array Access (Advanced)

For unit-aware operations without statistics methods:

```python
import numpy as np

# Get dimensional array directly
v_array_dimensional = v.array * v.scaling_coefficient  # NumPy array (no units)

# Create Pint array
v_quantity_array = uw.quantity(v_array_dimensional, "m/s")

# Now can use Pint operations
v_max_with_units = v_quantity_array.max()  # Returns Pint Quantity
v_in_cm_per_year = v_quantity_array.to("cm/year")
```

**Note**: This creates a copy of the entire array, which may be expensive for large datasets.

---

## Future Enhancement Ideas

Could add convenience methods:

```python
# Hypothetical future API:
v.max_with_units()  # Returns uw.quantity with proper units
T.to_units("degC")  # Convert entire field
```

But current pattern (explicit wrapping) is clear and performant.

---

## Summary

**The Pattern**:
1. Statistics methods return **plain Python types** (floats or tuples)
2. To get dimensional values: `uw.quantity(value * scaling_coefficient, units)`
3. For vectors: Use `max(v.max())` to get largest component

**Why**: Performance - avoids overhead of wrapping every operation in Pint quantities

**Workaround**: Explicit wrapping when unit conversion needed
