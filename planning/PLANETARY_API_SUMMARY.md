# Planetary Coordinate System API Summary

## Quick Reference

### Coordinate Access

```python
# Data arrays (numpy)
lon = mesh.planetary.lon      # Longitude (degrees East)
lat = mesh.planetary.lat      # Latitude (degrees North)
depth = mesh.planetary.depth  # Depth below surface (km)

# Symbolic (for equations)
λ_lon, λ_lat, λ_d = mesh.planetary[:]
```

### Basis Vectors (All Available Names)

#### Longitude Direction (East/West)
```python
mesh.planetary.unit_WE      # Primary (West→East)
mesh.planetary.unit_east    # Directional alias
mesh.planetary.unit_lon     # Coordinate alias
mesh.planetary.unit_west    # Opposite direction (-unit_WE)
```

#### Latitude Direction (North/South)
```python
mesh.planetary.unit_SN      # Primary (South→North)
mesh.planetary.unit_north   # Directional alias
mesh.planetary.unit_lat     # Coordinate alias
mesh.planetary.unit_south   # Opposite direction (-unit_SN)
```

#### Depth Direction (Up/Down)
```python
mesh.planetary.unit_down    # Primary (into planet)
mesh.planetary.unit_depth   # Coordinate alias
mesh.planetary.unit_up      # Opposite direction (-unit_down)
```

## Design Rationale

### Primary Names (`unit_WE`, `unit_SN`, `unit_down`)
- **Explicit about direction**: WE = West to East (positive East)
- **Indicates positivity**: SN = South to North (positive North)
- **Right-handed system**: WE × SN = down ✓
- **Unambiguous**: "down" clearly means into planet

### Directional Aliases (`unit_east`, `unit_north`, `unit_depth`)
- **More natural language**: "eastward flow"
- **Matches common usage**: "north-south gradient"
- **Clearer in code**: `flow * unit_east` vs `flow * unit_WE`

### Coordinate Aliases (`unit_lon`, `unit_lat`)
- **Coordinate-based thinking**: "velocity in lon direction"
- **Matches derivative notation**: `dT/dlon` uses `unit_lon`
- **Parallel with spherical**: Similar to `unit_e_0`, `unit_e_1`

### Opposite Directions (`unit_west`, `unit_south`, `unit_up`)
- **Convenience**: No need for explicit negation
- **Clearer intent**: `buoyancy * unit_up` vs `buoyancy * (-unit_down)`
- **Physics convention**: Upward buoyancy force, westward flow

## Usage Patterns

### Boundary Conditions
```python
# Surface no-penetration (perpendicular to surface)
v_bc = 0 * mesh.planetary.unit_up

# Fixed downward velocity
v_bc = -10 * mesh.planetary.unit_down  # or: 10 * mesh.planetary.unit_depth
```

### Horizontal Flow
```python
# Using primary names
v_horizontal = v_e * mesh.planetary.unit_WE + v_n * mesh.planetary.unit_SN

# Using directional aliases (clearer)
v_horizontal = v_east * mesh.planetary.unit_east + v_north * mesh.planetary.unit_north

# Using coordinate aliases
v_horizontal = v_lon * mesh.planetary.unit_lon + v_lat * mesh.planetary.unit_lat
```

### Forces
```python
# Gravitational acceleration (downward)
g_force = -9.81 * mesh.planetary.unit_down

# Buoyancy (upward)
buoyancy = rho * g * mesh.planetary.unit_up
```

### Gradients
```python
# Temperature gradient with depth
dT_dz = T.diff(λ_d)  # λ_d is symbolic depth
grad_T_vertical = dT_dz * mesh.planetary.unit_depth

# Horizontal temperature gradient
dT_dlon = T.diff(λ_lon)
dT_dlat = T.diff(λ_lat)
grad_T_horizontal = dT_dlon * mesh.planetary.unit_lon + dT_dlat * mesh.planetary.unit_lat
```

## Recommendation

**Use what makes your code clearest:**

- **Canonical names** (`unit_WE`, `unit_SN`, `unit_down`) for formal documentation
- **Directional aliases** (`unit_east`, `unit_north`) for readable physics code
- **Coordinate aliases** (`unit_lon`, `unit_lat`) for gradient/derivative contexts
- **Opposite directions** (`unit_up`, `unit_west`) to avoid explicit negation

All are equivalent - choose based on context and clarity.

## Examples from Your Workflow

**Before** (current - using old spherical basis):
```python
unit_vertical = cs_mesh.CoordinateSystem.unit_e_0
unit_SN = -cs_mesh.CoordinateSystem.unit_e_1  # Why negative??
unit_EW = cs_mesh.CoordinateSystem.unit_e_2
```

**After** (new - clear planetary basis):
```python
# Option 1: Primary names
unit_down = mesh.planetary.unit_down
unit_SN = mesh.planetary.unit_SN      # No sign confusion!
unit_WE = mesh.planetary.unit_WE

# Option 2: Directional aliases (recommended for clarity)
unit_depth = mesh.planetary.unit_depth
unit_north = mesh.planetary.unit_north
unit_east = mesh.planetary.unit_east

# Option 3: Coordinate aliases
unit_depth = mesh.planetary.unit_depth
unit_lat = mesh.planetary.unit_lat
unit_lon = mesh.planetary.unit_lon
```

All three options give the same vectors - choose what reads best!
