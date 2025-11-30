# Coordinate Interface Design

**Date**: 2025-01-10
**Status**: Design decision - to be implemented

## Design Decision: mesh.X vs swarm.coords

### Pattern: Honest Asymmetry

Meshes and swarms have fundamentally different relationships with coordinates, and the interface should reflect this honestly rather than forcing artificial symmetry.

## Mesh Coordinates: `mesh.X`

**Meshes ARE coordinate systems** - they carry geometric structure:
- Metric tensors (distances, angles)
- Orientation (coordinate frame)
- Continuous coordinate fields

**Interface:**
```python
mesh.X              # CoordinateSystem object (primary interface)
mesh.X[0]           # Symbolic x-coordinate function
mesh.X.coords       # Coordinate data array (spatial positions)
mesh.X.units        # Coordinate units (km, m, degrees, etc.)

# Example usage:
x, y = mesh.X                           # Unpack symbolic coordinates
expr = 1000 + 500 * mesh.X[0]          # Build symbolic expressions
positions = mesh.X.coords               # Get coordinate data
coord_units = mesh.X.units              # Query coordinate units
```

**Why `mesh.X` works:**
- `X` represents the coordinate system itself
- `mesh.X[0]` is meaningful: "the x-coordinate function"
- Symbolic operations make sense: `temperature.sym.diff(mesh.X[0])`

## Swarm Coordinates: `swarm.coords`

**Swarms are NOT coordinate systems** - they are collections of discrete points:
- No metric tensor
- No orientation
- No continuous coordinate fields
- Just data: particle positions that move

**Interface:**
```python
swarm.coords        # Particle position data (alias for swarm.data)
swarm.units         # Position units
# NO swarm.X - swarms don't define coordinate systems

# Example usage:
positions = swarm.coords                # Get particle positions
pos_units = swarm.units                 # Query position units
```

**Why NOT `swarm.X`:**
- Swarms don't carry geometric structure
- `swarm.X[0]` would be meaningless - particles don't define "x-coordinate"
- Would imply coordinate-geometry tracking that doesn't exist
- Would mislead users into thinking swarms have metric tensors

## The Fundamental Difference

### Meshes: Coordinate Systems
- **Continuous**: Defined at all points in space
- **Geometric**: Carry metric tensor, orientation
- **Symbolic**: `mesh.X[i]` represents coordinate functions
- **Structured**: Fixed topology and geometry

### Swarms: Point Collections
- **Discrete**: Only defined at particle locations
- **Non-geometric**: Just positions, no metric
- **Data-driven**: Direct position arrays
- **Dynamic**: Particles move, population changes

## Pattern Consistency

The interface pattern is:
- **Things with symbolic coordinate systems** → `.X` interface
- **Things with coordinate data only** → `.coords` + `.units` properties

**Variables follow the same pattern:**
```python
# MeshVariable - defined on coordinate system
temperature.sym         # Symbolic expression T(x,y)
temperature.array       # Temperature field data
temperature.units       # Temperature units
temperature.coords      # References mesh.X.coords (where it's defined)

# SwarmVariable - defined at discrete points
material.data           # Material property data
material.units          # Material units
material.coords         # References swarm.coords (where it's defined)

# Mesh coordinates - the coordinate system itself
mesh.X                  # Coordinate system
mesh.X.coords           # Coordinate data
mesh.X.units            # Coordinate units

# Swarm coordinates - just position data
swarm.coords            # Position data
swarm.units             # Position units
```

## Mathematical Implications

### Derivatives and Gradients

**Mesh variables have geometric derivatives:**
```python
# Gradient: ∂T/∂x requires coordinate system
dT_dx = temperature.sym.diff(mesh.X[0])
# Knows:
# - Numerator: temperature.units (K)
# - Denominator: mesh.X.units (km)
# - Result: K/km
```

**Swarm variables have no geometric derivatives:**
```python
# No swarm.X[0] to differentiate with respect to
# Swarms don't support spatial derivatives directly
# Must interpolate to mesh first via proxy variables
```

### Integration

**Mesh integration uses coordinate system:**
```python
# Integral over mesh domain uses metric
∫ T dV  # Requires mesh coordinate system and metric
```

**Swarm integration uses particle weights:**
```python
# Summation over particles with volumes
Σ (T_i * V_i)  # No coordinate system needed
```

## Implementation Notes

### Mesh.X as CoordinateSystem Object

Create a `CoordinateSystem` class that:
1. Behaves like current `mesh.X` for backward compatibility
2. Adds `.coords` and `.units` properties
3. Maintains symbolic coordinate interface

```python
class CoordinateSystem:
    """Mesh coordinate system with symbolic coordinates and data access."""

    def __init__(self, mesh):
        self._mesh = weakref.ref(mesh)
        self._symbols = ...  # Existing mesh.X symbolic coordinates

    def __getitem__(self, idx):
        """mesh.X[0] → x-coordinate symbol"""
        return self._symbols[idx]

    def __iter__(self):
        """x, y = mesh.X"""
        return iter(self._symbols)

    @property
    def coords(self):
        """Coordinate data array."""
        return self._mesh().points

    @property
    def units(self):
        """Coordinate units."""
        return self._mesh().units
```

### Swarm Coordinate Properties

Add convenience properties to Swarm:
```python
@property
def coords(self):
    """Particle position data. Alias for swarm.data."""
    return self.data

@property
def units(self):
    """Position units (inferred from mesh or explicitly set)."""
    return self._position_units or (
        self.mesh.X.units if hasattr(self, 'mesh') and self.mesh else None
    )
```

## Backward Compatibility

### Keep Existing Interfaces

```python
# These continue to work:
mesh.X[0]           # ✓ Via CoordinateSystem.__getitem__
x, y = mesh.X       # ✓ Via CoordinateSystem.__iter__
mesh.points         # ✓ Keep as alias for mesh.X.coords
mesh.units          # ✓ Keep as alias for mesh.X.units
swarm.data          # ✓ Keep as alias for swarm.coords (or vice versa)
```

### Deprecate Confusing Interface

```python
mesh.data           # Deprecate → use mesh.X.coords or mesh.points
                    # (mesh.data was ambiguous - coordinates? field data?)
```

## Design Philosophy

**Reflect Reality in the Interface:**
- Don't force false symmetry between different concepts
- Make geometric vs non-geometric distinction clear
- Guide users toward correct usage through interface design

**Key Principle:**
> Coordinate systems (mesh.X) carry geometric structure.
> Point collections (swarm.coords) do not.
> The interface should make this distinction obvious.

## Related Design Patterns

### Variables Pattern
All variables (mesh and swarm) follow same interface:
- `.sym` or `.data` → values
- `.array` → data access
- `.units` → value units
- `.coords` → where they're defined

### Coordinate Pattern (NEW)
Coordinates follow different patterns based on their nature:
- **Geometric (meshes)**: `.X` coordinate system object
- **Non-geometric (swarms)**: `.coords` + `.units` properties

This reflects the mathematical reality that coordinate systems are more than just position data.

## Future Extensions

### Curvilinear Coordinates
When implementing curvilinear coordinates, mesh.X would naturally extend:
```python
mesh.X.metric       # Metric tensor g_ij
mesh.X.jacobian     # Coordinate transformation Jacobian
mesh.X.basis        # Coordinate basis vectors
```

This structure doesn't make sense for swarms - confirming the design decision.

### Particle Tracking
Swarms might eventually track orientation or shape:
```python
swarm.coords        # Still just positions
swarm.orientation   # Particle orientation (if tracked)
swarm.shape         # Particle shape (if tracked)
```

But this still wouldn't make them coordinate systems - they'd be particle properties.

## Summary

**Recommended implementation:**
1. Make `mesh.X` return a `CoordinateSystem` object with `.coords` and `.units`
2. Add `swarm.coords` and `swarm.units` as properties
3. Keep `mesh.points`, `mesh.units`, `swarm.data` for backward compatibility
4. Deprecate `mesh.data` (ambiguous, replaced by `mesh.X.coords` or `mesh.points`)

**Key insight:**
The asymmetry between `mesh.X` and `swarm.coords` reflects the fundamental mathematical difference between coordinate systems (with geometric structure) and point collections (without geometric structure). This honest design guides users toward correct usage.
