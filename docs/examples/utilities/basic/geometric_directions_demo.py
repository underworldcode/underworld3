# %% [markdown]
"""
# 📐 Geometric Directions Demonstration

**PURPOSE:** Demonstrate the new geometric direction properties for mesh coordinate systems

## Description

This example shows how to use the enhanced geometric direction properties that are now
available for all mesh types. These properties provide intuitive access to natural
coordinate directions without needing to remember which unit vector corresponds to
which geometric direction.

**Key Features:**
- Automatic geometric direction properties for all coordinate systems
- Type-aware property availability (e.g., `unit_radial` only for cylindrical/spherical)
- Consistent naming across different mesh types
- Complete backward compatibility with existing `unit_e_0`, `unit_e_1`, etc.

**Mathematical Foundation:**

All geometric directions are built on the existing robust unit vector system:
- **Cartesian**: `unit_e_0` = x, `unit_e_1` = y, `unit_e_2` = z
- **Cylindrical**: `unit_e_0` = radial, `unit_e_1` = tangential  
- **Spherical**: `unit_e_0` = radial, `unit_e_1` = meridional, `unit_e_2` = azimuthal
"""

# %% [markdown]
"""
## Setup and Imports
"""

# %%
import underworld3 as uw
import sympy
import numpy as np

# %% [markdown]
"""
## Cartesian Coordinate System Properties

In Cartesian coordinates, geometric directions map naturally to x, y, z axes.
"""

# %%
print("=== CARTESIAN MESH (2D) ===")

# Create a simple structured box
box_2d = uw.meshing.StructuredQuadBox(
    elementRes=(4, 4),
    minCoords=(0.0, 0.0), 
    maxCoords=(1.0, 1.0)
)

cs_cart = box_2d.CoordinateSystem
print(f"Coordinate system: {cs_cart.coordinate_type}")
print(f"Geometric dimensions: {cs_cart.geometric_dimension_names}")
print()

# Access geometric directions
print("Geometric direction properties:")
print(f"  unit_horizontal:   {cs_cart.unit_horizontal}")
print(f"  unit_vertical:     {cs_cart.unit_vertical}")  
print(f"  unit_horizontal_1: {cs_cart.unit_horizontal_1}")
print()

# Show equivalence with existing unit vectors
print("Equivalence with existing unit vectors:")
print(f"  unit_horizontal == unit_e_0: {cs_cart.unit_horizontal.equals(cs_cart.unit_e_0)}")
print(f"  unit_vertical == unit_e_1:   {cs_cart.unit_vertical.equals(cs_cart.unit_e_1)}")
print()

# Show all available directions
directions = cs_cart.primary_directions
print(f"All available directions: {list(directions.keys())}")

# %% [markdown]
"""
## Cylindrical Coordinate System Properties

In cylindrical coordinates, we have natural radial and tangential directions.
"""

# %%  
print("\n=== CYLINDRICAL MESH (Annulus) ===")

# Create an annulus mesh
annulus = uw.meshing.Annulus(
    radiusOuter=1.0,
    radiusInner=0.5,
    cellSize=0.2
)

cs_cyl = annulus.CoordinateSystem
print(f"Coordinate system: {cs_cyl.coordinate_type}")
print(f"Geometric dimensions: {cs_cyl.geometric_dimension_names}")
print()

# Access cylindrical-specific geometric directions
print("Cylindrical-specific geometric directions:")
print(f"  unit_radial:     {cs_cyl.unit_radial}")
print(f"  unit_tangential: {cs_cyl.unit_tangential}")
print()

# General geometric directions
print("General geometric directions:")
print(f"  unit_horizontal:   {cs_cyl.unit_horizontal}")  # Maps to radial
print(f"  unit_vertical:     {cs_cyl.unit_vertical}")    # Cartesian y-direction
print(f"  unit_horizontal_1: {cs_cyl.unit_horizontal_1}") # Maps to tangential
print()

# Show equivalence with existing unit vectors
print("Equivalence with existing unit vectors:")
print(f"  unit_radial == unit_e_0:     {cs_cyl.unit_radial.equals(cs_cyl.unit_e_0)}")
print(f"  unit_tangential == unit_e_1: {cs_cyl.unit_tangential.equals(cs_cyl.unit_e_1)}")
print()

# Show all available directions
directions = cs_cyl.primary_directions
print(f"All available directions: {list(directions.keys())}")

# %% [markdown]
"""
## Spherical Coordinate System Properties

Spherical coordinates provide radial, meridional, and azimuthal directions.
"""

# %%
print("\n=== SPHERICAL MESH (CubedSphere) ===")

# Create a cubed sphere mesh
sphere = uw.meshing.CubedSphere(
    radiusOuter=1.0,
    radiusInner=0.5,
    numElements=3
)

cs_sph = sphere.CoordinateSystem
print(f"Coordinate system: {cs_sph.coordinate_type}")
print(f"Geometric dimensions: {cs_sph.geometric_dimension_names}")
print()

# Access spherical-specific geometric directions
print("Spherical-specific geometric directions:")
print(f"  unit_radial:      {cs_sph.unit_radial}")
print(f"  unit_meridional:  {cs_sph.unit_meridional}")
print(f"  unit_azimuthal:   {cs_sph.unit_azimuthal}")
print()

# General geometric directions
print("General geometric directions:")
print(f"  unit_horizontal:   {cs_sph.unit_horizontal}")  # Maps to meridional
print(f"  unit_vertical:     {cs_sph.unit_vertical}")    # Maps to radial
print(f"  unit_horizontal_1: {cs_sph.unit_horizontal_1}") # Maps to azimuthal
print()

# Show equivalence with existing unit vectors
print("Equivalence with existing unit vectors:")
print(f"  unit_radial == unit_e_0:      {cs_sph.unit_radial.equals(cs_sph.unit_e_0)}")
print(f"  unit_meridional == unit_e_1:  {cs_sph.unit_meridional.equals(cs_sph.unit_e_1)}")
print(f"  unit_azimuthal == unit_e_2:   {cs_sph.unit_azimuthal.equals(cs_sph.unit_e_2)}")
print()

# Show all available directions
directions = cs_sph.primary_directions
print(f"All available directions: {list(directions.keys())}")

# %% [markdown]
"""
## Practical Usage Examples

Here are some practical examples of using geometric directions in physics simulations.
"""

# %%
print("\n=== PRACTICAL USAGE EXAMPLES ===")

# Example 1: Radial body force in cylindrical coordinates
print("1. Radial body force in cylindrical coordinates:")
x, y = annulus.N
r = sympy.sqrt(x**2 + y**2)
radial_force_magnitude = r  # Force increases with radius

# OLD WAY: Need to remember that unit_e_0 is radial direction
radial_force_old = radial_force_magnitude * annulus.CoordinateSystem.unit_e_0

# NEW WAY: Use intuitive geometric property
radial_force_new = radial_force_magnitude * annulus.CoordinateSystem.unit_radial

print(f"  Using unit_e_0:    {radial_force_old}")
print(f"  Using unit_radial: {radial_force_new}")
print(f"  Results identical: {radial_force_old.equals(radial_force_new)}")
print()

# Example 2: Gravitational force in spherical coordinates
print("2. Gravitational force in spherical coordinates:")
x, y, z = sphere.N
r = sympy.sqrt(x**2 + y**2 + z**2)
gravity_magnitude = sympy.Symbol("g") / r**2  # Inverse square law

# NEW WAY: Clear, intuitive direction
gravity_force = -gravity_magnitude * sphere.CoordinateSystem.unit_radial
print(f"  Gravity force: {gravity_force}")
print()

# Example 3: Horizontal flow in Cartesian coordinates  
print("3. Horizontal flow in Cartesian coordinates:")
flow_speed = sympy.Symbol("U")

# NEW WAY: Clear geometric meaning
horizontal_flow = flow_speed * box_2d.CoordinateSystem.unit_horizontal
print(f"  Horizontal flow: {horizontal_flow}")
print()

# %% [markdown]
"""
## Error Handling and Type Safety

The geometric properties provide appropriate error messages when accessed
for unsupported coordinate systems.
"""

# %%
print("=== ERROR HANDLING EXAMPLES ===")

# Example: Trying to access cylindrical properties on Cartesian mesh
print("Accessing cylindrical properties on Cartesian mesh:")
try:
    radial_cartesian = box_2d.CoordinateSystem.unit_radial
    print(f"  Unexpected success: {radial_cartesian}")
except NotImplementedError as e:
    print(f"  Expected error: {e}")

print()

# Example: Trying to access spherical properties on cylindrical mesh  
print("Accessing spherical properties on cylindrical mesh:")
try:
    meridional_cylindrical = annulus.CoordinateSystem.unit_meridional
    print(f"  Unexpected success: {meridional_cylindrical}")
except NotImplementedError as e:
    print(f"  Expected error: {e}")

print()

# %% [markdown]
"""
## Summary and Benefits

The new geometric direction properties provide:

### ✅ **Intuitive Access**
- `unit_radial`, `unit_tangential` for cylindrical coordinates
- `unit_horizontal`, `unit_vertical` for all coordinate systems  
- `unit_meridional`, `unit_azimuthal` for spherical coordinates

### ✅ **Type Safety**  
- Properties only available for appropriate coordinate systems
- Clear error messages for unsupported combinations
- Compile-time property availability through `primary_directions`

### ✅ **Backward Compatibility**
- All existing `unit_e_0`, `unit_e_1`, `unit_e_2` code continues to work
- New properties are built on existing robust foundation
- No breaking changes to existing examples or user code

### ✅ **Consistent Interface**
- Same property names work across different coordinate systems
- `geometric_dimension_names` provides coordinate-system-specific naming
- `primary_directions` dictionary provides programmatic access

**Migration Path:**
- **Immediate benefit**: Use new properties in new code for clarity
- **Gradual adoption**: Replace `unit_e_0` with `unit_radial` etc. over time  
- **Full compatibility**: Both old and new approaches work indefinitely

The geometric direction properties make Underworld3 code more readable, 
maintainable, and accessible to users unfamiliar with coordinate system
implementation details.
"""

# %%
print("✅ Geometric Directions Demonstration Complete!")
print(f"New properties successfully demonstrated across {len([box_2d, annulus, sphere])} coordinate systems")