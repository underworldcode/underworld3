# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
r"""
# Spherical Mesh Examples

**PHYSICS:** coordinates
**DIFFICULTY:** basic

## Overview

Underworld3 provides two spherical mesh types:

1. **SphericalShell** — Complete spherical shell for global models
2. **RegionalSphericalBox** — Regional section for regional models

Both use spherical coordinates $(r, \theta, \phi)$ where $\theta$ is colatitude.

## Coordinate Access Pattern

**Cartesian coordinates are always available:**
- `mesh.X.coords` → $(x, y, z)$ data array
- `mesh.X[0], mesh.X[1], mesh.X[2]` → Symbolic $x, y, z$

**Spherical coordinates for SPHERICAL meshes:**
- `mesh.X.spherical.coords` → $(r, \theta, \phi)$ data array
- `mesh.X.spherical.view()` → See all available properties
"""

# %%
import nest_asyncio
nest_asyncio.apply()

import underworld3 as uw
import numpy as np

# %% [markdown]
"""
## Example 1: Global Spherical Shell

A complete spherical shell—useful for global mantle convection models.
"""

# %%
shell = uw.meshing.SphericalShell(
    radiusOuter=1.0,
    radiusInner=0.5,
    cellSize=0.2,  # Coarse for demonstration
    qdegree=2,
)

print(f"Coordinate system: {shell.CoordinateSystem.coordinate_type}")

# %% [markdown]
"""
## Accessing Spherical Coordinates

The `mesh.X.spherical` property provides natural coordinate access.
Use `.view()` to see everything available.
"""

# %%
# See what's available
shell.X.spherical.view()

# %%
# Two views of the same mesh data
sph = shell.X.spherical
print("Cartesian (internal):", shell.X.coords.shape)
print("Spherical (natural):", sph.coords.shape)

# Quick look at first 3 nodes
for i in range(3):
    xyz = shell.X.coords[i]
    rtp = sph.coords[i]
    print(f"Node {i}: ({xyz[0]:7.3f}, {xyz[1]:7.3f}, {xyz[2]:7.3f})  →  "
          f"(r={rtp[0]:.3f}, θ={np.degrees(rtp[1]):6.1f}°, φ={np.degrees(rtp[2]):7.1f}°)")

# %% [markdown]
"""
## Unit Vectors

Spherical basis vectors vary with position. These are essential for:
- Radial boundary conditions (free-slip on spherical surfaces)
- Interpreting velocity fields (radial vs tangential flow)
- Body forces (gravity pointing inward)
"""

# %%
# Evaluate the radial unit vector at a test point
test_point = np.array([[0.5, 0.5, 0.5]])
e_r = np.zeros(3)
for i in range(3):
    val = uw.function.evaluate(sph.unit_r[i], test_point, shell.X)
    e_r[i] = float(val.flat[0])

# The radial unit vector should point in the same direction as the position
pos_unit = test_point[0] / np.linalg.norm(test_point[0])
print(f"Radial unit vector at (0.5, 0.5, 0.5): {e_r}")
print(f"Position direction:                    {pos_unit}")
print(f"Match: {np.allclose(e_r, pos_unit)}")

# %% [markdown]
"""
## Example 2: Regional Spherical Box

A regional section of a sphere—useful for regional tectonic models.
Uses cubed-sphere projection for uniform element sizes.
"""

# %%
regional = uw.meshing.RegionalSphericalBox(
    radiusOuter=1.0,
    radiusInner=0.9,
    SWcorner=[130, -40],  # Southwest (lon, lat in degrees)
    NEcorner=[150, -25],  # Northeast
    numElementsLon=4,
    numElementsLat=4,
    numElementsDepth=2,
    degree=1,
)

# Show coordinate ranges in the natural spherical form
sph_reg = regional.X.spherical
print("Regional mesh coordinate ranges:")
print(f"  r:     [{sph_reg.r.min():.3f}, {sph_reg.r.max():.3f}]")
print(f"  theta: [{np.degrees(sph_reg.theta.min()):.1f}°, {np.degrees(sph_reg.theta.max()):.1f}°]")
print(f"  phi:   [{np.degrees(sph_reg.phi.min()):.1f}°, {np.degrees(sph_reg.phi.max()):.1f}°]")

# Convert colatitude to latitude for intuition
lat_min = 90 - np.degrees(sph_reg.theta.max())
lat_max = 90 - np.degrees(sph_reg.theta.min())
print(f"  (Latitude: [{lat_min:.1f}°, {lat_max:.1f}°])")

# %% [markdown]
r"""
## Practical Example: Radial Gravity

For spherical geometry, gravity points toward the center (negative radial).
The unit vectors let us express this naturally.
"""

# %%
# Gravity as a symbolic expression
g_magnitude = 1.0
gravity = -g_magnitude * sph_reg.unit_r  # Negative = toward center

# Evaluate at a test point
x, y, z = sph_reg.to_cartesian(0.95, np.radians(122.5), np.radians(140))
test_xyz = np.array([[x, y, z]])

grav_numeric = np.zeros(3)
for i in range(3):
    val = uw.function.evaluate(gravity[i], test_xyz, regional.X)
    grav_numeric[i] = float(val.flat[0])

print(f"Gravity at test point: {grav_numeric}")
print(f"Magnitude: {np.linalg.norm(grav_numeric):.4f}")
print("(Points toward origin, as expected)")

# %% [markdown]
r"""
## Notes

1. **Colatitude vs Latitude**: The spherical $\theta$ is colatitude (from north pole),
   not latitude. Convert: `latitude = 90° - θ`

2. **Internal representation**: Mesh stores Cartesian $(x, y, z)$ internally.
   Use `mesh.X.coords` for Cartesian, `mesh.X.spherical.coords` for spherical.

3. **For Earth with ellipsoidal shape**: Use `RegionalGeographicBox` instead
   (see `Ex_Geographic_Meshes.py`).

## See Also

- `Ex_Geographic_Meshes.py` — For ellipsoidal Earth geometry
- `mesh.X.spherical.view()` — Full property reference
"""
