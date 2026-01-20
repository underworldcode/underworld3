# Custom Mesh Creation

**When built-in meshes aren't enough**

Underworld3 provides several built-in mesh functions (`StructuredQuadBox`, `Annulus`, `SphericalShell`, etc.), but many research problems require custom geometries: circular inclusions, elliptical domains, grain packs, notched specimens, or terrain-following meshes.

This guide shows you how to create custom meshes using [gmsh](https://gmsh.info/) and load them into Underworld3.

---

## When Do You Need a Custom Mesh?

Use custom meshes when you need:

- **Complex domain shapes**: Ellipses, notches, irregular boundaries
- **Internal boundaries**: Inclusions, holes, embedded interfaces
- **Local refinement**: Fine mesh near features, coarse elsewhere
- **Multiple materials**: Distinct regions with different properties
- **Terrain/topography**: Meshes that follow real-world surfaces

If your domain is a simple box, annulus, or sphere, use the [built-in mesh functions](../beginner/tutorials/1-Meshes.ipynb) instead.

---

## Quick Start: Circular Inclusion

Here's a complete example of creating a channel with a circular obstacle:

```python
import underworld3 as uw
import numpy as np
from enum import Enum

# Step 1: Define boundary labels
class boundaries(Enum):
    bottom = 1
    right = 2
    top = 3
    left = 4
    inclusion = 5

# Step 2: Create mesh with gmsh (on rank 0 only)
if uw.mpi.rank == 0:
    import gmsh

    gmsh.initialize()
    gmsh.model.add("ChannelWithInclusion")

    # Domain parameters
    width, height = 2.0, 1.0
    cx, cy, radius = 0.5, 0.5, 0.15
    cell_size = 0.05
    fine_size = 0.02  # Finer near inclusion

    # Create rectangle (domain)
    p1 = gmsh.model.geo.addPoint(0, 0, 0, cell_size)
    p2 = gmsh.model.geo.addPoint(width, 0, 0, cell_size)
    p3 = gmsh.model.geo.addPoint(width, height, 0, cell_size)
    p4 = gmsh.model.geo.addPoint(0, height, 0, cell_size)

    bottom = gmsh.model.geo.addLine(p1, p2)
    right = gmsh.model.geo.addLine(p2, p3)
    top = gmsh.model.geo.addLine(p3, p4)
    left = gmsh.model.geo.addLine(p4, p1)

    # Create circle (inclusion)
    pc = gmsh.model.geo.addPoint(cx, cy, 0, fine_size)
    p5 = gmsh.model.geo.addPoint(cx + radius, cy, 0, fine_size)
    p6 = gmsh.model.geo.addPoint(cx, cy + radius, 0, fine_size)
    p7 = gmsh.model.geo.addPoint(cx - radius, cy, 0, fine_size)
    p8 = gmsh.model.geo.addPoint(cx, cy - radius, 0, fine_size)

    c1 = gmsh.model.geo.addCircleArc(p5, pc, p6)
    c2 = gmsh.model.geo.addCircleArc(p6, pc, p7)
    c3 = gmsh.model.geo.addCircleArc(p7, pc, p8)
    c4 = gmsh.model.geo.addCircleArc(p8, pc, p5)

    # Create surface with hole
    outer_loop = gmsh.model.geo.addCurveLoop([bottom, right, top, left])
    inner_loop = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
    surface = gmsh.model.geo.addPlaneSurface([outer_loop, inner_loop])

    gmsh.model.geo.synchronize()

    # Step 3: Define physical groups (CRITICAL!)
    # Tags MUST match your boundary enum values
    gmsh.model.addPhysicalGroup(1, [bottom], boundaries.bottom.value, name="bottom")
    gmsh.model.addPhysicalGroup(1, [right], boundaries.right.value, name="right")
    gmsh.model.addPhysicalGroup(1, [top], boundaries.top.value, name="top")
    gmsh.model.addPhysicalGroup(1, [left], boundaries.left.value, name="left")
    gmsh.model.addPhysicalGroup(1, [c1, c2, c3, c4], boundaries.inclusion.value, name="inclusion")
    gmsh.model.addPhysicalGroup(2, [surface], 99999, name="Elements")

    gmsh.model.mesh.generate(2)
    gmsh.write("channel_inclusion.msh")
    gmsh.finalize()

# Step 4: Load mesh into Underworld3
mesh = uw.discretisation.Mesh(
    "channel_inclusion.msh",
    boundaries=boundaries,
    useMultipleTags=True,
    useRegions=True,
)

# Now use as normal
v = uw.discretisation.MeshVariable("v", mesh, 2, degree=2)
p = uw.discretisation.MeshVariable("p", mesh, 1, degree=1)
```

---

## Key Concepts

### Physical Groups

Physical groups tell Underworld3 which mesh entities belong to which boundaries. **This is the most important concept for custom meshes.**

```python
# Physical group = (dimension, entities, tag, name)
gmsh.model.addPhysicalGroup(
    1,                          # dimension: 1 = edges/curves
    [line1, line2],             # entities: list of line IDs
    boundaries.MyBoundary.value, # tag: MUST match enum value
    name="MyBoundary"           # name: for debugging
)
```

**Dimension reference:**
- `0` = points (rarely used for BCs)
- `1` = lines/edges (2D boundary conditions)
- `2` = surfaces (3D boundary conditions, or 2D elements)
- `3` = volumes (3D elements)

### The Boundary Enum

Your boundary enum defines which boundaries exist and their numerical tags:

```python
from enum import Enum

class boundaries(Enum):
    Bottom = 1      # These values must match
    Top = 2         # the physical group tags
    Left = 3        # in your gmsh code
    Right = 4
    Inclusion = 5
```

**Common mistakes:**
- Tag mismatch: enum says `Bottom = 1` but gmsh uses tag `3`
- Missing Elements group: forgot to add the interior surface/volume
- Wrong dimension: using dim=2 for edge boundaries in 2D

### The Elements Group

You must define a physical group for the mesh interior:

```python
# For 2D meshes: surfaces are the elements
gmsh.model.addPhysicalGroup(2, [surface_id], 99999, name="Elements")

# For 3D meshes: volumes are the elements
gmsh.model.addPhysicalGroup(3, [volume_id], 99999, name="Elements")
```

The tag `99999` is convention but not required—just don't conflict with boundary tags.

---

## Common Patterns

### Elliptical Boundaries

```python
# Ellipse with semi-axes a (horizontal) and b (vertical)
a, b = 1.5, 1.0  # ellipticity = a/b

p0 = gmsh.model.geo.addPoint(0, 0, 0, cell_size)  # center
p1 = gmsh.model.geo.addPoint(a, 0, 0, cell_size)
p2 = gmsh.model.geo.addPoint(0, b, 0, cell_size)
p3 = gmsh.model.geo.addPoint(-a, 0, 0, cell_size)
p4 = gmsh.model.geo.addPoint(0, -b, 0, cell_size)

# Ellipse arcs: addEllipseArc(start, center, major_axis_point, end)
c1 = gmsh.model.geo.addEllipseArc(p1, p0, p1, p2)
c2 = gmsh.model.geo.addEllipseArc(p2, p0, p3, p3)
c3 = gmsh.model.geo.addEllipseArc(p3, p0, p3, p4)
c4 = gmsh.model.geo.addEllipseArc(p4, p0, p1, p1)
```

```{note}
For elliptical and curved boundaries, the mesh-derived normals (`mesh.Gamma`) may not accurately represent the true surface direction. See [Boundary Conditions on Curved Surfaces](curved-boundary-conditions.md) for detailed guidance on:
- Using projected normals (recommended)
- Computing analytical surface normals
- Understanding the accuracy trade-offs
```

### Multiple Inclusions

For many inclusions (grain packs, porous media), use OpenCASCADE boolean operations:

```python
import gmsh

gmsh.initialize()
gmsh.model.add("GrainPack")

# Create domain
domain = gmsh.model.occ.addRectangle(0, 0, 0, width, height)

# Create inclusions
inclusions = []
for i, (cx, cy, r) in enumerate(grain_positions):
    disk = gmsh.model.occ.addDisk(cx, cy, 0, r, r)
    inclusions.append((2, disk))

gmsh.model.occ.synchronize()

# Cut inclusions from domain
result, _ = gmsh.model.occ.cut([(2, domain)], inclusions)
gmsh.model.occ.synchronize()

# Now add physical groups...
```

### Local Mesh Refinement

Control mesh density by setting `meshSize` at points:

```python
# Coarse in bulk
p1 = gmsh.model.geo.addPoint(0, 0, 0, meshSize=0.1)

# Fine near feature
p2 = gmsh.model.geo.addPoint(cx, cy, 0, meshSize=0.01)
```

Or use gmsh fields for more control:

```python
# Refine near a point
gmsh.model.mesh.field.add("Distance", 1)
gmsh.model.mesh.field.setNumbers(1, "PointsList", [center_point])

gmsh.model.mesh.field.add("Threshold", 2)
gmsh.model.mesh.field.setNumber(2, "InField", 1)
gmsh.model.mesh.field.setNumber(2, "SizeMin", 0.01)
gmsh.model.mesh.field.setNumber(2, "SizeMax", 0.1)
gmsh.model.mesh.field.setNumber(2, "DistMin", 0.1)
gmsh.model.mesh.field.setNumber(2, "DistMax", 0.5)

gmsh.model.mesh.field.setAsBackgroundMesh(2)
```

---

## Mesh Callbacks

### Refinement Callback

When using mesh refinement, nodes on curved boundaries may drift. Use a callback to snap them back:

```python
def snap_to_circle(dm):
    """Snap inclusion boundary nodes back to exact circle."""
    coords_vec = dm.getCoordinatesLocal()
    coords = coords_vec.array.reshape(-1, 2)

    # Find points on the inclusion boundary
    inclusion_points = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
        dm, "inclusion"
    )

    # Compute current radius and snap to target
    dx = coords[inclusion_points, 0] - cx
    dy = coords[inclusion_points, 1] - cy
    r_current = np.sqrt(dx**2 + dy**2).reshape(-1, 1)

    coords[inclusion_points, 0] = cx + dx * radius / r_current.flatten()
    coords[inclusion_points, 1] = cy + dy * radius / r_current.flatten()

    coords_vec.array[...] = coords.reshape(-1)
    dm.setCoordinatesLocal(coords_vec)

# Use when loading mesh
mesh = uw.discretisation.Mesh(
    "mesh.msh",
    refinement=2,
    refinement_callback=snap_to_circle,
    boundaries=boundaries,
)
```

### Return Coords to Bounds

For simulations with inflow boundaries, particles may escape. Use this callback to push them back:

```python
def keep_particles_inside(coords):
    """Push escaped particles back inside domain."""
    # Particles that went past left boundary
    escaped_left = coords[:, 0] < 0.0
    coords[escaped_left, 0] = 0.001

    # Particles that went past right boundary
    escaped_right = coords[:, 0] > width
    coords[escaped_right, 0] = width - 0.001

    return coords

mesh = uw.discretisation.Mesh(
    "mesh.msh",
    return_coords_to_bounds=keep_particles_inside,
    boundaries=boundaries,
)
```

---

## Using pygmsh (Alternative)

[pygmsh](https://pygmsh.readthedocs.io/) provides a more Pythonic interface:

```python
import pygmsh

with pygmsh.geo.Geometry() as geom:
    # Create shapes
    circle = geom.add_circle([0.5, 0.5, 0], 0.15, mesh_size=0.02)
    rect = geom.add_rectangle(0, 2, 0, 1, z=0, mesh_size=0.05)

    # Embed circle in rectangle (internal boundary, not a hole)
    for curve in circle.curve_loop.curves:
        geom.in_surface(curve, rect)

    # Physical groups (uses labels, auto-assigns tags)
    geom.add_physical(rect.surface.curve_loop.curves[0], label="bottom")
    geom.add_physical(rect.surface.curve_loop.curves[1], label="right")
    geom.add_physical(rect.surface.curve_loop.curves[2], label="top")
    geom.add_physical(rect.surface.curve_loop.curves[3], label="left")
    geom.add_physical(circle.curve_loop.curves, label="inclusion")
    geom.add_physical(rect.surface, label="Elements")

    geom.generate_mesh(dim=2)
    geom.save_geometry("mesh.msh")
```

**Note**: pygmsh auto-assigns physical group tags, so you'll need to discover them or use Underworld's auto-boundary feature (see Troubleshooting).

---

## Troubleshooting

### "Boundary not found" error

**Cause**: Physical group tag doesn't match boundary enum value.

**Fix**: Ensure your enum values match the tags in `addPhysicalGroup()`:
```python
class boundaries(Enum):
    Bottom = 1  # Must match tag in addPhysicalGroup(..., 1, ...)
```

### Mesh loads but boundaries don't work

**Cause**: Physical groups not created, or wrong dimension.

**Fix**: Check that:
1. You called `addPhysicalGroup()` for each boundary
2. You used dimension 1 for edges (2D) or dimension 2 for faces (3D)
3. You included the Elements group

### Nodes drift off curved boundaries

**Cause**: Mesh refinement moves nodes, but they're not snapped back.

**Fix**: Use a `refinement_callback` (see above).

### Particles escape at inflow

**Cause**: Advection pushes particles outside domain.

**Fix**: Use `return_coords_to_bounds` callback (see above).

### "No elements found" error

**Cause**: Missing Elements physical group.

**Fix**: Add the interior elements group:
```python
gmsh.model.addPhysicalGroup(2, [surface], 99999, name="Elements")
```

---

## Tips for Success

1. **Start simple**: Get a basic mesh working before adding complexity
2. **Visualize early**: Use `gmsh.fltk.run()` to see your mesh before saving
3. **Check physical groups**: Use `gmsh.model.getPhysicalGroups()` to verify
4. **Match tags exactly**: Enum values must equal physical group tags
5. **Test boundaries**: Apply simple BCs to verify each boundary works
6. **Use rank 0 only**: Wrap gmsh code in `if uw.mpi.rank == 0:`

---

## Further Reading

- [Gmsh Python API Tutorial](https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-API)
- [pygmsh Documentation](https://pygmsh.readthedocs.io/)
- [Developer Guide: Gmsh Integration](../developer/guides/GMSH_INTEGRATION_GUIDE.md) — Internal architecture details

---

*Last updated: January 2026*
