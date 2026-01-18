# Gmsh Integration Developer Guide

**Status**: Draft
**Date**: January 2026
**Purpose**: Technical guide for developers working with gmsh meshes in Underworld3

---

## Overview

Underworld3 uses [gmsh](https://gmsh.info/) for mesh generation, with meshes ultimately stored in PETSc's DMPlex format. This guide explains the complete pipeline from gmsh geometry to solver-ready mesh, and how to work with external gmsh files.

```{mermaid}
flowchart LR
    subgraph Input["Mesh Sources"]
        direction TB
        G1["gmsh Python API"]
        G2["pygmsh"]
        G3["External .msh file"]
        G4["Gmsh GUI (.geo)"]
    end

    subgraph Convert["Conversion Layer"]
        direction TB
        C1["_from_gmsh()"]
        C2["PETSc.DMPlex.createFromFile()"]
    end

    subgraph UW["Underworld Mesh"]
        direction TB
        U1["UW_Boundaries label"]
        U2["Coordinate system"]
        U3["Mesh variables"]
    end

    Input --> Convert
    Convert --> UW
```

---

## The Two Gmsh Interfaces

### 1. gmsh Python API (Direct)

The native gmsh Python interface provides full control over mesh generation:

```python
import gmsh

gmsh.initialize()
gmsh.model.add("MyMesh")

# Create geometry
p1 = gmsh.model.geo.add_point(0, 0, 0, meshSize=0.1)
p2 = gmsh.model.geo.add_point(1, 0, 0, meshSize=0.1)
# ... more geometry ...

# Create physical groups (CRITICAL for boundaries)
gmsh.model.add_physical_group(1, [line_id], tag=1)
gmsh.model.set_physical_name(1, 1, "Bottom")

gmsh.model.geo.synchronize()
gmsh.model.mesh.generate(2)
gmsh.write("mesh.msh")
gmsh.finalize()
```

**Underworld uses this directly** in all built-in mesh functions (`StructuredQuadBox`, `Annulus`, etc.).

### 2. pygmsh (Higher-Level Wrapper)

pygmsh provides a more Pythonic interface:

```python
import pygmsh

with pygmsh.geo.Geometry() as geom:
    # Create geometry with cleaner syntax
    rect = geom.add_rectangle(0, 1, 0, 1, z=0, mesh_size=0.1)

    # Physical groups
    geom.add_physical(rect.lines[0], label="Bottom")
    geom.add_physical(rect.lines[2], label="Top")

    mesh = geom.generate_mesh()
    mesh.write("mesh.msh")
```

**pygmsh ultimately calls gmsh** — it's a convenience wrapper, not a separate mesher.

### Key Difference for Underworld

| Aspect | gmsh API | pygmsh |
|--------|----------|--------|
| Physical group tags | Explicit control via `tag=` parameter | Auto-assigned (less predictable) |
| Integration with UW | Direct — UW mesh functions use this | Requires tag discovery |
| Flexibility | Full gmsh feature access | Simplified common operations |

**Recommendation**: For new Underworld mesh types, use the **gmsh API directly** to maintain explicit control over physical group tags.

---

## Physical Groups: The Critical Concept

Physical groups are how gmsh marks boundaries and regions. They are **essential** for Underworld boundary conditions.

### Anatomy of a Physical Group

```python
# In gmsh:
gmsh.model.add_physical_group(
    dim,      # 0=point, 1=line/edge, 2=surface, 3=volume
    tags,     # List of entity IDs to include
    tag=N     # The physical group's numerical identifier
)
gmsh.model.set_physical_name(dim, N, "Name")
```

### How Underworld Uses Physical Groups

1. **Tag values become boundary enum values**:
   ```python
   # In mesh creation:
   class Boundaries(Enum):
       Bottom = 1
       Right = 2
       Top = 3
       Left = 4

   # In gmsh geometry:
   gmsh.model.add_physical_group(1, [bottom_line], tag=1)  # Must match!
   gmsh.model.add_physical_group(1, [right_line], tag=2)
   ```

2. **Names become boundary enum names** (for documentation/debugging)

3. **All boundaries consolidated into `UW_Boundaries`**:
   ```python
   # Internally, Underworld creates:
   dm.createLabel("UW_Boundaries")
   for b in boundaries:
       # Each boundary's entities get tagged with b.value
       stacked_bc_label.setStratumIS(b.value, entity_indices)
   ```

### The Elements Physical Group

Underworld requires a physical group for the mesh interior:

```python
# 2D mesh: surface elements
gmsh.model.addPhysicalGroup(2, [surface_id], 99999)
gmsh.model.setPhysicalName(2, 99999, "Elements")

# 3D mesh: volume elements
gmsh.model.addPhysicalGroup(3, [volume_id], 99999)
gmsh.model.setPhysicalName(3, 99999, "Elements")
```

**Tag 99999** is convention but not required — it just needs to not conflict with boundary tags.

---

## The Conversion Pipeline

### Step 1: Gmsh to .msh File

```python
gmsh.write("mesh.msh")  # Gmsh's native format (version 4.x)
```

### Step 2: .msh to PETSc DMPlex

```python
# In _from_gmsh() (discretisation_mesh.py):

# Key PETSc options for gmsh import:
options["dm_plex_gmsh_multiple_tags"] = True   # Allow entities in multiple groups
options["dm_plex_gmsh_use_regions"] = True     # Import physical groups as labels

# Create DMPlex from file:
plex = PETSc.DMPlex().createFromFile(filename, interpolate=True)

# Mark all boundary faces (edges in 2D):
plex.markBoundaryFaces("All_Boundaries", 1001)
```

### Step 3: DMPlex to Underworld Mesh

```python
# In Mesh.__init__():

# Create unified boundary label:
self.dm.createLabel("UW_Boundaries")
stacked_bc_label = self.dm.getLabel("UW_Boundaries")

# Populate from individual gmsh physical group labels:
for b in self.boundaries:
    lab = self.dm.getLabel(b.name)  # gmsh physical group name
    if lab:
        lab_is = lab.getStratumIS(b.value)
        if lab_is:
            stacked_bc_label.setStratumIS(b.value, lab_is)
```

### The Label Structure

After conversion, the DMPlex has these labels:

| Label Name | Purpose | Values |
|------------|---------|--------|
| `UW_Boundaries` | Unified boundary label | One value per boundary type |
| `All_Boundaries` | All exterior faces/edges | 1001 |
| `Bottom`, `Top`, etc. | Individual gmsh physical groups | Original tag values |
| `Elements` | Interior elements | 99999 (convention) |

---

## Loading External Gmsh Files

### Current Approach

```python
from enum import Enum

# 1. Define boundaries enum matching gmsh physical groups
class MyBoundaries(Enum):
    Inlet = 1
    Outlet = 2
    Wall = 3
    # Values MUST match gmsh physical group tags!

# 2. Load mesh
mesh = uw.discretisation.Mesh(
    "external_mesh.msh",
    boundaries=MyBoundaries,
    coordinate_system_type=uw.coordinates.CoordinateSystemType.CARTESIAN
)
```

### The Problem

You need to know the exact physical group tags in the gmsh file. Currently there's no easy way to discover these.

---

## Proposed Utility API

To simplify working with external gmsh files:

### Inspection

```python
import underworld3 as uw

# Discover what's in a gmsh file
info = uw.meshing.inspect_gmsh("mesh.msh")

# Returns dict like:
{
    'dimension': 2,
    'physical_groups': [
        {'dim': 1, 'tag': 1, 'name': 'Bottom', 'entity_count': 10},
        {'dim': 1, 'tag': 2, 'name': 'Right', 'entity_count': 10},
        {'dim': 1, 'tag': 3, 'name': 'Top', 'entity_count': 10},
        {'dim': 1, 'tag': 4, 'name': 'Left', 'entity_count': 10},
        {'dim': 2, 'tag': 99999, 'name': 'Elements', 'entity_count': 200},
    ],
    'nodes': 231,
    'elements': {'triangles': 200, 'lines': 40}
}
```

### Automatic Boundary Enum Creation

```python
# Create boundaries enum from gmsh file
boundaries = uw.meshing.boundaries_from_gmsh("mesh.msh")

# Returns Enum like:
# class Boundaries(Enum):
#     Bottom = 1
#     Right = 2
#     Top = 3
#     Left = 4

# Then use it:
mesh = uw.discretisation.Mesh("mesh.msh", boundaries=boundaries)
```

### Validation

```python
# Check if file is compatible before loading
issues = uw.meshing.validate_gmsh("mesh.msh")

# Returns list of issues:
# - "No physical groups for dimension 1 (boundaries)"
# - "No 'Elements' physical group found"
# - "Physical group tag 0 is reserved"
```

### Convenience Loader

```python
# One-step loading with auto-discovery
mesh = uw.meshing.load_gmsh("mesh.msh")

# Equivalent to:
boundaries = uw.meshing.boundaries_from_gmsh("mesh.msh")
mesh = uw.discretisation.Mesh("mesh.msh", boundaries=boundaries)
```

---

## Creating New Mesh Types

When adding a new mesh function to Underworld, follow this pattern:

### 1. Define the Boundaries Enum

```python
from enum import Enum

class MyMeshBoundaries(Enum):
    """Boundaries for MyMesh geometry."""
    Inner = 1
    Outer = 2
    Side_A = 3
    Side_B = 4
```

### 2. Create Gmsh Geometry with Matching Tags

```python
def MyMesh(inner_radius, outer_radius, cellSize, ...):
    import gmsh

    boundaries = MyMeshBoundaries

    gmsh.initialize()
    gmsh.model.add("MyMesh")

    # Create geometry...
    inner_arc = gmsh.model.geo.add_circle_arc(...)
    outer_arc = gmsh.model.geo.add_circle_arc(...)

    # Physical groups with MATCHING tags
    gmsh.model.add_physical_group(1, [inner_arc], tag=boundaries.Inner.value)
    gmsh.model.set_physical_name(1, boundaries.Inner.value, boundaries.Inner.name)

    gmsh.model.add_physical_group(1, [outer_arc], tag=boundaries.Outer.value)
    gmsh.model.set_physical_name(1, boundaries.Outer.value, boundaries.Outer.name)

    # Don't forget the elements group!
    gmsh.model.addPhysicalGroup(2, [surface], 99999)
    gmsh.model.setPhysicalName(2, 99999, "Elements")

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(2)

    # Save and convert
    filename = f".meshes/my_mesh_{params}.msh"
    gmsh.write(filename)
    gmsh.finalize()

    # Create Underworld mesh
    mesh = uw.discretisation.Mesh(
        filename,
        boundaries=boundaries,
        coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
        ...
    )

    return mesh
```

### 3. Key Requirements Checklist

- [ ] Boundaries enum defined with unique positive integer values
- [ ] Physical group tags match enum values exactly
- [ ] Physical group names match enum names
- [ ] Elements physical group included (dim=2 for 2D, dim=3 for 3D)
- [ ] No tag value conflicts (avoid 0, 1001, 666 which are reserved)
- [ ] Coordinate system type specified correctly

---

## PETSc DMPlex Internals

### Label System

DMPlex uses "labels" to mark mesh entities:

```python
# Create a label
dm.createLabel("MyLabel")
label = dm.getLabel("MyLabel")

# Mark entities (points in PETSc terminology)
label.setValue(point_id, value)

# Or mark many at once with IndexSet
label.setStratumIS(value, index_set)

# Query
values = label.getNonEmptyStratumValuesIS().getIndices()
points = label.getStratumIS(value).getIndices()
```

### Entity Numbering

DMPlex numbers entities in a specific order:
- **Cells** (highest dimension): 0 to nCells-1
- **Faces** (codimension 1): nCells to nCells+nFaces-1
- **Edges** (codimension 2, 3D only): ...
- **Vertices**: last

```python
# Get ranges
cStart, cEnd = dm.getHeightStratum(0)  # Cells
fStart, fEnd = dm.getHeightStratum(1)  # Faces
vStart, vEnd = dm.getDepthStratum(0)   # Vertices
```

### Coordinate Access

```python
# Get coordinate section and vector
coordSection = dm.getCoordinateSection()
coordVec = dm.getCoordinatesLocal()

# For a vertex, get its coordinates
coords = dm.getCoordinates().array.reshape(-1, dim)
```

---

## Troubleshooting

### "Boundary not found" errors

**Cause**: Physical group tag doesn't match boundary enum value.

**Solution**: Use `inspect_gmsh()` to check tags, or ensure enum values match gmsh tags exactly.

### Missing boundaries after loading

**Cause**: Physical groups not created in gmsh, or wrong dimension.

**Solution**: Ensure `add_physical_group` is called for each boundary with correct dimension (1 for 2D mesh boundaries, 2 for 3D mesh boundaries).

### "Elements" group issues

**Cause**: No physical group for the mesh interior.

**Solution**: Add physical group for surfaces (2D) or volumes (3D) with tag 99999.

### Parallel loading issues

**Cause**: DMPlex distribution happening before labels are set up.

**Solution**: The `_from_gmsh()` function handles this by loading on rank 0, saving to HDF5, then loading in parallel. Don't modify this pattern.

---

## Reserved Values

| Value | Purpose | Don't Use For |
|-------|---------|---------------|
| 0 | Often used by gmsh for "no group" | User boundaries |
| 666 | `Null_Boundary` in Underworld | User boundaries |
| 1001 | `All_Boundaries` (all exterior) | User boundaries |
| 99999 | `Elements` (convention) | User boundaries |

---

## Advanced Patterns from Examples

The following patterns are used in the `docs/examples/` notebooks for creating unusual meshes.

### Elliptical Geometries

Create elliptical boundaries using `add_ellipse_arc()`:

```python
# From Ex_Stokes_Ellipse_Cartesian.py
# Inner ellipse with variable ellipticity
p0 = gmsh.model.geo.add_point(0.0, 0.0, 0.0, meshSize=cellSize)  # Center
p1 = gmsh.model.geo.add_point(radius * ellipticity, 0.0, 0.0, meshSize=cellSize)
p2 = gmsh.model.geo.add_point(0.0, radius, 0.0, meshSize=cellSize)
p3 = gmsh.model.geo.add_point(-radius * ellipticity, 0.0, 0.0, meshSize=cellSize)
p4 = gmsh.model.geo.add_point(0.0, -radius, 0.0, meshSize=cellSize)

# Quarter arcs: add_ellipse_arc(start, center, major_axis_point, end)
c1 = gmsh.model.geo.add_ellipse_arc(p1, p0, p1, p2)
c2 = gmsh.model.geo.add_ellipse_arc(p2, p0, p3, p3)
c3 = gmsh.model.geo.add_ellipse_arc(p3, p0, p3, p4)
c4 = gmsh.model.geo.add_ellipse_arc(p4, p0, p1, p1)

cl = gmsh.model.geo.add_curve_loop([c1, c2, c3, c4])
```

**Key insight**: For elliptical boundaries, surface normals must be computed analytically:
```python
# Analytical surface normal for ellipse
Gamma_N = sympy.Matrix([2*x / ellipticity**2, 2*y]).T
Gamma_N = Gamma_N / sympy.sqrt(Gamma_N.dot(Gamma_N))
```

### Boolean Operations for Complex Domains

Use OpenCASCADE (`gmsh.model.occ`) for boolean operations:

```python
# From Ex_Explicit_Flow_Grains.py
# Create domain and cut out circular inclusions
domain_loop = gmsh.model.occ.add_curve_loop((bottom, right, top, left))
gmsh.model.occ.add_plane_surface([domain_loop])

# Create circular inclusions
inclusions = []
for row in range(rows):
    for col in range(cols):
        # Create circle using quarter arcs
        inclusion_loop = gmsh.model.occ.add_curve_loop(quarter_circles)
        inclusion = gmsh.model.occ.add_plane_surface([inclusion_loop])
        inclusions.append((2, inclusion))

# Cut inclusions from domain
domain_cut, index = gmsh.model.occ.cut([(2, domain_loop)], inclusions)
```

**Tracking boundaries after cut**: Bounding boxes can change after boolean operations:
```python
# Save bounding boxes BEFORE cut
bboxes = [gmsh.model.get_bounding_box(1, line) for line in [bottom, right, top, left]]

# After cut, match boundaries by bounding box
for new_line in new_boundary_lines:
    new_bbox = gmsh.model.occ.get_bounding_box(1, new_line)
    original_index = bboxes.index(new_bbox)  # Find matching original
```

### Refinement Callbacks

Snap nodes back to curved boundaries after mesh refinement:

```python
# From Ex_Stokes_Flow_Internal_BC.py
def mesh_refinement_callback(dm):
    """Ensure circular boundary stays on circle after refinement."""
    c2 = dm.getCoordinatesLocal()
    coords = c2.array.reshape(-1, 2) - centre

    R = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2).reshape(-1, 1)

    # Find points on the circular boundary
    circle_indices = uw.cython.petsc_discretisation.petsc_dm_find_labeled_points_local(
        dm, "inclusion"
    )

    # Snap to exact radius
    coords[circle_indices] *= radius / R[circle_indices]
    coords = coords + centre

    c2.array[...] = coords.reshape(-1)
    dm.setCoordinatesLocal(c2)

# Use in mesh creation
mesh = uw.discretisation.Mesh(
    "mesh.msh",
    refinement=2,
    refinement_callback=mesh_refinement_callback,
    ...
)
```

### Return Coords to Bounds Callback

Restore particles that escape the domain (important for inflow boundaries):

```python
# From Ex_Stokes_Flow_Internal_BC.py
def return_coords_to_bounds(coords):
    """Restore particles that moved past inflow boundary."""
    lefty_troublemakers = coords[:, 0] < 0.0
    coords[lefty_troublemakers, 0] = 0.0001  # Push back inside
    return coords

mesh = uw.discretisation.Mesh(
    "mesh.msh",
    return_coords_to_bounds=return_coords_to_bounds,
    ...
)
```

### Embedding Curves in Surfaces (pygmsh)

For internal boundaries that don't split the domain:

```python
# From Ex_Stokes_Flow_Internal_BC.py (pygmsh)
with pygmsh.geo.Geometry() as geom:
    inclusion = geom.add_circle((cx, cy, 0.0), radius, mesh_size=fine_size)
    domain = geom.add_rectangle(xmin=0, xmax=w, ymin=0, ymax=h, mesh_size=coarse_size)

    # Embed inclusion curves IN the domain surface (doesn't cut through)
    for curve in inclusion.curve_loop.curves:
        geom.in_surface(curve, domain)

    # Physical groups
    geom.add_physical(inclusion.curve_loop.curves, label="inclusion")
    geom.add_physical(domain.surface, label="Elements")
```

### Complex Notch Geometry

Rounded corners using circle arcs (from `.geo` file patterns):

```python
# From Ex_Shear_Band_Notch_Benchmark.py
# Points at corner with different refinement levels
Point_outer = gmsh.model.geo.add_point(x1, y1, 0, cl_coarse)
Point_inner = gmsh.model.geo.add_point(x2, y2, 0, cl_fine)
Point_center = gmsh.model.geo.add_point(xc, yc, 0, cl_fine)

# Rounded corner using circle arc
Circle = gmsh.model.geo.addCircleArc(Point_outer, Point_center, Point_inner)
```

### pygmsh vs gmsh API: Tag Handling

**pygmsh** auto-assigns tags (convenient but less predictable):
```python
# pygmsh - tags auto-assigned, use labels for boundaries
geom.add_physical(domain.surface.curve_loop.curves[0], label="bottom")
geom.add_physical(domain.surface.curve_loop.curves[1], label="right")
```

**gmsh API** requires explicit tags (more control):
```python
# gmsh API - explicit tags matching boundary enum
gmsh.model.addPhysicalGroup(1, [bottom_line], boundaries.Bottom.value, name="Bottom")
gmsh.model.addPhysicalGroup(1, [right_line], boundaries.Right.value, name="Right")
```

**Recommendation**: Use gmsh API for new Underworld mesh types (explicit control). Use pygmsh for quick prototyping.

### Manifold Meshes (Surfaces in 3D)

For spherical or other surface meshes, use built-in functions:

```python
# From Manifold_S2.py
# 2D surface mesh embedded in 3D space
mesh = uw.meshing.SegmentedSphericalSurface2D(
    cellSize=0.05,
    numSegments=3,
    qdegree=3,
    filename="manifold.msh"
)

# Coordinates are 2D (lon, lat) but embedded in 3D
lon, lat = mesh.CoordinateSystem.N
```

---

## Example Files Reference

These example notebooks demonstrate advanced mesh patterns (in `docs/examples/`):

| Example | Key Patterns | Location |
|---------|--------------|----------|
| **Elliptical annulus** | Ellipse arcs, analytical normals, free-slip BCs | `fluid_mechanics/intermediate/Ex_Stokes_Ellipse_Cartesian.py` |
| **Grain pack flow** | Boolean cuts, random geometry, many inclusions | `porous_flow/advanced/Ex_Explicit_Flow_Grains.py` |
| **Internal boundaries** | pygmsh, refinement callback, embedded curves | `fluid_mechanics/intermediate/Ex_Stokes_Flow_Internal_BC.py` |
| **Circular obstruction** | pygmsh, refinement callback, DFG benchmark | `fluid_mechanics/advanced/Ex_Navier_Stokes_Benchmarks_NS_DFG_2d.py` |
| **Notch geometry** | Complex .geo-style, rounded corners, multiple cell sizes | `solid_mechanics/advanced/Ex_Shear_Band_Notch_Benchmark.py` |
| **Terrain mesh** | Discrete entities, 3D mesh from topography | `fluid_mechanics/intermediate/Ex_ChannelFlow_IrregularBase.py` |
| **Spherical manifold** | Surface mesh in 3D, segmented sphere | `utilities/advanced/Manifold_S2.py` |

---

## References

- [Gmsh Python API Tutorial](https://gmsh.info/doc/texinfo/gmsh.html#Gmsh-API)
- [pygmsh Documentation](https://pygmsh.readthedocs.io/)
- [PETSc DMPlex Manual](https://petsc.org/main/manual/dmplex/)
- Underworld source: `src/underworld3/discretisation/discretisation_mesh.py`
- Underworld source: `src/underworld3/meshing/` (mesh function examples)

---

## Future Work

### Utility Functions (Implementation Priority)

1. **`inspect_gmsh()`**: Discover physical groups, tags, and element types in a `.msh` file
2. **`boundaries_from_gmsh()`**: Auto-generate boundary enum from `.msh` file
3. **`validate_gmsh()`**: Check file compatibility before loading
4. **`load_gmsh()`**: One-step loading with auto-discovery

### Infrastructure Improvements

5. **Support for .geo files**: Call gmsh to mesh on load
6. **meshio integration**: Alternative import path for other mesh formats
7. **Better error messages**: Catch tag mismatches early with helpful diagnostics

### User-Facing Documentation

**Created**: See [Custom Mesh Creation](../../advanced/custom-meshes.md) in the Advanced Usage section.

Covers practical usage for researchers who need custom geometries but aren't contributing to the codebase.

---

*Last updated: January 2026*
