# Mesh Geometry System Audit

## Implementation Location (Updated 2025-09-29)
**Current Status:** Mesh generation functions reorganized into specialized modules:
- **Cartesian**: `src/underworld3/meshing/cartesian.py` - UnstructuredSimplexBox, StructuredQuadBox, BoxInternalBoundary
- **Spherical**: `src/underworld3/meshing/spherical.py` - SphericalShell, SphericalShellInternalBoundary, SegmentofSphere, CubedSphere
- **Annulus**: `src/underworld3/meshing/annulus.py` - Annulus, QuarterAnnulus, SegmentofAnnulus, AnnulusWithSpokes, AnnulusInternalBoundary, DiscInternalBoundaries
- **Geographic**: `src/underworld3/meshing/geographic.py` - RegionalSphericalBox
- **Segmented**: `src/underworld3/meshing/segmented.py` - SegmentedSphericalSurface2D, SegmentedSphericalShell, SegmentedSphericalBall

All functions maintain backward compatibility via `src/underworld3/meshing/__init__.py` imports.

## Current Mesh Types and Their Geometric Definitions

### Cartesian Meshes
- **UnstructuredSimplexBox** (`CoordinateSystemType.CARTESIAN`)
- **StructuredQuadBox** (`CoordinateSystemType.CARTESIAN`)
- **BoxInternalBoundary** (`CoordinateSystemType.CARTESIAN`)

**Confirmed unit vectors:** ✅
- `unit_e_0` = [1, 0] or [1, 0, 0] (x-direction)
- `unit_e_1` = [0, 1] or [0, 1, 0] (y-direction) 
- `unit_e_2` = [0, 0, 1] (z-direction, 3D only)

**Implementation:** `_rRotN = sympy.eye(mesh.dim)` (identity matrix)

**Proposed standardization:**
```python
@property
def unit_horizontal(self):
    return self.CoordinateSystem.unit_e_0  # Primary horizontal

@property  
def unit_vertical(self):
    return self.CoordinateSystem.unit_e_1  # Upward direction (2D)
    # return self.CoordinateSystem.unit_e_2  # Upward direction (3D)
```

### Cylindrical/Polar Meshes  
- **Annulus** (`CoordinateSystemType.CYLINDRICAL2D`)
- **QuarterAnnulus** (`CoordinateSystemType.CYLINDRICAL2D`)
- **SegmentofAnnulus** (`CoordinateSystemType.CYLINDRICAL2D`)
- **AnnulusWithSpokes** (`CoordinateSystemType.CYLINDRICAL2D`)
- **AnnulusInternalBoundary** (`CoordinateSystemType.CYLINDRICAL2D`)

**Confirmed unit vectors:** ✅
- `unit_e_0` = [cos(θ), sin(θ)] (radial direction, outward from center)
- `unit_e_1` = [-sin(θ), cos(θ)] (tangential direction, counter-clockwise)

**Implementation:** 
```python
_rRotN = sympy.Matrix([
    [sympy.cos(theta), sympy.sin(theta)],
    [-sympy.sin(theta), sympy.cos(theta)]
])
# where theta = sympy.atan2(y, x)
```

**Proposed standardization:**
```python
@property
def unit_radial(self):
    return self.CoordinateSystem.unit_e_0
    
@property
def unit_tangential(self):
    return self.CoordinateSystem.unit_e_1
    
@property
def unit_vertical(self):
    # INTENTIONAL: "vertical" enables coordinate-system-independent mathematics
    # In 2D annulus context, "vertical" means Cartesian y-direction
    # This allows writing: force = gravity * mesh.unit_vertical
    # The same code works across different mesh types with appropriate interpretation
    return sympy.Matrix([0, 1])  # Cartesian vertical interpretation
```

### Spherical Meshes
- **CubedSphere** (`CoordinateSystemType.SPHERICAL`)
- **RegionalSphericalBox** (`CoordinateSystemType.SPHERICAL` or `SPHERICAL_NATIVE`)  
- **SegmentofSphere** (`CoordinateSystemType.SPHERICAL`)

**Confirmed unit vectors:** ✅
- `unit_e_0` = [x/r, y/r, z/r] (radial direction, outward from center)
- `unit_e_1` = spherical theta direction (colatitude/meridional)
- `unit_e_2` = spherical phi direction (azimuthal/longitude)

**Implementation:**
```python
# For SPHERICAL coordinate system:
_rRotN = sympy.Matrix([
    [x/r, y/r, z/r],                    # radial
    [theta_direction_components],        # meridional  
    [phi_direction_components]           # azimuthal
])
# where r = sqrt(x² + y² + z²)
```

**Proposed standardization:**
```python
@property
def unit_radial(self):
    return self.CoordinateSystem.unit_e_0
    
@property
def unit_meridional(self):
    return self.CoordinateSystem.unit_e_1
    
@property  
def unit_azimuthal(self):
    return self.CoordinateSystem.unit_e_2
    
@property
def unit_vertical(self):
    # "Vertical" means "upward" consistently across all coordinate systems
    # In spherical: radially outward from center = "up" (away from planet center)
    # This maintains consistency with Cartesian "up" direction
    return self.CoordinateSystem.unit_e_0  # Radially outward (upward)
```

## Design Philosophy: Mathematical Independence

### INTENTIONAL: Coordinate-System-Independent Mathematics
The apparent "ambiguity" of terms like "vertical" is actually **intentional design** to enable mathematical expressions that work across different coordinate systems:

```python
# This code works regardless of mesh type:
gravitational_force = density * gravity * mesh.unit_vertical
buoyancy_force = density_difference * gravity * mesh.unit_vertical
thermal_diffusion = temperature.diff(mesh.unit_vertical)

# Different interpretations by coordinate system:
# - Cartesian: unit_vertical = [0,1] or [0,0,1] (y or z direction)
# - Spherical: unit_vertical = unit_radial (outward from center)
# - Annulus: unit_vertical = [0,1] (Cartesian y-direction)
```

**Benefits:**
- **Mathematical Abstraction**: Write physics equations once, work everywhere
- **Code Reuse**: Same solver code works with different mesh geometries
- **Physical Intuition**: "Vertical" always means "upward" regardless of coordinate system
- **Flexibility**: Mesh determines appropriate geometric interpretation

### Example: Gravity-Driven Flow
```python
# Works for any mesh type - Cartesian, spherical, annulus, etc.
def setup_gravity_flow(mesh, density_field, gravity_magnitude):
    gravity_vector = gravity_magnitude * mesh.unit_vertical
    body_force = density_field * gravity_vector
    return body_force
```

This approach prioritizes **mathematical elegance** and **coordinate-system independence** over rigid geometric specificity.

## Proposed Mesh Interface

### Base Mesh Geometric Interface
```python
class Mesh:
    # Required geometric properties (all meshes must implement)
    @property
    def geometric_dimension_names(self):
        """Return names of geometric dimensions for this mesh type"""
        raise NotImplementedError
        
    @property  
    def primary_directions(self):
        """Return dict of primary geometric directions"""
        raise NotImplementedError
        
    # Sampling interface
    def create_line_sample(self, start, direction_expr, length, num_points=50):
        """Create sample points along sympy-defined direction"""
        # Generic implementation using coordinate system
        
    def create_profile_sample(self, profile_type, **params):
        """Create sample for common profile types"""
        # Mesh-specific implementation
```

### Mesh-Specific Implementations

#### Cartesian Meshes
```python
class StructuredQuadBox(Mesh):
    @property
    def geometric_dimension_names(self):
        return ['horizontal', 'vertical'] if self.dim == 2 else ['horizontal_x', 'horizontal_y', 'vertical']
        
    @property
    def primary_directions(self):
        if self.dim == 2:
            return {
                'unit_horizontal': self.CoordinateSystem.unit_e_0,
                'unit_vertical': self.CoordinateSystem.unit_e_1,
                'unit_x': self.CoordinateSystem.unit_e_0,  # alias
                'unit_y': self.CoordinateSystem.unit_e_1   # alias
            }
        else:  # 3D
            return {
                'unit_horizontal_x': self.CoordinateSystem.unit_e_0,
                'unit_horizontal_y': self.CoordinateSystem.unit_e_1, 
                'unit_vertical': self.CoordinateSystem.unit_e_2,
                'unit_x': self.CoordinateSystem.unit_e_0,  # alias
                'unit_y': self.CoordinateSystem.unit_e_1,  # alias
                'unit_z': self.CoordinateSystem.unit_e_2   # alias
            }
            
    def create_profile_sample(self, profile_type, **params):
        if profile_type == 'vertical':
            x_pos = params.get('x_position', 0.5)
            y_range = params.get('range', (self.minCoords[1], self.maxCoords[1]))
            return self.create_line_sample(
                start=[x_pos, y_range[0]],
                direction_expr=self.primary_directions['unit_vertical'],
                length=y_range[1] - y_range[0],
                num_points=params.get('num_points', 50)
            )
        elif profile_type == 'horizontal':
            y_pos = params.get('y_position', 0.5) 
            x_range = params.get('range', (self.minCoords[0], self.maxCoords[0]))
            return self.create_line_sample(
                start=[x_range[0], y_pos],
                direction_expr=self.primary_directions['unit_horizontal'],
                length=x_range[1] - x_range[0],
                num_points=params.get('num_points', 50)
            )
        # etc.
```

#### Annulus Meshes
```python
class Annulus(Mesh):
    @property
    def geometric_dimension_names(self):
        return ['radial', 'tangential']
        
    @property
    def primary_directions(self):
        return {
            'unit_radial': self.CoordinateSystem.unit_e_0,
            'unit_tangential': self.CoordinateSystem.unit_e_1,
            'unit_vertical': sympy.Matrix([0, 1]),  # Cartesian vertical
        }
        
    def create_profile_sample(self, profile_type, **params):
        if profile_type == 'radial':
            theta = params.get('theta', 0)
            r_range = params.get('range', (self.r_inner, self.r_outer))
            start_point = [r_range[0] * sympy.cos(theta), r_range[0] * sympy.sin(theta)]
            direction = sympy.Matrix([sympy.cos(theta), sympy.sin(theta)])
            return self.create_line_sample(
                start=start_point,
                direction_expr=direction,
                length=r_range[1] - r_range[0], 
                num_points=params.get('num_points', 50)
            )
        elif profile_type == 'tangential':
            radius = params.get('radius', (self.r_inner + self.r_outer)/2)
            theta_range = params.get('range', (0, 2*sympy.pi))
            # Arc sampling implementation
            # ...
```

## Action Plan

### 1. Immediate: Standardize Existing Properties
```python
# Add to all mesh types as properties
@property 
def unit_vertical(self):
    # Mesh-specific implementation
    
@property
def geometric_directions(self):
    # Return dict of available directions for this mesh type
```

### 2. Implement Generic Line Sampling
```python
def create_line_sample(self, start, direction_expr, length, num_points=50):
    """Generic implementation using sympy evaluation"""
    # This can be implemented once in base class
```

### 3. Update Visualization Library
```python
# Updated parallel visualization
uw.visualization.parallel.parallel_profile_plot(
    field, mesh, 
    profile_spec={'type': 'vertical', 'x_position': 0.5},
    title="Vertical Profile"
)
```

## Audit Results Summary

### ✅ Current System Strengths
- **Complete unit vector foundation**: All mesh types have proper `unit_e_0`, `unit_e_1`, `unit_e_2` definitions
- **Consistent coordinate system architecture**: Each mesh type correctly implements `CoordinateSystemType`
- **Proper transformation matrices**: `_rRotN` matrices correctly transform between natural and Cartesian coordinates
- **Boundary normal integration**: Existing boundary_normals use `unit_e_0` appropriately (e.g., radial boundaries)
- **Real-world usage**: Examples like `Ex_Stokes_Disk_CylCoords.py` demonstrate `unit_e_0` as radial direction

### ✅ **IMPLEMENTED: Geometric Direction Properties** (Verified 2025-09-29)
- **✅ High-level geometric properties**: `unit_vertical`, `unit_horizontal`, `unit_radial`, `unit_tangential` available in `coordinates.py`
- **✅ Geometric dimension naming**: `geometric_dimension_names` property exposes natural dimension names
- **✅ Complete property dictionary**: `primary_directions` provides programmatic access to all available directions
- **✅ Type-aware properties**: Coordinate-system-specific properties with appropriate error handling
- **✅ Backward compatibility**: All existing `unit_e_0` etc. usage continues to work unchanged

### ✅ **IMPLEMENTED: Complete Sampling Infrastructure** (Verified 2025-09-29)
- **✅ Generic line sampling**: `create_line_sample()` implemented in `coordinates.py:646+`
- **✅ Mesh-specific profile sampling**: `create_profile_sample()` implemented in `coordinates.py:769+`
- **✅ Dual coordinate output**: Both Cartesian (for `global_evaluate()`) and natural (for plotting) coordinates
- **✅ Coordinate conversion**: Automatic transformation between Cartesian and natural coordinate systems

### 🔧 Implementation Status: GEOMETRY COMPLETE ✅ | SAFETY PENDING ⚠️
1. **✅ ~~Geometric direction standardization~~**: **COMPLETED** - All mesh-specific properties implemented
2. **✅ ~~Sampling infrastructure~~**: **COMPLETED** - Full sampling system with `global_evaluate()` integration
3. **✅ ~~Dimension naming~~**: **COMPLETED** - `geometric_dimension_names` property implemented
4. **✅ ~~2D vs 3D handling~~**: **COMPLETED** - Clear conventions implemented for all dimensions
5. **⚠️ Boundary name safety**: **CRITICAL ISSUE** - Case sensitivity causes PETSc crashes (detailed above)
6. **📋 Semantic boundary labels**: **PLANNED** - Coordinate-independent boundary naming system

### 📊 Coordinate System Validation
**All coordinate systems properly implemented:**
- ✅ **CARTESIAN**: Identity matrix, standard x/y/z directions
- ✅ **CYLINDRICAL2D**: Proper radial/tangential transformation matrix
- ✅ **SPHERICAL**: Complete radial/meridional/azimuthal basis
- ✅ **Native coordinate variants**: Proper handling of natural coordinate storage

This approach:
- ✅ **Builds on existing system** (`unit_e_0`, `unit_e_1`, etc.)
- ✅ **Uses sympy expressions** for geometric definitions
- ✅ **Mesh-centric authority** over geometry  
- ✅ **Handles 2D/3D** through geometric naming conventions
- ✅ **Extensible** to new mesh types
- ✅ **Backward compatible** with existing code

## ⚠️ Critical Issue: Boundary Name Case Sensitivity

### Current Risk: Silent Failure → PETSc Crash
**Problem**: Boundary condition names are **case-sensitive** with no validation, leading to catastrophic failures:

```python
# THIS WORKS:
solver.add_dirichlet_bc(1.0, "Top")

# THIS CRASHES PETSc:
solver.add_dirichlet_bc(1.0, "top")     # Wrong case
solver.add_dirichlet_bc(1.0, "TOP")     # Wrong case
solver.add_dirichlet_bc(1.0, "Topp")    # Typo
```

### Technical Root Cause
**Location**: `petsc_generic_snes_solvers.pyx:706, 746, 1291, etc.`
```python
# Direct dictionary lookup with no error handling:
value = mesh.boundaries[bc.boundary].value  # KeyError if wrong case → PETSc crash
```

**Failure Chain:**
1. User passes wrong case: `"top"` instead of `"Top"`
2. Python dictionary lookup fails: `KeyError: 'top'`
3. Exception propagates to PETSc C code
4. PETSc crashes with unhelpful error message
5. User gets cryptic crash, no indication of boundary name issue

### Impact Assessment
- **Silent Failures**: No validation means typos go undetected until runtime crash
- **Debugging Nightmare**: PETSc crashes don't indicate boundary name problems
- **User Experience**: Extremely frustrating for new users
- **Production Risk**: Models can fail catastrophically in batch runs

### Proposed Solutions

#### Short-Term Fix: Better Error Messages
```python
# In add_condition() method, add validation:
if bc.boundary not in mesh.boundaries:
    available = list(mesh.boundaries.keys())
    raise ValueError(f"Boundary '{bc.boundary}' not found. "
                    f"Available boundaries: {available}. "
                    f"Note: boundary names are case-sensitive.")
```

#### Medium-Term: Case-Insensitive Lookup
```python
# Create case-insensitive boundary accessor:
def get_boundary(mesh, name):
    # Try exact match first
    if name in mesh.boundaries:
        return mesh.boundaries[name]

    # Try case-insensitive match
    name_lower = name.lower()
    matches = [k for k in mesh.boundaries.keys() if k.lower() == name_lower]

    if len(matches) == 1:
        return mesh.boundaries[matches[0]]
    elif len(matches) > 1:
        raise ValueError(f"Ambiguous boundary name '{name}'. Matches: {matches}")
    else:
        raise ValueError(f"Boundary '{name}' not found. Available: {list(mesh.boundaries.keys())}")
```

#### Long-Term: Standardized Boundary Names
- Implement semantic boundary aliases (see next section)
- Consistent naming conventions across mesh types
- Automatic validation during mesh creation

## Boundary Naming Standardization Challenge

### Current State: Mesh-Specific Names
Different mesh types use different boundary naming conventions:

| Mesh Type | "Upper" Boundary | "Lower" Boundary | Sides |
|-----------|------------------|------------------|-------|
| **Cartesian** | `"Top"` | `"Bottom"` | `"Left"`, `"Right"`, `"Front"`, `"Back"` |
| **Spherical** | `"Upper"` | `"Lower"` | `"North"`, `"South"`, `"East"`, `"West"` |
| **Annulus** | `"Outer"` | `"Inner"` | Various angular boundaries |

### Proposed: Semantic Boundary Aliases
**Goal**: Enable coordinate-independent boundary specifications:

```python
# This should work for any mesh type:
bc_bottom = uw.DirichletBC(mesh.boundary_labels.gravity_aligned_lower, value=0)
bc_top = uw.DirichletBC(mesh.boundary_labels.gravity_aligned_upper, value=1)

# Equivalent coordinate-specific access still available:
bc_cartesian = uw.DirichletBC(mesh.boundaries.Top, value=1)        # Cartesian
bc_spherical = uw.DirichletBC(mesh.boundaries.Upper, value=1)      # Spherical
bc_annulus = uw.DirichletBC(mesh.boundaries.Outer, value=1)        # Annulus
```

### Technical Challenge: gmsh + PETSc Label Integration
**Issue**: PETSc supports duplicate boundary labels, but gmsh integration complicates this:

- **PETSc Capability**: Can create multiple labels for the same boundary
  ```c
  // PETSc can handle:
  DMSetLabelValue(dm, "Top", point, 1);           // gmsh-specific name
  DMSetLabelValue(dm, "gravity_upper", point, 1); // semantic alias
  ```

- **gmsh Integration**: Generates mesh with specific boundary names
  - gmsh creates boundaries with fixed names during mesh generation
  - Post-processing needed to add semantic aliases

### Implementation Strategy

#### Phase 1: Post-Processing Semantic Labels
```python
# After gmsh mesh creation, add semantic aliases:
def add_semantic_boundary_labels(mesh):
    """Add coordinate-independent boundary aliases"""

    # Cartesian mesh
    if mesh.coordinate_system_type == CoordinateSystemType.CARTESIAN:
        mesh.add_boundary_alias("gravity_aligned_upper", "Top")
        mesh.add_boundary_alias("gravity_aligned_lower", "Bottom")
        mesh.add_boundary_alias("horizontal_left", "Left")
        mesh.add_boundary_alias("horizontal_right", "Right")

    # Spherical mesh
    elif mesh.coordinate_system_type == CoordinateSystemType.SPHERICAL:
        mesh.add_boundary_alias("gravity_aligned_upper", "Upper")
        mesh.add_boundary_alias("gravity_aligned_lower", "Lower")
        mesh.add_boundary_alias("meridional_north", "North")
        mesh.add_boundary_alias("meridional_south", "South")

    # Annulus mesh
    elif mesh.coordinate_system_type == CoordinateSystemType.CYLINDRICAL2D:
        mesh.add_boundary_alias("radial_outer", "Outer")
        mesh.add_boundary_alias("radial_inner", "Inner")
        # gravity_aligned depends on orientation
```

#### Phase 2: Enhanced API Access
```python
class Mesh:
    @property
    def boundary_labels(self):
        """Access semantic boundary labels"""
        return self._semantic_boundaries

    def add_boundary_alias(self, semantic_name, gmsh_name):
        """Create semantic alias for existing gmsh boundary"""
        if gmsh_name not in self.boundaries:
            raise ValueError(f"Boundary '{gmsh_name}' not found")

        # Use PETSc label duplication
        original_value = self.boundaries[gmsh_name].value
        self.dm.setLabelValue(semantic_name, original_value)
        self._semantic_boundaries[semantic_name] = self.boundaries[gmsh_name]
```

#### Phase 3: Validation and Safety
```python
def validate_boundary_access(mesh, boundary_name):
    """Validate boundary exists and provide helpful errors"""
    # Check exact match
    if boundary_name in mesh.boundaries:
        return mesh.boundaries[boundary_name]

    # Check semantic aliases
    if hasattr(mesh, '_semantic_boundaries') and boundary_name in mesh._semantic_boundaries:
        return mesh._semantic_boundaries[boundary_name]

    # Provide helpful error with suggestions
    all_boundaries = list(mesh.boundaries.keys())
    if hasattr(mesh, '_semantic_boundaries'):
        all_boundaries.extend(mesh._semantic_boundaries.keys())

    # Suggest case-insensitive matches
    suggestions = [b for b in all_boundaries if b.lower() == boundary_name.lower()]

    error_msg = f"Boundary '{boundary_name}' not found.\n"
    error_msg += f"Available boundaries: {all_boundaries}\n"
    if suggestions:
        error_msg += f"Did you mean: {suggestions}? (case-sensitive)"

    raise ValueError(error_msg)
```

### Benefits of This Approach
1. **Backward Compatibility**: Existing gmsh-specific names continue to work
2. **Mathematical Independence**: Same BC code works across mesh types
3. **Gradual Migration**: Can implement incrementally without breaking changes
4. **Extensible**: Easy to add new semantic categories as needed
5. **Validation**: Centralized error handling with helpful messages

---

## ⚠️ Future Work: Internal Surface Normal Orientation

**Status**: Planning - May require PETSc team discussion
**Date Added**: 2025-01-18
**Related**: `docs/developer/design/PROJECTED_NORMALS_API_DESIGN.md`

### Background

Investigation into boundary normal accuracy for curved surfaces revealed:
- **Boundary surfaces**: PETSc handles orientation correctly for surfaces that lie on the domain boundary
- **Internal surfaces**: Orientation is problematic for internal surfaces (e.g., fault planes, material interfaces)

### The Problem

For **boundary surfaces** (e.g., outer surface of a mesh):
- PETSc provides consistent outward-pointing normals via `mesh.Gamma`
- These are facet-based but correctly oriented

For **internal surfaces** (e.g., embedded faults, material interfaces):
- PETSc may not provide consistent orientation
- No guaranteed "outward" direction for internal surfaces
- This affects:
  - Free-slip conditions on fault planes
  - Flux conditions across material interfaces
  - Traction boundary conditions on embedded surfaces

### Proposed Solutions

#### Option 1: User-Specified Orientation Vector
```python
# User provides reference direction for orientation
n_proj = mesh.project_surface_normals(
    surfaces=["Fault"],
    orientation_reference=sympy.Matrix([1, 0]),  # User-specified
)
```
**Pros**: Explicit, user controls behavior
**Cons**: Extra parameter required, may be confusing

#### Option 2: PETSc Enhancement Request
Discuss with PETSc team about:
- Consistent internal surface labeling
- Orientation metadata in DMPlex labels
- API for specifying surface orientation

#### Option 3: Post-Processing Heuristics
```python
# Detect and correct orientation based on geometry
def correct_internal_orientation(normals, surface_points, reference_point):
    """Ensure normals point away from reference point."""
    for i, (n, p) in enumerate(zip(normals, surface_points)):
        direction = p - reference_point
        if np.dot(n, direction) < 0:
            normals[i] = -n
    return normals
```
**Pros**: Automatic for simple cases
**Cons**: Fails for complex geometries

### Action Items

1. **Document Use Cases**: Gather specific examples where internal surface orientation matters
2. **Evaluate PETSc Capabilities**: Check current PETSc features for internal surface handling
3. **PETSc Team Discussion**: If needed, prepare feature request or discussion topic
4. **Interim Solution**: Implement user-specified orientation as immediate workaround

### Related Documentation

- `docs/advanced/curved-boundary-conditions.md` - User-facing documentation on curved boundaries
- `docs/developer/design/PROJECTED_NORMALS_API_DESIGN.md` - API design for `mesh.project_surface_normals()`

### Notes

This issue was identified during investigation of boundary normal accuracy on elliptical meshes. While projected normals significantly improve accuracy for curved boundaries (~99.8% improvement over raw facet normals), the approach relies on consistent orientation from PETSc, which is only guaranteed for domain boundary surfaces.

*Added 2025-01-18: Investigation into mesh.Gamma normalization and projected normals*