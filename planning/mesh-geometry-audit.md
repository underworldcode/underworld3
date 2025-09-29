# Mesh Geometry System Audit

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
    # In 2D annulus, "vertical" is ambiguous - could be:
    # 1. Cartesian y-direction: [0, 1]
    # 2. Radial direction: unit_e_0  
    # 3. Not defined (raise NotImplementedError)
    return sympy.Matrix([0, 1])  # Cartesian vertical
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
    # "Vertical" in spherical could mean:
    # 1. Radial direction (toward center): -unit_e_0
    # 2. Gravity direction (depends on location)
    # 3. Local "up" direction  
    return -self.CoordinateSystem.unit_e_0  # Toward center
```

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

### ✅ **IMPLEMENTED: Geometric Direction Properties**
- **✅ High-level geometric properties**: `unit_vertical`, `unit_horizontal`, `unit_radial`, `unit_tangential` now available
- **✅ Geometric dimension naming**: `geometric_dimension_names` property exposes natural dimension names
- **✅ Complete property dictionary**: `primary_directions` provides programmatic access to all available directions
- **✅ Type-aware properties**: Coordinate-system-specific properties with appropriate error handling
- **✅ Backward compatibility**: All existing `unit_e_0` etc. usage continues to work unchanged

### ✅ **IMPLEMENTED: Complete Sampling Infrastructure**
- **✅ Generic line sampling**: `create_line_sample()` for arbitrary sympy-defined directions
- **✅ Mesh-specific profile sampling**: `create_profile_sample()` with coordinate-system-aware profiles
- **✅ Dual coordinate output**: Both Cartesian (for `global_evaluate()`) and natural (for plotting) coordinates
- **✅ Coordinate conversion**: Automatic transformation between Cartesian and natural coordinate systems

### 🔧 Implementation Status: COMPLETE ✅
1. **✅ ~~Geometric direction standardization~~**: **COMPLETED** - All mesh-specific properties implemented
2. **✅ ~~Sampling infrastructure~~**: **COMPLETED** - Full sampling system with `global_evaluate()` integration
3. **✅ ~~Dimension naming~~**: **COMPLETED** - `geometric_dimension_names` property implemented
4. **✅ ~~2D vs 3D handling~~**: **COMPLETED** - Clear conventions implemented for all dimensions

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