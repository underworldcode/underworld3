# Mesh Coordinate Units Interface Design

## Problem Statement

Currently, mesh coordinates (`mesh.data`, `mesh.points`, `mesh.X`) have no unit information, creating a significant user-interface gap for physical modeling:

1. **`mesh.data`** - Returns raw numpy arrays without units
2. **`mesh.points`** - Same as mesh.data, no unit context
3. **`mesh.X`** - SymPy coordinate symbols without unit metadata

## Use Cases Requiring Mesh Units

### 1. Physical Scale Understanding
```python
# User creates a mantle convection mesh
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(64, 64),
    minCoords=(0.0, 0.0),
    maxCoords=(2900.0, 2900.0)  # What units? km? m? arbitrary?
)

# Current: User has no way to know the physical scale
# Proposed: Explicit units make scale clear
```

### 2. Data Import/Export with Proper Units
```python
# Loading GIS data or experimental coordinates
coords_from_file = load_coordinates("survey_data.csv")  # In degrees
mesh = create_mesh_from_coordinates(coords_from_file)

# Current: Units lost, must track separately
# Proposed: Units preserved through the workflow
```

### 3. Multi-scale Modeling
```python
# Regional model: kilometers
regional_mesh = create_mesh(..., units="km")

# Local model: meters
local_mesh = create_mesh(..., units="m")

# Current: No way to ensure consistent coordinate transformations
# Proposed: Automatic unit conversion and validation
```

### 4. Physical Interpretation of Results
```python
# Gradient calculations need coordinate units
velocity = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")
strain_rate = velocity.gradient()  # Units should be 1/s = (m/s)/m

# Current: Strain rate units unclear without mesh coordinate units
# Proposed: Automatic unit derivation from mesh + variable units
```

## Proposed Interface Design

### 1. Mesh Constructor with Units
```python
# Simple unit specification
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(32, 32),
    minCoords=(0.0, 0.0),
    maxCoords=(2900.0, 2900.0),
    units="km"  # New parameter
)
```

### 2. Units Property Interface
```python
# Queryable and settable units
print(f"Mesh coordinates in: {mesh.units}")  # "kilometer"

# Change units after creation
mesh.units = "m"  # Converts coordinates automatically
```

### 3. Unit-Aware Coordinate Access
```python
# Option A: mesh.data returns UWQuantity arrays
coords = mesh.data  # UWQuantity array with units
print(f"Max extent: {coords.max()}")  # "2900.0 kilometer"

# Option B: Separate units metadata
coords = mesh.data  # Still numpy array
units = mesh.coordinate_units  # Separate units query
```

### 4. Unit-Aware Coordinate Symbols
```python
# mesh.X includes unit context
x, y = mesh.X
print(f"Coordinate x has units: {x.units}")  # Future enhancement

# Or coordinate system level units
print(f"Coordinate system units: {mesh.CoordinateSystem.units}")
```

### 5. Unit Conversion Methods
```python
# Convert mesh to different units
mesh_meters = mesh.to_units("m")
assert mesh_meters.data.max() == 2900000.0  # Converted to meters

# In-place conversion
mesh.convert_units("cm")
```

## Implementation Strategy

### Phase 1: Basic Units Support
1. **Add `units` parameter to mesh constructors**
2. **Add `units` property to Mesh class**
3. **Store units metadata in mesh objects**
4. **Update mesh.data to return unit-aware arrays (UWQuantity)**

### Phase 2: Unit-Aware Operations
1. **Implement `to_units()` and `convert_units()` methods**
2. **Update coordinate access to include units**
3. **Integrate with mesh variable gradient calculations**
4. **Add units to mesh save/load operations**

### Phase 3: Advanced Features
1. **Unit-aware coordinate symbols (mesh.X)**
2. **Automatic unit derivation for derived quantities**
3. **Visualization with unit-aware axes and labels**
4. **GIS and scientific data format integration**

## Technical Implementation Details

### 1. Mesh Class Modifications
```python
class Mesh:
    def __init__(self, ..., units=None):
        self._coordinate_units = units
        self._original_coords = None  # Store for unit conversion

    @property
    def units(self):
        return self._coordinate_units

    @units.setter
    def units(self, new_units):
        if self._coordinate_units is not None:
            # Convert existing coordinates
            self._convert_coordinates(self._coordinate_units, new_units)
        self._coordinate_units = new_units

    @property
    def data(self):
        """Return unit-aware coordinate array."""
        if self._coordinate_units is not None:
            return uw.create_quantity(self._points, self._coordinate_units)
        return self._points  # Backward compatibility
```

### 2. Unit-Aware Coordinate Arrays
```python
# mesh.data returns UWQuantity for unit-aware meshes
coords = mesh.data  # Type: UWQuantity with units
x_coords = coords[:, 0]  # Still has units
max_x = x_coords.max()  # UWQuantity with proper units

# Backward compatibility: unitless meshes return numpy arrays
unitless_mesh = uw.meshing.StructuredQuadBox(...)
coords = unitless_mesh.data  # Type: numpy.ndarray (as before)
```

### 3. Integration with Model Units
```python
# Mesh units should integrate with model scaling
model = uw.Model("convection")
model.set_reference_quantities(mantle_depth=2900*uw.units.km)

# Mesh created with compatible units
mesh = uw.meshing.StructuredQuadBox(..., units="km")

# Automatic integration
mesh_model_units = model.to_model_units(mesh.data)
```

## Backward Compatibility

### Existing Code Continues to Work
```python
# Existing unitless meshes work exactly as before
mesh = uw.meshing.StructuredQuadBox(
    elementRes=(4, 4),
    minCoords=(0.0, 0.0),
    maxCoords=(1.0, 1.0)
    # No units parameter - works as before
)

coords = mesh.data  # Still returns numpy array
assert isinstance(coords, np.ndarray)  # True
```

### Gradual Migration Path
```python
# Users can gradually add units to existing code
mesh.units = "m"  # Add units to existing mesh

# Or specify units when creating new meshes
mesh = uw.meshing.StructuredQuadBox(..., units="m")
```

## Benefits

### 1. Improved User Experience
- **Clear physical interpretation** of mesh scales
- **Automatic unit validation** prevents errors
- **Integrated workflow** with physical quantities

### 2. Better Scientific Workflow
- **Data import/export** preserves units
- **Multi-scale modeling** with proper transformations
- **Physical equations** with correct dimensional analysis

### 3. Enhanced Visualization
- **Axis labels** with correct units
- **Scale bars** showing physical dimensions
- **Coordinate readouts** in meaningful units

### 4. Future-Proof Design
- **Extensible** to 3D coordinates and curvilinear systems
- **Compatible** with existing units system
- **Integrates** with model scaling and reference quantities

## Open Design Questions

### 1. Default Behavior
- Should meshes without units be "dimensionless" or None?
- How to handle mixed unit systems (x in km, y in degrees)?

### 2. Performance Impact
- UWQuantity arrays vs numpy arrays performance
- Caching strategy for unit conversions
- Memory overhead for unit metadata

### 3. Coordinate System Integration
- How to integrate with CoordinateSystem classes?
- Support for curvilinear coordinates (spherical, cylindrical)?
- Geographic coordinate systems (lat/lon)?

### 4. Advanced Features
- Automatic unit derivation for mesh metrics (area, volume)?
- Integration with finite element calculations?
- Support for time-dependent coordinate transformations?

## Implementation Status

### ✅ COMPLETED (2025-10-11)

**Phase 1: Basic Units Support - COMPLETE**
- ✅ Added `units` parameter to mesh constructors
- ✅ Added `units` property to Mesh class
- ✅ Store units metadata in mesh objects
- ✅ Updated mesh.points to return unit-aware arrays (UnitAwareArray)

**KDTree Unit-Aware Implementation - COMPLETE**
- ✅ Both ckdtree (Cython) and kdtree (Python) implementations support units
- ✅ KDTree stores coordinate units from construction
- ✅ Automatic unit conversion for queries
- ✅ Unit-aware distance results
- ✅ Unit mismatch detection with clear errors
- ✅ RBF interpolation handles units correctly

**Files Modified**:
1. `src/underworld3/discretisation/discretisation_mesh.py`
   - Line 1810: KD-tree construction uses `self._points` (raw array)
   - Line 2404: Cell size calculation uses `self._points` (raw array)
   - Pattern: Internal operations use `_points`, external API uses `points`

2. `src/underworld3/ckdtree.pyx`
   - Added `coord_units` attribute
   - Unit detection and storage on construction
   - `_convert_coords_to_tree_units()` method
   - Unit-aware `query()` and `rbf_interpolator_local_from_kdtree()`

3. `src/underworld3/kdtree.py`
   - Identical changes to ckdtree (both implementations consistent)

**Test Coverage**: test_0620_mesh_units_interface.py validates all functionality

### Key Implementation Patterns

**Internal vs External Access**:
```python
# Internal mesh operations: Use raw arrays
self._points  # No unit wrapping, fast

# External API: Unit-aware arrays
self.points   # Unit-aware when units set
```

**KDTree Usage**:
```python
# Create from mesh
mesh = uw.meshing.StructuredQuadBox(..., units="kilometer")
kd = uw.kdtree.KDTree(mesh.points)  # Stores 'kilometer' units

# Query with automatic conversion
query_m = UnitAwareArray([[100000.0, 50000.0]], units="meter")
dist, idx = kd.query(query_m)  # Auto-converts m→km, returns km distances
```

## Remaining Work

### Phase 2: Unit-Aware Operations (Future)
1. **Implement `to_units()` and `convert_units()` methods** (not yet started)
2. **Integrate with mesh variable gradient calculations** (partial - needs review)
3. **Add units to mesh save/load operations** (not yet started)

### Phase 3: Advanced Features (Future)
1. **Unit-aware coordinate symbols (mesh.X)** (not yet started)
2. **Automatic unit derivation for derived quantities** (not yet started)
3. **Visualization with unit-aware axes and labels** (not yet started)
4. **GIS and scientific data format integration** (not yet started)

## Next Steps

1. ✅ **Create test suite** for proposed interface (Done in test_0620_mesh_units_interface.py)
2. ✅ **Implement Phase 1** basic units support (COMPLETE)
3. 🔄 **Validate with realistic use cases** (mantle convection, geology) - Ongoing
4. ⏳ **Gather user feedback** on interface design
5. ⏳ **Extend to advanced features** based on user needs

This design provides a comprehensive solution to the mesh coordinate units gap while maintaining backward compatibility and enabling future enhancements.