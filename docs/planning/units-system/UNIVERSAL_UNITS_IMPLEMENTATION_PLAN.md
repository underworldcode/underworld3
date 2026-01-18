# Universal Units Implementation Plan

## 🎯 **Core Philosophy: Everything Has Physical Units**

**Key Insight**: There are no "model units" vs "physical units" - only different physical unit systems with automatic conversion between them.

- **Internal units** = chosen physical units for numerical accuracy (e.g., Mm, Ma, GPa)
- **User units** = convenient physical units (km, years, Pa)
- **Unit tracking** = every quantity knows its units → trivial conversion
- **Universal approach** = works for any scaling (1.5 Earth-mass planets, etc.)

---

## 📋 **Priority 1: Unit-Aware Functions**

### **Core Principle**
Every function that accepts coordinates/quantities should:
1. **Detect input units** automatically
2. **Convert to required units** (usually internal mesh units)
3. **Return results with appropriate units**

### **1.1 Unit-Aware evaluate() Functions**

**Target**: `uw.function.evaluate()` and `global_evaluate()`

```python
def evaluate(expr, coords, coord_sys=None, ...):
    """
    Unit-aware evaluation that auto-converts coordinates.

    Parameters
    ----------
    coords : array-like with units or plain array
        Coordinates in any physical unit system
    """
    # Detect coordinate units
    if has_units(coords):
        target_units = get_mesh_coordinate_units(coord_sys or expr)
        internal_coords = convert_units(coords, target_units)
    else:
        # Assume coords are already in mesh units
        internal_coords = coords

    # Evaluate using internal units
    result = _evaluate_internal(expr, internal_coords, ...)

    # Tag result with appropriate units based on expression
    return add_expression_units(result, expr)
```

**Implementation Steps**:
- [ ] Add unit detection to coordinate inputs
- [ ] Add automatic unit conversion to mesh coordinate system
- [ ] Add result unit tagging based on expression dimensionality
- [ ] Backward compatibility: plain arrays assumed to be in mesh units

### **1.2 Unit-Aware Mesh Geometry Functions**

**Targets**:
- `mesh.points_in_domain(points)`
- `mesh.test_if_points_in_cells(points, cells)`
- Any function accepting coordinate inputs

```python
def points_in_domain(self, points, strict_validation=True):
    """Check if points lie in domain, with automatic unit conversion."""
    # Convert input coordinates to mesh coordinate units
    if has_units(points):
        mesh_units = self.get_coordinate_units()
        points_internal = convert_units(points, mesh_units)
    else:
        points_internal = points

    # Perform domain check in mesh units
    return self._points_in_domain_internal(points_internal, strict_validation)
```

### **1.3 Unit-Aware Visualization Functions**

**Target**: `visualisation.py` functions that pass coordinates to `evaluate()`

```python
def scalar_fn_to_pv_points(pv_mesh, uw_fn, dim=None, simplify=True):
    """Evaluate UW function at mesh points with automatic unit handling."""
    coords = pv_mesh.points[:, 0:dim]

    # Coords from PyVista are in physical units (from mesh.points)
    # Add unit metadata so evaluate() can handle conversion
    coords_with_units = add_coordinate_units(coords, pv_mesh.unit_metadata)

    # evaluate() will auto-convert to mesh units
    scalar_values = uw.function.evaluate(uw_fn, coords_with_units, evalf=True)

    return scalar_values
```

---

## 📋 **Priority 2: Helper Functions for Easy Conversion**

### **2.1 Array Conversion Utilities**

```python
# Convert any array to any unit system
def convert_array_units(array, from_units, to_units):
    """Convert array from one unit system to another."""

def auto_convert_to_mesh_units(array, mesh):
    """Convert array coordinates to mesh unit system."""

def convert_evaluation_result(result, target_units):
    """Convert evaluation results to target unit system."""
```

### **2.2 Quantity Conversion Utilities**

```python
def convert_quantity_units(quantity, target_units):
    """Convert UWQuantity or Pint quantity to target units."""

def detect_quantity_units(obj):
    """Detect units of any object (UWQuantity, Pint, array with metadata)."""

def make_dimensionless(quantity, reference_scales):
    """Convert physical quantity to dimensionless using reference scales."""
```

### **2.3 Unit Detection and Metadata**

```python
def has_units(obj):
    """Check if object has unit information."""

def get_units(obj):
    """Extract unit information from any object."""

def add_units(array, units_str):
    """Add unit metadata to plain array."""
```

---

## 📋 **Priority 3: Unit-Aware NDArray System**

### **Core Concept**
Create array equivalent of `UWQuantity` that seamlessly integrates with existing `NDArray_With_Callback` system.

### **3.1 UnitAwareArray Design**

```python
class UnitAwareArray:
    """
    Array that tracks units and automatically converts when needed.

    Like Pint's Quantity arrays but integrated with UW3 NDArray system.
    """
    def __init__(self, array, units):
        self._array = array
        self._units = units

    def to(self, target_units):
        """Convert to different units."""

    def to_base_units(self):
        """Convert to base SI units."""

    def __array__(self):
        """NumPy array interface - return array in current units."""

    # Automatic unit conversion for operations
    def __add__(self, other):
        """Addition with automatic unit compatibility checking."""

    def __mul__(self, other):
        """Multiplication with unit propagation."""
```

### **3.2 Integration with NDArray_With_Callback**

```python
class UnitAwareNDArray(NDArray_With_Callback):
    """
    NDArray_With_Callback that tracks units and converts automatically.

    Maintains all existing callback functionality while adding unit awareness.
    """
    def __init__(self, array, units, callback=None):
        super().__init__(array, callback)
        self._units = units

    def convert_to(self, target_units):
        """Convert array to different units, triggering callbacks."""

    def __setitem__(self, key, value):
        """Auto-convert incoming values to array units."""
        if has_units(value):
            value = convert_units(value, self._units)
        super().__setitem__(key, value)
```

### **3.3 Seamless Integration Points**

- **mesh.points**: Return `UnitAwareNDArray` instead of plain array
- **swarm.points**: Return `UnitAwareNDArray` with automatic conversion
- **Variable.array**: Track units for physical quantities
- **Evaluation results**: Auto-wrapped in `UnitAwareArray`

---

## 📋 **Priority 4: Serialization with Units**

### **4.1 Mesh Serialization**

```python
# In mesh save operations
def save_mesh_with_units(mesh, filename):
    metadata = {
        'coordinate_units': mesh.get_coordinate_units(),
        'length_scale': mesh.get_length_scale(),
        'coordinate_system': mesh.CoordinateSystem.type,
        'scaling_applied': getattr(mesh.CoordinateSystem, '_scaled', False)
    }
    # Save geometry + metadata
```

**Implementation**:
- [ ] Add unit metadata to HDF5/XDMF mesh files
- [ ] Store coordinate system and scaling information
- [ ] Preserve physical unit information for external tools

### **4.2 Swarm Serialization**

```python
# In swarm save operations
def save_swarm_with_units(swarm, filename):
    metadata = {
        'coordinate_units': swarm.mesh.get_coordinate_units(),
        'particle_coordinate_units': swarm.get_coordinate_units(),
        'variable_units': {var.name: var.get_units() for var in swarm.variables}
    }
```

**Implementation**:
- [ ] Save particle coordinate units
- [ ] Save swarm variable units
- [ ] Link to mesh unit information

### **4.3 Variable Serialization**

```python
# In variable save operations
def save_variable_with_units(variable, filename):
    metadata = {
        'variable_units': variable.get_units(),
        'mesh_coordinate_units': variable.mesh.get_coordinate_units(),
        'physical_dimensions': variable.get_dimensionality()
    }
```

**Implementation**:
- [ ] Save variable physical units
- [ ] Save coordinate system context
- [ ] Enable unit validation on load

---

## 📋 **Priority 5: Missing Features Analysis**

### **5.1 Integration Points We Haven't Considered**

#### **A. Boundary Conditions**
```python
# BCs specified with coordinates - need unit awareness
mesh.add_dirichlet_bc(coords=boundary_points, ...)  # What units?
```

#### **B. Initial Conditions**
```python
# ICs may use coordinate-dependent expressions
T.array[...] = some_function(mesh.points)  # Units matter
```

#### **C. Material Properties**
```python
# Properties may be coordinate-dependent
density = f(x, y, z)  # What coordinate units?
```

#### **D. Solver Interfaces**
```python
# Solvers may need coordinate information
stokes.solve()  # Internal coordinates for PETSc
```

### **5.2 Performance Considerations**

#### **A. Unit Conversion Overhead**
- **Caching**: Avoid repeated conversions
- **Lazy conversion**: Convert only when needed
- **Batch operations**: Convert arrays efficiently

#### **B. Memory Usage**
- **Unit metadata storage**: Minimal overhead
- **Array copying**: Avoid unnecessary copies during conversion
- **Reference sharing**: Share unit information across related objects

### **5.3 User Experience Considerations**

#### **A. Error Messages**
```python
# Clear unit mismatch errors
"Cannot evaluate expression: coordinates in 'km' but mesh expects 'Mm'"
```

#### **B. Unit Discovery**
```python
# Easy way to check what units things expect/provide
mesh.coordinate_units  # 'Mm'
result.units          # 'Pa*s'
```

#### **C. Conversion Utilities**
```python
# Easy conversion at user level
coords_km = coords_m.to('km')
mesh_coords = coords.to_mesh_units(mesh)
```

### **5.4 Edge Cases and Special Scenarios**

#### **A. Mixed Unit Systems**
- Multiple meshes with different unit systems
- Swarms moving between meshes
- Variable interpolation across unit systems

#### **B. External Tool Integration**
- PyVista: Needs consistent physical units
- ParaView: Human-readable units in saved files
- Post-processing: Unit preservation through workflows

#### **C. Backward Compatibility**
- Existing code without unit awareness
- Gradual migration path
- Fallback behaviors for plain arrays

---

## 🎯 **Implementation Sequence**

### **Phase 1: Core Unit Awareness (Immediate)**
1. ✅ Fix `evaluate()` functions with unit auto-conversion
2. ✅ Fix visualization coordinate handling
3. ✅ Add unit detection utilities
4. ✅ Test with coordinate scaling validation scripts

### **Phase 2: Helper Functions (Short-term)**
1. ⏳ Array conversion utilities
2. ⏳ Quantity conversion utilities
3. ⏳ Unit detection and metadata system
4. ⏳ Integration with existing coordinate functions

### **Phase 3: Unit-Aware Arrays (Medium-term)**
1. ⏳ Design `UnitAwareArray` class
2. ⏳ Integrate with `NDArray_With_Callback`
3. ⏳ Update mesh.points and swarm.points to return unit-aware arrays
4. ⏳ Test seamless integration

### **Phase 4: Serialization (Medium-term)**
1. ⏳ Add unit metadata to mesh save/load
2. ⏳ Add unit metadata to swarm save/load
3. ⏳ Add unit metadata to variable save/load
4. ⏳ Test round-trip unit preservation

### **Phase 5: Comprehensive Integration (Long-term)**
1. ⏳ Address all identified integration points
2. ⏳ Performance optimization
3. ⏳ User experience enhancements
4. ⏳ Comprehensive testing and validation

---

## ✅ **Success Criteria**

1. **Universal Unit Interface**: All user-facing functions accept any physical unit system
2. **Automatic Conversion**: No manual unit conversion required by users
3. **Backward Compatibility**: Existing code continues to work
4. **External Tool Integration**: PyVista, ParaView get human-readable units
5. **Persistent Units**: Save/load preserves unit information
6. **Performance**: Minimal overhead from unit tracking
7. **Clear Errors**: Helpful messages for unit mismatches
8. **Comprehensive Coverage**: All coordinate-dependent functions are unit-aware

**Goal**: Users work entirely in physical units they understand, with seamless automatic conversion throughout the system.