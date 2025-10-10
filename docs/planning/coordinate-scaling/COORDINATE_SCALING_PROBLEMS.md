# CRITICAL: Coordinate Scaling Interface Problems

## 🚨 **MAJOR INTERFACE INCONSISTENCY DISCOVERED**

After implementing coordinate scaling for `mesh.X`, `mesh.points`, and `swarm.points`, a **major interface breaking change** has been identified. The coordinate scaling creates an inconsistent interface where some functions expect physical coordinates while others expect model coordinates.

## 📋 **Summary of Changes Made**

### ✅ **What Was Implemented**
1. **`mesh.X`**: Now represents physical coordinates (scaled SymPy expressions)
2. **`mesh.points`**: Now returns/accepts physical coordinates (automatic conversion)
3. **`swarm.points`**: Now returns/accepts physical coordinates (automatic conversion)
4. **`model.to_model_units()`**: Smart coercion function (safe to call repeatedly)

### ❌ **What This Breaks**

## 🔍 **Coordinate Usage Audit Results**

### 1. **uw.function.evaluate() and global_evaluate()** ❌ **BROKEN**

**Location**: `/src/underworld3/function/_function.pyx:336,192`

**Documentation**:
- Line 336: `coord_sys: mesh.N vector coordinate system`
- Line 192: `coord_sys: mesh.N vector coordinate system`

**Problem**: Both functions expect **model coordinates** (`mesh.N` coordinates), but users will now pass **physical coordinates** from `mesh.points`.

**Impact**:
- `uw.function.evaluate(expr, mesh.points)` will fail or give wrong results
- All evaluation at user-provided coordinates will be broken

### 2. **Visualization Functions** ❌ **BROKEN**

**Location**: `/src/underworld3/visualisation/visualisation.py:212-213`

**Code**:
```python
coords = pv_mesh.points[:, 0:dim]  # Gets coordinates from PyVista mesh
scalar_values = uw.function.evaluate(uw_fn, coords, evalf=True)  # Passes to evaluate
```

**Problem**: PyVista mesh gets coordinates from `mesh.points` (now physical), but `evaluate()` expects model coordinates.

**Impact**: All visualization will be broken when coordinate scaling is enabled.

### 3. **Mesh Geometric Methods** ❌ **PROBABLY BROKEN**

**Location**: `/src/underworld3/discretisation/discretisation_mesh.py`

**Methods Found**:
- `points_in_domain(self, points, strict_validation=True)` (line 2103)
- `test_if_points_in_cells(self, points, cells)` (line 1982)

**Problem**: These methods take coordinate inputs but don't specify coordinate system. They likely expect model coordinates for internal mesh operations.

**Impact**: Domain testing and cell location functions will fail with physical coordinates.

### 4. **Integration and Boundary Conditions** ⚠️ **UNKNOWN**

**Potential Issues**:
- Boundary condition specification by coordinates
- Integration domain boundaries
- Initial condition specification
- Particle seeding operations

**Need Investigation**: These haven't been audited yet but likely have coordinate dependencies.

## 🎯 **Root Cause Analysis**

The fundamental problem is that **coordinate scaling was implemented at the interface level only**, without considering the **internal ecosystem of coordinate-dependent functions**.

### **Current Mixed Interface**:
- **Physical coordinate interface**: `mesh.X`, `mesh.points`, `swarm.points`
- **Model coordinate interface**: `evaluate()`, visualization, mesh methods, etc.

### **Expected User Workflow (Now Broken)**:
```python
# User gets physical coordinates
coords = mesh.points  # Physical coordinates

# User tries to evaluate at these coordinates
values = uw.function.evaluate(expr, coords)  # ❌ BROKEN - expects model coordinates
```

## 🔧 **Solution Strategies**

### **Option 1: Universal Physical Interface (Recommended)**
Convert ALL coordinate-dependent functions to expect physical coordinates:

1. **Modify evaluate functions** to automatically convert physical → model coordinates
2. **Modify visualization functions** to handle the conversion
3. **Modify mesh methods** to expect physical coordinates
4. **Update all documentation** to specify physical coordinate expectations

**Pros**: Consistent user interface, physical coordinates everywhere
**Cons**: Requires extensive changes throughout codebase

### **Option 2: Provide Both Interfaces**
Keep model coordinate functions but add physical coordinate versions:

1. **Add `mesh.model_points`** property for model coordinates
2. **Add `evaluate_physical()`** functions that handle conversion
3. **Keep existing functions unchanged** for backward compatibility

**Pros**: Backward compatible, explicit about coordinate systems
**Cons**: Confusing dual interface, more maintenance

### **Option 3: Revert Coordinate Scaling**
Remove coordinate scaling and use explicit conversion functions:

1. **Revert `mesh.points` and `swarm.points`** to model coordinates
2. **Keep `mesh.X` physical scaling** (for symbolic expressions only)
3. **Provide explicit conversion**: `model.physical_coordinates(mesh.points)`

**Pros**: Minimal breaking changes, explicit conversions
**Cons**: Less intuitive interface, manual conversions required

## 🚦 **Immediate Action Required**

### **Phase 1: Assessment**
1. ✅ **Audit coordinate usage** - In progress
2. ⏳ **Test current breakage** - Create comprehensive test to demonstrate problems
3. ⏳ **Measure impact** - Determine how many functions are affected

### **Phase 2: Decision**
1. ⏳ **Choose solution strategy** - Based on audit results
2. ⏳ **Plan implementation** - Systematic approach to fixes
3. ⏳ **Validate approach** - Ensure no other breakage

### **Phase 3: Implementation**
1. ⏳ **Fix coordinate-dependent functions**
2. ⏳ **Update documentation**
3. ⏳ **Comprehensive testing**

## 📊 **Impact Assessment**

### **Critical Functions Affected**:
- ✅ `uw.function.evaluate()` and `global_evaluate()` ❌ **BROKEN**
- ✅ Visualization functions ❌ **BROKEN**
- ✅ Mesh geometric methods ❌ **PROBABLY BROKEN**
- ⏳ Integration functions ❌ **UNKNOWN**
- ⏳ Boundary condition functions ❌ **UNKNOWN**
- ⏳ Save/load operations ❌ **UNKNOWN**

### **User Experience Impact**:
- **Existing tutorials/examples** will break
- **User code** relying on coordinate functions will break
- **Visualization** will produce wrong results
- **Scientific results** could be incorrect if scaling is unnoticed

## 🎯 **Recommendation**

**Implement Option 1 (Universal Physical Interface)** with systematic conversion:

1. **Modify `evaluate()` functions** to detect and convert physical coordinates
2. **Add coordinate system detection** to automatically handle conversions
3. **Update all coordinate-dependent functions** to expect physical coordinates
4. **Comprehensive testing** with all three validation strategies
5. **Clear documentation** about the coordinate system used

This provides the most intuitive and consistent user interface while fixing all identified problems.

---

**Status**: 🚨 **CRITICAL ISSUE** - Coordinate scaling implementation incomplete and breaking existing functionality.