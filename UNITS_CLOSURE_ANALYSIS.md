# Units System Closure Analysis

**Date:** 2025-11-08
**Purpose:** Comprehensive analysis of closure properties for unit-aware operations
**Definition:** A system has "closure" when operations on unit-aware objects return unit-aware objects

---

## Table of Contents

1. [Unit-Aware Object Types](#unit-aware-object-types)
2. [Closure Property Matrix](#closure-property-matrix)
3. [Operation Types and Results](#operation-types-and-results)
4. [Conversion Methods](#conversion-methods)
5. [Current Gaps and Phase 4 Work](#current-gaps-and-phase-4-work)

---

## Unit-Aware Object Types

### Core Types

| Type | Class | Use Case | Stores Units | Tested in Closure? |
|------|-------|----------|--------------|-------------------|
| **Variable** | `MeshVariable`, `SwarmVariable` | Field data on mesh/swarm | ✅ Yes (`.units` property) | ✅ Yes |
| **Quantity** | `UWQuantity` | Scalar/vector constants | ✅ Yes (`.units` property) | ✅ Yes |
| **Array** | `UnitAwareArray` | NumPy arrays with units | ✅ Yes (`.units` property) | ✅ Yes |
| **Expression** | `UnitAwareExpression` | SymPy expressions with units | ✅ Yes (`.units` property) | ❌ **NO** - Not tested! |
| **Symbol** | SymPy `Symbol` from `.sym` | Mathematical symbols | ⚠️ Via Variable lookup | N/A |

### Non-Unit-Aware Types

| Type | Examples | Can Convert? |
|------|----------|--------------|
| **Plain numbers** | `int`, `float`, `complex` | Yes → `UWQuantity` |
| **Plain arrays** | `np.ndarray` | Yes → `UnitAwareArray` |
| **SymPy expressions** | `sympy.Symbol`, `sympy.Expr` | Maybe (if contains unit-aware vars) |
| **Pint quantities** | `pint.Quantity` | Yes → `UWQuantity` |

---

## Closure Property Matrix

### Variable Operations

| Operation | Input Types | Output Type | Has Units? | Status |
|-----------|-------------|-------------|------------|--------|
| `var1 * var2` | Variable × Variable | SymPy Matrix | ❌ No | **MISSING** (Phase 4) |
| `var * scalar` | Variable × number | SymPy Matrix | ❌ No | **MISSING** (Phase 4) |
| `var * quantity` | Variable × UWQuantity | SymPy Matrix | ❌ No | **MISSING** (Phase 4) |
| `var + var` | Variable + Variable | SymPy Matrix | ❌ No | **MISSING** (Phase 4) |
| `var - var` | Variable - Variable | SymPy Matrix | ❌ No | **MISSING** (Phase 4) |
| `var / var` | Variable / Variable | SymPy Matrix | ❌ No | **MISSING** (Phase 4) |
| `var ** 2` | Variable power | SymPy Matrix | ❌ No | **MISSING** (Phase 4) |
| `var[0]` | Component access | SymPy expression | ❌ No | **MISSING** (Phase 4) |

**Note:** Variable arithmetic currently returns plain SymPy objects. Phase 4 will wrap these in `UnitAwareExpression` for closure.

### SymPy Expression Operations

| Operation | Input Types | Output Type | Has Units? | Status |
|-----------|-------------|-------------|------------|--------|
| `expr * expr` | Expression × Expression | SymPy | ⚠️ Via `uw.get_units()` | **PARTIAL** |
| `expr + expr` | Expression + Expression | SymPy | ⚠️ Via `uw.get_units()` | **PARTIAL** |
| `expr.diff(x)` | Derivative | SymPy | ⚠️ Via `uw.get_units()` | **PARTIAL** |
| `uw.get_units(expr)` | Extract units | `pint.Unit` or str | ✅ Yes | ✅ **WORKING** |

**Note:** SymPy expressions don't carry units directly, but units can be extracted via `uw.get_units()`. This is less convenient than having unit-aware expression objects.

### UnitAwareArray Operations

| Operation | Input Types | Output Type | Has Units? | Status |
|-----------|-------------|-------------|------------|--------|
| `arr1 + arr2` | Array + Array | `UnitAwareArray` | ✅ Yes | ✅ **WORKING** |
| `arr1 - arr2` | Array - Array | `UnitAwareArray` | ✅ Yes | ✅ **WORKING** |
| `arr1 * arr2` | Array × Array | `UnitAwareArray` | ✅ Yes | ✅ **WORKING** |
| `arr1 / arr2` | Array / Array | `UnitAwareArray` | ✅ Yes | ✅ **WORKING** |
| `arr * scalar` | Array × number | `UnitAwareArray` | ✅ Yes | ✅ **WORKING** |
| `arr.max()` | Reduction | `UWQuantity` | ✅ Yes | ✅ **WORKING** |
| `arr.mean()` | Reduction | `UWQuantity` | ✅ Yes | ✅ **WORKING** |
| `arr.sum()` | Reduction | `UWQuantity` | ✅ Yes | ✅ **WORKING** |
| `arr.to("km")` | Unit conversion | `UnitAwareArray` | ✅ Yes | ✅ **WORKING** |

**Status:** ✅ **COMPLETE CLOSURE** - All array operations return unit-aware objects

### UWQuantity Operations

| Operation | Input Types | Output Type | Has Units? | Status |
|-----------|-------------|-------------|------------|--------|
| `qty1 + qty2` | Quantity + Quantity | `UWQuantity` | ✅ Yes | ✅ **WORKING** |
| `qty1 - qty2` | Quantity - Quantity | `UWQuantity` | ✅ Yes | ✅ **WORKING** |
| `qty1 * qty2` | Quantity × Quantity | `UWQuantity` | ✅ Yes | ✅ **WORKING** |
| `qty1 / qty2` | Quantity / Quantity | `UWQuantity` | ✅ Yes | ✅ **WORKING** |
| `qty ** 2` | Power | `UWQuantity` | ✅ Yes | ✅ **WORKING** |
| `qty.to("km")` | Unit conversion | `UWQuantity` | ✅ Yes | ✅ **WORKING** |

**Status:** ✅ **COMPLETE CLOSURE** - All quantity operations return unit-aware objects

### Mixed Operations

| Operation | Input Types | Output Type | Has Units? | Status |
|-----------|-------------|-------------|------------|--------|
| `array + quantity` | Array + Quantity | `UnitAwareArray` | ✅ Yes | ✅ **WORKING** |
| `array * quantity` | Array × Quantity | `UnitAwareArray` | ✅ Yes | ✅ **WORKING** |
| `variable.array` | Extract array | `UnitAwareArray` | ✅ Yes | ✅ **WORKING** |
| `uw.evaluate(expr)` | Evaluate on mesh | `UnitAwareArray` | ✅ Yes | ✅ **WORKING** |

---

## Operation Types and Results

### 1. Arithmetic Operations

**Inputs:** Unit-aware objects (Variables, Quantities, Arrays)

#### Addition / Subtraction

**Rule:** Operands must have **compatible units** (same dimensionality)

```python
# WORKING (UnitAwareArray)
length1 = UnitAwareArray([1, 2, 3], units="m")
length2 = UnitAwareArray([10, 20, 30], units="cm")
total = length1 + length2  # Auto-converts, result in meters ✓

# WORKING (UWQuantity)
distance1 = uw.quantity(5, "km")
distance2 = uw.quantity(500, "m")
total = distance1 + distance2  # Result: 5.5 km ✓

# NOT WORKING (Variables) - Returns plain SymPy, not unit-aware
T1 = uw.discretisation.MeshVariable("T1", mesh, units="kelvin")
T2 = uw.discretisation.MeshVariable("T2", mesh, units="kelvin")
result = T1 + T2  # Returns sympy.Matrix, not UnitAwareExpression ❌
```

**Result units:** Same as operands

#### Multiplication

**Rule:** Units **multiply** (dimensional analysis)

```python
# WORKING (UnitAwareArray)
distance = UnitAwareArray([1, 2, 3], units="m")
time = UnitAwareArray([0.5, 1.0, 1.5], units="s")
speed = distance / time  # Result: m/s ✓

# WORKING (UWQuantity)
length = uw.quantity(5, "m")
width = uw.quantity(3, "m")
area = length * width  # Result: 15 m² ✓

# NOT WORKING (Variables)
velocity = mesh_var("v", mesh, 2, units="m/s")
time_var = mesh_var("t", mesh, 1, units="s")
result = velocity * time_var  # Plain SymPy, no units ❌
```

**Result units:** Product of operand units (e.g., `m × s` → `m·s`)

#### Division

**Rule:** Units **divide** (dimensional analysis)

```python
# WORKING (UnitAwareArray)
distance = UnitAwareArray([100, 200], units="km")
time = UnitAwareArray([2, 4], units="h")
speed = distance / time  # Result: km/h ✓

# Result units: Quotient (e.g., `m / s` → `m/s` or `m·s⁻¹`)
```

#### Power

**Rule:** Units raised to power

```python
# WORKING (UWQuantity)
length = uw.quantity(5, "m")
area = length ** 2  # Result: 25 m² ✓

# WORKING (get_units on expressions)
T = mesh_var("T", mesh, units="kelvin")
T_squared = T.sym ** 2
units = uw.get_units(T_squared)  # Result: 'kelvin ** 2' ✓

# Result units: Base units to the power (e.g., `m²`, `m⁻¹`, `m^0.5`)
```

### 2. Derivative Operations

**Rule:** Derivative units = variable units / coordinate units

```python
# Coordinate has units
mesh = uw.meshing.StructuredQuadBox(...)  # With reference quantities
T = uw.discretisation.MeshVariable("T", mesh, units="kelvin")

# Derivative
dT_dx = T.sym.diff(mesh.N.x)
units = uw.get_units(dT_dx)  # Result: 'kelvin / kilometer' ✓

# Chain rule applied automatically!
```

### 3. Component Access

**Variables are matrices:**

```python
velocity = uw.discretisation.MeshVariable("v", mesh, 2, units="m/s")
# velocity.sym is Matrix([[v_x], [v_y]])

v_x = velocity[0]  # Component access
# Result: SymPy expression (NOT unit-aware) ❌

# Phase 4 fix: Return UnitAwareExpression
```

### 4. Evaluation

**Convert symbolic expressions to numerical arrays:**

```python
T = mesh_var("T", mesh, units="kelvin")
x = mesh.X[0]  # Has units if reference quantities set

# Evaluate expression
result = uw.function.evaluate(T.sym / x)
# Result: UnitAwareArray with units 'kelvin / kilometer' ✓
```

---

## Conversion Methods

### To Unit-Aware Types

#### 1. Plain Number → UWQuantity

```python
# Method 1: uw.quantity()
distance = uw.quantity(5.0, "km")

# Method 2: Direct construction
distance = UWQuantity(5.0, "km")
```

#### 2. Plain Array → UnitAwareArray

```python
import numpy as np

# Method 1: Direct construction
data = np.array([1, 2, 3])
length = UnitAwareArray(data, units="m")

# Method 2: From existing array
length = UnitAwareArray([1, 2, 3], units="m")

# Method 3: Variable.array property (automatically unit-aware)
var = mesh_var("v", mesh, units="m/s")
array = var.array  # Returns UnitAwareArray ✓
```

#### 3. SymPy Expression → UnitAwareExpression

```python
# Phase 4 functionality (NOT YET IMPLEMENTED):
T = mesh_var("T", mesh, units="kelvin")
v = mesh_var("v", mesh, 2, units="m/s")

# Current: Returns plain SymPy
expr = T * v[0]  # Returns: sympy.Mul ❌

# Phase 4: Will return UnitAwareExpression
expr = T * v[0]  # Will return: UnitAwareExpression ✓
expr.units  # Will work ✓
```

#### 4. Pint Quantity → UWQuantity

```python
import pint

ureg = pint.UnitRegistry()
pint_qty = 5 * ureg.meter

# Convert to UWQuantity
uw_qty = UWQuantity(pint_qty.magnitude, str(pint_qty.units))
```

### From Unit-Aware Types

#### 1. Extract Magnitude (Remove Units)

```python
# UWQuantity
distance = uw.quantity(5, "km")
value = distance.value  # 5.0 (float)

# UnitAwareArray
lengths = UnitAwareArray([1, 2, 3], units="m")
plain_array = lengths.magnitude  # np.ndarray([1, 2, 3])
# Or:
plain_array = lengths.view(np.ndarray)
```

#### 2. Unit Conversion

```python
# UWQuantity
distance_km = uw.quantity(5, "km")
distance_m = distance_km.to("m")  # UWQuantity(5000, "m")

# UnitAwareArray
lengths_m = UnitAwareArray([1000, 2000], units="m")
lengths_km = lengths_m.to("km")  # UnitAwareArray([1, 2], units="km")

# Variables (via .array property)
var_m = mesh_var("x", mesh, units="m")
var_m.array = [1000, 2000, 3000]
# To convert: extract, convert, assign
data_km = var_m.array.to("km")  # UnitAwareArray([1, 2, 3], "km")
```

#### 3. Check Units

```python
# Any object
obj = ...  # Variable, Quantity, Array, Expression

# Get units
units = uw.get_units(obj)
# Returns: pint.Unit, str, or None

# Check if has units
has_units = uw.get_units(obj) is not None
```

---

## Current Gaps and Phase 4 Work

### Phase 4 Goal: Complete Closure for Variable Operations

**The Problem:**
Variable arithmetic returns plain SymPy objects without unit awareness:

```python
T = mesh_var("T", mesh, units="kelvin")
v = mesh_var("v", mesh, 2, units="m/s")

# Current behavior:
result = T * v[0]
type(result)  # sympy.Mul ❌
hasattr(result, 'units')  # False ❌
uw.get_units(result)  # Works via analysis, but not direct property ⚠️

# Desired behavior (Phase 4):
result = T * v[0]
type(result)  # UnitAwareExpression ✓
result.units  # 'kelvin * meter / second' ✓
result.to("...")  # Unit conversion ✓
```

### Implementation Plan

**Part B of Phase 4:** Complete Closure Property

1. **Create `UnitAwareExpression` wrapper class** (already exists, needs enhancement)
2. **Modify `MathematicalMixin.__add__`, `__mul__`, etc.** to return `UnitAwareExpression`
3. **Update component access `__getitem__`** to return `UnitAwareExpression`
4. **Ensure all operations preserve unit-awareness**

**Files to modify:**
- `src/underworld3/utilities/mathematical_mixin.py` - Arithmetic operators
- `src/underworld3/expression/unit_aware_expression.py` - Expression wrapper

**Target:** 30/30 closure tests passing (currently ~24/30)

---

## Summary Table: Closure Status

| Object Type | Arithmetic | Derivatives | Components | Conversions | Tested? | Status |
|-------------|-----------|-------------|------------|-------------|---------|--------|
| **UnitAwareArray** | ✅ | N/A | N/A | ✅ | ✅ | **COMPLETE** |
| **UWQuantity** | ✅ | N/A | N/A | ✅ | ✅ | **COMPLETE** |
| **UnitAwareExpression** | ✅ | ✅ | ✅ | ✅ | ❌ | **EXISTS BUT NOT USED** |
| **Variable** | ❌ | ⚠️ | ❌ | ✅ | ✅ | **INCOMPLETE** (returns plain SymPy) |
| **Plain SymPy Expression** | ⚠️ | ⚠️ | ❌ | ⚠️ | ✅ | **PARTIAL** (units via `get_units()`) |

**Legend:**
- ✅ Complete closure (all operations return unit-aware)
- ⚠️ Partial (units extractable via `uw.get_units()`, but not direct properties)
- ❌ Missing (returns plain objects without units)
- N/A Not applicable

---

## Quick Reference: How to Check Units

```python
import underworld3 as uw

# Any object type:
obj = ...  # Variable, Quantity, Array, Expression

# Extract units (unified API):
units = uw.get_units(obj)
# Returns: pint.Unit, str, or None

# Check if has units:
if uw.get_units(obj) is not None:
    print(f"Object has units: {uw.get_units(obj)}")
else:
    print("Object has no units")

# For direct property access (works on some types):
if hasattr(obj, 'units'):
    print(f"Direct units property: {obj.units}")
```

---

**Next:** Phase 4 implementation to achieve complete closure for all variable operations.
