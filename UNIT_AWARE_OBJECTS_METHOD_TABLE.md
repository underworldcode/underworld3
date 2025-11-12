# Unit-Aware Objects and Methods - Comprehensive Table

**Generated:** 2025-01-07
**Purpose:** Complete inventory of all unit-aware classes and their methods

---

## Summary Table

| Class | Location | Status | Primary Use | Methods Count |
|-------|----------|--------|-------------|---------------|
| **UnitAwareArray** | `utilities/unit_aware_array.py` | ✅ Active | Data arrays with units | 45+ |
| **UnitAwareMixin** | `utilities/units_mixin.py` | ⚠️ Deprecated | Legacy mixin for variables | 20+ |
| **UWQuantity** | `function/quantities.py` | ✅ Active | Scalar quantities with units | 30+ |
| **DimensionalityMixin** | `utilities/dimensionality_mixin.py` | ✅ Active | Non-dimensional scaling | 15+ |
| **UnitAwareExpression** | `expression/unit_aware_expression.py` | ✅ Active | SymPy expressions with units | 20+ |
| **MathematicalExpression** | `expression/unit_aware_expression.py` | ✅ Active | Math ops on expressions | 10+ |
| **EnhancedMeshVariable** | `discretisation/enhanced_variables.py` | ✅ Active | Mesh variables with units | Inherits all |
| **EnhancedSwarmVariable** | `discretisation/enhanced_variables.py` | ✅ Active | Swarm variables with units | Inherits all |

---

## Detailed Method Tables

### 1. UnitAwareArray (`utilities/unit_aware_array.py`)

**Purpose:** NumPy arrays with automatic unit tracking and conversion
**Inheritance:** NDArray_With_Callback → numpy.ndarray
**Status:** ✅ **Core Production Class** (used everywhere)

#### Core Properties
| Method/Property | Type | Description |
|----------------|------|-------------|
| `units` | property | Get/set units string (e.g., "m", "m/s") |
| `has_units` | property | Boolean - does array have units? |
| `dimensionality` | property | Pint dimensionality dict |
| `magnitude` | property | Raw numpy array without units |
| `unit_checking` | property | Enable/disable unit compatibility checks |
| `auto_convert` | property | Enable/disable automatic unit conversion |

#### Unit Conversion
| Method | Description |
|--------|-------------|
| `to(target_units)` | Convert to different units (Pint-compatible) |

#### Statistical Methods (All preserve units)
| Method | Description |
|--------|-------------|
| `max(axis, ...)` | Local maximum |
| `min(axis, ...)` | Local minimum |
| `mean(axis, ...)` | Local mean |
| `sum(axis, ...)` | Local sum |
| `std(axis, ...)` | Local standard deviation |
| `var(axis, ...)` | Local variance |

#### MPI-Aware Global Methods (Parallel computing)
| Method | Description |
|--------|-------------|
| `global_max(axis, ...)` | Global maximum across all MPI ranks |
| `global_min(axis, ...)` | Global minimum across all MPI ranks |
| `global_sum(axis, ...)` | Global sum across all MPI ranks |
| `global_mean(axis, ...)` | Global mean across all MPI ranks |
| `global_var(axis, ...)` | Global variance across all MPI ranks |
| `global_std(axis, ...)` | Global standard deviation across all MPI ranks |
| `global_norm(ord)` | Global vector norm |
| `global_size()` | Global array size |
| `global_rms()` | Global root mean square |

#### Arithmetic Operators (All unit-aware)
| Operator | Method | Description |
|----------|--------|-------------|
| `+` | `__add__`, `__radd__` | Addition (requires compatible units) |
| `-` | `__sub__`, `__rsub__` | Subtraction (requires compatible units) |
| `*` | `__mul__`, `__rmul__` | Multiplication (units multiply) |
| `/` | `__truediv__`, `__rtruediv__` | Division (units divide) |

#### Array Methods (Preserve units)
| Method | Description |
|--------|-------------|
| `copy(order)` | Create copy with same units |
| `astype(dtype, ...)` | Convert dtype, preserve units |
| `view(dtype, type)` | Create view, preserve units |
| `reshape(*shape)` | Reshape, preserve units |
| `flatten(order)` | Flatten, preserve units |
| `squeeze(axis)` | Remove dimensions, preserve units |
| `transpose(*axes)` | Transpose, preserve units |

#### Special Methods
| Method | Description |
|--------|-------------|
| `__repr__()` | String representation with units |
| `__str__()` | User-friendly string |
| `__array_function__()` | NumPy function dispatch |

---

### 2. UnitAwareMixin (`utilities/units_mixin.py`)

**Purpose:** Add units capability to any class (legacy approach)
**Status:** ⚠️ **Deprecated** (being replaced by enhanced_variables.py hierarchy)
**Note:** Documentation says "preserved only for historical reference"

#### Core Properties
| Method/Property | Description |
|----------------|-------------|
| `units` | Get units string |
| `dimensionality` | Get Pint dimensionality |
| `has_units` | Boolean - has units? |
| `scale_factor` | Scaling factor for non-dimensionalization |

#### Methods
| Method | Description | Status |
|--------|-------------|--------|
| `set_units(units, backend)` | Set units with backend | ✅ Active |
| `create_quantity(value)` | Create Pint quantity | ✅ Active |
| `non_dimensional_value(value)` | Convert to non-dimensional | ✅ Active |
| `dimensional_value(non_dim_value)` | Convert from non-dimensional | ✅ Active |
| `check_units_compatibility(other)` | Check if units compatible | ✅ Active |
| `units_repr()` | String representation with units | ✅ Active |
| ~~`to_units(target_units)`~~ | **REMOVED Phase 2** | ❌ Deleted |

#### Arithmetic (Partial)
| Operator | Method | Description |
|----------|--------|-------------|
| `*` | `__mul__` | Multiplication with unit algebra |
| `+` | `__add__` | Addition (requires compatible units) |
| `-` | `__sub__` | Subtraction (requires compatible units) |

---

### 3. UWQuantity (`function/quantities.py`)

**Purpose:** Scalar quantities with units (wrapper around Pint Quantity)
**Inheritance:** DimensionalityMixin, UnitAwareMixin
**Status:** ✅ **Core Production Class** (ubiquitous for scalar values)

#### Core Properties
| Method/Property | Description |
|----------------|-------------|
| `value` | Get numeric value |
| `units` | Get units string |
| `has_units` | Boolean - has units? |
| `sym` | SymPy symbol representation |
| `dimensionality` | Pint dimensionality dict |

#### Unit Conversion
| Method | Description |
|--------|-------------|
| `to(target_units)` | Convert to different units |
| `to_compact()` | Auto-select best units (e.g., 1000m → 1km) |
| `to_nice_units()` | Convert to "nice" display units |
| ~~`to_units(target_units)`~~ | **REMOVED Phase 2** |

#### Arithmetic Operators (Full support)
| Operator | Method | Description |
|----------|--------|-------------|
| `+` | `__add__`, `__radd__` | Addition with unit checking |
| `-` | `__sub__`, `__rsub__` | Subtraction with unit checking |
| `*` | `__mul__`, `__rmul__` | Multiplication (units multiply) |
| `/` | `__truediv__`, `__rtruediv__` | Division (units divide) |
| `**` | `__pow__` | Power (units to power) |
| `-` | `__neg__` | Negation (preserve units) |

#### SymPy Integration
| Method | Description |
|--------|-------------|
| `_sympify_()` | Convert to SymPy symbol for expressions |
| `diff(*args, **kwargs)` | Differentiation |
| `atoms(*types)` | Get atomic symbols |
| `is_number()` | Check if numeric |

#### Utility
| Method | Description |
|--------|-------------|
| `copy(other)` | Copy from another quantity |
| `__float__()` | Convert to float (magnitude only) |
| `__format__(format_spec)` | Custom formatting |
| `__repr__()`, `__str__()` | String representations |

---

### 4. DimensionalityMixin (`utilities/dimensionality_mixin.py`)

**Purpose:** Non-dimensional scaling and reference values
**Status:** ✅ **Active** (used for non-dimensional physics)

#### Core Properties
| Method/Property | Description |
|----------------|-------------|
| `dimensionality` | Get Pint dimensionality |
| `scaling_coefficient` | Get/set scaling coefficient |
| `is_nondimensional` | Boolean - is non-dimensional? |

#### Methods
| Method | Description | Status |
|--------|-------------|--------|
| `nd_array` | Get non-dimensional array | ✅ Active |
| `from_nd(nd_value)` | Convert from non-dimensional | ✅ Active |
| `set_reference_scale(scale)` | Set reference scaling value | ✅ Active |
| ~~`to_nd()`~~ | **REMOVED Phase 2** | ❌ Deleted |

---

### 5. UnitAwareExpression (`expression/unit_aware_expression.py`)

**Purpose:** SymPy expressions that track units through operations
**Status:** ✅ **Active** (partially integrated, Phase 4 work pending)

#### Core Properties
| Method/Property | Description |
|----------------|-------------|
| `sym` | SymPy expression |
| `units` | Pint units object |
| `args` | SymPy expression arguments |

#### Arithmetic Operators (All preserve units)
| Operator | Method | Description |
|----------|--------|-------------|
| `+` | `__add__`, `__radd__` | Addition with unit checking |
| `-` | `__sub__`, `__rsub__` | Subtraction with unit checking |
| `*` | `__mul__`, `__rmul__` | Multiplication (units multiply) |
| `/` | `__truediv__`, `__rtruediv__` | Division (units divide) |
| `**` | `__pow__` | Power (units to power) |
| `-` | `__neg__` | Negation (preserve units) |

#### SymPy Operations (Units tracked)
| Method | Description |
|--------|-------------|
| `diff(var)` | Differentiation with unit algebra |
| `integrate(var)` | Integration with unit algebra |
| `expand()` | Expand expression, preserve units |
| `simplify()` | Simplify expression, preserve units |
| `subs(substitutions)` | Substitute values, track units |
| `_sympify_()` | SymPy protocol support |

---

### 6. MathematicalExpression (`expression/unit_aware_expression.py`)

**Purpose:** Extends UnitAwareExpression with evaluation capability
**Inheritance:** UnitAwareExpression
**Status:** ✅ **Active**

#### Additional Methods
| Method | Description |
|--------|-------------|
| `evaluate(coords, **kwargs)` | Evaluate expression at coordinates |
| `min()` | Evaluate and find minimum |
| `max()` | Evaluate and find maximum |

---

### 7. EnhancedMeshVariable & EnhancedSwarmVariable

**Purpose:** Production mesh/swarm variables with full unit support
**Location:** `discretisation/enhanced_variables.py`
**Status:** ✅ **Active** (production classes)

**Inheritance Chain:**
- **EnhancedMeshVariable**: UnitAwareMixin → DimensionalityMixin → MathematicalMixin → _MeshVariable
- **EnhancedSwarmVariable**: UnitAwareMixin → _SwarmVariable

**Methods:** Inherits all methods from parent classes + variable-specific methods

---

## Method Categories Summary

### 1. Unit Conversion Methods (All Classes)
- ✅ **`.to(target_units)`** - Standard Pint-compatible conversion
- ❌ ~~`.to_units(target_units)`~~ - **REMOVED Phase 2** (deprecated alias)
- ✅ **`.to_compact()`** - Auto-select nice units (UWQuantity only)
- ✅ **`.to_nice_units()`** - User-friendly units (UWQuantity only)
- ❌ ~~`.to_nd()`~~ - **REMOVED Phase 2** (broken symbolic approach)

### 2. Unit Inspection Methods
- ✅ **`.units`** - Get units string (all classes)
- ✅ **`.has_units`** - Boolean check (all classes)
- ✅ **`.dimensionality`** - Pint dimensionality dict (all classes)
- ✅ **`.magnitude`** - Raw value without units (UnitAwareArray, UWQuantity)

### 3. Arithmetic Operations (Unit-Aware)
All classes support unit-aware arithmetic:
- **Addition/Subtraction**: Requires compatible units
- **Multiplication/Division**: Units follow algebra rules
- **Power**: Units raised to power

### 4. SymPy Integration
- ✅ **`_sympify_()`** - SymPy protocol (UWQuantity, UnitAwareExpression)
- ✅ **`.diff(var)`** - Differentiation (UWQuantity, UnitAwareExpression)
- ✅ **`.sym`** - SymPy representation (UWQuantity, UnitAwareExpression, Variables)

### 5. MPI-Aware Methods (UnitAwareArray only)
All global_ methods work across MPI ranks:
- `global_max()`, `global_min()`, `global_mean()`
- `global_sum()`, `global_std()`, `global_var()`
- `global_norm()`, `global_size()`, `global_rms()`

---

## Architectural Observations

### Strengths ✅
1. **Comprehensive UnitAwareArray**: Full NumPy compatibility with units
2. **Pint Integration**: Standard scientific Python units library
3. **SymPy Integration**: Mathematical expressions with units
4. **MPI Support**: Parallel-aware statistical methods
5. **Consistent API**: `.to()` method across all classes (after Phase 2)

### Issues Identified ⚠️
1. **Multiple Inheritance Complexity**: DimensionalityMixin + UnitAwareMixin + MathematicalMixin
2. **Deprecated UnitAwareMixin**: Should be replaced by enhanced_variables.py hierarchy
3. **Incomplete Closure**: Variable operations don't always return unit-aware results (Phase 4 work)
4. **Documentation Gaps**: Some methods lack proper docstrings

### Phase 2 Cleanup Results ✅
- **Removed**: `.to_units()` from 3 classes (standardized on `.to()`)
- **Removed**: `.to_nd()` from DimensionalityMixin (broken implementation)
- **Deleted**: test_0814_dimensionality_nondimensional.py (tested broken feature)
- **Result**: Cleaner API, Pint-compatible

---

## Usage Recommendations

### For Data Arrays
✅ **Use:** `UnitAwareArray`
```python
length = UnitAwareArray([1, 2, 3], units="m")
length_km = length.to("km")  # Convert units
```

### For Scalar Quantities
✅ **Use:** `UWQuantity`
```python
import underworld3 as uw
depth = uw.quantity(1000, "km")
depth_m = depth.to("m")
```

### For Mesh/Swarm Variables
✅ **Use:** `EnhancedMeshVariable` or `EnhancedSwarmVariable`
```python
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2, units="kelvin")
```

### For Expressions
✅ **Use:** Variable arithmetic directly (returns UnitAwareExpression)
```python
momentum = density * velocity  # Units tracked automatically
```

---

## Future Work (Phases 3-5)

### Phase 3: Bug Fixes
- Investigate power units computation (`T**2` should return `'kelvin²'`)
- Fix evaluation dimensionalization

### Phase 4: Complete Closure
- Make all variable operations return `UnitAwareExpression`
- Replace `UnitAwareMixin` with enhanced_variables.py hierarchy
- Ensure 100% unit-aware operation closure

### Phase 5: Validation
- Comprehensive test suite
- Performance benchmarking
- Documentation updates

---

**End of Table**
