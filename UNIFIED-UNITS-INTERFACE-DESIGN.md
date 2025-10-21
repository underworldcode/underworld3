# Unified Units Interface Design

**Date**: 2025-10-16
**Status**: Design Document
**Priority**: HIGH - Foundation for explicit model units system

## Philosophy

**Every object with units should have the same interface.**

Focus energy on making the units interface:
- **Reliable**: Consistently works across all object types
- **Safe**: Handles unitless objects gracefully
- **Helpful**: Clear patterns, good error messages

## Core Patterns

### Pattern 1: `.to()` - Universal Unit Conversion

**Principle**: If an object has units, it should have `.to(target_units)`

```python
# Physical quantities (already works)
quantity = uw.quantity(5, "cm/yr")
quantity.to("m/s")  # ✓

# Variable values
T_value = T.array[0]  # Should return UWQuantity
T_value.to("K")  # Should work

# Coordinate arrays
coords = mesh.X.coords  # Should return UnitAwareArray
coords.to("km")  # Should work

# Evaluation results
result = uw.function.evaluate(grad, coords)  # Should return UWQuantity
result.to("K/km")  # Should work

# Mesh variable arrays
values = T.array  # Should return UnitAwareArray
values.to("K")  # Should work
```

**Implementation requirement**: Return unit-aware objects everywhere, not bare numpy arrays or floats.

### Pattern 2: `uw.get_units()` - Universal Unit Query

**Principle**: Works on anything, safely returns units or None

```python
# With units
uw.get_units(uw.quantity(5, "m/s"))  # → 'm/s'
uw.get_units(T.sym)  # → 'kelvin'
uw.get_units(mesh.X[0])  # → 'meter' (or model units)
uw.get_units(T.sym.diff(x))  # → 'kelvin / meter'

# Unitless (safe!)
uw.get_units(2.5)  # → None
uw.get_units(np.array([1, 2, 3]))  # → None
uw.get_units(sympy.Symbol('x'))  # → None (unless patched)
```

**Implementation requirement**: Never throw exceptions, always return str or None.

### Pattern 3: Unit Validation Helpers

```python
# Check if units are compatible
uw.check_units(value, "m/s")  # Raises ValueError if incompatible

# Check dimensionality
uw.check_dimensionality(value, "[length]/[time]")  # Checks dimensional consistency

# Assert units for defensive programming
uw.assert_units(boundary_value, "m/s", "Boundary condition")
# Raises with helpful message if wrong
```

## Implementation Details

### 1. UWQuantity - Base Class

**Current state**: Already has `.to()` method via Pint

**Requirements**: Ensure all numeric results return UWQuantity

```python
class UWQuantity:
    def to(self, target_units):
        """Convert to target units (already implemented via Pint)"""

    @property
    def units(self):
        """Return units as string"""

    def has_units(self):
        """Check if quantity has units"""
```

### 2. UnitAwareArray - For Array Data

**Current state**: Exists but may need enhancements

**Requirements**:
- Should wrap numpy arrays with units
- Should have `.to()` method
- Should preserve units through operations

```python
class UnitAwareArray:
    def __init__(self, data, units):
        self._data = np.asarray(data)
        self._units = units

    def to(self, target_units):
        """Convert to target units"""
        if self._units is None:
            raise ValueError("Cannot convert array without units")
        converted = uw.quantity(1.0, self._units).to(target_units).magnitude
        return UnitAwareArray(self._data * converted, target_units)

    @property
    def magnitude(self):
        """Get raw numpy array"""
        return self._data

    @property
    def units(self):
        """Get units string"""
        return self._units
```

### 3. Function Evaluation Returns UWQuantity

**Current**: May return bare numpy arrays

**Target**: Always return unit-aware results

```python
def evaluate(fn, coords, **kwargs):
    """
    Evaluate function at coordinates.

    Returns:
        UWQuantity or UnitAwareArray with appropriate units
    """
    # Get units of expression
    result_units = uw.get_units(fn)

    # Evaluate (internal PETSc machinery)
    raw_result = _evaluate_internal(fn, coords)

    # Wrap in unit-aware object
    if result_units:
        if np.isscalar(raw_result):
            return uw.quantity(raw_result, result_units)
        else:
            return UnitAwareArray(raw_result, result_units)
    else:
        # No units - return as-is
        return raw_result
```

### 4. MeshVariable.array Returns Unit-Aware

**Current**: Returns bare numpy array

**Target**: Return UnitAwareArray

```python
class MeshVariable:
    @property
    def array(self):
        """
        Get variable data with units.

        Returns:
            UnitAwareArray with units from self.units
        """
        raw_data = self._get_raw_array()

        if self.units:
            return UnitAwareArray(raw_data, self.units)
        else:
            return raw_data

    @array.setter
    def array(self, value):
        """
        Set variable data, converting units if needed.

        Args:
            value: UWQuantity, UnitAwareArray, or raw array
        """
        if isinstance(value, (UWQuantity, UnitAwareArray)):
            # Convert to variable's units
            if self.units and value.units != self.units:
                converted = value.to(self.units)
                self._set_raw_array(converted.magnitude)
            else:
                self._set_raw_array(value.magnitude)
        else:
            # Raw data - assume already in correct units
            self._set_raw_array(value)
```

### 5. Mesh Coordinates Return UnitAwareArray

**Current**: May return bare arrays or patched objects

**Target**: Consistent UnitAwareArray

```python
class Mesh:
    @property
    def X(self):
        """Coordinate accessor"""
        return CoordinateAccessor(self)

class CoordinateAccessor:
    @property
    def coords(self):
        """
        Get coordinate array with units.

        Returns:
            UnitAwareArray with mesh.units
        """
        raw_coords = self.mesh._get_raw_coords()
        return UnitAwareArray(raw_coords, self.mesh.units)
```

### 6. Universal `uw.get_units()` Implementation

**File**: `src/underworld3/function/unit_conversion.py`

**Current**: Works for many types, may throw errors on some

**Target**: Safe for all types

```python
def get_units(obj):
    """
    Get units from any object, safely.

    Args:
        obj: Any object (quantity, expression, array, scalar, etc.)

    Returns:
        str: Units string (e.g., 'meter', 'kelvin/meter')
        None: If object has no units or units cannot be determined

    Never raises exceptions.
    """
    try:
        # Priority 1: Direct units attribute
        if hasattr(obj, 'units') and obj.units is not None:
            return str(obj.units)

        # Priority 2: UWQuantity/Pint quantity
        if isinstance(obj, UWQuantity):
            if obj.has_units:
                return str(obj.units)
            return 'dimensionless'

        # Priority 3: UnitAwareArray
        if isinstance(obj, UnitAwareArray):
            return obj.units

        # Priority 4: Symbolic expressions (compute from expression tree)
        if hasattr(obj, '_sympify_'):
            # Use existing compute_expression_units
            units = compute_expression_units(obj)
            return units if units else None

        # Priority 5: SymPy expressions
        if isinstance(obj, (sympy.Basic, sympy.Matrix)):
            units = compute_expression_units(obj)
            return units if units else None

        # Priority 6: Numpy arrays (check for units attribute)
        if isinstance(obj, np.ndarray):
            if hasattr(obj, 'units'):
                return obj.units
            return None

        # Priority 7: Scalars (no units)
        if isinstance(obj, (int, float, complex)):
            return None

        # Fallback: No units
        return None

    except Exception as e:
        # Never raise - log warning and return None
        import warnings
        warnings.warn(f"Could not determine units for {type(obj)}: {e}")
        return None
```

### 7. Unit Validation Helpers

**File**: `src/underworld3/units.py`

```python
def check_units(obj, expected_units):
    """
    Check if object has compatible units.

    Args:
        obj: Object to check
        expected_units: Expected units (str)

    Raises:
        ValueError: If units are incompatible
    """
    obj_units = get_units(obj)

    if obj_units is None:
        raise ValueError(f"Object has no units (expected {expected_units})")

    # Check dimensionality
    try:
        obj_qty = uw.quantity(1.0, obj_units)
        exp_qty = uw.quantity(1.0, expected_units)
        obj_qty.to(expected_units)  # Will raise if incompatible
    except Exception as e:
        raise ValueError(
            f"Incompatible units: got '{obj_units}', expected '{expected_units}'"
        ) from e


def assert_units(obj, expected_units, context=""):
    """
    Assert object has expected units (defensive programming).

    Args:
        obj: Object to check
        expected_units: Expected units
        context: Description for error message

    Raises:
        AssertionError: If units don't match
    """
    try:
        check_units(obj, expected_units)
    except ValueError as e:
        context_msg = f" in {context}" if context else ""
        raise AssertionError(f"Unit validation failed{context_msg}: {e}") from e


def check_dimensionality(obj, expected_dimensionality):
    """
    Check dimensional consistency.

    Args:
        obj: Object to check
        expected_dimensionality: String like '[length]/[time]'

    Raises:
        ValueError: If dimensionality doesn't match
    """
    obj_units = get_units(obj)

    if obj_units is None:
        raise ValueError(f"Object has no units")

    obj_qty = uw.quantity(1.0, obj_units)
    obj_dim = obj_qty.dimensionality

    # Parse expected dimensionality
    # (Implementation depends on Pint's dimensionality format)
    expected_dim = _parse_dimensionality(expected_dimensionality)

    if obj_dim != expected_dim:
        raise ValueError(
            f"Wrong dimensionality: got {obj_dim}, expected {expected_dim}"
        )
```

## Usage Patterns

### Pattern: Data Input

```python
# Read data with units
temperature_data = np.loadtxt("temp.csv")
temperature = uw.quantity(temperature_data, "K")

# Convert to model units
model_units = model.get_model_units()['temperature']
temperature_model = temperature.to(model_units)

# Set on variable
T.array[...] = temperature_model
```

### Pattern: Computation

```python
# Compute in model units (automatic)
grad = mesh.vector.gradient(T.sym)

# Check what units we got
print(f"Gradient units: {uw.get_units(grad)}")  # e.g., 'kK/Mm'

# Evaluate
grad_values = uw.function.evaluate(grad, coords)
# grad_values is UnitAwareArray with units='kK/Mm'
```

### Pattern: Output Conversion

```python
# Convert to desired output units
grad_physical = grad_values.to("K/km")

# Or convert to SI
grad_si = grad_values.to("K/m")

# Save to file
np.savetxt("gradient.csv", grad_physical.magnitude)
# Also save units metadata
with open("gradient_units.txt", "w") as f:
    f.write(f"Units: {grad_physical.units}\n")
```

### Pattern: Validation

```python
# Defensive programming in boundary conditions
def add_dirichlet_bc(self, value, boundary):
    # Validate units
    expected = model.get_model_units()['velocity']

    try:
        uw.check_units(value, expected)
    except ValueError as e:
        raise ValueError(
            f"Invalid boundary condition units: {e}\n"
            f"Hint: Convert to model units with .to('{expected}')"
        )

    # Apply BC
    self._apply_bc(value, boundary)
```

## Testing Strategy

### Unit Tests for Core Interface

```python
def test_quantity_to():
    """Test UWQuantity.to() conversion"""
    q = uw.quantity(5, "cm/yr")
    q_si = q.to("m/s")
    assert np.isclose(q_si.magnitude, 1.58e-9)

def test_array_to():
    """Test UnitAwareArray.to() conversion"""
    arr = UnitAwareArray([1, 2, 3], "km")
    arr_m = arr.to("m")
    assert np.allclose(arr_m.magnitude, [1000, 2000, 3000])
    assert arr_m.units == "m"

def test_evaluate_returns_units():
    """Test evaluate returns unit-aware results"""
    result = uw.function.evaluate(T.sym, coords)
    assert isinstance(result, (UWQuantity, UnitAwareArray))
    assert result.units is not None

def test_get_units_safe():
    """Test get_units handles all types safely"""
    assert uw.get_units(2.5) is None
    assert uw.get_units(np.array([1, 2])) is None
    assert uw.get_units(uw.quantity(1, "m")) == "m"
    assert uw.get_units(T.sym) is not None
```

### Integration Tests

```python
def test_workflow_with_units():
    """Test complete workflow with unit conversions"""
    # Input
    mesh = uw.meshing.Box(maxCoords=(1000*uw.units.km, 500*uw.units.km))
    T = uw.discretisation.MeshVariable("T", mesh, 1, units="K")
    T.array[...] = uw.quantity(300, "K")

    # Compute
    grad = mesh.vector.gradient(T.sym)
    grad_value = uw.function.evaluate(grad, coords)

    # Check units preserved
    assert uw.get_units(grad_value) is not None

    # Output conversion
    grad_physical = grad_value.to("K/km")
    assert hasattr(grad_physical, 'magnitude')
    assert hasattr(grad_physical, 'units')
```

## Implementation Priority

### Phase 1: Core Interface (HIGH PRIORITY)
1. ✅ Ensure `uw.get_units()` is safe for all types
2. ✅ Make `uw.function.evaluate()` return UWQuantity/UnitAwareArray
3. ✅ Ensure UnitAwareArray has `.to()` method

### Phase 2: Variable Integration
4. Make `MeshVariable.array` return UnitAwareArray
5. Make `mesh.X.coords` return UnitAwareArray
6. Test round-trip conversions

### Phase 3: Validation Helpers
7. Implement `check_units()` and `assert_units()`
8. Add to boundary conditions
9. Add to solver inputs

### Phase 4: Documentation
10. Document the `.to()` pattern
11. Document `uw.get_units()` usage
12. Provide examples for common workflows

## Benefits

### For Users
- **Predictable**: Same interface everywhere
- **Safe**: `get_units()` never crashes
- **Discoverable**: `.to()` is intuitive
- **Flexible**: Convert to any compatible units

### For Development
- **Reliable**: Easy to test unit handling
- **Maintainable**: Clear patterns to follow
- **Extensible**: Easy to add units to new types
- **Debuggable**: Units visible throughout

## Integration with Explicit Model Units

This interface supports the explicit model units system:

```python
# Model defines units
model_units = model.get_model_units()  # {'length': 'Mm', ...}

# Everything uses model units internally
mesh.units  # 'Mm'
T.units  # 'kK'

# But users can work in any units
T.array[...] = uw.quantity(300, "K")  # Converts K → kK

# Results in model units, convert on output
grad_value = uw.function.evaluate(grad, coords)  # In kK/Mm
grad_physical = grad_value.to("K/km")  # Convert to desired units
```

## Related Design Documents

- `EXPLICIT-MODEL-UNITS-DESIGN.md` - Model unit system architecture
- `DERIVATIVE_UNITS_SUMMARY.md` - Derivative units implementation
- `docs/developer/COORDINATE-UNITS-TECHNICAL-NOTE.md` - Technical details

## Decision: Mesh Vector Patching

**Recommendation**: **Only patch as last resort**

The coordinate patching system was an attempt to hide model units. With explicit model units, patching is less necessary:

```python
# Without patching (explicit model units)
x, y = mesh.X.coords[:, 0], mesh.X.coords[:, 1]  # UnitAwareArray in Mm
uw.get_units(x)  # 'Mm' (clear and explicit)

# With patching (hidden model units)
x, y = mesh.X  # Patched symbols
uw.get_units(x)  # 'meter' (hides that it's actually scaled)
```

**Decision**: Remove patching, use UnitAwareArray everywhere instead. Simpler and more consistent with explicit model units philosophy.

## Next Steps

1. Audit current `uw.get_units()` implementation for safety
2. Ensure `evaluate()` returns unit-aware objects
3. Add `.to()` to all array types
4. Implement validation helpers
5. Update documentation with patterns
6. Test thoroughly with explicit model units system
