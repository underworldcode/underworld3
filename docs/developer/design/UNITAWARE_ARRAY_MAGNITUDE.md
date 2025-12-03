# UnitAwareArray .magnitude Property

**Date**: 2025-10-12
**Status**: ✅ IMPLEMENTED

## Problem

When using meshes with coordinate units, `mesh.X.coords` returns a `UnitAwareArray` with units. This caused a pattern break in tutorials where users would extract coordinates for dimensionless arithmetic:

```python
# Create mesh with coordinate units
mesh = uw.meshing.UnstructuredSimplexBox(..., units="km")

# Extract coordinates - returns UnitAwareArray with units="km"
x, y = mesh.X.coords[:, 0], mesh.X.coords[:, 1]

# This FAILS - mixing dimensionless scalar with unit-aware array
temperature.array[:, 0, 0] = 300 + 2.6 * y  # ValueError!
```

**Error**:
```
ValueError: Cannot add dimensionless scalar 300 to array with units 'km'.
Use array.to_units('dimensionless') or multiply by appropriate units.
```

**User's feedback**: "This creates x,y as unitAwareNDArray ... but there is not an obvious way to access just the coord data and strip out the units. The array method of a meshVariable is the raw data."

## Solution

Added `.magnitude` property to `UnitAwareArray` class (following Pint library convention).

### Implementation

**File**: `src/underworld3/utilities/unit_aware_array.py` (lines 174-195)

```python
@property
def magnitude(self):
    """
    Get the numerical values without units (like Pint's .magnitude).

    This returns a plain numpy array view of the data, stripping units.
    Useful when you need raw numerical values for dimensionless calculations.

    Returns
    -------
    np.ndarray
        Plain numpy array without unit tracking

    Examples
    --------
    >>> coords = mesh.X.coords  # UnitAwareArray with units="km"
    >>> x, y = coords[:, 0].magnitude, coords[:, 1].magnitude  # Plain arrays
    >>> temperature.array[:, 0, 0] = 300 + 2.6 * y  # Works - no units
    """
    # Use numpy's asarray to get a plain numpy array
    # This avoids our overridden view() method which preserves units
    return np.asarray(self)
```

### Why `np.asarray()`?

Initially tried `self.view(np.ndarray)`, but `UnitAwareArray` overrides `.view()` to preserve units. Using `np.asarray(self)` bypasses the override and returns a plain numpy array without unit tracking.

## Usage Pattern

### Correct Pattern (with `.magnitude`)
```python
# Create mesh with coordinate units
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0),
    maxCoords=(1000.0, 500.0),
    cellSize=50.0,
    units="km"
)

# Extract raw coordinate values
coords = mesh.X.coords
x = coords[:, 0].magnitude  # Plain numpy array, no units
y = coords[:, 1].magnitude  # Plain numpy array, no units

# Dimensionless arithmetic works
temperature.array[:, 0, 0] = 300 + 2.6 * y  # ✅ Works!
```

### Pattern to Avoid
```python
# This FAILS - coordinates still have units
x, y = mesh.X.coords[:, 0], mesh.X.coords[:, 1]
temperature.array[:, 0, 0] = 300 + 2.6 * y  # ❌ ValueError!

# This is VERBOSE and doesn't work (can't convert length to dimensionless)
y.to_units('dimensionless')  # ❌ DimensionalityError!
```

## Benefits

1. **Follows Pint Convention** - Familiar pattern for users who know Pint library
2. **Simple API** - Single property access to strip units
3. **Clear Intent** - `.magnitude` clearly signals "get raw numerical values"
4. **Efficient** - Returns view, not copy (when possible)
5. **Safe** - Original array retains units, magnitude is separate object

## Tutorial Updates

**File**: `docs/beginner/tutorials/12-Units_System.ipynb`

Updated cell 9 to demonstrate correct pattern:

```python
# Initialize with some values
# Note: Use .magnitude to get raw coordinate values for dimensionless arithmetic
with uw.synchronised_array_update():
    coords = mesh.X.coords
    x = coords[:, 0].magnitude  # Plain numpy array without units
    y = coords[:, 1].magnitude  # Plain numpy array without units

    temperature.array[:, 0, 0] = 300 + 2.6 * y  # Linear temperature profile
    velocity.array[:, 0, 0] = 5.0  # Constant x-velocity
    velocity.array[:, 0, 1] = 0.0  # No y-velocity
```

Updated summary (cell 19) to include:
```
4. **Extracting raw values**: Use `.magnitude` property to get plain numpy arrays from unit-aware arrays
```

## Testing

**Test script**: `/tmp/test_magnitude.py`

Results:
```
✅ coords[:, 0].magnitude works!
   type(x) = <class 'numpy.ndarray'>  # Plain numpy, not UnitAwareArray
   x.shape = (80,)
   x[0:3] = [   0. 1000.    0.]  # No units in output

✅ Original coords retains units (expected error):
   ValueError: Cannot add dimensionless scalar 300 to array with units 'km'
```

## Related Systems

- **Pint library** - Uses identical `.magnitude` property pattern
- **UWQuantity** - Also has `.magnitude` for extracting dimensionless values
- **mesh.X.coords** - Returns `UnitAwareArray` when mesh has coordinate units
- **MeshVariable.array** - Returns plain `NDArray_With_Callback` (no units)

## Design Rationale

### Why Not Convert Units?

Cannot convert dimensional quantity (length) to dimensionless:
```python
coords[:, 1].to_units('dimensionless')  # DimensionalityError!
```

### Why Not Override Arithmetic?

Could make `300 + unit_array` automatically strip units, but:
- Defeats purpose of unit checking
- Silences dimensional errors
- Makes debugging harder
- Breaks principle of least surprise

### Why `.magnitude` is Better

- **Explicit is better than implicit** - User explicitly requests raw values
- **Preserves unit safety** - Original array still protected
- **Familiar pattern** - Pint users already know this
- **Clear code** - `coords[:, 0].magnitude` clearly shows intent

## Future Considerations

- Consider adding `.m` as shorthand alias (Pint uses this: `quantity.m == quantity.magnitude`)
- Document pattern in variable initialization guide
- Add to "Common Patterns" section of documentation

---

**Status**: ✅ Implemented and tested. Tutorial updated. Ready for use.
