# Units System Design Principles

**Date**: 2025-01-10
**Context**: Clarification during units system debugging and test fixes

## Core Concept: Unit System Transformations

The Underworld3 units system is **NOT** about:
- Making quantities dimensionless
- Normalizing values to 0→1 ranges
- Removing physical meaning from coordinates

The units system **IS** about:
- **Transforming between different unit systems**: model units ↔ user units
- Providing problem-appropriate scaling for numerical stability
- Maintaining full dimensional analysis throughout

## Reference Quantities Define a Unit System

When you call:
```python
model.set_reference_quantities(
    characteristic_length=1000 * uw.units.km,
    plate_velocity=5 * uw.units.cm / uw.units.year,
    mantle_temperature=1500 * uw.units.kelvin
)
```

You are defining a **model unit system**:
- 1 model length unit = 1000 km
- 1 model velocity unit = 5 cm/year
- 1 model temperature unit = 1500 K

This is analogous to defining:
- 1 parsec = 3.26 light-years
- 1 astronomical unit = 149.6 million km
- 1 electron-volt = 1.602×10⁻¹⁹ joules

The reference quantities are **conversion factors**, not normalization parameters.

## Two Unit Systems in UW3

### 1. Model Units (Internal/Computational)
- Used internally by PETSc, solvers, and mesh coordinates
- Defined by reference quantities
- Chosen for **numerical convenience** (e.g., avoiding very large/small numbers)
- Often use powers of 10 for human-friendly reference scales

**Example**: A mesh created with `minCoords=(0, 0), maxCoords=(2.9, 2.9)`:
- These are in **model length units**
- With `characteristic_length=1000 km`, this is a 2900 km × 2900 km domain
- Could equally be `maxCoords=(1, 1)` for 1000 km × 1000 km - **0→1 is not special**

### 2. User Units (External/Physical)
- SI units (meters, kilograms, seconds) or human-friendly units (km, years, GPa)
- What users work with when specifying boundary conditions, initial conditions
- What gets displayed in visualization and output

**Example**: User specifies a temperature boundary condition:
- Input: `T_surface = 300 * uw.units.kelvin`
- Stored internally: `300/1500 = 0.2` (in model temperature units)
- Output/visualization: `300 K` (converted back to user units)

## Coordinate System Transformations

### mesh._points (Model Coordinates)
- Coordinates as stored internally in PETSc
- In **model units** as specified at mesh creation
- Example: `(0.5, 0.5)` in model units

### mesh.data / mesh.points (User Coordinates)
- Coordinates in **user units** (typically meters or specified units)
- Automatically converted from model units using `length_scale`
- Example: `(500000, 500000)` meters = 500 km

### mesh.X (Symbolic Coordinates)
- **Without scaling**: `mesh.X[0] = N.x` (symbolic model coordinate)
- **With scaling**: `mesh.X[0] = 1000000.0*N.x` (symbolic physical coordinate in meters)
- Designed for user-facing symbolic expressions in physical units

## The 0→1 Range is NOT Special

Common misconception: "Model coordinates are dimensionless and range from 0 to 1"

**Reality**:
- Model coordinates are **fully dimensional** (they have length units)
- The range depends on mesh creation: could be 0→1, 0→2.9, -1→1, etc.
- 0→1 is just a **convention** often used for normalized/non-dimensional problems
- The reference quantities define the **conversion factor** to physical units

**Example**:
```python
# Mesh spanning 0→2.9 in model units
mesh = uw.meshing.StructuredQuadBox(
    minCoords=(0.0, 0.0),
    maxCoords=(2.9, 2.9)  # 2.9 model length units
)

# With characteristic_length=1000 km:
# - Model coordinates: 0 to 2.9 (in model length units)
# - Physical coordinates: 0 to 2900 km
# - mesh.data returns coordinates in meters: 0 to 2,900,000
```

## Design Philosophy

### Why This Approach?

1. **Numerical Stability**: Working in model units prevents:
   - Very large numbers (10⁶ meters)
   - Very small numbers (10⁻¹⁵ Pa·s)
   - Loss of precision in floating-point arithmetic

2. **Problem-Appropriate Scaling**: Choose reference scales that make sense:
   - Mantle convection: characteristic_length ~ 1000 km (mantle depth)
   - Lithosphere deformation: characteristic_length ~ 100 km
   - Laboratory experiments: characteristic_length ~ 1 cm

3. **Human-Friendly References**: Powers of 10 make unit conversions intuitive:
   - 1000 km, not 1.234 km
   - 1500 K, not 1234.5 K
   - Makes mental arithmetic easier

4. **Full Dimensional Analysis**: Never lose physical meaning:
   - Model units have dimensions: [length], [time], [temperature], [mass]
   - Can check dimensional consistency
   - Can convert to any compatible unit system

## Implementation: Unit Conversions in evaluate()

When calling `uw.function.evaluate()`:

```python
# Case 1: Model coordinates (no coord_units specified)
result = uw.function.evaluate(temp.sym, coords)
# - coords interpreted as model units
# - No coordinate conversion
# - Result units determined by variable's units attribute

# Case 2: Physical coordinates (coord_units specified)
result = uw.function.evaluate(temp.sym, coords, coord_units='km')
# - coords converted: km → meters → model units
# - Evaluation in model units
# - Result units determined by variable's units attribute
```

### Variable Units Determine Result Units

- Variable with `units="kelvin"` → result is UWQuantity in kelvin
- Variable with `units=None` → result is plain numpy array (no units)
- The input coordinate units do NOT affect output units
- Output units come from the **expression being evaluated**, not the coordinates

## Key Takeaway

**The units system transforms between problem-appropriate model units and human-readable user units.**

It's fundamentally about **unit system conversions**, not normalization or removing dimensions. Both model and user units are fully dimensional - they just use different reference scales optimized for different purposes.

## References for Documentation

When documenting the units system:
- ✅ DO: Emphasize "unit system transformation"
- ✅ DO: Show examples with different domain ranges (not just 0→1)
- ✅ DO: Explain reference quantities as "conversion factors"
- ❌ DON'T: Say "dimensionless" or "normalized" coordinates
- ❌ DON'T: Imply 0→1 is required or special
- ❌ DON'T: Suggest model units lack physical meaning

## Related Files

- `src/underworld3/function/unit_conversion.py` - Core unit conversion logic
- `src/underworld3/model.py` - Reference quantities and fundamental scales
- `tests/test_0804_backward_compatibility_units.py` - Tests for evaluate() with scaling
- `tests/test_0720_coordinate_units_gradients.py` - Coordinate unit conversion tests
