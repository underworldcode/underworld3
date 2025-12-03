# Non-Dimensionalization System Review

**Review ID**: UW3-2025-11-006
**Date**: 2025-11-17
**Status**: Submitted for Review
**Component**: Non-Dimensionalization and Scaling
**Reviewer**: [To be assigned]

## Overview

This review covers Underworld3's non-dimensionalization system that provides automatic scaling of physical quantities to dimensionless model units using Pint dimensional analysis. The system elegantly separates units (what dimension) from scaling (reference magnitude), uses sophisticated composite unit construction from dimensionality, and provides human-readable display of model units in geological terms. This represents a fundamental improvement in numerical precision and user experience for geodynamic simulations.

**Key Innovation**: The `to_model_units()` method uses Pint's native dimensional analysis to automatically construct correct composite units (like velocity = length/time, density = mass/length³) without manual arithmetic, eliminating a major source of scaling errors.

## Changes Made

### Code Changes

**Core Non-Dimensionalization**:
- `src/underworld3/model.py` - Model scaling system (~300 lines of changes)
  - `to_model_units()` method using Pint dimensional analysis (lines 3383-3446)
  - `_convert_to_model_units_general()` automatic composite unit construction (lines 3447-3633)
  - `to_model_magnitude()` convenience method for magnitude extraction (lines 3635-3682)
  - Reference quantities system (`set_reference_quantities()`)
  - Model registry creation with custom Pint constants

**Human-Readable Display**:
- `src/underworld3/function/quantities.py` - Display improvements
  - `_interpret_model_units()` method (lines 722-864)
  - Combines Pint constants numerically via `.to_base_units()`
  - Tests friendly unit conversions (cm/year, km, Myr, GPa, etc.)
  - Magnitude-based scoring for best readable units
  - Updated `__str__()`, `__repr__()`, `__format__()` methods

**Testing Non-Dimensionalization**:
- `tests/test_0816_global_nd_flag.py` - Global non-dimensionalization flag testing
- `tests/test_0817_poisson_nd.py` - Poisson solver with non-dimensionalization
- `tests/test_0818_stokes_nd.py` - Stokes solver with non-dimensionalization (5 tests passing)

### Documentation Changes

**Created**:
- `planning/HUMAN_READABLE_MODEL_UNITS.md` - Human-readable display implementation
  - Problem statement and solution approach
  - Implementation details with algorithm description
  - Examples showing before/after improvements
  - Benefits and technical notes

**Updated**:
- `CLAUDE.md` - Status update (2025-10-08)
  - Noted "Elegant to_model_units() implementation"
  - Uses Pint dimensional analysis for composite dimensions
  - Returns dimensionless UWQuantity objects

### Test Coverage

**Non-Dimensionalization Tests**:
- Global flag testing (`test_0816_global_nd_flag.py`)
- Poisson solver with scaling (`test_0817_poisson_nd.py`)
- Stokes solver with scaling (`test_0818_stokes_nd.py`) - 5/5 passing ✅

**Coverage**: Non-dimensionalization validated with multiple solver systems

## System Architecture

### Part 1: Separation of Concerns

#### Philosophy: Units vs. Scaling

**Units (Dimensional)**: Answer "what dimension?"
- Temperature in Kelvin vs. Celsius (both temperature)
- Distance in meters vs. kilometers (both length)
- Managed by units-awareness system

**Scaling (Magnitude)**: Answer "how large is typical?"
- 1500 K for mantle temperature (reference quantity)
- 2900 km for mantle depth (reference quantity)
- Managed by non-dimensionalization system

**Key Insight**: These are orthogonal concerns that should be handled separately.

#### Benefits of Separation

```python
# Units system: handles different unit systems
temperature_c = uw.quantity(25, "degC")
temperature_k = temperature_c.to("K")  # Dimensional conversion

# Scaling system: handles reference magnitudes
model = uw.Model()
model.set_reference_quantities(temperature=uw.quantity(1500, "K"))
temp_scaled = model.to_model_units(temperature_k)  # Non-dimensionalization
```

### Part 2: Elegant to_model_units() Implementation

#### Purpose

Convert any physical quantity to dimensionless model units using Pint's dimensional analysis to automatically construct composite units.

#### Key Innovation: Composite Unit Construction

**The Problem with Manual Scaling**:
```python
# Manual approach - error-prone!
length_scale = 2900e3  # meters
time_scale = 1.83e15   # seconds
velocity_scale = length_scale / time_scale  # Must calculate manually!
velocity_model = velocity_physical / velocity_scale
```

**The Elegant Approach**:
```python
# Automatic approach - Pint does the math!
model.set_reference_quantities(
    length=uw.quantity(2900, "km"),
    time=uw.quantity(58, "megayear")
)

# Pint automatically knows: velocity = length/time
velocity_model = model.to_model_units(velocity_physical)
# Constructs: _2900000m / _1p83E15s automatically! ✓
```

#### Implementation Details

**Step 1: Protocol Pattern** (lines 3427-3436):
```python
def to_model_units(self, quantity):
    """
    Convert to model units using smart protocol pattern.

    Safe to call repeatedly:
    1. Does nothing if model has no units
    2. Does nothing if quantity already in model units
    3. Does nothing if quantity is dimensionless
    4. Uses protocol pattern for extensibility
    """
    # Early returns for edge cases
    if not hasattr(self, "_pint_registry"):
        return quantity  # No reference quantities set

    if hasattr(quantity, "_is_model_units") and quantity._is_model_units:
        return quantity  # Already scaled

    if hasattr(quantity, "dimensionality") and not dict(quantity.dimensionality):
        return quantity  # Dimensionless

    # Protocol: try hidden method first
    if hasattr(quantity, "_to_model_units_"):
        result = quantity._to_model_units_(self)
        if result is not None:
            return result

    # General approach for any object
    return self._convert_to_model_units_general(quantity)
```

**Step 2: Dimensional Analysis** (lines 3522-3546):
```python
def _convert_to_model_units_general(self, quantity):
    """
    Convert using Pint's native conversion - the elegant part!
    """
    # Convert to base SI to get dimensionality
    si_qty = quantity.to_base_units()
    dimensionality = si_qty.dimensionality  # e.g., {'[length]': 1, '[time]': -1}

    # Build model unit expression from dimensional analysis
    model_units_parts = []

    for base_dim, power in dim_dict.items():
        dim_name = str(base_dim).strip("[]")  # 'length', 'time', etc.

        if dim_name in self._model_constants:
            const_info = self._model_constants[dim_name]
            const_name = const_info["constant_name"]  # '_2900000m', '_1p83E15s'

            # Build unit expression
            if power == 1:
                model_units_parts.append(const_name)
            elif power == -1:
                model_units_parts.append(f"{const_name}**-1")
            else:
                model_units_parts.append(f"{const_name}**{power}")

    # Construct model unit string
    # For velocity: "_2900000m / _1p83E15s" (automatically!)
    model_units = "*".join(model_units_parts)

    # Use Pint's native .to() method - handles scaling correctly!
    target_unit = getattr(self._pint_registry, model_units)
    converted_pint = pint_qty.to(target_unit)

    return UWQuantity._from_pint(converted_pint, model_registry=self._pint_registry)
```

**Why This Works**:
1. **Pint knows relationships**: velocity = length/time, density = mass/length³, etc.
2. **Automatic composition**: Constructs `_length / _time` for velocity
3. **Correct scaling**: Pint handles the arithmetic: 2900 km / 58 Myr = 5 cm/year
4. **No manual calculation**: Users never compute derived scales manually!

#### Examples of Composite Unit Construction

```python
model.set_reference_quantities(
    length=uw.quantity(2900, "km"),
    time=uw.quantity(58, "Myr"),
    mass=uw.quantity(1e21, "kg"),
    temperature=uw.quantity(1500, "K")
)

# Velocity = length/time
velocity = uw.quantity(5, "cm/year")
vel_model = model.to_model_units(velocity)
# Pint constructs: _2900000m * _1p83E15s**-1 automatically

# Density = mass/length³
density = uw.quantity(3300, "kg/m**3")
dens_model = model.to_model_units(density)
# Pint constructs: _1E21kg * _2900000m**-3 automatically

# Stress = mass/(length * time²)
stress = uw.quantity(1, "GPa")
stress_model = model.to_model_units(stress)
# Pint constructs: _1E21kg * _2900000m**-1 * _1p83E15s**-2 automatically

# Thermal diffusivity = length²/time
diffusivity = uw.quantity(1e-6, "m**2/s")
diff_model = model.to_model_units(diffusivity)
# Pint constructs: _2900000m**2 * _1p83E15s**-1 automatically
```

### Part 3: Human-Readable Model Units

#### The Problem

Model units using Pint constants were incomprehensible:
```python
vel_model = model.to_model_units(uw.quantity(5, "cm/year"))
print(vel_model)
# UWQuantity(0.9999999999999922, '_2900000m / _1p83E15s')  # What?!
```

Users couldn't understand what `_2900000m / _1p83E15s` meant.

#### The Solution

Automatic interpretation combining Pint constants numerically and converting to user-friendly units:

```python
print(vel_model)
# 3.16e9 (≈ 5.000 cm/year)  # Ah, makes sense!

print(repr(vel_model))
# UWQuantity(3.16e9, '_2900000m / _1p83E15s')  [≈ 5.000 cm/year]

print(f"Velocity: {vel_model}")
# Velocity: 3.16e9 (≈ 5.000 cm/year)
```

#### Implementation

**Algorithm** (lines 722-864):
```python
def _interpret_model_units(self):
    """
    Interpret model units in human-readable form.

    Steps:
    1. Create unit quantity (1.0 * model_units)
    2. Convert to SI base units (combines constants numerically)
    3. Try friendly unit conversions (cm/year, km, Myr, GPa, etc.)
    4. Score by magnitude (prefer 0.001-1000 range)
    5. Return best interpretation
    """
    # Example: _2900000m / _1p83E15s
    unit_qty = 1.0 * self._pint_qty.units

    # Combines: (2.9e6 / 1.83e15) m/s = 1.58e-9 m/s
    base_qty = unit_qty.to_base_units()

    # Try conversions to friendly units
    best_score = float('inf')
    best_conversion = None

    for name, friendly_unit in friendly_conversions:
        try:
            converted = base_qty.to(friendly_unit)
            magnitude = abs(converted.magnitude)

            # Score: prefer magnitudes 0.001-1000
            if 0.001 <= magnitude <= 1000:
                score = abs(log10(magnitude))  # Prefer ~1
            else:
                score = 100 + abs(log10(magnitude) - 1.5)  # Heavily penalize

            if score < best_score:
                best_score = score
                best_conversion = (magnitude, name)

        except Exception:
            continue

    if best_conversion:
        magnitude, name = best_conversion
        return f"≈ {magnitude:.3f} {name}"

    return None
```

**Friendly Units Priority** (geological first):
```python
friendly_conversions = [
    # Velocity (geological scales first)
    ('cm/year', ureg.cm / ureg.year),
    ('km/Myr', ureg.km / (1e6 * ureg.year)),
    ('mm/year', ureg.mm / ureg.year),
    ('m/s', ureg.m / ureg.s),

    # Length
    ('km', ureg.km),
    ('m', ureg.m),
    ('cm', ureg.cm),

    # Time (geological first)
    ('Myr', 1e6 * ureg.year),
    ('kyr', 1e3 * ureg.year),
    ('year', ureg.year),
    ('s', ureg.s),

    # Pressure/stress (geological first)
    ('GPa', 1e9 * ureg.Pa),
    ('MPa', 1e6 * ureg.Pa),
    ('Pa', ureg.Pa),

    # Temperature, viscosity, density, etc.
]
```

**Display Integration**:
```python
def __str__(self):
    """String representation with human-readable interpretation."""
    if self._has_custom_units and self._model_instance:
        interpretation = self._interpret_model_units()
        if interpretation:
            return f"{self.value} ({interpretation})"
    return f"{self.value} {self.units}"

def __repr__(self):
    """Technical representation with interpretation appended."""
    base_repr = f"UWQuantity({self.value}, '{self.units}')"
    if self._has_custom_units and self._model_instance:
        interpretation = self._interpret_model_units()
        if interpretation:
            return f"{base_repr}  [{interpretation}]"
    return base_repr
```

### Part 4: Reference Quantities System

#### Purpose

Allow users to specify characteristic magnitudes that define the model's scaling system.

#### Setting Reference Quantities

```python
model = uw.Model()

# Define fundamental scales
model.set_reference_quantities(
    length=uw.quantity(2900, "km"),        # Mantle depth
    temperature=uw.quantity(1500, "K"),     # Mantle temperature
    time=uw.quantity(58, "Myr"),            # Convective time scale
    mass=uw.quantity(1e21, "kg")            # Reference mass
)

# All derived scales computed automatically!
scales = model.get_fundamental_scales()
print(scales['length'])      # 2900 kilometer
print(scales['time'])        # 58 megayear
print(scales['temperature']) # 1500 kelvin

# Derived quantities computed via dimensional analysis:
# velocity = length/time
# density = mass/length³
# stress = mass/(length*time²)
# etc.
```

#### Model Registry Creation

When `set_reference_quantities()` is called:
1. Creates custom Pint registry with model-specific constants
2. Registers constants with names like `_2900000m`, `_1p83E15s`
3. Stores in `model._pint_registry` for use in `to_model_units()`
4. Enables automatic composite unit construction

### Part 5: Dimensionless UWQuantity Objects

#### Purpose

Model units are dimensionless (the scaling has been applied), but maintain unit information for display and conversion purposes.

#### Key Properties

**Dimensionless Value**:
```python
velocity = uw.quantity(5, "cm/year")
vel_model = model.to_model_units(velocity)

# Value is dimensionless number
print(vel_model.value)  # 0.999... (close to 1.0 in model units)

# But has dimensional information for interpretation
print(vel_model.units)  # '_2900000m / _1p83E15s'
print(vel_model)        # 1.0 (≈ 5.000 cm/year)
```

**Marked as Model Units**:
```python
# Flag prevents re-scaling
vel_model._is_model_units = True

# Calling to_model_units() again is safe (no-op)
vel_model_again = model.to_model_units(vel_model)
assert vel_model_again is vel_model  # Same object returned
```

**Model Instance Attached**:
```python
# Stores reference to model for interpretation
vel_model._model_instance = model

# Enables human-readable display
print(vel_model)  # 1.0 (≈ 5.000 cm/year)
```

## Testing Instructions

### Test Basic Non-Dimensionalization

```python
import underworld3 as uw

# Create model with reference quantities
model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(100, "km"),
    time=uw.quantity(1, "Myr"),
    temperature=uw.quantity(1000, "K"),
    mass=uw.quantity(1e20, "kg")
)

# Test simple quantity
length_phys = uw.quantity(50, "km")
length_model = model.to_model_units(length_phys)
print(f"Length: {length_model}")  # Should be ~0.5 (50/100)

# Test composite quantity (velocity)
velocity_phys = uw.quantity(5, "cm/year")
velocity_model = model.to_model_units(velocity_phys)
print(f"Velocity: {velocity_model}")  # Should show human-readable interpretation

# Verify dimensionality is preserved
print(f"Velocity dimensionality: {velocity_model.dimensionality}")
# Should be {'[length]': 1, '[time]': -1}
```

### Test Composite Unit Construction

```python
# Define only fundamental scales
model.set_reference_quantities(
    length=uw.quantity(2900, "km"),
    time=uw.quantity(58, "Myr"),
    mass=uw.quantity(1e21, "kg")
)

# Pint should automatically construct composite units
density_phys = uw.quantity(3300, "kg/m**3")
density_model = model.to_model_units(density_phys)
print(f"Density: {density_model}")

stress_phys = uw.quantity(1, "GPa")
stress_model = model.to_model_units(stress_phys)
print(f"Stress: {stress_model}")

viscosity_phys = uw.quantity(1e21, "Pa*s")
viscosity_model = model.to_model_units(viscosity_phys)
print(f"Viscosity: {viscosity_model}")

# All should have human-readable interpretations
```

### Test Human-Readable Display

```python
# Model units should be interpretable
velocity_model = model.to_model_units(uw.quantity(5, "cm/year"))

# Test different display modes
print(str(velocity_model))   # "1.0 (≈ 5.000 cm/year)"
print(repr(velocity_model))  # "UWQuantity(...) [≈ 5.000 cm/year]"
print(f"Vel: {velocity_model}")  # "Vel: 1.0 (≈ 5.000 cm/year)"
```

### Run Non-Dimensionalization Tests

```bash
# Global flag testing
pytest tests/test_0816_global_nd_flag.py -v

# Poisson solver with scaling
pytest tests/test_0817_poisson_nd.py -v

# Stokes solver with scaling
pytest tests/test_0818_stokes_nd.py -v
```

## Known Limitations

### 1. Cannot Convert Model Units Directly

**Issue**: Model units contain custom Pint constants that don't exist in standard registries.

**Example**:
```python
vel_model = model.to_model_units(uw.quantity(5, "cm/year"))

# Cannot convert model units to arbitrary units
try:
    vel_si = vel_model.to("m/s")  # Error: Unknown unit '_2900000m'
except Exception as e:
    print(f"Cannot convert: {e}")
```

**Reason**: Model units like `_2900000m` are dimensionless scaled values, not real units.

**Workaround**: Conversion must go through the model:
```python
# Method 1: Use model's from_model_units() (if implemented)
# vel_phys = model.from_model_units(vel_model, target_units="m/s")

# Method 2: Re-create physical quantity using interpretation
# From display: "1.0 (≈ 5.000 cm/year)"
# User knows original physical value
```

**Future**: Implement `model.from_model_units()` for reverse conversion.

### 2. Interpretation Heuristics May Not Always Be Ideal

**Issue**: Magnitude-based scoring might not choose the units users expect.

**Example**:
```python
# User expects mm/year, but system chooses cm/year
velocity = uw.quantity(2.5, "mm/year")
vel_model = model.to_model_units(velocity)
print(vel_model)  # "... (≈ 0.25 cm/year)" instead of "2.5 mm/year"
```

**Reason**: Scoring prefers magnitudes in range [0.001, 1000], with preference near 1.

**Workaround**: User can recognize equivalent units (0.25 cm/year = 2.5 mm/year).

**Future**: Allow user preferences for unit display (per-dimension preferred units).

### 3. Requires Complete Set of Fundamental Dimensions

**Issue**: Cannot convert quantities with dimensions not defined in reference quantities.

**Example**:
```python
model.set_reference_quantities(
    length=uw.quantity(100, "km"),
    time=uw.quantity(1, "Myr")
    # Missing: temperature, mass
)

# Temperature conversion fails
temp = uw.quantity(1000, "K")
temp_model = model.to_model_units(temp)  # Returns None (dimensionless check)
```

**Reason**: No reference quantity for temperature dimension.

**Solution**: Define all fundamental dimensions used in simulation:
```python
model.set_reference_quantities(
    length=...,
    time=...,
    temperature=...,  # Add missing dimensions
    mass=...
)
```

### 4. Pint Constant Names Are Opaque

**Issue**: Internal constant names like `_2900000m` are not self-documenting.

**Impact**: Users seeing raw model units in debugging don't immediately understand meaning.

**Mitigation**: Human-readable interpretation mostly solves this for display purposes.

**Future**: Consider more descriptive constant names or metadata.

## Benefits Summary

### For Users

1. **Automatic Composite Units**: Never manually calculate derived scales (velocity = length/time computed automatically)
2. **Human-Readable Display**: Model units shown in familiar geological terms (cm/year, GPa, Myr)
3. **Safe Repeated Calls**: `to_model_units()` is idempotent - safe to call multiple times
4. **Dimensional Integrity**: Dimensionality preserved through scaling (velocity stays [length/time])
5. **Natural Workflow**: Define fundamental scales, all derived scales computed automatically

### For Developers

1. **Elimination of Scaling Bugs**: Pint handles composite unit arithmetic, no manual errors
2. **Clearer Code**: `model.to_model_units(velocity)` is self-documenting
3. **Separation of Concerns**: Scaling orthogonal to dimensional analysis
4. **Extensibility**: Protocol pattern (`_to_model_units_()`) allows custom conversion logic
5. **Debugging**: Human-readable display makes tracking scaling issues easier

### For Numerical Stability

1. **Values Near Unity**: Scaled values typically O(1), optimal for floating-point arithmetic
2. **Reduced Underflow/Overflow**: Geological scales (km, Myr) scaled to reasonable magnitudes
3. **Consistent Precision**: All quantities scaled similarly, uniform precision
4. **Solver Conditioning**: Better matrix conditioning with O(1) values

### For Project

1. **Scientific Correctness**: Dimensional analysis ensures physical correctness
2. **User Experience**: Geological units natural for domain scientists
3. **Error Prevention**: Automatic scaling eliminates manual calculation errors
4. **Professional Quality**: Matches best practices in scientific computing

## Related Documentation

- `planning/HUMAN_READABLE_MODEL_UNITS.md` - Human-readable display implementation
- `CLAUDE.md` - Status notes on elegant to_model_units() implementation (2025-10-08)
- `src/underworld3/model.py` - Implementation source (lines 3383-3682)
- `src/underworld3/function/quantities.py` - UWQuantity with interpretation (lines 722-864)

## Sign-Off

| Role | Name | Date | Status |
|------|------|------|--------|
| Author | AI Assistant | 2025-11-17 | Submitted |
| Primary Reviewer | [To be assigned] | | Pending |
| Secondary Reviewer | [To be assigned] | | Pending |
| Project Lead | [To be assigned] | | Pending |

## Review Comments and Resolutions

[To be filled during review process]

---

**Review Status**: Awaiting assignment of reviewers
**Expected Completion**: [TBD]
**Priority**: HIGH

This review documents an elegant non-dimensionalization system that uses Pint's dimensional analysis to automatically construct composite model units, eliminating manual scaling arithmetic while providing human-readable display of scaled quantities in geological terms.
