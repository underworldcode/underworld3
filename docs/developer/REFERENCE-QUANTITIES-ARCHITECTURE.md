# Reference Quantities and Dimensional Analysis Architecture

## Overview

Underworld3's reference quantities system implements **pure dimensional analysis** to automatically derive fundamental scales [L], [M], [T], [θ] from user-provided quantities. The system is **completely domain-agnostic** and requires **no hard-coded parameter names**.

## Key Architectural Insight

Users provide **what they know** about their problem using **domain-specific quantities**:

```python
# Geoscience domain
model.set_reference_quantities(
    domain_depth=uw.quantity(2900, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
    mantle_viscosity=uw.quantity(1e21, "Pa*s"),
    temperature_scale=uw.quantity(1500, "K")
)

# Aerodynamics domain (SAME MATHEMATICS, DIFFERENT TERMINOLOGY)
model.set_reference_quantities(
    wingspan=uw.quantity(10, "m"),
    airspeed=uw.quantity(50, "m/s"),
    air_density=uw.quantity(1.225, "kg/m**3"),
    reference_temperature=uw.quantity(300, "K")
)

# Materials science domain (AGAIN, SAME MATH)
model.set_reference_quantities(
    specimen_length=uw.quantity(0.1, "m"),
    strain_rate=uw.quantity(1e-3, "1/s"),
    material_density=uw.quantity(7850, "kg/m**3"),
    max_temperature=uw.quantity(1000, "K")
)
```

The system **automatically derives** the fundamental scales without caring about parameter names. This is the fundamental principle of dimensional analysis.

## How It Works: The Three Phases

### Phase 1: Direct Dimension Mapping

First, the system looks for quantities that ARE fundamental dimensions:

```python
# These are directly mapped (if found)
[temperature] ← Any quantity with dimensionality of kelvin
[length]      ← Any quantity with dimensionality of meter
[time]        ← Any quantity with dimensionality of second
[mass]        ← Any quantity with dimensionality of kilogram
```

**Example:**
```python
model.set_reference_quantities(
    domain_depth=uw.quantity(2900, "km"),  # → [length] = 2900000 m
    temperature_scale=uw.quantity(1500, "K"),  # → [temperature] = 1500 K
)
```

### Phase 2: Composite Quantity Analysis

Then, the system analyzes composite quantities (velocities, viscosities, densities) to derive missing scales:

| If You Provide | Can Derive |
|----------------|-----------|
| Velocity (L/T) + Length | Time = Length / Velocity |
| Velocity (L/T) + Time | Length = Velocity * Time |
| Viscosity (M/L/T) + Length + Time | Mass = Viscosity * Length * Time |
| Density (M/L³) + Length | Mass = Density * Length³ |

**Example:**
```python
model.set_reference_quantities(
    domain_depth=uw.quantity(2900, "km"),        # [L]
    plate_velocity=uw.quantity(5, "cm/year"),    # [L]/[T]
)
# Derived: [T] = [L] / [L/T] = 2900 km / (5 cm/year) ≈ 1.83×10^15 seconds
```

### Phase 3: Complete System Derivation

If the system is **fully determined**, all fundamental scales are derived:

```python
model.set_reference_quantities(
    domain_depth=uw.quantity(2900, "km"),            # [L] = 2.9×10^6 m
    plate_velocity=uw.quantity(5, "cm/year"),        # [L]/[T] ratio
    mantle_viscosity=uw.quantity(1e21, "Pa*s"),      # [M]/[L]/[T]
    temperature_scale=uw.quantity(1500, "K"),        # [θ] = 1500 K
)
```

Result:
- **Length**: 2,900,000 m
- **Time**: 1.83×10^15 seconds (≈ 58 million years!)
- **Mass**: 5.3×10^42 kg
- **Temperature**: 1500 K

## Solvability: When Does Dimensional Analysis Work?

### Fully Determined Systems (Always Solvable)

A system is **fully determined** when you can solve for all fundamental scales uniquely.

**Example 1: Geoscience (4 quantities, 4 unknowns)**
```
domain_depth ← [L]
plate_velocity ← [L]/[T]
mantle_viscosity ← [M]/[L]/[T]
temperature_scale ← [θ]
```
✅ Fully solvable → All scales derived

**Example 2: Aerodynamics (4 quantities, 4 unknowns)**
```
wingspan ← [L]
airspeed ← [L]/[T]
air_density ← [M]/[L]³
temperature ← [θ]
```
✅ Fully solvable → All scales derived

### Underdetermined Systems (Gracefully Handled)

A system is **underdetermined** when you have fewer constraints than unknowns.

**Example: Only Velocity**
```python
model.set_reference_quantities(
    plate_velocity=uw.quantity(5, "cm/year")  # Only [L]/[T]
)
```

Problem: The equation V = L/T has infinite solutions!
- Option 1: L=1 km/year, T=1 year/second (??)
- Option 2: L=1 m, T=1 second
- Option 3: L=10 km, T=10 years
- ... infinitely many more

Solution: **System returns empty** (None for all scales)

**User action**: Provide at least one dimensional anchor
```python
model.set_reference_quantities(
    domain_depth=uw.quantity(2900, "km"),      # ← Anchor: [L]
    plate_velocity=uw.quantity(5, "cm/year"),  # ← Ratio: [L]/[T]
)
# Now solvable: T = L / (L/T) = 2900 km / (5 cm/year)
```

## Design Principles

### 1. No Hard-Coded Names

The system uses **`**quantities` (arbitrary keyword arguments)**, not hard-coded parameter names:

```python
# ✅ WORKS: Any name is fine
model.set_reference_quantities(wingspan=..., airspeed=...)
model.set_reference_quantities(specimen_length=..., strain_rate=...)

# ✅ WORKS: Multiple names for same domain
model.set_reference_quantities(domain_depth=..., characteristic_length=...)

# ✅ WORKS: Arbitrary ordering
model.set_reference_quantities(b=..., a=..., c=...)
```

### 2. Pure Dimensional Analysis

The system ONLY uses **dimensionality**, never parameter names:

```python
# These work identically (system sees only dimensionality)
model.set_reference_quantities(L=uw.quantity(100, "km"))
model.set_reference_quantities(domain_depth=uw.quantity(100, "km"))
model.set_reference_quantities(my_characteristic_length=uw.quantity(100, "km"))
```

### 3. Intelligent Defaults

When possible, the system derives missing scales. When not possible, it gracefully returns what it can:

```python
model.set_reference_quantities(
    L=uw.quantity(100, "km"),
    V=uw.quantity(1, "cm/year"),
    T_ref=uw.quantity(1500, "K"),
)
# Derives: [length]=100km, [time]=100km/(1cm/year), [temperature]=1500K
# Cannot derive: [mass] (not enough info)
```

## Practical Workflow: How Users Should Think About This

### Step 1: Identify Your Domain Quantities

What quantities define your problem?

```python
# Geoscience: What do you measure?
# - How deep is your model domain?
# - How fast do plates move?
# - What's your reference viscosity?
# - What's your temperature range?

# Aerodynamics: What do you measure?
# - What's the wing size?
# - What's the flight speed?
# - What's the air density?
# - What's the reference temperature?
```

### Step 2: Specify Those Quantities (With ANY Names)

```python
model.set_reference_quantities(
    your_length_quantity=uw.quantity(...),
    your_velocity_quantity=uw.quantity(...),
    your_viscosity_quantity=uw.quantity(...),
    your_temperature_quantity=uw.quantity(...),
)
```

### Step 3: System Derives Fundamental Scales

Automatically. No further action needed.

### Step 4 (If Needed): Handle Underdetermined Systems

If system returns `None` for some scales:

1. **Identify what's missing**: Which scales failed to derive?
2. **Provide anchors**: Add quantities that directly constrain those dimensions
3. **Re-check**: System should now be fully determined

## FAQ: Why No Hard-Coded Names?

**Q: The notebook shows `temperature_difference` and `plate_velocity`. Are these required?**

A: No. Those are just EXAMPLES. You can use ANY names:

```python
# All of these work identically:
model.set_reference_quantities(temperature_difference=...)
model.set_reference_quantities(temp_scale=...)
model.set_reference_quantities(T_ref=...)
model.set_reference_quantities(hottest_temp=...)  # Or literally anything

# System derives [θ] scale from the VALUE's dimensionality (Kelvin),
# not from the parameter NAME.
```

**Q: Can I mix domains in one model?**

A: Yes, as long as dimensionalities don't conflict:

```python
model.set_reference_quantities(
    # Geoscience terminology
    domain_depth=uw.quantity(100, "km"),
    # Aerodynamics terminology
    airspeed=uw.quantity(50, "m/s"),
    # Physics terminology
    reference_mass=uw.quantity(1e10, "kg"),
)
# All work together through dimensional analysis
```

## Technical Details

### Implementation: `_derive_fundamental_scales()`

Located in `src/underworld3/model.py` (lines 871-1005), this method:

1. **Phase 1** (lines 907-922): Identifies direct dimension quantities
2. **Phase 2** (lines 924-991): Analyzes composite quantities and derives missing scales
3. **Phase 3** (lines 993-1005): Converts to base SI units and returns results

### Supported Derivations

The system can derive scales from these quantity types:

| Quantity Type | Dimensionality | Used For |
|---------------|-----------------|----------|
| Length | [L] | Direct [L] |
| Time | [T] | Direct [T] |
| Mass | [M] | Direct [M] |
| Temperature | [θ] | Direct [θ] |
| Velocity | [L]/[T] | Derive [L] or [T] |
| Viscosity | [M]/[L]/[T] | Derive [M] |
| Density | [M]/[L]³ | Derive [M] |
| Strain rate | [T]⁻¹ | Derive [T] |

### Future Enhancement

The system could be enhanced with **linear algebra solver** for more complex systems:

```python
# Currently underdetermined (3 equations, 4 unknowns)
model.set_reference_quantities(
    velocity=uw.quantity(1, "m/s"),      # [L]/[T]
    viscosity=uw.quantity(1e-3, "Pa*s"), # [M]/[L]/[T]
    density=uw.quantity(1000, "kg/m**3"),# [M]/[L]³
    # Missing: direct [L], [M], [T], or [θ]
)

# Future: Could ask user for one anchor, solve rest
# OR: Could provide interactive prompts
# OR: Could accept defaults (e.g., assume T=1 second as reference)
```

## Summary

**The user's architectural vision is already implemented!**

- ✅ No hard-coded parameter names required
- ✅ Pure dimensional analysis from first principles
- ✅ Works with any domain terminology
- ✅ Automatically derives [L], [M], [T], [θ] when possible
- ✅ Gracefully handles underdetermined systems
- ✅ Completely domain-agnostic

Users specify what they **know** about their problem using their domain terminology. The system figures out the fundamental scales through pure mathematics.
