# Complete Units System Guide

## The Big Picture: Two Separate Systems

Underworld3 has **two distinct unit simplification systems** that work on different types of quantities:

### 1. Rounding System (for Model Units)

**What it affects:** Model unit **creation only**

**When it's used:** When you call `model.set_reference_quantities()`

**What it does:** Determines how reference quantities are rounded to create clean model unit names

**Example:**
```python
model = uw.get_default_model()
model.unit_rounding_mode = "powers_of_10"  # Default

model.set_reference_quantities(
    domain_depth=uw.quantity(499.999, "m"),  # Floating point error!
    # ... other quantities
)

# Rounding system creates clean name: _1km instead of _499p9999m
L = model.to_model_units(uw.quantity(1000, "m"))
print(L.units)  # → _1km (clean!)
```

**Modes:**
- `"powers_of_10"` (default): 1, 10, 100, 1000, ...
- `"engineering"`: 1, 1000, 1e6, 1e9, ... (powers of 1000)

**Limitation:** Only affects model unit names, not regular quantities!

---

### 2. Compact Units (for Regular Quantities)

**What it affects:** Regular Pint quantities display and conversion

**When it's used:** When you call `quantity.to_compact()` or `quantity.to_nice_units()`

**What it does:** Automatically finds the best unit representation for display

**Example:**
```python
# Create quantity with awkward magnitude
pressure = uw.quantity(1e-9, "GPa")
print(pressure)  # → 1e-09 gigapascal (awkward!)

# Simplify to nice units
nice = pressure.to_compact()
print(nice)  # → 1.0 pascal (perfect!)
```

**How it works:** Uses Pint's built-in algorithm to choose the most readable units

**Limitation:** Only works for regular Pint quantities, not model units!

---

## Quick Decision Tree

```
Do you have a quantity with awkward magnitude?
│
├─ YES → Is it a model unit (created by to_model_units())?
│        │
│        ├─ YES → Model units already show interpretation automatically
│        │        Example: 2.0 (≈ 0.5000 km)
│        │        No action needed!
│        │
│        └─ NO → It's a regular quantity
│                 Use: qty.to_compact()
│                 Example: 1e-9 GPa → 1.0 Pa
│
└─ NO → Creating a new model with set_reference_quantities()?
         │
         ├─ YES → Set rounding mode FIRST
         │        model.unit_rounding_mode = "engineering"  # or "powers_of_10"
         │        Then call set_reference_quantities()
         │
         └─ NO → Just use quantities normally
                  Convert if needed: qty.to("target_units")
```

---

## Complete API Reference

### Model Unit Rounding

```python
# Get/set rounding mode (affects model unit creation)
model.unit_rounding_mode                    # Get current mode
model.unit_rounding_mode = "engineering"    # Set to engineering mode
model.unit_rounding_mode = "powers_of_10"   # Set to powers of 10 (default)

# Must be set BEFORE calling set_reference_quantities()
model.set_reference_quantities(...)
```

### Regular Quantity Conversion

```python
# Automatic compact units (NEW!)
qty.to_compact()                            # Pint's automatic algorithm
qty.to_nice_units()                         # Alias for to_compact()

# Manual conversion to specific units
qty.to("Pa")                                # Convert to pascals
qty.to("km/Myr")                            # Convert to km per Myr

# Advanced Pint access
qty._pint_qty.to_compact()                  # Direct Pint (verbose)
qty._pint_qty.to_base_units()               # Convert to SI base units

# Quantity properties
qty.has_units                               # Check if has units
qty.units                                   # Get units string
qty.value                                   # Get numerical value
```

---

## Examples

### Example 1: Simplifying Awkward Regular Quantities

```python
# Problem: Display shows awkward magnitude
pressure = uw.quantity(1e-9, "GPa")
distance = uw.quantity(1000000, "mm")
time = uw.quantity(86400, "s")

print(pressure)   # 1e-09 gigapascal
print(distance)   # 1000000.0 millimeter
print(time)       # 86400.0 second

# Solution: Use to_compact()
print(pressure.to_compact())   # 1.0 pascal
print(distance.to_compact())   # 1.0 kilometer
print(time.to_compact())       # 86.4 kilosecond
```

### Example 2: Creating Model with Engineering Rounding

```python
# Set rounding mode BEFORE setting reference quantities
uw.reset_default_model()
model = uw.get_default_model()
model.unit_rounding_mode = "engineering"  # Powers of 1000

# Now set reference quantities
model.set_reference_quantities(
    domain_depth=uw.quantity(7500, "m"),  # Will round to 1km (not 10km)
    reference_viscosity=uw.quantity(1, "ZPa.s"),
    reference_density=uw.quantity(3000, "kg/(m^3)"),
    mantle_temperature=uw.quantity(1000, "K"),
)

# Model units now use engineering rounding
L = model.to_model_units(uw.quantity(5000, "m"))
print(L.units)  # _1km (engineering: round to 1000, not 100 or 10000)
```

### Example 3: Model Units Already Show Nice Interpretation

```python
# Model units automatically display with interpretation
model.set_reference_quantities(...)
velocity_model = model.to_model_units(uw.quantity(5, "cm/year"))

# Model units show both technical and human-readable forms
print(velocity_model)
# Output: 1.58 (≈ 5.000 cm/year)

# No need to call to_compact() - already nice!
```

---

## Common Mistakes

### ❌ Mistake 1: Trying to use rounding mode for regular quantities

```python
# WRONG: Rounding mode doesn't affect regular quantities
model.unit_rounding_mode = "engineering"
qty = uw.quantity(1e-9, "GPa")
print(qty)  # Still shows: 1e-09 gigapascal
```

**Fix:** Use `to_compact()` for regular quantities
```python
qty.to_compact()  # → 1.0 pascal
```

### ❌ Mistake 2: Setting rounding mode after set_reference_quantities()

```python
# WRONG: Too late!
model.set_reference_quantities(...)
model.unit_rounding_mode = "engineering"  # Has no effect
```

**Fix:** Set mode BEFORE
```python
model.unit_rounding_mode = "engineering"
model.set_reference_quantities(...)  # Now it works
```

### ❌ Mistake 3: Expecting model units to respond to to_compact()

```python
# WRONG: Model units don't work with to_compact()
model_qty = model.to_model_units(uw.quantity(5, "cm/year"))
model_qty.to_compact()  # ERROR: Not available for model units
```

**Fix:** Model units already show interpretation automatically
```python
print(model_qty)  # Already shows: 1.58 (≈ 5.000 cm/year)
```

---

## Summary

| Quantity Type | For Nice Display | When to Set |
|---------------|------------------|-------------|
| **Regular quantities** | `qty.to_compact()` | Anytime |
| **Model units** | Set `model.unit_rounding_mode` | Before `set_reference_quantities()` |

**Key Insight:** The rounding system and compact units are **separate** - one for creating model units, one for displaying regular quantities. Use the right tool for the right job!
