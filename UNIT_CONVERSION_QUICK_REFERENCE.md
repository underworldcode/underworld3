# Unit Conversion Quick Reference

## Accessing and Changing the Rounding Mode

```python
import underworld3 as uw

# Get the current model
model = uw.get_default_model()

# Check current rounding mode
print(model.unit_rounding_mode)  # Default: "powers_of_10"

# Change to engineering mode (powers of 1000)
model.unit_rounding_mode = "engineering"

# Change back to powers of 10
model.unit_rounding_mode = "powers_of_10"
```

**Important:** Set the rounding mode BEFORE calling `model.set_reference_quantities()`. The mode affects how reference quantities are rounded when creating model units.

## Converting Quantities to Better Units

### ⭐ NEW: Automatic Compact Units (Easiest!)

```python
# Create a quantity with awkward magnitude
pressure = uw.quantity(1e-9, "GPa")
print(pressure)                  # → 1e-09 gigapascal (awkward!)

# Automatically find best units
nice = pressure.to_compact()
print(nice)                      # → 1.0 pascal (perfect!)

# Alias method
better = pressure.to_nice_units()
print(better)                    # → 1.0 pascal (same result)
```

**More examples:**
```python
uw.quantity(1000000, "mm").to_compact()     # → 1.0 kilometer
uw.quantity(0.001, "km").to_compact()       # → 1.0 meter
uw.quantity(86400, "s").to_compact()        # → 86.4 kilosecond
uw.quantity(0.000001, "MPa").to_compact()   # → 1.0 pascal
```

### Basic Unit Conversion (When You Know Target Units)

```python
# Create a quantity with units
velocity = uw.quantity(5, "cm/year")

# Convert to specific units using .to() method
velocity_mps = velocity.to("m/s")           # → 1.58e-09 m/s
velocity_mmyr = velocity.to("mm/year")      # → 50.0 mm/year
velocity_kmmyr = velocity.to("km/Myr")      # → 50.0 km/Myr
```

### Advanced: Direct Pint Access

```python
velocity = uw.quantity(5, "cm/year")

# Access underlying Pint quantity
pint_qty = velocity._pint_qty

# Pint's compact (same as to_compact() but more verbose)
compact = pint_qty.to_compact()             # → 50.0 mm/year

# Convert to SI base units
base = pint_qty.to_base_units()             # → 1.58e-09 m/s
```

## Checking Quantity Properties

```python
qty = uw.quantity(5, "cm/year")

# Check if quantity has units
qty.has_units                    # → True

# Get units string
qty.units                        # → "centimeter / year"

# Get numerical value
qty.value                        # → 5.0

# Access Pint quantity (if available)
qty._pint_qty                    # → Pint Quantity object
```

## Working with Model Units

Model units (created by `model.to_model_units()`) automatically display with human-readable interpretation:

```python
model.set_reference_quantities(
    domain_depth=uw.quantity(2900, "km"),
    # ... other quantities
)

# Convert to model units
L_model = model.to_model_units(uw.quantity(1000, "m"))

# Display automatically shows interpretation
print(L_model)
# Output: 0.3448... (≈ 1000 m)

# Access technical units
print(L_model.units)
# Output: _1km  (or _100km, etc. depending on rounding mode)
```

## Practical Workflows

### Workflow 1: Display Simulation Results in Readable Units

```python
# You have a simulation result (in model units or SI)
result_mps = 1.58e-9  # meters per second

# Convert to geological units for readability
result_physical = uw.quantity(result_mps, "m/s")
result_geol = result_physical.to("cm/year")
print(result_geol)  # → 4.99 cm/year (much more intuitive!)
```

### Workflow 2: Change Rounding Mode for a New Model

```python
# Start fresh
uw.reset_default_model()
model = uw.get_default_model()

# Set to engineering mode FIRST
model.unit_rounding_mode = "engineering"

# THEN set reference quantities
model.set_reference_quantities(
    domain_depth=uw.quantity(7500, "m"),  # → _1km (not _10km)
    # ...
)
```

### Workflow 3: Find the Best Unit Representation

```python
value = uw.quantity(0.0000000158, "m/s")

# Try different units to see which looks best
print(value.to("m/s"))        # → 1.58e-08 m/s (too small)
print(value.to("cm/year"))    # → 4.99 cm/year (perfect!)
print(value.to("km/Myr"))     # → 499 km/Myr (also good for geology)

# Or let Pint choose automatically
print(value._pint_qty.to_compact())  # → automatic selection
```

## Common Unit Names

**Length:**
- `"m"`, `"km"`, `"cm"`, `"mm"`

**Time:**
- `"s"`, `"year"`, `"Myr"` (million years), `"kyr"` (thousand years)

**Velocity:**
- `"m/s"`, `"cm/year"`, `"km/Myr"`, `"mm/year"`

**Pressure:**
- `"Pa"`, `"kPa"`, `"MPa"`, `"GPa"`

**Viscosity:**
- `"Pa*s"`, `"Pa.s"`

**Density:**
- `"kg/m**3"`, `"kg/m^3"`, `"g/cm**3"`

**Temperature:**
- `"K"`, `"degC"` (not "Celsius"), `"degF"` (not "Fahrenheit")

## Summary Table

| Operation | Code | Result |
|-----------|------|--------|
| Check rounding mode | `model.unit_rounding_mode` | `"powers_of_10"` or `"engineering"` |
| Set rounding mode | `model.unit_rounding_mode = "engineering"` | Changes mode |
| **⭐ Auto compact** | **`qty.to_compact()`** | **Best representation (NEW!)** |
| **⭐ Nice units** | **`qty.to_nice_units()`** | **Alias for to_compact() (NEW!)** |
| Convert units | `qty.to("km/Myr")` | New quantity in target units |
| SI base units | `qty._pint_qty.to_base_units()` | Fundamental SI units |
| Check has units | `qty.has_units` | `True` or `False` |
| Get units string | `qty.units` | `"cm/year"` etc. |
| Get value | `qty.value` | Numerical value |

## Tips

1. **Set rounding mode before** `set_reference_quantities()` - it affects how model units are created
2. **Use `.to()` for explicit conversion** - most straightforward approach
3. **Use `.to_compact()` for auto-selection** - when you want Pint to choose the best units
4. **Model units auto-interpret** - they show human-readable form automatically
5. **Check `.has_units`** before conversion to avoid errors on dimensionless quantities
