# Underworld3 Coding Principles

## 1. Avoid Domain-Specific Hard-Coding in General Infrastructure

**Problem Identified**: The units system's `_simple_dimensional_analysis()` method hard-codes geological terms like `'depth'`, `'length'`, `'domain'` in name pattern matching, causing it to fail when users use different but equivalent terms like `'thickness'`.

### The Fragility

```python
# FRAGILE: Hard-coded domain assumptions
elif 'depth' in name.lower() or 'length' in name.lower() or 'domain' in name.lower():
    fundamental_scales['length'] = qty
```

**Issues:**
- Fails for `crustal_thickness`, `layer_width`, `specimen_height`, etc.
- Assumes geological terminology in general-purpose code
- Requires maintenance for every domain UW3 is used in
- Breaks user workflows based on naming choices

### The Principle

**General-purpose infrastructure code must never hard-code domain-specific terminology.**

Instead:
1. **Use dimensional analysis**: Detect fundamental dimensions from the quantity's units, not its name
2. **User-provided metadata**: Allow users to specify what role a quantity plays
3. **Descriptive feedback**: Use the user's chosen names in messages and documentation

### Better Approach

```python
# ROBUST: Dimension-based detection
def _analyze_by_dimensions(self, quantities):
    """Detect fundamental scales by dimensional analysis, not naming conventions."""
    scales = {}

    for name, qty in quantities.items():
        dimensionality = qty.dimensionality

        # Use physics, not naming conventions
        if dimensionality == self._length_dimensionality:
            if 'length' not in scales:  # Use first length scale found
                scales['length'] = qty
                scales['_length_source'] = name  # Remember user's chosen name
        elif dimensionality == self._time_dimensionality:
            if 'time' not in scales:
                scales['time'] = qty
                scales['_time_source'] = name

    return scales

# User feedback uses THEIR terminology
def get_scale_summary(self):
    length_source = self._scales.get('_length_source', 'length_scale')
    return f"Length scale derived from your '{length_source}' parameter"
```

### Alternative: Explicit User Specification

```python
# Let users be explicit about roles
model.set_reference_quantities(
    typical_length=crustal_thickness,  # Clear intent
    characteristic_time=mountain_building_time,
    reference_temperature=surface_temperature
)

# Or with metadata
model.set_reference_quantities(
    crustal_thickness=35*uw.units.km,    # User's descriptive name
    mountain_building_time=10*uw.units.Myr,
    roles={
        'crustal_thickness': 'length',   # Explicit role assignment
        'mountain_building_time': 'time'
    }
)
```

### Broader Implications

**Analysis of Current Codebase**: This fragile pattern is pervasive in `model.py`:

```python
# Lines 657-659: Temperature and length detection
if 'temperature' in name.lower():
    fundamental_scales['temperature'] = qty
elif 'depth' in name.lower() or 'length' in name.lower() or 'domain' in name.lower():
    fundamental_scales['length'] = qty

# Lines 668-671: Time detection
if 'time' in name.lower():
    fundamental_scales['time'] = qty

# Lines 677: Viscosity detection
if 'viscosity' in name.lower():
    # derive mass from viscosity

# Lines 682: Density detection
if 'density' in name.lower() and 'length' in fundamental_scales:
    # derive mass from density
```

**Problems with each**:
- `'depth'` misses `thickness`, `height`, `width`, `radius`, `diameter`
- `'viscosity'` misses `eta`, `mu`, `dynamic_viscosity`
- `'density'` misses `rho`, `mass_density`, `bulk_density`
- `'temperature'` misses `temp`, `thermal`, `kelvin_scale`

**System-wide fragility**: Every domain that uses different terminology breaks the system.

### Systematic Solution

Replace **all** name-based detection with **dimensional analysis**:

```python
def _robust_dimensional_analysis(self) -> dict:
    """Dimension-based scale detection - domain agnostic."""
    fundamental_scales = {}

    # Define fundamental dimension patterns
    patterns = {
        'length': self._units.meter.dimensionality,
        'time': self._units.second.dimensionality,
        'mass': self._units.kilogram.dimensionality,
        'temperature': self._units.kelvin.dimensionality,
        'velocity': (self._units.meter / self._units.second).dimensionality,
        'viscosity': (self._units.pascal * self._units.second).dimensionality,
        'density': (self._units.kilogram / self._units.meter**3).dimensionality,
    }

    # Detect scales by physics, not naming
    for name, qty in self._reference_quantities.items():
        dimensionality = qty.dimensionality

        for scale_type, pattern in patterns.items():
            if dimensionality == pattern and scale_type not in fundamental_scales:
                fundamental_scales[scale_type] = qty
                fundamental_scales[f'_{scale_type}_source'] = name  # User's name
                break

    # Derive compound scales using established physics
    self._derive_compound_scales(fundamental_scales)
    return fundamental_scales
```

This pattern should be systematically eliminated:
- ✅ **Variable naming conventions** - Use dimensional analysis
- ✅ **Parameter detection** - Use units, not names
- ✅ **Physical assumptions** - Use established physics relationships
- ❌ **Domain-specific vocabulary** - Never hard-code field terminology

**Core Rule**: Infrastructure code should be domain-agnostic. Domain knowledge belongs in:
- User interfaces and documentation
- Domain-specific examples and tutorials
- Optional convenience functions clearly marked as domain-specific
- User-provided metadata and naming choices

---

*Identified during units system debugging - warning generation failed due to geological term assumptions in general dimensional analysis code. Pattern found to be systematic across model.py dimensional analysis.*