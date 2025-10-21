#!/usr/bin/env python3
"""
Demonstration of where the rounding system IS and ISN'T used.
"""

import underworld3 as uw

print("="*70)
print("ROUNDING SYSTEM: WHERE IT'S USED vs WHERE IT'S NOT")
print("="*70)
print()

# ============================================================================
# WHERE IT IS USED: Model unit creation
# ============================================================================
print("1. WHERE ROUNDING IS USED: Model Unit Creation")
print("-" * 70)

uw.reset_default_model()
model = uw.get_default_model()

print("When you call set_reference_quantities(), the rounding system")
print("determines how to create clean model unit names:")
print()

model.set_reference_quantities(
    reference_viscosity=uw.quantity(1, "ZPa.s"),
    reference_density=uw.quantity(3000, "kg/(m^3)"),
    domain_depth=uw.quantity(499.999, "m"),  # Gets rounded to 500m → _1km
    mantle_temperature=uw.quantity(1000, "K"),
)

L = model.to_model_units(uw.quantity(1000, "m"))
print(f"Result: {L}")
print(f"Unit name: {L.units}")
print()
print("✓ The rounding system created '_1km' instead of '_499p9999m'")
print()

# ============================================================================
# WHERE IT'S NOT USED: Regular quantity display
# ============================================================================
print("="*70)
print("2. WHERE ROUNDING IS NOT USED: Regular Quantities")
print("-" * 70)
print()

# Create a quantity with awkward magnitude
quant = uw.quantity(1e-9, 'GPa')

print(f"Regular quantity: {quant}")
print(f"  Type: {type(quant)}")
print(f"  Display: {quant}")
print()
print("❌ NOT simplified automatically!")
print("❌ No 'nice version' shown adjacent!")
print()

# The underlying Pint quantity
print(f"Underlying Pint: {quant._pint_qty}")
print()

# ============================================================================
# CURRENT WORKAROUND: Manual conversion
# ============================================================================
print("="*70)
print("3. CURRENT WORKAROUND: Manual Conversion")
print("-" * 70)
print()

print("You have to manually convert to better units:")
quant_pa = quant.to("Pa")
quant_mpa = quant.to("MPa")

print(f"Original:        {quant}")
print(f"Convert to Pa:   {quant_pa}")
print(f"Convert to MPa:  {quant_mpa}")
print()

# ============================================================================
# PINT'S BUILT-IN SOLUTION: to_compact()
# ============================================================================
print("="*70)
print("4. PINT'S BUILT-IN SOLUTION: to_compact()")
print("-" * 70)
print()

print("Pint has a to_compact() method, but you need direct access:")
print()

quant = uw.quantity(1e-9, 'GPa')
print(f"Original:           {quant}")
print(f"Via _pint_qty:      {quant._pint_qty.to_compact()}")
print()

# More examples
examples = [
    uw.quantity(1000000, "mm"),
    uw.quantity(0.001, "km"),
    uw.quantity(1e-9, "GPa"),
    uw.quantity(86400, "s"),
]

print("More examples using to_compact():")
for ex in examples:
    compact = ex._pint_qty.to_compact()
    print(f"  {str(ex):30s} → {compact}")
print()

# ============================================================================
# THE PROBLEM
# ============================================================================
print("="*70)
print("5. THE PROBLEM")
print("-" * 70)
print()

print("The fancy rounding system (powers of 10 vs engineering) only affects:")
print("  ✓ Model unit creation (_1km, _100m, etc.)")
print()
print("It does NOT affect:")
print("  ❌ Display of regular UWQuantity objects")
print("  ❌ Automatic simplification of awkward magnitudes")
print("  ❌ Finding 'nice' units for user-created quantities")
print()
print("So uw.quantity(1e-9, 'GPa') shows as '1e-9 GPa', not '1.0 Pa'")
print()

# ============================================================================
# WHAT'S MISSING
# ============================================================================
print("="*70)
print("6. WHAT'S MISSING")
print("-" * 70)
print()

print("Users need:")
print("  1. Easy access to to_compact() without ._pint_qty")
print("  2. Automatic display of 'nice' version for awkward magnitudes")
print("  3. A way to 're-ground' quantities to better units")
print()

print("Desired API:")
print("  quant = uw.quantity(1e-9, 'GPa')")
print("  print(quant.to_compact())         # → 1.0 Pa")
print("  print(quant.to_nice_units())      # → 1.0 Pa")
print()
print("Or automatic display:")
print("  print(quant)  # → 1e-9 GPa  [≈ 1.0 Pa]")
print()

print("="*70)
