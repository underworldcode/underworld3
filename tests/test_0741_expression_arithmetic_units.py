import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
#!/usr/bin/env python3
"""
Test units propagation in UWexpression arithmetic.
"""
import underworld3 as uw

print("=" * 70)
print("Testing Units Propagation in UWexpression Arithmetic")
print("=" * 70)

# Setup
model = uw.Model()
model.set_reference_quantities(
    length=uw.quantity(2900, "km"),
    time=uw.quantity(1, "Myr"),
)

# Test 1: Quantity * Quantity
print("\n1. UWQuantity * UWQuantity:")
v1 = uw.quantity(5, "cm/year")
t1 = uw.quantity(1, "Myr")

result1 = v1 * t1
print(f"   {v1.units} * {t1.units}")
print(f"   Result: {uw.get_units(result1)}")
print(f"   Expected: centimeter")
print(f"   ✓ Correct!" if "centimeter" in str(uw.get_units(result1)) else "   ✗ WRONG!")

# Test 2: Quantity * Expression
print("\n2. UWQuantity * UWexpression:")
v2 = uw.quantity(5, "cm/year")
t2 = uw.expression(r"t_\textrm{now}", 1, "Current time", units="Myr")

print(f"   velocity: {v2} (type: {type(v2).__name__})")
print(f"   time: {t2} (type: {type(t2).__name__})")
print(f"   velocity units: {uw.get_units(v2)}")
print(f"   time units: {uw.get_units(t2)}")

result2 = v2 * t2
print(f"   Result: {result2}")
print(f"   Result type: {type(result2).__name__}")
print(f"   Result units: {uw.get_units(result2)}")
print(f"   Expected: centimeter (cm/year * Myr = cm)")
print(f"   ✓ Correct!" if "centimeter" in str(uw.get_units(result2)) else "   ✗ WRONG!")

# Test 3: Expression * Quantity (reverse)
print("\n3. UWexpression * UWQuantity (reverse order):")
result3 = t2 * v2
print(f"   Result: {result3}")
print(f"   Result type: {type(result3).__name__}")
print(f"   Result units: {uw.get_units(result3)}")
print(f"   Expected: centimeter")
print(f"   ✓ Correct!" if "centimeter" in str(uw.get_units(result3)) else "   ✗ WRONG!")

# Test 4: Expression * Expression
print("\n4. UWexpression * UWexpression:")
v3 = uw.expression("v", 5, "velocity", units="cm/year")
t3 = uw.expression("t", 1, "time", units="Myr")

result4 = v3 * t3
print(f"   Result: {result4}")
print(f"   Result type: {type(result4).__name__}")
print(f"   Result units: {uw.get_units(result4)}")
print(f"   Expected: centimeter")
print(f"   ✓ Correct!" if "centimeter" in str(uw.get_units(result4)) else "   ✗ WRONG!")

# Test 5: Check if it's a SymPy vs Pint issue
print("\n5. Internal state inspection:")
v_qty = uw.quantity(5, "cm/year")
t_expr = uw.expression("t", 1, "time", units="Myr")
product = v_qty * t_expr

print(f"   Product type: {type(product)}")
print(f"   Has _pint_qty? {hasattr(product, '_pint_qty')}")
if hasattr(product, '_pint_qty'):
    print(f"   _pint_qty: {product._pint_qty}")
    print(f"   _pint_qty.units: {product._pint_qty.units}")
print(f"   Has .sym? {hasattr(product, 'sym')}")
if hasattr(product, 'sym'):
    print(f"   .sym: {product.sym}")
    print(f"   get_units(product.sym): {uw.get_units(product.sym) if hasattr(product, 'sym') else 'N/A'}")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
