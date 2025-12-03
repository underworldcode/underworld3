"""
Test composite expressions with mixed units evaluate correctly.

This tests the fix for expressions containing quantities with non-SI units
(e.g., km instead of m) being properly converted to base units before evaluation.

Key Behavior:
- When UWexpressions with units are substituted, values are converted to base SI units
- Results are returned as plain arrays (no unit wrapping) to avoid unit mismatches
- Numeric values are correct in base SI units

Tests both patterns:
1. Direct UWQuantity arithmetic
2. UWexpression-wrapped quantities (lazy evaluation pattern)
3. Direct composite evaluation (user's pattern)
"""

import pytest

# Units system tests - intermediate complexity
pytestmark = pytest.mark.level_2
import numpy as np
import underworld3 as uw


def test_composite_direct_quantities():
    """Test composite expressions with direct UWQuantity objects."""
    # Create quantities with mixed units (L in km)
    rho0 = uw.quantity(3300, "kg/m^3")
    alpha = uw.quantity(3e-5, "1/K")
    g = uw.quantity(10, "m/s^2")
    DeltaT = uw.quantity(1300, "K")
    L = uw.quantity(2900, "km")  # ← Non-SI unit
    eta0 = uw.quantity(1e21, "Pa*s")
    kappa = uw.quantity(1e-6, "m^2/s")

    # Calculate Rayleigh number
    Ra = (rho0 * alpha * g * DeltaT * L**3) / (eta0 * kappa)

    # Evaluate at a point
    coords = np.array([[0.0, 0.0]])
    result = uw.function.evaluate(Ra, coords)

    # Expected value (manual calculation with proper unit conversion)
    L_meters = 2900000  # km to m
    Ra_expected = (3300 * 3e-5 * 10 * 1300 * L_meters**3) / (1e21 * 1e-6)

    # Verify result
    assert abs(result[0, 0, 0] - Ra_expected) / Ra_expected < 1e-6, \
        f"Expected {Ra_expected:.6e}, got {result[0, 0, 0]:.6e}"


def test_composite_wrapped_expressions():
    """Test composite expressions with UWexpression-wrapped quantities (lazy evaluation)."""
    # Create UWexpressions wrapping quantities (user's preferred pattern)
    rho0 = uw.expression(r"\rho_0", sym=uw.quantity(3300, "kg/m^3"))
    alpha = uw.expression(r"\alpha", sym=uw.quantity(3e-5, "1/K"))
    g = uw.expression("g", sym=uw.quantity(9.8, "m/s^2"))
    kappa = uw.expression(r"\kappa", sym=uw.quantity(1e-6, "m^2/s"))
    eta0 = uw.expression(r"\eta_0", sym=uw.quantity(1e21, "Pa*s"))
    DeltaT = uw.expression(r"\Delta T", sym=uw.quantity(3000, "K"))
    L = uw.expression("L", sym=uw.quantity(2900, "km"))  # ← Non-SI unit

    # Create composite expression
    Ra_se = uw.expression("Ra", (rho0 * alpha * g * DeltaT * L**3) / (eta0 * kappa))

    # Evaluate at a point
    coords = np.array([[0.0, 0.0]])
    result = uw.function.evaluate(Ra_se, coords)

    # Expected value (manual calculation with proper unit conversion)
    L_meters = 2900000  # km to m
    Ra_expected = (3300 * 3e-5 * 9.8 * 3000 * L_meters**3) / (1e21 * 1e-6)

    # Verify result
    assert abs(result[0, 0, 0] - Ra_expected) / Ra_expected < 1e-6, \
        f"Expected {Ra_expected:.6e}, got {result[0, 0, 0]:.6e}"


def test_direct_composite_evaluation():
    """Test direct evaluation of composite expressions (no outer wrapper)."""
    # Create UWexpressions wrapping quantities
    rho0 = uw.expression(r"\rho_0", sym=uw.quantity(3300, "kg/m^3"))
    alpha = uw.expression(r"\alpha", sym=uw.quantity(3e-5, "1/K"))
    g = uw.expression("g", sym=uw.quantity(9.8, "m/s^2"))
    kappa = uw.expression(r"\kappa", sym=uw.quantity(1e-6, "m^2/s"))
    eta0 = uw.expression(r"\eta_0", sym=uw.quantity(1e21, "Pa*s"))
    DeltaT = uw.expression(r"\Delta T", sym=uw.quantity(3000, "K"))
    L = uw.expression("L", sym=uw.quantity(2900, "km"))  # ← Non-SI unit

    # Evaluate composite directly (no outer uw.expression wrapper)
    coords = np.array([[0.0, 0.0]])
    result = uw.function.evaluate((rho0 * alpha * g * DeltaT * L**3) / (eta0 * kappa), coords)

    # Expected value
    L_meters = 2900000  # km to m
    Ra_expected = (3300 * 3e-5 * 9.8 * 3000 * L_meters**3) / (1e21 * 1e-6)

    # Verify: Should return plain array (not UWQuantity with wrong units!)
    assert isinstance(result, np.ndarray), f"Expected ndarray, got {type(result)}"
    assert not hasattr(result, 'units'), "Result should not have units attribute"

    # Verify numeric value is correct
    assert abs(result[0, 0, 0] - Ra_expected) / Ra_expected < 1e-6, \
        f"Expected {Ra_expected:.6e}, got {result[0, 0, 0]:.6e}"


def test_mixed_units_various_scales():
    """Test that various non-SI units are correctly converted."""
    # Distance in different units
    L_km = uw.expression("L_km", sym=uw.quantity(1, "km"))
    L_cm = uw.expression("L_cm", sym=uw.quantity(100, "cm"))
    L_mm = uw.expression("L_mm", sym=uw.quantity(1000, "mm"))

    coords = np.array([[0.0, 0.0]])

    # All should evaluate to base SI units (meters)
    result_km = uw.function.evaluate(L_km, coords)[0, 0, 0]
    result_cm = uw.function.evaluate(L_cm, coords)[0, 0, 0]
    result_mm = uw.function.evaluate(L_mm, coords)[0, 0, 0]

    assert abs(result_km - 1000) < 1e-6, f"km: expected 1000 m, got {result_km}"
    assert abs(result_cm - 1.0) < 1e-6, f"cm: expected 1.0 m, got {result_cm}"
    assert abs(result_mm - 1.0) < 1e-6, f"mm: expected 1.0 m, got {result_mm}"


if __name__ == "__main__":
    test_composite_direct_quantities()
    print("✅ test_composite_direct_quantities passed")

    test_composite_wrapped_expressions()
    print("✅ test_composite_wrapped_expressions passed")

    test_direct_composite_evaluation()
    print("✅ test_direct_composite_evaluation passed")

    test_mixed_units_various_scales()
    print("✅ test_mixed_units_various_scales passed")

    print("\n🎉 All composite expression unit tests passed!")
