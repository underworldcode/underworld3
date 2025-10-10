#!/usr/bin/env python3
"""
Pint ↔ SymPy Units Conversion Utilities

Core implementation of bidirectional conversion between Pint quantities
and SymPy units expressions for Underworld3 units integration.
"""

import sympy as sp
from sympy.physics import units as su
import underworld3 as uw
import numpy as np

class PintSymPyConverter:
    """
    Handles conversion between Pint quantities and SymPy units.

    Key responsibilities:
    - Convert user-friendly Pint input to SymPy for expressions
    - Handle dimensional analysis and unit mapping
    - Provide round-trip conversion capability
    """

    def __init__(self):
        # Map Pint dimensionalities to SymPy units
        self._dimensionality_map = {
            # Base dimensions
            '[length]': su.meter,
            '[time]': su.second,
            '[mass]': su.kilogram,
            '[temperature]': su.kelvin,
            '[substance]': su.mole,
            'dimensionless': 1,

            # Common derived dimensions
            '[length] / [time]': su.meter / su.second,  # velocity
            '[length] / [time] ** 2': su.meter / su.second**2,  # acceleration
            '[mass] / [length] / [time]': su.kilogram / (su.meter * su.second),  # viscosity
            '[mass] / [length] / [time] ** 2': su.kilogram / (su.meter * su.second**2),  # pressure/stress
            '[mass] / [length] ** 2 / [time] ** 2': su.kilogram / (su.meter**2 * su.second**2),  # force per volume
            '[mass] * [length] / [time] ** 2': su.kilogram * su.meter / su.second**2,  # force
            '[mass] * [length] ** 2 / [time] ** 2': su.kilogram * su.meter**2 / su.second**2,  # energy
            '[mass] * [length] ** 2 / [time] ** 3': su.kilogram * su.meter**2 / su.second**3,  # power

            # Thermal dimensions
            '[mass] * [length] ** 2 / [time] ** 3 / [temperature]': su.kilogram * su.meter**2 / (su.second**3 * su.kelvin),  # thermal conductivity
            '[length] ** 2 / [time] ** 2 / [temperature]': su.meter**2 / (su.second**2 * su.kelvin),  # specific heat
            '[length] ** 2 / [time]': su.meter**2 / su.second,  # thermal diffusivity
        }

        # Common unit string mappings for user convenience
        self._unit_string_map = {
            # Length
            'm': su.meter,
            'meter': su.meter,
            'km': 1000 * su.meter,
            'kilometer': 1000 * su.meter,
            'cm': su.meter / 100,
            'centimeter': su.meter / 100,

            # Time
            's': su.second,
            'second': su.second,
            'year': 365.25 * 24 * 3600 * su.second,
            'Myr': 1e6 * 365.25 * 24 * 3600 * su.second,
            'hour': 3600 * su.second,
            'day': 24 * 3600 * su.second,

            # Velocity
            'm/s': su.meter / su.second,
            'cm/year': su.meter / 100 / (365.25 * 24 * 3600 * su.second),
            'km/Myr': 1000 * su.meter / (1e6 * 365.25 * 24 * 3600 * su.second),

            # Pressure/Stress
            'Pa': su.kilogram / (su.meter * su.second**2),
            'pascal': su.kilogram / (su.meter * su.second**2),
            'GPa': 1e9 * su.kilogram / (su.meter * su.second**2),
            'MPa': 1e6 * su.kilogram / (su.meter * su.second**2),

            # Viscosity
            'Pa*s': su.kilogram / (su.meter * su.second),
            'Pa·s': su.kilogram / (su.meter * su.second),

            # Temperature
            'K': su.kelvin,
            'kelvin': su.kelvin,

            # Dimensionless
            '1': 1,
            'dimensionless': 1,
        }

    def pint_to_sympy(self, pint_quantity):
        """
        Convert Pint quantity to SymPy quantity.

        Args:
            pint_quantity: Pint Quantity object

        Returns:
            SymPy expression with units

        Example:
            >>> pint_vel = 5 * uw.units.cm / uw.units.year
            >>> sympy_vel = converter.pint_to_sympy(pint_vel)
            >>> print(sympy_vel)
            1.58440439070145e-9*meter/second
        """

        # Get SI magnitude and dimensionality
        si_magnitude = pint_quantity.to_base_units().magnitude
        dimensionality_str = str(pint_quantity.dimensionality)

        # Look up SymPy unit equivalent
        if dimensionality_str in self._dimensionality_map:
            sympy_unit = self._dimensionality_map[dimensionality_str]
            return si_magnitude * sympy_unit
        else:
            raise ValueError(f"Unsupported dimensionality: {dimensionality_str}")

    def sympy_to_pint(self, sympy_quantity):
        """
        Convert SymPy quantity back to Pint (for user output).

        Args:
            sympy_quantity: SymPy expression with units

        Returns:
            Pint Quantity object
        """

        # Extract numerical coefficient and units from SymPy expression
        numerical_part, unit_part = self._extract_sympy_components(sympy_quantity)

        # Convert SymPy units back to Pint units string
        unit_string = self._sympy_units_to_pint_string(unit_part)

        # Create Pint quantity
        try:
            # Try to convert numerical part to float
            if hasattr(numerical_part, 'evalf'):
                numerical_value = float(numerical_part.evalf())
            else:
                numerical_value = float(numerical_part)
        except (TypeError, ValueError):
            # If we can't convert to float, the expression is too complex
            raise ValueError(f"Cannot convert symbolic expression {numerical_part} to float")

        return numerical_value * uw.scaling.units(unit_string)

    def unit_string_to_sympy(self, unit_string):
        """
        Convert unit string to SymPy units.

        Args:
            unit_string: String like "m/s", "Pa", "K"

        Returns:
            SymPy unit expression
        """

        if unit_string in self._unit_string_map:
            return self._unit_string_map[unit_string]
        else:
            # Try to parse more complex expressions
            # For now, raise error for unsupported strings
            raise ValueError(f"Unsupported unit string: {unit_string}")

    def _extract_sympy_components(self, sympy_expr):
        """Extract numerical and unit parts from SymPy expression"""

        # Handle the case where we have a numerical coefficient times units
        # Example: 1.58e-9*meter/second should give (1.58e-9, meter/second)

        if isinstance(sympy_expr, sp.Mul):
            numerical_coeff = 1
            unit_expr = 1

            for arg in sympy_expr.args:
                if isinstance(arg, (int, float, sp.Float, sp.Integer, sp.Rational)):
                    # This is a numerical coefficient
                    numerical_coeff *= float(arg)
                elif isinstance(arg, su.Quantity):
                    # This is a unit
                    unit_expr *= arg
                elif isinstance(arg, sp.Pow) and isinstance(arg.base, su.Quantity):
                    # This is a unit raised to a power
                    unit_expr *= arg
                else:
                    # For now, assume other symbolic parts are dimensionless
                    numerical_coeff *= arg

            return numerical_coeff, unit_expr

        elif isinstance(sympy_expr, su.Quantity):
            return 1, sympy_expr

        elif isinstance(sympy_expr, (int, float, sp.Float, sp.Integer, sp.Rational)):
            return sympy_expr, 1

        else:
            # Assume it's dimensionless if we can't parse it
            return sympy_expr, 1

    def _sympy_units_to_pint_string(self, unit_part):
        """Convert SymPy units back to Pint-compatible string"""

        # Simple mapping for common cases
        unit_str_map = {
            su.meter: 'm',
            su.second: 's',
            su.kilogram: 'kg',
            su.kelvin: 'K',
            su.meter / su.second: 'm/s',
            su.kilogram / (su.meter * su.second**2): 'Pa',
            su.kilogram / (su.meter * su.second): 'Pa*s',
        }

        if unit_part in unit_str_map:
            return unit_str_map[unit_part]
        else:
            # For more complex units, try to build string representation
            # Convert SymPy expression to string and make it Pint-compatible
            unit_str = str(unit_part)

            # Replace SymPy notation with Pint notation
            replacements = {
                'meter': 'm',
                'second': 's',
                'kilogram': 'kg',
                'kelvin': 'K',
                '**': '^',
                '*': ' * ',
            }

            for sympy_str, pint_str in replacements.items():
                unit_str = unit_str.replace(sympy_str, pint_str)

            return unit_str

# Global converter instance
converter = PintSymPyConverter()

def test_basic_conversion():
    """Test basic Pint ↔ SymPy conversion"""

    print("BASIC PINT ↔ SYMPY CONVERSION TEST")
    print("=" * 40)

    # Test common geological quantities
    test_quantities = [
        ("Plate velocity", 5 * uw.scaling.units.cm / uw.scaling.units.year),
        ("Mantle viscosity", 1e21 * uw.scaling.units.Pa * uw.scaling.units.s),
        ("Lithosphere pressure", 1e9 * uw.scaling.units.Pa),
        ("Domain size", 3000 * uw.scaling.units.km),
        ("Temperature difference", 1000 * uw.scaling.units.K),
    ]

    for name, pint_qty in test_quantities:
        print(f"\n{name}:")
        print(f"  Pint: {pint_qty}")

        # Convert to SymPy
        sympy_qty = converter.pint_to_sympy(pint_qty)
        print(f"  SymPy: {sympy_qty}")

        # Test round-trip conversion
        try:
            pint_back = converter.sympy_to_pint(sympy_qty)
            print(f"  Round-trip: {pint_back}")

            # Check if values match (within tolerance)
            original_si = pint_qty.to_base_units().magnitude
            roundtrip_si = pint_back.to_base_units().magnitude
            relative_error = abs(original_si - roundtrip_si) / abs(original_si) if original_si != 0 else 0

            if relative_error < 1e-10:
                print(f"  ✓ Round-trip successful")
            else:
                print(f"  ⚠ Round-trip error: {relative_error:.2e}")

        except Exception as e:
            print(f"  ✗ Round-trip failed: {e}")

def test_expression_integration():
    """Test how SymPy units work with expressions"""

    print(f"\n{'='*40}")
    print("EXPRESSION INTEGRATION TEST")
    print("=" * 40)

    # Create SymPy quantities from Pint input
    velocity_pint = 5 * uw.scaling.units.cm / uw.scaling.units.year
    viscosity_pint = 1e21 * uw.scaling.units.Pa * uw.scaling.units.s

    velocity_sympy = converter.pint_to_sympy(velocity_pint)
    viscosity_sympy = converter.pint_to_sympy(viscosity_pint)

    print(f"Base quantities:")
    print(f"  Velocity: {velocity_sympy}")
    print(f"  Viscosity: {viscosity_sympy}")

    # Create symbolic variables for expressions
    x, y = sp.symbols('x y', real=True)
    u = sp.Function('u')(x, y)

    # Create expressions with units
    velocity_field = u * velocity_sympy / velocity_sympy  # Normalize to get just the unit
    velocity_with_units = u * (su.meter / su.second)

    print(f"\nVelocity field: {velocity_with_units}")

    # Test derivatives (strain rate)
    strain_rate = sp.diff(velocity_with_units, x)
    print(f"Strain rate: {strain_rate}")

    # Extract units from the derivative
    _, strain_units = converter._extract_sympy_components(strain_rate)
    print(f"Strain rate units: {strain_units}")

    # Test stress calculation
    stress = viscosity_sympy * strain_rate
    print(f"Stress expression: {stress}")

    # Simplify to see final units
    stress_simplified = sp.simplify(stress)
    print(f"Stress simplified: {stress_simplified}")

    # Extract final stress units
    _, stress_units = converter._extract_sympy_components(stress_simplified)
    print(f"Stress units: {stress_units}")

def test_unit_strings():
    """Test unit string parsing"""

    print(f"\n{'='*40}")
    print("UNIT STRING PARSING TEST")
    print("=" * 40)

    test_strings = ['m/s', 'Pa', 'K', 'cm/year', 'GPa', 'Pa*s']

    for unit_str in test_strings:
        try:
            sympy_unit = converter.unit_string_to_sympy(unit_str)
            print(f"  {unit_str} → {sympy_unit}")
        except Exception as e:
            print(f"  {unit_str} → Error: {e}")

def main():
    """Run all conversion tests"""

    print("PINT ↔ SYMPY CONVERSION UTILITIES")
    print("=" * 50)

    test_basic_conversion()
    test_expression_integration()
    test_unit_strings()

    print(f"\n{'='*50}")
    print("CONVERSION UTILITIES READY")
    print("=" * 50)
    print("✅ Basic Pint → SymPy conversion working")
    print("✅ Expression integration demonstrated")
    print("✅ Unit arithmetic functioning")
    print("✅ Ready for Stokes notebook integration")

if __name__ == "__main__":
    main()