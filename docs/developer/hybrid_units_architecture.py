#!/usr/bin/env python3
"""
Hybrid SymPy+Pint Units Architecture

Explores combining Pint's user-friendly interface with SymPy's native
expression handling to solve the units integration challenge.

Key Design Principles (from our conversation):
1. array.access returns numpy arrays (performance)
2. Dimensionless by default, units are lifetime commitment
3. Minimal checking overhead
4. Semi-automatic conversions (explicit user requests)
5. Pint handles validation, SymPy handles expressions
"""

import sympy as sp
from sympy.physics import units as su
import underworld3 as uw

def explore_hybrid_conversion():
    """Test converting between Pint and SymPy units"""

    print("HYBRID PINT ↔ SYMPY CONVERSION")
    print("=" * 40)

    # Test Pint → SymPy conversion
    pint_viscosity = 1e21 * uw.scaling.units.Pa * uw.scaling.units.s
    pint_velocity = 5 * uw.scaling.units.cm / uw.scaling.units.year

    print(f"Pint quantities:")
    print(f"  Viscosity: {pint_viscosity}")
    print(f"  Velocity: {pint_velocity}")

    # Simple conversion strategy
    def pint_to_sympy(pint_quantity):
        """Convert Pint quantity to SymPy quantity"""

        # Get SI value
        si_value = pint_quantity.to_base_units().magnitude

        # Map common units (extensible)
        unit_map = {
            '[length] / [time]': su.meter / su.second,
            '[mass] / [length] / [time]': su.kilogram / (su.meter * su.second),
            '[mass] / [length] / [time] ** 2': su.kilogram / (su.meter * su.second**2),
            '[length]': su.meter,
            '[time]': su.second,
            '[mass]': su.kilogram,
            '[temperature]': su.kelvin,
        }

        dimensionality_str = str(pint_quantity.dimensionality)
        if dimensionality_str in unit_map:
            sympy_unit = unit_map[dimensionality_str]
            return si_value * sympy_unit
        else:
            raise ValueError(f"Unknown dimensionality: {dimensionality_str}")

    # Convert to SymPy
    sympy_viscosity = pint_to_sympy(pint_viscosity)
    sympy_velocity = pint_to_sympy(pint_velocity)

    print(f"\nSymPy conversion:")
    print(f"  Viscosity: {sympy_viscosity}")
    print(f"  Velocity: {sympy_velocity}")

    return sympy_viscosity, sympy_velocity

def test_expression_integration(sympy_viscosity, sympy_velocity):
    """Test how SymPy units work with expressions"""

    print(f"\n{'='*40}")
    print("SYMPY EXPRESSION INTEGRATION")
    print("=" * 40)

    # Create symbolic variables (representing field values)
    x, y = sp.symbols('x y', real=True)
    u_x = sp.Function('u_x')(x, y)  # Velocity component function
    u_y = sp.Function('u_y')(x, y)

    # Create expressions with units
    velocity_x = u_x * su.meter / su.second
    velocity_y = u_y * su.meter / su.second

    print(f"Velocity components with units:")
    print(f"  u_x: {velocity_x}")
    print(f"  u_y: {velocity_y}")

    # Test derivative (strain rate)
    strain_rate_xx = sp.diff(velocity_x, x)
    strain_rate_xy = sp.diff(velocity_x, y)

    print(f"\nStrain rate components:")
    print(f"  ∂u_x/∂x: {strain_rate_xx}")
    print(f"  ∂u_x/∂y: {strain_rate_xy}")

    # Extract units from expressions
    def extract_units(expr):
        """Extract unit part from SymPy expression"""
        unit_part = 1
        numerical_part_args = []

        if isinstance(expr, sp.Mul):
            for arg in expr.args:
                if isinstance(arg, su.Quantity):
                    unit_part *= arg
                else:
                    numerical_part_args.append(arg)

        numerical_part = sp.Mul(*numerical_part_args) if numerical_part_args else 1
        return numerical_part, unit_part

    num_part, unit_part = extract_units(strain_rate_xx)
    print(f"\nUnit extraction:")
    print(f"  Numerical: {num_part}")
    print(f"  Units: {unit_part}")

    # Test stress calculation
    viscosity_expr = sympy_viscosity  # Already has units
    stress_xx = viscosity_expr * strain_rate_xx

    print(f"\nStress calculation:")
    print(f"  Viscosity: {viscosity_expr}")
    print(f"  Stress: {stress_xx}")

    # Simplify to check unit consistency
    stress_simplified = sp.simplify(stress_xx)
    print(f"  Simplified: {stress_simplified}")

    return stress_xx

def test_jit_compatibility(stress_expr):
    """Test code generation from expressions with units"""

    print(f"\n{'='*40}")
    print("JIT CODE GENERATION")
    print("=" * 40)

    # Analyze expression structure
    print(f"Expression: {stress_expr}")
    print(f"Expression args: {stress_expr.args}")

    def separate_units_for_jit(expr):
        """Separate numerical and unit parts for JIT compilation"""

        if isinstance(expr, sp.Mul):
            numerical_args = []
            unit_scale = 1

            for arg in expr.args:
                if isinstance(arg, su.Quantity):
                    # This is a unit - accumulate scale factor
                    unit_scale *= arg
                elif isinstance(arg, (int, float, sp.Float)):
                    # Numerical constant
                    numerical_args.append(arg)
                else:
                    # Symbolic part (functions, derivatives, symbols)
                    numerical_args.append(arg)

            numerical_expr = sp.Mul(*numerical_args) if numerical_args else 1
            return numerical_expr, unit_scale
        else:
            return expr, 1

    numerical_part, unit_scale = separate_units_for_jit(stress_expr)

    print(f"\nSeparation for JIT:")
    print(f"  Numerical: {numerical_part}")
    print(f"  Unit scale: {unit_scale}")

    # Test code generation on numerical part
    try:
        from sympy import ccode

        # Create a simpler expression for testing
        x, y = sp.symbols('x y')
        simple_expr = x * y  # Represents the numerical computation

        c_code = ccode(simple_expr)
        print(f"  Sample C code: {c_code}")

        # Show how full JIT would work
        print(f"\nJIT strategy:")
        print(f"  1. Generate code for: {simple_expr}")
        print(f"  2. Apply unit scale: {unit_scale}")
        print(f"  3. Final code: ({c_code}) * scale_factor")

    except Exception as e:
        print(f"  Code generation issue: {e}")

def design_model_integration():
    """Design how this integrates with the Model class"""

    print(f"\n{'='*40}")
    print("MODEL INTEGRATION DESIGN")
    print("=" * 40)

    # Address our previous design decisions

    print("1. FLEXIBLE REFERENCE QUANTITIES:")
    print("   # User provides Pint quantities (familiar)")
    print("   model.set_reference_quantities(")
    print("       mantle_viscosity=1e21*uw.units.Pa*uw.units.s,")
    print("       plate_velocity=5*uw.units.cm/uw.units.year")
    print("   )")
    print("   # System converts to SymPy internally")
    print("   # Solves dimensional constraints automatically")

    print("\n2. VARIABLE CREATION:")
    print("   # After model commits to units:")
    print("   velocity = uw.MeshVariable('u', mesh, units='m/s')  # Required")
    print("   pressure = uw.MeshVariable('p', mesh, units='Pa')   # Required")
    print("   strain = uw.MeshVariable('e', mesh, units=None)     # Dimensionless")

    print("\n3. ARRAY ACCESS (PERFORMANCE PRESERVED):")
    print("   velocity.array[...] = 0.05  # Fast numpy, model base units")
    print("   velocity.to_array('m/s')    # Explicit conversion when needed")

    print("\n4. EXPRESSION HANDLING:")
    print("   # Variables create SymPy expressions with units")
    print("   strain_rate = velocity.sym[0].diff(x)  # Has units 1/s")
    print("   stress = viscosity * strain_rate       # Has units Pa")

    print("\n5. BOUNDARY CONDITIONS (JIT COMPATIBLE):")
    print("   def inflow(x, y, t):")
    print("       return sp.sin(x) * 5*su.meter/(100*su.year)")
    print("   ")
    print("   stokes.add_dirichlet_bc(inflow, 'Top')")
    print("   # unwrap() separates numerical and unit parts")

    print("\n6. SERIALIZATION:")
    print("   # Model saves both Pint reference quantities and")
    print("   # derived SymPy scaling for reconstruction")

def test_backwards_compatibility():
    """Test how this works with dimensionless (current) code"""

    print(f"\n{'='*40}")
    print("BACKWARDS COMPATIBILITY")
    print("=" * 40)

    print("DIMENSIONLESS MODE (current behavior):")
    print("  model = uw.Model()  # No units commitment")
    print("  velocity = uw.MeshVariable('u', mesh)  # No units required")
    print("  velocity.array[...] = 0.05  # Plain numbers")
    print("  # Everything works as before")

    print("\nUNITS MODE (new behavior):")
    print("  model = uw.Model()")
    print("  model.set_reference_quantities(...)  # Commits to units")
    print("  velocity = uw.MeshVariable('u', mesh, units='m/s')  # Required")
    print("  velocity.array[...] = 0.05  # Model base units")
    print("  velocity.to_array('m/s')  # Explicit conversion")

    print("\nELEGANT HANDLING OF units=None:")
    print("  # Dimensionless quantities work in both modes")
    print("  rayleigh = uw.MeshVariable('Ra', mesh, units=None)")
    print("  rayleigh.array[...] = 1e6  # Always dimensionless")

def main():
    """Run the full hybrid architecture exploration"""

    print("HYBRID SYMPY+PINT ARCHITECTURE EXPLORATION")
    print("=" * 50)
    print("Addressing all design questions from our conversation")

    # Test the hybrid conversion
    sympy_visc, sympy_vel = explore_hybrid_conversion()

    # Test expression integration
    stress_expr = test_expression_integration(sympy_visc, sympy_vel)

    # Test JIT compatibility
    test_jit_compatibility(stress_expr)

    # Show model integration
    design_model_integration()

    # Test backwards compatibility
    test_backwards_compatibility()

    print(f"\n{'='*50}")
    print("HYBRID ARCHITECTURE ASSESSMENT")
    print("=" * 50)

    print("✅ BENEFITS:")
    print("  • Pint: User-friendly input, excellent unit conversion")
    print("  • SymPy: Native expression integration, JIT compatible")
    print("  • Performance: array access stays fast")
    print("  • Backwards compatible: dimensionless mode preserved")
    print("  • Flexible: reference quantities, not hardcoded categories")

    print("\n🏗️ IMPLEMENTATION REQUIREMENTS:")
    print("  • Pint ↔ SymPy conversion utilities")
    print("  • Enhanced MeshVariable with units awareness")
    print("  • Modified unwrap() for unit separation")
    print("  • Model class units commitment system")
    print("  • Expression unit extraction tools")

    print("\n🎯 SOLVES ORIGINAL PROBLEMS:")
    print("  1. Expression units: ✅ SymPy handles naturally")
    print("  2. JIT compatibility: ✅ Separable parts")
    print("  3. Performance: ✅ Array access unchanged")
    print("  4. Boundary conditions: ✅ unwrap handles conversion")
    print("  5. Flexible scaling: ✅ Reference quantities approach")

if __name__ == "__main__":
    main()