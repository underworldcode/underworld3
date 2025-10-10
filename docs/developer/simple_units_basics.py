# %% [markdown]
# # Simple Units Basics
#
# Very simple examples showing one concept at a time:
# 1. How to add units to a MeshVariable
# 2. How to access units
# 3. How to change scaling systems
# 4. Basic operations

# %%
# Basic setup
import underworld3 as uw
from underworld3.discretisation.enhanced_variables import EnhancedMeshVariable

# Create simple mesh
mesh = uw.meshing.StructuredQuadBox(elementRes=(2, 2), minCoords=(0, 0), maxCoords=(1, 1))
print(f"✅ Created simple 2x2 mesh")

# %% [markdown]
# ## 1. How to Add Units to a MeshVariable

# %%
# Regular MeshVariable (no units)
velocity_regular = uw.discretisation.MeshVariable("velocity", mesh, 2)
print(f"Regular variable: {velocity_regular.name}")
print(f"Has units: {hasattr(velocity_regular, 'has_units')}")

# Enhanced MeshVariable with units
velocity_with_units = EnhancedMeshVariable("velocity_units", mesh, 2, units="m/s")
print(f"\nEnhanced variable: {velocity_with_units.name}")
print(f"Has units: {velocity_with_units.has_units}")

# %% [markdown]
# ## 2. How to Access Units

# %%
# Access units information
print(f"Units: {velocity_with_units.units}")
print(f"Dimensionality: {velocity_with_units.dimensionality}")
print(f"Backend type: {type(velocity_with_units._units_backend).__name__}")

# Create variables with different units
pressure = EnhancedMeshVariable("pressure", mesh, 1, units="Pa")
temperature = EnhancedMeshVariable("temperature", mesh, 1, units="K")

print(f"\nPressure units: {pressure.units}")
print(f"Temperature units: {temperature.units}")

# %% [markdown]
# ## 3. Different Units Backends

# %%
# Same variable with different backends
velocity_pint = EnhancedMeshVariable("vel_pint", mesh, 2, units="m/s", units_backend="pint")
velocity_sympy = EnhancedMeshVariable("vel_sympy", mesh, 2, units="m/s", units_backend="sympy")

print(f"Pint backend units: {velocity_pint.units}")
print(f"SymPy backend units: {velocity_sympy.units}")
print(f"Pint dimensionality: {velocity_pint.dimensionality}")
print(f"SymPy dimensionality: {velocity_sympy.dimensionality}")

# %% [markdown]
# ## 4. Units Compatibility

# %%
# Check if variables have compatible units
velocity1 = EnhancedMeshVariable("vel1", mesh, 2, units="m/s")
velocity2 = EnhancedMeshVariable("vel2", mesh, 2, units="m/s")
pressure_var = EnhancedMeshVariable("press", mesh, 1, units="Pa")

print(f"velocity1 vs velocity2 (both m/s): {velocity1.check_units_compatibility(velocity2)}")
print(f"velocity1 vs pressure (m/s vs Pa): {velocity1.check_units_compatibility(pressure_var)}")

# %% [markdown]
# ## 5. Creating Quantities with Units

# %%
# Create dimensional quantities from values
velocity_var = EnhancedMeshVariable("vel", mesh, 2, units="m/s")

# Create a quantity using the variable's units
speed_value = 5.0  # Just a number
speed_with_units = velocity_var.create_quantity(speed_value)
print(f"Value: {speed_value}")
print(f"With units: {speed_with_units}")

# %%
from sympy import symbols
from sympy.physics.units.systems import SI
from sympy.physics.units import length, mass, acceleration, force
from sympy.physics.units import gravitational_constant as G
from sympy.physics.units.systems.si import dimsys_SI
F = mass*acceleration
F

# %%
# Note: These sympy examples require proper imports and variable definitions
# import sympy.physics.units
# sympy.physics.units.decimeter
# sympy.physics.units.convert_to(vel.sym, sympy.physics.units.kilometer / sympy.physics.units.hour)

# %% [markdown]
# ## 6. Different Unit Strings

# %%
# Test various unit string formats
unit_examples = [
    ("velocity", "m/s"),
    ("pressure", "Pa"),
    ("viscosity", "Pa*s"),
    ("temperature", "K"),
    ("length", "m")
]

print("Unit string examples:")
for name, unit_str in unit_examples:
    try:
        var = EnhancedMeshVariable(f"test_{name}", mesh, 1, units=unit_str)
        print(f"  {name}: '{unit_str}' → {var.units}")
    except Exception as e:
        print(f"  {name}: '{unit_str}' → Error: {e}")

# %% [markdown]
# ## 7. Array Access (Performance Critical)

# %%
# Show that array access works normally
velocity_test = EnhancedMeshVariable("vel_test", mesh, 2, units="m/s")

print(f"Array shape: {velocity_test.array.shape}")
print(f"Array access works: {type(velocity_test.array)}")
print(f"First element: {velocity_test.array[0]}")

# Arrays work exactly like regular MeshVariables
velocity_test.array[0, 0, 0] = 1.5
velocity_test.array[0, 0, 1] = 2.0
print(f"After assignment: {velocity_test.array[0]}")

# %%
# Enhanced view() method shows units information
print("=== Enhanced Variable View ===")
velocity_test.view()

# %%
# Important edge case: Mathematical operations return pure SymPy objects
print("=== Mathematical Operations Edge Case ===")
scaled_expr = 2 * velocity_pint
print(f"2 * velocity_pint:")
print(f"  Type: {type(scaled_expr)}")
print(f"  Value: {scaled_expr}")
print(f"  Has units: {hasattr(scaled_expr, 'units')}")

# To get units information, access the original variable
print(f"\\nOriginal variable units: {velocity_pint.units}")
print(f"Expression represents: 2 * (quantity with {velocity_pint.units})")

# This is correct behavior for JIT compilation - expressions are pure SymPy

# %% [markdown]
# ## 7a. Scale Factors for Compilation (Latest Feature)

# %%
# Demonstration of new scale factor functionality
print("=== Scale Factor Demonstration ===")

# Create variables with different units to show scale factor calculation
velocity_ms = EnhancedMeshVariable("vel_ms", mesh, 2, units="m/s", units_backend="sympy")
velocity_cmy = EnhancedMeshVariable("vel_cmy", mesh, 2, units="cm/year", units_backend="sympy")
pressure_pa = EnhancedMeshVariable("pressure_pa", mesh, 1, units="Pa", units_backend="sympy")
pressure_gpa = EnhancedMeshVariable("pressure_gpa", mesh, 1, units="GPa", units_backend="sympy")

print("Scale factors for different units:")
print(f"  m/s: {velocity_ms.scale_factor}")
print(f"  cm/year: {velocity_cmy.scale_factor}")
print(f"  Pa: {pressure_pa.scale_factor}")
print(f"  GPa: {pressure_gpa.scale_factor}")

# Demonstrate reference scaling for geological problems
print(f"\nBefore reference scaling: {velocity_cmy.scale_factor}")
velocity_cmy.set_reference_scaling(5.0)  # Typical plate velocity: 5 cm/year
print(f"After reference scaling (5 cm/year → O(1)): {velocity_cmy.scale_factor}")

print("\n💡 Key Point: Scale factors are SymPy-friendly for symbolic cancellation")
print("   This enables automatic scaling during unwrap/compilation")

# %% [markdown]
# ## 7b. Units in Mathematical Expressions (Design Notes)
#
# **Important**: Mathematical operations return pure SymPy objects without units.
# This is intentional for JIT compilation compatibility.

# %%
# Demonstrating the design rationale
print("=== Why Mathematical Expressions Don't Have Units ===")
print("1. JIT Compatibility:")
print("   • SymPy expressions can be compiled to fast numerical code")
print("   • Units would interfere with code generation")
print("   • Solver expects dimensionless expressions")

print("\\n2. Getting Units Information:")
velocity_expr = velocity_pint[0]  # Component access
momentum_expr = velocity_pint * 1000  # Some operation
print(f"   • Original variable: {velocity_pint.units}")
print(f"   • Expression is symbolic: {momentum_expr}")
print(f"   • Units meaning: 1000 * {velocity_pint.units}")

print("\\n3. Component Access Edge Case:")
scaled_expr = 2 * velocity_pint
try:
    component = scaled_expr[0]
    print(f"   • (2 * velocity)[0]: {type(component)}")
    component.units  # This will fail
except AttributeError as e:
    print(f"   • Error: {e}")
    print(f"   • SymPy Mul objects don't have units")

print("\\n4. For Numerical Calculations with Units:")
# Use create_quantity for values with units
speed_with_units = velocity_pint.create_quantity(5.0)
print(f"   • Numerical value with units: {speed_with_units}")
print(f"   • This preserves units: {speed_with_units.units}")

print("\\n5. Summary:")
print("   • Variables have units: ✅")
print("   • Expressions are pure SymPy: ✅ (by design)")
print("   • Components of expressions: ✅ (pure SymPy)")
print("   • This enables JIT compilation: ✅")

# %% [markdown]
# ## 8. Mathematical Operations

# %%
# Test basic mathematical operations
vel = EnhancedMeshVariable("vel_math", mesh, 2, units="m/s", varsymbol=r"v_\textrm{math}")

# Component access
print(f"Component access: vel[0] = {vel[0]}")

# Mathematical operations (if available)
if hasattr(vel, 'dot'):
    print(f"Dot product available: {type(vel.dot)}")

# Scalar multiplication
try:
    scaled = 2.0 * vel
    print(f"Scalar multiplication: 2.0 * vel = {type(scaled)}")
except Exception as e:
    print(f"Scalar multiplication: {e}")

# %%
vel

# %%

# %%

# %% [markdown]
# ## 9. Model Reference Quantities (Simple)

# %%
# Simple model setup
model = uw.Model("simple_test")

# Check if new method is available (needs rebuild)
if hasattr(model, 'set_reference_quantities'):
    print("✅ Model reference quantities available")
    try:
        model.set_reference_quantities(
            test_velocity=1.0 * uw.scaling.units.m / uw.scaling.units.s,
            test_pressure=1000.0 * uw.scaling.units.Pa
        )
        print(f"Model has units: {model.has_units()}")
        print(f"Reference quantities: {list(model.get_reference_quantities().keys())}")
    except Exception as e:
        print(f"Error: {e}")
else:
    print("⚠️  Model.set_reference_quantities() not available")
    print("📝 Run 'pixi run underworld-build' to enable")

# %% [markdown]
# ## 10. Summary

# %%
print("📋 SIMPLE UNITS BASICS SUMMARY")
print("=" * 30)
print("✅ Basic concepts demonstrated:")
print("  • Adding units to MeshVariable")
print("  • Accessing units information")
print("  • Different backends (Pint vs SymPy)")
print("  • Units compatibility checking")
print("  • Array access preservation")
print("  • Mathematical operations")
print("  • Scale factors for compilation")
print("  • Reference scaling for typical values")
print("\n🎯 Key Points:")
print("  • EnhancedMeshVariable adds units to regular MeshVariable")
print("  • Array access performance unchanged")
print("  • Units are optional - regular variables work as before")
print("  • Both Pint and SymPy backends supported")
print("  • Scale factors enable automatic unwrap scaling")
print("  • Powers-of-ten scaling for clean numerical values")

# %%
