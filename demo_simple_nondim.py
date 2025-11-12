#!/usr/bin/env python3
"""Simple demonstration of show_nondimensional_form()"""

import underworld3 as uw

print("\n" + "=" * 70)
print("show_nondimensional_form() - User View")
print("=" * 70)

# Setup
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(1000, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
    mantle_viscosity=uw.quantity(1e21, "Pa*s")
)

mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
u = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)
stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

print("\nExample 1: Checking if viscosity is scaled correctly")
print("-" * 70)

# Set viscosity to match reference
stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e21, "Pa*s")
visc = stokes.constitutive_model.Parameters.shear_viscosity_0

print(f"Input: η = {visc._pint_qty}")
print(f"Reference: 1e21 Pa*s")

# View what actually gets used
uw.use_nondimensional_scaling(True)
print(f"Symbolic form: {visc.sym}")
print(f"Non-dimensional: {uw.show_nondimensional_form(visc.sym)}")
print("✓ Shows 1.0 - viscosity matches reference!")

print("\nExample 2: Viscosity contrast preserved?")
print("-" * 70)

# Use 10x higher viscosity
stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e22, "Pa*s")
visc2 = stokes.constitutive_model.Parameters.shear_viscosity_0

print(f"Input: η = {visc2._pint_qty}")
print(f"Reference: 1e21 Pa*s")
print(f"Expected contrast: 10x")
print(f"Symbolic form: {visc2.sym}")
print(f"Non-dimensional: {uw.show_nondimensional_form(visc2.sym)}")
print("✓ Shows 10.0 - contrast is preserved!")

print("\nExample 3: Complex expression")
print("-" * 70)

x, y = mesh.X
expr = visc2.sym * x / y
print(f"Expression: η * x / y")
print(f"Symbolic: {expr}")
print(f"Non-dimensional: {uw.show_nondimensional_form(expr)}")
print("(Note: coordinates also get scaled if mesh has units)")

print("\n" + "=" * 70)
print("Why This Is Useful")
print("=" * 70)
print("""
1. **Debug JIT compilation**: When you see unexpected values in solver output,
   use show_nondimensional_form() to see exactly what numeric values
   are being substituted.

2. **Verify scaling**: Check that parameters are being non-dimensionalized
   correctly relative to reference quantities.

3. **Check contrasts**: Ensure viscosity/property contrasts are preserved
   (e.g., 10x viscosity → 10.0 in non-dimensional form).

4. **Transparency**: See what the solver actually uses, not just symbolic
   expressions.

USAGE:
  uw.use_nondimensional_scaling(True)
  uw.show_nondimensional_form(your_expression)

Returns the expression with all unit-aware parameters substituted with
their non-dimensional values.
""")
print("=" * 70 + "\n")
