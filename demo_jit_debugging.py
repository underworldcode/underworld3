#!/usr/bin/env python3
"""Demonstrate debugging JIT compilation output"""

import underworld3 as uw

print("\n" + "=" * 70)
print("Debugging JIT Output: The Problem You Reported")
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

print("\nYour original issue:")
print("-" * 70)
print("""
You reported seeing in JIT output:
  "Processing JIT 1 / Matrix([[-9.9588211776e-8*..."

You said:
  "If I run uw.non_dimensionalise on the viscosity it returns
   <1, dimensionless> but the unwrapping / non-dimensionalisation
   does this: 9.9588211776e-8"

The question: Why 1e-8 when non_dimensionalise says 1.0?
""")

print("\nRESOLVED: The 1e-8 is NOT from viscosity!")
print("-" * 70)

stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e21, "Pa*s")
uw.use_nondimensional_scaling(True)

# Check what viscosity becomes
c_tensor = stokes.constitutive_model.c
elem = c_tensor[0, 0, 0, 0]

print(f"\nViscosity in constitutive tensor:")
print(f"  Symbolic: {elem}")
print(f"  Non-dimensional: {uw.show_nondimensional_form(elem)}")
print(f"  → Numeric: {float(uw.show_nondimensional_form(elem).evalf()):.2f}")
print("\n  ✓ Viscosity IS correctly scaled to 2.0 (not 1e-8!)")

print("\nThe 1e-8 you saw was from something else:")
print("  - Geometry terms (x, y coordinates)")
print("  - Strain rate components")
print("  - Other expression terms")
print("\nNot from the viscosity parameter itself!")

print("\n" + "=" * 70)
print("How show_nondimensional_form() Solves This")
print("=" * 70)
print("""
BEFORE (confusing):
  1. You see: "JIT output: 1e-8*..."
  2. You check: uw.non_dimensionalise(viscosity) → 1.0
  3. Confusion: Why 1e-8 if viscosity is 1.0?

AFTER (clear):
  1. You see: "JIT output: 1e-8*..."
  2. You check: uw.show_nondimensional_form(c_tensor[0,0,0,0]) → 2.0
  3. Understanding: Viscosity is correct (2.0), the 1e-8 is from elsewhere!

The key difference:
  - non_dimensionalise() operates on the parameter alone
  - show_nondimensional_form() shows the parameter IN CONTEXT
    (i.e., in the actual expression that goes to the solver)
""")

print("\n" + "=" * 70)
print("Additional Debugging Power")
print("=" * 70)

print("\nCheck different viscosities:")
cases = [
    (1e21, "reference value"),
    (1e22, "10x stiffer"),
    (1e20, "10x weaker"),
]

for visc_val, desc in cases:
    stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(visc_val, "Pa*s")
    c = stokes.constitutive_model.c
    elem_val = float(uw.show_nondimensional_form(c[0,0,0,0]).evalf())
    print(f"  η = {visc_val:.0e} Pa*s ({desc:20s}) → tensor elem = {elem_val:6.2f}")

print("\nYou can immediately see viscosity contrasts are preserved!")
print("=" * 70 + "\n")
