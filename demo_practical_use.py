#!/usr/bin/env python3
"""Practical use case: Inspecting what goes into the solver"""

import underworld3 as uw

print("\n" + "=" * 70)
print("Practical Use: Inspect Solver Expressions")
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

print("\nSituation: You want to verify the constitutive tensor is correct")
print("-" * 70)

# Case 1: Viscosity at reference value
print("\nCase 1: η = 1e21 Pa*s (matches reference)")
stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e21, "Pa*s")

# Enable non-dimensional scaling
uw.use_nondimensional_scaling(True)

# Get the constitutive tensor (what actually goes into the solver)
c_tensor = stokes.constitutive_model.c
elem = c_tensor[0, 0, 0, 0]

print(f"  Tensor element (symbolic): {elem}")
nondim = uw.show_nondimensional_form(elem)
print(f"  Non-dimensional form: {nondim}")
print(f"  → Numeric: {float(nondim.evalf()):.2f}")
print("  ✓ Expected: 2.0 (because c_ijkl = 2*I_ijkl*η, and η → 1.0)")

# Case 2: Different viscosity
print("\nCase 2: η = 5e21 Pa*s (5x reference)")
stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(5e21, "Pa*s")

c_tensor2 = stokes.constitutive_model.c
elem2 = c_tensor2[0, 0, 0, 0]

print(f"  Tensor element (symbolic): {elem2}")
nondim2 = uw.show_nondimensional_form(elem2)
print(f"  Non-dimensional form: {nondim2}")
print(f"  → Numeric: {float(nondim2.evalf()):.2f}")
print("  ✓ Expected: 10.0 (because η → 5.0, and 2*5.0 = 10.0)")

# Case 3: Much lower viscosity (weak layer)
print("\nCase 3: η = 1e20 Pa*s (0.1x reference - weak layer)")
stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e20, "Pa*s")

c_tensor3 = stokes.constitutive_model.c
elem3 = c_tensor3[0, 0, 0, 0]

print(f"  Tensor element (symbolic): {elem3}")
nondim3 = uw.show_nondimensional_form(elem3)
print(f"  Non-dimensional form: {nondim3}")
print(f"  → Numeric: {float(nondim3.evalf()):.2f}")
print("  ✓ Expected: 0.2 (because η → 0.1, and 2*0.1 = 0.2)")

print("\n" + "=" * 70)
print("Real-World Usage Scenario")
print("=" * 70)
print("""
You run a simulation and see in the JIT output:
  "Processing JIT 1 / Matrix([[-9.9588e-8*x*y, ...]])"

You think: "Wait, why is the viscosity so small? I set η = 1e21!"

Debug process:
  1. Enable scaling: uw.use_nondimensional_scaling(True)
  2. Get tensor: c = stokes.constitutive_model.c
  3. Inspect: uw.show_nondimensional_form(c[0,0,0,0])
  4. See: "2.0" (or "10.0", or "0.2")

Now you understand:
  - 2.0 → Your viscosity matches the reference scale
  - 10.0 → Your viscosity is 10x the reference
  - 0.2 → Your viscosity is 0.1x the reference

The small numbers in JIT output are from other terms (geometry, etc.),
not from the viscosity!
""")
print("=" * 70)

print("\nKEY INSIGHT:")
print("  show_nondimensional_form() reveals what numeric values")
print("  are ACTUALLY used during compilation after scaling.")
print("  Parameters with units → dimensionless numbers")
print("=" * 70 + "\n")
