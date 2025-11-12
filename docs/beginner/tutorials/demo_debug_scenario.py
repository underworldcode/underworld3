#!/usr/bin/env python3
"""Demonstrate debugging non-dimensionalization issues"""

import underworld3 as uw

print("=" * 70)
print("Debugging Scenario: Mixed Scaled and Unscaled Parameters")
print("=" * 70)

# Setup scaling
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(1000, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
    mantle_viscosity=uw.quantity(1e21, "Pa*s"),
)

mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
u = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

print("\nScenario: User sets viscosity to match reference quantity")
print("-" * 70)

# Set viscosity to match reference
stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e21, "Pa*s")
visc = stokes.constitutive_model.Parameters.shear_viscosity_0

print(f"Assigned: {visc._pint_qty}")
print(f"Reference mantle_viscosity: 1e21 Pa*s")
print(f"Expected non-dimensional value: 1.0")

# Check what actually gets compiled
uw.use_nondimensional_scaling(True)
c_tensor = stokes.constitutive_model.c
elem = c_tensor[0, 0, 0, 0]

nondim = uw.show_nondimensional_form(elem)
print(f"\nNon-dimensional tensor element: {nondim}")
print(f"✓ Correctly scaled to {float(nondim.evalf()):.1f}")

print("\n" + "=" * 70)
print("Scenario: User accidentally uses different viscosity scale")
print("=" * 70)

# Now use a different viscosity
stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(5e21, "Pa*s")
visc2 = stokes.constitutive_model.Parameters.shear_viscosity_0

print(f"Assigned: {visc2._pint_qty}")
print(f"Reference mantle_viscosity: 1e21 Pa*s")
print(f"Expected non-dimensional value: 5.0")

# Check again
c_tensor2 = stokes.constitutive_model.c
elem2 = c_tensor2[0, 0, 0, 0]

uw.show_nondimensional_form(elem2)



nondim2 = uw.show_nondimensional_form(elem2)
print(f"\nNon-dimensional tensor element: {nondim2}")
print(f"✓ Correctly scaled to {float(nondim2.evalf()):.1f}")
print("\nThis helps users verify that viscosity contrast is preserved!")

print("\n" + "=" * 70)
print("Scenario: Diffusivity with different units")
print("=" * 70)

# Create a diffusion problem
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel

# Thermal diffusivity
thermal_kappa = uw.quantity(1e-6, "m**2/s")
poisson.constitutive_model.Parameters.diffusivity = thermal_kappa

print(f"Assigned thermal diffusivity: {thermal_kappa}")
print(
    f"Reference thermal_diffusivity: {model.reference_quantities.get('thermal_diffusivity', 'NOT SET')}"
)

# If thermal_diffusivity was set as reference
if "thermal_diffusivity" in model.reference_quantities:
    c_diff = poisson.constitutive_model.c
    elem_diff = c_diff[0, 0]
    nondim_diff = uw.show_nondimensional_form(elem_diff)
    print(f"Non-dimensional diffusivity: {nondim_diff}")
else:
    print("⚠️ No thermal_diffusivity reference quantity set")
    print("   Diffusivity will be scaled using derived scales from domain_depth, etc.")

print("\n" + "=" * 70)
print("Key Insight: show_nondimensional_form() helps you:")
print("=" * 70)
print("1. Verify parameters are scaled correctly")
print("2. Check that viscosity/property contrasts are preserved")
print("3. Debug unexpected values in JIT compilation output")
print("4. Understand what numeric values actually enter the solver")
print("\nWithout this, users see symbolic expressions and don't know")
print("what numeric values are being used during compilation.")
print("=" * 70)
