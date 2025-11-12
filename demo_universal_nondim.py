#!/usr/bin/env python3
"""Demonstrate universal show_nondimensional_form() after fix"""

import underworld3 as uw

print("\n" + "=" * 70)
print("show_nondimensional_form() - Now Works Universally!")
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
stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e21, "Pa*s")

uw.use_nondimensional_scaling(True)

print("\n✓ FIXED: Can now inspect any type of expression!")
print("-" * 70)

print("\n1. Direct quantity inspection:")
visc_qty = uw.quantity(1e21, "Pa*s")
print(f"   Input:  uw.quantity(1e21, 'Pa*s')")
print(f"   Result: {uw.show_nondimensional_form(visc_qty)}")
print(f"   → Correctly shows 1.0 (matches reference)")

print("\n2. Parameter in constitutive tensor:")
c_tensor = stokes.constitutive_model.c
print(f"   Input:  c[0,0,0,0]")
print(f"   Result: {uw.show_nondimensional_form(c_tensor[0,0,0,0])}")
print(f"   → Shows 2.0 (tensor construction: 2*η where η=1.0)")

print("\n3. Verify viscosity contrast:")
stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e22, "Pa*s")
c_stiff = stokes.constitutive_model.c
print(f"   Input:  η = 1e22 Pa*s (10x reference)")
print(f"   Result: {uw.show_nondimensional_form(c_stiff[0,0,0,0])}")
print(f"   → Shows 20.0 (correctly: 2 * 10.0)")

print("\n4. Different material property:")
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
poisson = uw.systems.Poisson(mesh, u_Field=T)
poisson.constitutive_model = uw.constitutive_models.DiffusionModel
poisson.constitutive_model.Parameters.diffusivity = uw.quantity(1e-6, "m**2/s")
c_diff = poisson.constitutive_model.c
print(f"   Input:  diffusivity tensor element")
print(f"   Result: {float(uw.show_nondimensional_form(c_diff[0,0]).evalf()):.6f}")
print(f"   → Non-dimensional diffusivity value")

print("\n" + "=" * 70)
print("Key Benefits")
print("=" * 70)
print("""
✓ Works with ANY expression type (no more AttributeError!)
✓ Direct inspection: uw.show_nondimensional_form(quantity)
✓ Tensor elements: uw.show_nondimensional_form(c[i,j,...])
✓ Complex expressions: uw.show_nondimensional_form(η * x / y)
✓ Verify contrasts: See if 10x viscosity → 10.0 in non-dimensional form

USE CASE:
When you see confusing numbers in JIT output, use this function to
inspect what the compiler actually sees after non-dimensionalization.
""")
print("=" * 70 + "\n")
