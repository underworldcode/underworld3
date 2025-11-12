#!/usr/bin/env python3
"""Demonstrate the user-facing non-dimensional form viewer"""

import underworld3 as uw

print("=" * 70)
print("Non-Dimensional Form Viewer Demo")
print("=" * 70)

# Setup scaling system
uw.reset_default_model()
model = uw.get_default_model()
model.set_reference_quantities(
    domain_depth=uw.quantity(1000, "km"),
    plate_velocity=uw.quantity(5, "cm/year"),
    mantle_viscosity=uw.quantity(1e21, "Pa*s")
)

print("\n1. Reference Quantities Set")
print("-" * 70)
print(f"Domain depth: 1000 km")
print(f"Plate velocity: 5 cm/year")
print(f"Mantle viscosity: 1e21 Pa*s")

# Create mesh and Stokes system
mesh = uw.meshing.StructuredQuadBox(elementRes=(4, 4))
u = uw.discretisation.MeshVariable("U", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1)

stokes = uw.systems.Stokes(mesh, velocityField=u, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel

print("\n2. Assign Viscosity with Units")
print("-" * 70)
stokes.constitutive_model.Parameters.shear_viscosity_0 = uw.quantity(1e21, "Pa*s")
visc_param = stokes.constitutive_model.Parameters.shear_viscosity_0

print(f"Viscosity parameter: {visc_param}")
print(f"  Type: {type(visc_param).__name__}")
print(f"  Units: {visc_param._pint_qty if hasattr(visc_param, '_pint_qty') else 'None'}")

print("\n3. View Symbolic Form (What User Normally Sees)")
print("-" * 70)
print(f"Symbolic: {visc_param.sym}")

print("\n4. View Non-Dimensional Form (New Feature!)")
print("-" * 70)
uw.use_nondimensional_scaling(True)

# Get a tensor element containing the viscosity
c_tensor = stokes.constitutive_model.c
sample_elem = c_tensor[0, 0, 0, 0]

print(f"Tensor element (symbolic): {sample_elem}")
print(f"  Type: {type(sample_elem)}")

# Show the non-dimensional form
nondim_form = uw.show_nondimensional_form(sample_elem)
print(f"\nNon-dimensional form: {nondim_form}")
print(f"  Type: {type(nondim_form)}")

if hasattr(nondim_form, 'evalf'):
    numeric = float(nondim_form.evalf())
    print(f"  Numeric value: {numeric}")
    print(f"  Expected: 2.0 (since 2 * (1e21 / 1e21) = 2.0)")
    if abs(numeric - 2.0) < 0.01:
        print("  ✓ Correct!")

print("\n5. Demonstrate with Complex Expression")
print("-" * 70)

# Create a more complex expression
x, y = mesh.X
complex_expr = 2 * visc_param.sym * x**2 / y

print(f"Expression: 2 * η * x² / y")
print(f"  Symbolic: {complex_expr}")
print(f"  Non-dimensional: {uw.show_nondimensional_form(complex_expr)}")

print("\n6. Use Case: Debug Non-Dimensionalization")
print("-" * 70)
print("When JIT compilation shows unexpected values, use:")
print("  uw.show_nondimensional_form(expression)")
print("\nThis reveals what values are being substituted during compilation.")
print("Helps identify if:")
print("  - Parameters are being non-dimensionalized correctly (should be ~1.0)")
print("  - Units are consistent across expression")
print("  - Scaling is working as expected")

print("\n" + "=" * 70)
print("Demo Complete")
print("=" * 70)
