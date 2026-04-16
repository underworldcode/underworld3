"""Minimal VEP timing test — isolate where time is spent.

Run with: pixi run -e amr-dev python tests/minimal_vep_timing.py
"""

import time
import sympy
import underworld3 as uw
from underworld3.systems import VE_Stokes

t0 = time.time()

# --- Mesh ---
mesh = uw.meshing.UnstructuredSimplexBox(
    minCoords=(0.0, 0.0), maxCoords=(1.0, 1.0),
    cellSize=1.0 / 8, qdegree=3,
)
print(f"Mesh: {time.time() - t0:.1f}s")

# --- Variables ---
v = uw.discretisation.MeshVariable("U", mesh, 2, degree=2, vtype=uw.VarType.VECTOR)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True,
                                    vtype=uw.VarType.SCALAR)
print(f"Variables: {time.time() - t0:.1f}s")

# --- Solver + constitutive model ---
stokes = VE_Stokes(mesh, velocityField=v, pressureField=p, order=1)
stokes.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.constitutive_model.Parameters.shear_modulus = 1.0
stokes.constitutive_model.Parameters.shear_viscosity_min = 1.0e-3
stokes.constitutive_model.Parameters.strainrate_inv_II_min = 1.0e-10
stokes.saddle_preconditioner = 1.0
stokes.tolerance = 1.0e-4
print(f"Solver setup: {time.time() - t0:.1f}s")

# --- Yield stress: Piecewise (fault layer) ---
x, y = mesh.X
tau_y = sympy.Piecewise(
    (0.3, (y >= 0.45) & (y <= 0.55)),
    (1.0e6, True),
)
stokes.constitutive_model.Parameters.yield_stress = tau_y
print(f"Yield stress set: {time.time() - t0:.1f}s")

# --- BCs ---
stokes.add_essential_bc(sympy.Matrix([0.5, 0.0]), "Top")
stokes.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
stokes.add_essential_bc((sympy.oo, 0.0), "Left")
stokes.add_essential_bc((sympy.oo, 0.0), "Right")
stokes.bodyforce = sympy.Matrix([0.0, 0.0])
stokes.petsc_options["ksp_type"] = "fgmres"
print(f"BCs set: {time.time() - t0:.1f}s")

# --- First solve (includes JIT compilation) ---
# Set dt_elastic explicitly to work around elastic_dt alias bug in VE_Stokes.solve()
t1 = time.time()
stokes.solve(timestep=0.02, zero_init_guess=True)
print(f"First solve (incl JIT): {time.time() - t1:.1f}s")

# --- Second solve (cached JIT) ---
t2 = time.time()
stokes.solve(timestep=0.02, zero_init_guess=False)
print(f"Second solve (cached): {time.time() - t2:.1f}s")

# --- Compare: pure VE (no yield) ---
stokes2 = VE_Stokes(mesh, velocityField=v, pressureField=p, order=1)
stokes2.constitutive_model = uw.constitutive_models.ViscoElasticPlasticFlowModel
stokes2.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes2.constitutive_model.Parameters.shear_modulus = 1.0
# yield_stress defaults to sympy.oo — pure VE
stokes2.saddle_preconditioner = 1.0
stokes2.tolerance = 1.0e-4
stokes2.add_essential_bc(sympy.Matrix([0.5, 0.0]), "Top")
stokes2.add_essential_bc(sympy.Matrix([0.0, 0.0]), "Bottom")
stokes2.add_essential_bc((sympy.oo, 0.0), "Left")
stokes2.add_essential_bc((sympy.oo, 0.0), "Right")
stokes2.bodyforce = sympy.Matrix([0.0, 0.0])
stokes2.petsc_options["ksp_type"] = "fgmres"

t3 = time.time()
stokes2.solve(timestep=0.02, zero_init_guess=True)
print(f"Pure VE first solve (incl JIT): {time.time() - t3:.1f}s")

t4 = time.time()
stokes2.solve(timestep=0.02, zero_init_guess=False)
print(f"Pure VE second solve (cached): {time.time() - t4:.1f}s")

print(f"\nTotal: {time.time() - t0:.1f}s")
