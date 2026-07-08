"""
Re-run rock-only submesh and Nitsche air-layer with normalised Gamma_N.
Both use identical penalty=1e4, tol=1e-4.
Checkpoints saved for notebook visualisation.
"""

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3.cython.petsc_discretisation import petsc_dm_filter_by_label
from underworld3.discretisation import Mesh
from underworld3.coordinates import CoordinateSystemType
import numpy as np
import sympy
import os
from enum import Enum

r_outer_full = 1.5; r_internal = 1.0; r_inner = 0.5
cellsize = 1/16; n = 2; k = 1
vel_penalty = 1e4; stokes_tol = 1e-4; eta_air = 1e-3

# =====================================================================
# Full mesh (shared by both solves)
# =====================================================================
print("Creating full mesh...", flush=True)
full_mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer_full, radiusInternal=r_internal,
    radiusInner=r_inner, cellSize=cellsize)

# =====================================================================
# 1. Rock-only submesh solve
# =====================================================================
print("\n--- Rock-only submesh ---", flush=True)
subdm = petsc_dm_filter_by_label(full_mesh.dm, "Inner", 101)
subdm.markBoundaryFaces("All_Boundaries", 1001)

class sub_bd(Enum):
    Lower = 1       # r = r_inner (from full mesh "Lower")
    Internal = 2    # r = r_internal (from full mesh "Internal" — submesh outer boundary)

rock_mesh = Mesh(subdm, degree=1, qdegree=2, boundaries=sub_bd,
                 coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D)

v_rock = uw.discretisation.MeshVariable("V", rock_mesh, rock_mesh.dim, degree=2)
p_rock = uw.discretisation.MeshVariable("P", rock_mesh, 1, degree=1, continuous=True)

r_s, th_s = rock_mesh.CoordinateSystem.xR
unit_r_s = rock_mesh.CoordinateSystem.unit_e_0
G_N_s = rock_mesh.Gamma_N
v_theta_s = r_s * rock_mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

stokes_rock = Stokes(rock_mesh, velocityField=v_rock, pressureField=p_rock)
stokes_rock.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_rock.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes_rock.saddle_preconditioner = 1.0
rho_s = ((r_s / r_internal) ** k) * sympy.cos(n * th_s)
stokes_rock.bodyforce = rho_s * (-1.0 * unit_r_s)
stokes_rock.add_natural_bc(vel_penalty * G_N_s.dot(v_rock.sym) * G_N_s, "Internal")
stokes_rock.add_natural_bc(vel_penalty * G_N_s.dot(v_rock.sym) * G_N_s, "Lower")
stokes_rock.tolerance = stokes_tol
stokes_rock.petsc_options["snes_type"] = "newtonls"
stokes_rock.petsc_options["ksp_type"] = "fgmres"
stokes_rock.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes_rock.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

print("Solving submesh...", flush=True)
stokes_rock.solve(verbose=False)

# Null space removal
I0 = uw.maths.Integral(rock_mesh, v_theta_s.dot(v_rock.sym))
ns = I0.evaluate()
I0.fn = v_theta_s.dot(v_theta_s)
ns_norm = I0.evaluate()
dv = uw.function.evaluate(ns * v_theta_s, v_rock.coords).reshape(-1, 2) / ns_norm
v_rock.data[...] -= dv

v_l2_rock = np.sqrt(uw.maths.Integral(rock_mesh, v_rock.sym.dot(v_rock.sym)).evaluate())
print(f"Rock submesh velocity L2: {v_l2_rock:.6e}", flush=True)

out_rock = "./output/normalised_rock/"
if uw.mpi.rank == 0:
    os.makedirs(out_rock, exist_ok=True)
rock_mesh.write_timestep("rock", meshVars=[v_rock, p_rock], outputPath=out_rock, index=0)

# =====================================================================
# 2. Nitsche air-layer solve on full mesh
# =====================================================================
print("\n--- Nitsche air-layer (full mesh) ---", flush=True)
v_nit = uw.discretisation.MeshVariable("V", full_mesh, full_mesh.dim, degree=2)
p_nit = uw.discretisation.MeshVariable("P", full_mesh, 1, degree=1, continuous=False)
eta_var = uw.discretisation.MeshVariable("eta", full_mesh, 1, degree=1, continuous=False)
bf_mask = uw.discretisation.MeshVariable("mask", full_mesh, 1, degree=1, continuous=False)

r_at = np.sqrt(eta_var.coords[:, 0]**2 + eta_var.coords[:, 1]**2)
is_rock = r_at < r_internal
eta_var.data[is_rock, 0] = 1.0
eta_var.data[~is_rock, 0] = eta_air
bf_mask.data[is_rock, 0] = 1.0
bf_mask.data[~is_rock, 0] = 0.0

r_f, th_f = full_mesh.CoordinateSystem.xR
unit_r_f = full_mesh.CoordinateSystem.unit_e_0
G_N_f = full_mesh.Gamma_N
v_theta_f = r_f * full_mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

stokes_nit = Stokes(full_mesh, velocityField=v_nit, pressureField=p_nit)
stokes_nit.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_nit.constitutive_model.Parameters.shear_viscosity_0 = eta_var.sym[0, 0]
stokes_nit.saddle_preconditioner = 1.0 / eta_var.sym[0, 0]
rho_f = ((r_f / r_internal) ** k) * sympy.cos(n * th_f)
stokes_nit.bodyforce = bf_mask.sym[0, 0] * rho_f * (-1.0 * unit_r_f)
stokes_nit.add_natural_bc(vel_penalty * G_N_f.dot(v_nit.sym) * G_N_f, "Upper")
stokes_nit.add_natural_bc(vel_penalty * G_N_f.dot(v_nit.sym) * G_N_f, "Lower")
stokes_nit.add_natural_bc(vel_penalty * v_nit.sym.dot(unit_r_f) * unit_r_f, "Internal")
stokes_nit.tolerance = stokes_tol
stokes_nit.petsc_options["snes_type"] = "newtonls"
stokes_nit.petsc_options["ksp_type"] = "fgmres"
stokes_nit.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes_nit.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes_nit.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes_nit.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes_nit.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes_nit.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes_nit.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

print("Solving Nitsche...", flush=True)
stokes_nit.solve(verbose=False)

# Null space removal
I0_f = uw.maths.Integral(full_mesh, v_theta_f.dot(v_nit.sym))
ns_f = I0_f.evaluate()
I0_f.fn = v_theta_f.dot(v_theta_f)
ns_norm_f = I0_f.evaluate()
dv_f = uw.function.evaluate(ns_f * v_theta_f, v_nit.coords).reshape(-1, 2) / ns_norm_f
v_nit.data[...] -= dv_f

v_l2_nit_inner = np.sqrt(uw.maths.Integral(
    full_mesh,
    sympy.Piecewise((1.0, r_f < r_internal), (0.0, True)) * v_nit.sym.dot(v_nit.sym)
).evaluate())
print(f"Nitsche inner velocity L2: {v_l2_nit_inner:.6e}", flush=True)
print(f"Relative error: {abs(v_l2_nit_inner - v_l2_rock) / v_l2_rock:.4e}", flush=True)

out_nit = "./output/normalised_nitsche/"
if uw.mpi.rank == 0:
    os.makedirs(out_nit, exist_ok=True)
full_mesh.write_timestep("nitsche", meshVars=[v_nit, p_nit, eta_var], outputPath=out_nit, index=0)

print("\nCheckpoints saved.", flush=True)
