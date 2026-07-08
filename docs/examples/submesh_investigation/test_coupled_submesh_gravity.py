"""
Coupled submesh demonstration: thermal-Stokes with gravity.

Workflow:
1. Create full mesh with internal boundary
2. Extract rock submesh
3. Set temperature on rock submesh (analytical)
4. Prolongate temperature to full mesh (zero in air)
5. Solve Poisson gravity on full mesh using T-derived density
6. Restrict gravity to rock submesh
7. Solve Stokes on rock submesh with gravity as buoyancy

This exercises the full data flow: extract_region, prolongate,
restrict, copy_into, and solving on both meshes.

Usage:
    pixi run -e default python tests/test_coupled_submesh_gravity.py
"""

import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
import os

# --- Parameters ---

r_outer = 1.5
r_internal = 1.0
r_inner = 0.5
cellsize = 1/12
vel_penalty = 1e4

# --- Step 1: Create meshes ---

uw.pprint(0, "Step 1: Creating meshes...")

full_mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer,
    radiusInternal=r_internal,
    radiusInner=r_inner,
    cellSize=cellsize,
)

rock_mesh = full_mesh.extract_region("Inner")

uw.pprint(0, f"  Full mesh: {full_mesh.dm.getChart()}")
uw.pprint(0, f"  Rock submesh: {rock_mesh.dm.getChart()}")

# --- Step 2: Variables ---

# Temperature on rock submesh
T_rock = uw.discretisation.MeshVariable("T_rock", rock_mesh, 1, degree=2)

# Temperature on full mesh (for gravity source)
T_full = uw.discretisation.MeshVariable("T_full", full_mesh, 1, degree=2)

# Gravity potential on full mesh
phi_full = uw.discretisation.MeshVariable("phi", full_mesh, 1, degree=2)

# Gravity potential restricted to rock submesh
phi_rock = uw.discretisation.MeshVariable("phi_rock", rock_mesh, 1, degree=2)

# Stokes variables on rock submesh
v_rock = uw.discretisation.MeshVariable("V_rock", rock_mesh, rock_mesh.dim, degree=2)
p_rock = uw.discretisation.MeshVariable("P_rock", rock_mesh, 1, degree=1, continuous=True)

# --- Step 3: Set temperature on rock submesh ---

uw.pprint(0, "Step 3: Setting temperature on rock submesh...")

r_rock_coords = np.sqrt(T_rock.coords[:, 0]**2 + T_rock.coords[:, 1]**2)
th_rock_coords = np.arctan2(T_rock.coords[:, 1], T_rock.coords[:, 0])

# Temperature: hot blob near the inner boundary
T_rock.data[:, 0] = np.cos(2 * th_rock_coords) * (1.0 - (r_rock_coords - r_inner) / (r_internal - r_inner))

uw.pprint(0, f"  T_rock range: [{T_rock.data.min():.4f}, {T_rock.data.max():.4f}]")

# --- Step 4: Prolongate temperature to full mesh ---

uw.pprint(0, "Step 4: Prolongating T to full mesh...")

T_full.data[:] = 0.0  # zero in air
rock_mesh.prolongate(T_rock, T_full)

r_full_coords = np.sqrt(T_full.coords[:, 0]**2 + T_full.coords[:, 1]**2)
rock_mask = r_full_coords < r_internal + 1e-6
uw.pprint(0, f"  T_full rock region: [{T_full.data[rock_mask].min():.4f}, {T_full.data[rock_mask].max():.4f}]")
uw.pprint(0, f"  T_full air region max: {np.abs(T_full.data[~rock_mask]).max():.2e}")

# --- Step 5: Solve Poisson gravity on full mesh ---

uw.pprint(0, "Step 5: Solving Poisson gravity on full mesh...")

gravity = uw.systems.Poisson(full_mesh, u_Field=phi_full)
gravity.constitutive_model = uw.constitutive_models.DiffusionModel
gravity.constitutive_model.Parameters.diffusivity = 1.0
gravity.f = T_full.sym[0, 0]  # density source = temperature

# Zero potential on outer boundary
gravity.add_dirichlet_bc(0.0, "Upper")

gravity.tolerance = 1e-6
gravity.petsc_options["snes_type"] = "newtonls"
gravity.petsc_options["ksp_type"] = "fgmres"

gravity.solve(verbose=False)

uw.pprint(0, f"  phi range: [{phi_full.data.min():.4e}, {phi_full.data.max():.4e}]")

# --- Step 6: Restrict gravity to rock submesh ---

uw.pprint(0, "Step 6: Restricting gravity to rock submesh...")

rock_mesh.restrict(phi_full, phi_rock)

err = np.abs(phi_rock.data[:, 0] - phi_full.data[rock_mask, 0][:phi_rock.data.shape[0]]).max()
uw.pprint(0, f"  phi_rock range: [{phi_rock.data.min():.4e}, {phi_rock.data.max():.4e}]")

# --- Step 7: Solve Stokes on rock submesh ---

uw.pprint(0, "Step 7: Solving Stokes on rock submesh...")

r_s, th_s = rock_mesh.CoordinateSystem.xR
G_N = rock_mesh.Gamma_N
unit_r = rock_mesh.CoordinateSystem.unit_e_0

stokes = Stokes(rock_mesh, velocityField=v_rock, pressureField=p_rock)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.saddle_preconditioner = 1.0

# Buoyancy from gravity gradient (simplified: use T directly as density)
# In a real model: bodyforce = -rho * grad(phi)
# Here we use T as a proxy for density-driven flow
stokes.bodyforce = T_rock.sym[0, 0] * (-unit_r)

stokes.add_natural_bc(vel_penalty * G_N.dot(v_rock.sym) * G_N, "Internal")
stokes.add_natural_bc(vel_penalty * G_N.dot(v_rock.sym) * G_N, "Lower")

stokes.tolerance = 1e-4
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

stokes.solve(verbose=False)

# Null space removal
v_theta = r_s * rock_mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))
I0 = uw.maths.Integral(rock_mesh, v_theta.dot(v_rock.sym))
ns = I0.evaluate()
I0.fn = v_theta.dot(v_theta)
nn = I0.evaluate()
dv = uw.function.evaluate(ns * v_theta, v_rock.coords).reshape(-1, 2) / nn
v_rock.data[...] -= dv

vmag = np.sqrt(v_rock.data[:, 0]**2 + v_rock.data[:, 1]**2)
uw.pprint(0, f"  Stokes max |v|: {vmag.max():.6e}")

# --- Step 8: Prolongate velocity back to full mesh for visualisation ---

uw.pprint(0, "Step 8: Prolongating velocity to full mesh...")

v_full = uw.discretisation.MeshVariable("V_full", full_mesh, full_mesh.dim, degree=2)
v_full.data[:] = 0.0
rock_mesh.prolongate(v_rock, v_full)

vmag_full = np.sqrt(v_full.data[:, 0]**2 + v_full.data[:, 1]**2)
uw.pprint(0, f"  Full mesh: rock |v| max={vmag_full[rock_mask[:v_full.data.shape[0]]].max():.6e}")
uw.pprint(0, f"  Full mesh: air |v| max={vmag_full[~rock_mask[:v_full.data.shape[0]]].max():.2e}")

# --- Checkpoint ---

out = "./output/coupled_submesh/"
if uw.mpi.rank == 0:
    os.makedirs(out, exist_ok=True)

full_mesh.write_timestep("coupled", meshVars=[T_full, phi_full, v_full], outputPath=out, index=0)
rock_mesh.write_timestep("coupled_rock", meshVars=[T_rock, phi_rock, v_rock, p_rock], outputPath=out, index=0)

uw.pprint(0, f"\nCheckpoints saved to {out}")
uw.pprint(0, "Done — coupled submesh workflow complete.")
