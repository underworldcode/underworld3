"""
Bootstrap through decreasing air viscosity contrasts.

Start from eta_air=1e-3 checkpoint, solve at 1e-4, use that to
initialise 1e-5, and so on down to 1e-6. Each step uses the
previous solution as initial guess (zero_init_guess=False).

All use dP1 pressure, normalised Gamma_N, penalty=1e4.
"""

import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
import os

r_internal = 1.0; r_inner = 0.5; r_outer_full = 1.5; cellsize = 1/16
n = 2; k = 1; vel_penalty = 1e4; stokes_tol = 1e-4

# --- Create mesh ---
print("Creating mesh...", flush=True)
mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer_full, radiusInternal=r_internal,
    radiusInner=r_inner, cellSize=cellsize)

v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=False)
eta_var = uw.discretisation.MeshVariable("eta", mesh, 1, degree=1, continuous=False)
bf_mask = uw.discretisation.MeshVariable("mask", mesh, 1, degree=1, continuous=False)

r_at = np.sqrt(eta_var.coords[:, 0]**2 + eta_var.coords[:, 1]**2)
is_rock = r_at < r_internal
bf_mask.data[is_rock, 0] = 1.0
bf_mask.data[~is_rock, 0] = 0.0

r_f, th_f = mesh.CoordinateSystem.xR
unit_r_f = mesh.CoordinateSystem.unit_e_0
G_N = mesh.Gamma_N
v_theta = r_f * mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

# --- Load eta=1e-3 checkpoint as starting point ---
print("Loading eta=1e-3 checkpoint...", flush=True)
v.read_timestep("nitsche", "V", 0, outputPath="output/normalised_nitsche/")
p.read_timestep("nitsche", "P", 0, outputPath="output/normalised_nitsche/")

# Set initial viscosity
eta_var.data[is_rock, 0] = 1.0
eta_var.data[~is_rock, 0] = 1e-3

# --- Bootstrap through decreasing viscosity ---
eta_steps = [1e-4, 1e-5, 1e-6]

for eta_air in eta_steps:
    print(f"\n{'='*60}", flush=True)
    print(f"Solving with eta_air = {eta_air:.0e}", flush=True)
    print(f"{'='*60}", flush=True)

    # Update viscosity
    eta_var.data[~is_rock, 0] = eta_air

    # Create fresh solver (needed because constitutive model refs change)
    stokes = Stokes(mesh, velocityField=v, pressureField=p)
    stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
    stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_var.sym[0, 0]
    stokes.saddle_preconditioner = 1.0 / eta_var.sym[0, 0]

    rho_f = ((r_f / r_internal) ** k) * sympy.cos(n * th_f)
    stokes.bodyforce = bf_mask.sym[0, 0] * rho_f * (-1.0 * unit_r_f)

    stokes.add_natural_bc(vel_penalty * G_N.dot(v.sym) * G_N, "Upper")
    stokes.add_natural_bc(vel_penalty * G_N.dot(v.sym) * G_N, "Lower")
    stokes.add_natural_bc(vel_penalty * v.sym.dot(unit_r_f) * unit_r_f, "Internal")

    stokes.tolerance = stokes_tol
    stokes.petsc_options["snes_type"] = "newtonls"
    stokes.petsc_options["ksp_type"] = "fgmres"
    stokes.petsc_options["ksp_monitor"] = None
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
    stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
    stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
    stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
    stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

    # Use previous solution as initial guess
    stokes.solve(zero_init_guess=False, verbose=True)

    # Null space removal
    I0 = uw.maths.Integral(mesh, v_theta.dot(v.sym))
    ns = I0.evaluate()
    I0.fn = v_theta.dot(v_theta)
    nn = I0.evaluate()
    dv = uw.function.evaluate(ns * v_theta, v.coords).reshape(-1, 2) / nn
    v.data[...] -= dv

    # Norms
    rock_mask = sympy.Piecewise((1.0, r_f < r_internal), (0.0, True))
    v_l2 = np.sqrt(uw.maths.Integral(mesh, rock_mask * v.sym.dot(v.sym)).evaluate())
    print(f"  Inner velocity L2: {v_l2:.6e}", flush=True)

    # Checkpoint
    eta_str = f"{eta_air:.0e}".replace("-", "m")
    out_dir = f"./output/bootstrap_eta{eta_str}/"
    if uw.mpi.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    mesh.write_timestep(f"eta{eta_str}", meshVars=[v, p, eta_var], outputPath=out_dir, index=0)
    print(f"  Checkpoint: {out_dir}", flush=True)

print("\nDone.", flush=True)
