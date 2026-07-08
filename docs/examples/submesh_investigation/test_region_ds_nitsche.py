"""
Nitsche BC on internal boundary with viscosity contrast.

Same setup as test_region_ds_air_layer.py but uses add_nitsche_bc()
on the internal boundary instead of the simple velocity penalty.

With a viscosity contrast, the Nitsche consistency term (sigma.n.d)
may correctly weight the stress from each side, potentially giving
better results than the simple penalty.

Usage:
    pixi run -e default python tests/test_region_ds_nitsche.py
"""

import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
import os

# --- Parameters ---

r_outer_full = 1.5
r_internal = 1.0
r_inner = 0.5
cellsize = 1/16
n = 2
k = 1
stokes_tol = 1.0e-4
eta_air = 1.0e-3

output_dir = "./output/region_ds_nitsche/"
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# --- Mesh ---

uw.pprint(0, f"Creating full mesh: r_inner={r_inner}, r_internal={r_internal}, r_outer={r_outer_full}")

mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer_full,
    radiusInternal=r_internal,
    radiusInner=r_inner,
    cellSize=cellsize,
)

# --- Variables ---

v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)
eta_var = uw.discretisation.MeshVariable("eta", mesh, 1, degree=1, continuous=False)
bf_mask_var = uw.discretisation.MeshVariable("mask", mesh, 1, degree=1, continuous=False)

# --- Assign viscosity and mask by radius ---

r_at_eta = np.sqrt(eta_var.coords[:, 0]**2 + eta_var.coords[:, 1]**2)
is_rock = r_at_eta < r_internal
eta_var.data[is_rock, 0] = 1.0
eta_var.data[~is_rock, 0] = eta_air
bf_mask_var.data[is_rock, 0] = 1.0
bf_mask_var.data[~is_rock, 0] = 0.0

uw.pprint(0, f"Viscosity: {is_rock.sum()} rock DOFs (eta=1), {(~is_rock).sum()} air DOFs (eta={eta_air})")

# --- Coordinate system ---

unit_rvec = mesh.CoordinateSystem.unit_e_0
r, th = mesh.CoordinateSystem.xR
Gamma = mesh.Gamma

v_theta_fn_xy = r * mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

# --- Stokes solver ---

stokes = Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_var.sym[0, 0]
stokes.saddle_preconditioner = 1.0 / eta_var.sym[0, 0]

rho = ((r / r_internal) ** k) * sympy.cos(n * th)
stokes.bodyforce = bf_mask_var.sym[0, 0] * rho * (-1.0 * unit_rvec)

# Free-slip on outer and inner (penalty — these are exterior boundaries)
stokes.add_natural_bc(1e4 * Gamma.dot(v.sym) * Gamma, "Upper")
stokes.add_natural_bc(1e4 * Gamma.dot(v.sym) * Gamma, "Lower")

# Nitsche free-slip on internal boundary
# Uses constitutive model viscosity, so it sees the contrast
stokes.add_nitsche_bc("Internal", direction=unit_rvec, gamma=10.0, theta=1)

# --- Solver options ---

stokes.tolerance = stokes_tol
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_type"] = "newtonls"
stokes.petsc_options["ksp_type"] = "fgmres"

stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes.petsc_options.setValue("fieldsplit_velocity_pc_mg_cycle_type", "w")
stokes.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"
stokes.petsc_options["fieldsplit_velocity_ksp_type"] = "fcg"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_type"] = "chebyshev"
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_max_it"] = 5
stokes.petsc_options["fieldsplit_velocity_mg_levels_ksp_converged_maxits"] = None

# --- Solve ---

uw.pprint(0, "Solving with Nitsche BC on internal boundary...")
stokes.solve(verbose=True)

# --- Null space removal ---

I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()
dv = uw.function.evaluate(norm * v_theta_fn_xy, v.coords).reshape(-1, 2) / vnorm
v.data[...] -= dv

# --- Norms ---

sharp_mask = sympy.Piecewise((1.0, r < r_internal), (0.0, True))
v_l2_inner = np.sqrt(uw.maths.Integral(mesh, sharp_mask * v.sym.dot(v.sym)).evaluate())
p_l2_inner = np.sqrt(uw.maths.Integral(mesh, sharp_mask * p.sym.dot(p.sym)).evaluate())

r_vals = uw.function.evaluate(r, v.coords)
inner_mask = r_vals.flatten() < r_internal
v_mag = np.sqrt(v.data[:, 0]**2 + v.data[:, 1]**2)
v_max_inner = v_mag[inner_mask].max()

ref_v_l2 = 1.8061681957e-03
ref_p_l2 = 1.1796447277e-01
ref_v_max = 2.1782171120e-03

uw.pprint(0, "=" * 60)
uw.pprint(0, f"Nitsche BC on Internal (eta_air={eta_air})")
uw.pprint(0, f"  Rock-region norms:")
uw.pprint(0, f"    Velocity L2:  {v_l2_inner:.10e}  (ref: {ref_v_l2:.10e})")
uw.pprint(0, f"    Pressure L2:  {p_l2_inner:.10e}  (ref: {ref_p_l2:.10e})")
uw.pprint(0, f"    Max |v|:      {v_max_inner:.10e}  (ref: {ref_v_max:.10e})")
uw.pprint(0, f"  Relative errors:")
uw.pprint(0, f"    Velocity L2:  {abs(v_l2_inner - ref_v_l2) / ref_v_l2:.4e}")
uw.pprint(0, f"    Pressure L2:  {abs(p_l2_inner - ref_p_l2) / ref_p_l2:.4e}")
uw.pprint(0, f"    Max |v|:      {abs(v_max_inner - ref_v_max) / ref_v_max:.4e}")
uw.pprint(0, "=" * 60)

# --- Checkpoint ---
mesh.write_timestep("nitsche", meshVars=[v, p, eta_var], outputPath=output_dir, index=0)
uw.pprint(0, f"Checkpoint saved to {output_dir}")
