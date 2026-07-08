"""
Low-viscosity air layer comparison for Region DS verification.

Solves Stokes on the full AnnulusInternalBoundary mesh with:
  - Rock (inner): viscosity=1.0, body force active
  - Air (outer):  viscosity=eta_air (very low), zero body force

Viscosity is set element-by-element using a P0 (discontinuous degree-1)
MeshVariable, assigned from the cell region labels.

Compares inner-region velocity/pressure norms against the rock-only
reference solution. As eta_air -> 0, the inner-region solution should
converge to the rock-only reference.

Usage:
    pixi run -e default python tests/test_region_ds_air_layer.py
"""

import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
import os

# --- Parameters ---

r_outer_full = 1.5   # Full mesh outer radius
r_internal = 1.0     # Internal boundary (rock/air interface)
r_inner = 0.5        # Inner radius
cellsize = 1/16
n = 2
k = 1
vel_penalty = 1.0e4
stokes_tol = 1.0e-4
eta_air = 1.0e-3     # Low viscosity for air layer

output_dir = "./output/region_ds_air_layer/"
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

uw.pprint(0, f"Mesh chart: {mesh.dm.getChart()}")
uw.pprint(0, f"Regions: {[r.name for r in mesh.regions]}")

# --- Variables ---

v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)

# P0-like viscosity field (discontinuous, degree 1 — lowest available)
eta_var = uw.discretisation.MeshVariable("eta", mesh, 1, degree=1, continuous=False)

# P0-like body force mask
bf_mask_var = uw.discretisation.MeshVariable("mask", mesh, 1, degree=1, continuous=False)

# --- Assign viscosity and mask by radius ---

r_at_eta = np.sqrt(eta_var.coords[:, 0]**2 + eta_var.coords[:, 1]**2)
is_rock = r_at_eta < r_internal

eta_var.data[is_rock, 0] = 1.0
eta_var.data[~is_rock, 0] = eta_air

bf_mask_var.data[is_rock, 0] = 1.0
bf_mask_var.data[~is_rock, 0] = 0.0

n_rock = is_rock.sum()
n_air = (~is_rock).sum()
uw.pprint(0, f"Viscosity assigned: {n_rock} rock DOFs (eta=1), {n_air} air DOFs (eta={eta_air})")

# --- Coordinate system ---

unit_rvec = mesh.CoordinateSystem.unit_e_0
r, th = mesh.CoordinateSystem.xR
Gamma = mesh.Gamma

# Null space: constant v_theta in x,y coordinates
v_theta_fn_xy = r * mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

# --- Stokes solver ---

stokes = Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = eta_var.sym[0, 0]
stokes.saddle_preconditioner = 1.0 / eta_var.sym[0, 0]

# Body force only in rock region
rho = ((r / r_internal) ** k) * sympy.cos(n * th)
gravity_fn = -1.0 * unit_rvec
stokes.bodyforce = bf_mask_var.sym[0, 0] * rho * gravity_fn

# Free-slip on outer and inner boundaries (Gamma-based)
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Upper")
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Lower")

# Penalty on radial velocity at internal boundary (analytical radial direction
# avoids Gamma support-ordering ambiguity on internal faces)
stokes.add_natural_bc(vel_penalty * v.sym.dot(unit_rvec) * unit_rvec, "Internal")

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

uw.pprint(0, f"Solving Stokes with eta_air={eta_air}...")
stokes.solve(verbose=True)

# --- Null space removal ---

I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()

dv = uw.function.evaluate(norm * v_theta_fn_xy, v.coords).reshape(-1, 2) / vnorm
v.data[...] -= dv

# --- Compute norms on INNER region only ---

# Use Piecewise mask for integration over inner region
sharp_mask = sympy.Piecewise((1.0, r < r_internal), (0.0, True))

# Inner-region velocity L2 norm
v_l2_inner_integral = uw.maths.Integral(mesh, sharp_mask * v.sym.dot(v.sym))
v_l2_inner = np.sqrt(v_l2_inner_integral.evaluate())

# Inner-region pressure L2 norm
p_l2_inner_integral = uw.maths.Integral(mesh, sharp_mask * p.sym.dot(p.sym))
p_l2_inner = np.sqrt(p_l2_inner_integral.evaluate())

# Full-domain norms
v_l2_full = np.sqrt(uw.maths.Integral(mesh, v.sym.dot(v.sym)).evaluate())
p_l2_full = np.sqrt(uw.maths.Integral(mesh, p.sym.dot(p.sym)).evaluate())

# Velocity magnitude stats in inner region
r_vals = uw.function.evaluate(r, v.coords)
inner_mask = r_vals.flatten() < r_internal
v_mag = np.sqrt(v.data[:, 0] ** 2 + v.data[:, 1] ** 2)
v_max_inner = v_mag[inner_mask].max() if inner_mask.any() else 0.0
v_max_air = v_mag[~inner_mask].max() if (~inner_mask).any() else 0.0

# --- Report ---

# Reference values from rock-only solve (cellsize=1/16, n=2, k=1)
ref_v_l2 = 1.8061681957e-03
ref_p_l2 = 1.1796447277e-01
ref_v_max = 2.1782171120e-03

uw.pprint(0, "=" * 60)
uw.pprint(0, f"Air layer comparison (eta_air={eta_air})")
uw.pprint(0, f"  r_inner={r_inner}, r_internal={r_internal}, r_outer={r_outer_full}")
uw.pprint(0, "")
uw.pprint(0, "  Inner-region norms:")
uw.pprint(0, f"    Velocity L2:  {v_l2_inner:.10e}  (ref: {ref_v_l2:.10e})")
uw.pprint(0, f"    Pressure L2:  {p_l2_inner:.10e}  (ref: {ref_p_l2:.10e})")
uw.pprint(0, f"    Max |v|:      {v_max_inner:.10e}  (ref: {ref_v_max:.10e})")
uw.pprint(0, "")
uw.pprint(0, "  Relative errors:")
uw.pprint(0, f"    Velocity L2:  {abs(v_l2_inner - ref_v_l2) / ref_v_l2:.4e}")
uw.pprint(0, f"    Pressure L2:  {abs(p_l2_inner - ref_p_l2) / ref_p_l2:.4e}")
uw.pprint(0, f"    Max |v|:      {abs(v_max_inner - ref_v_max) / ref_v_max:.4e}")
uw.pprint(0, "")
uw.pprint(0, "  Full-domain norms:")
uw.pprint(0, f"    Velocity L2:  {v_l2_full:.10e}")
uw.pprint(0, f"    Pressure L2:  {p_l2_full:.10e}")
uw.pprint(0, f"    Max |v| air:  {v_max_air:.10e}")
uw.pprint(0, "=" * 60)

# --- Save checkpoint ---
mesh.write_timestep("air_layer", meshVars=[v, p, eta_var], outputPath=output_dir, index=0)
uw.pprint(0, f"Checkpoint saved to {output_dir}")
