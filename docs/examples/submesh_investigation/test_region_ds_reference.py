"""
Reference Stokes solution on a rock-only annulus mesh.

This establishes ground-truth velocity and pressure norms for verifying
the Region DS subdomain solving approach. The rock-only annulus here
corresponds to the Inner region of an AnnulusInternalBoundary mesh.

Test problem: Isoviscous Stokes with smooth density, free-slip BCs.

    Rock region: r_inner=0.5, r_outer=1.0 (= r_internal of full mesh)
    Density:     rho = cos(n*theta) * (r/r_outer)^k
    Body force:  -rho * unit_r (radial gravity)
    BCs:         Free-slip (penalty) on both boundaries
    Viscosity:   1.0

Usage:
    pixi run -e default python tests/test_region_ds_reference.py
"""

import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
import os

# --- Parameters ---

r_outer = 1.0    # Outer radius (= r_internal of full mesh)
r_inner = 0.5    # Inner radius
cellsize = 1/16  # Mesh resolution
n = 2            # Wave number
k = 1            # Power exponent for density
vel_penalty = 1.0e6
stokes_tol = 1.0e-6

output_dir = "./output/region_ds_reference/"
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# --- Mesh ---

uw.pprint(0, f"Creating rock-only annulus: r_inner={r_inner}, r_outer={r_outer}")

mesh = uw.meshing.Annulus(
    radiusOuter=r_outer,
    radiusInner=r_inner,
    cellSize=cellsize,
)

uw.pprint(0, f"Mesh chart: {mesh.dm.getChart()}")

# --- Variables ---

v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)

# --- Coordinate system ---

unit_rvec = mesh.CoordinateSystem.unit_e_0
r, th = mesh.CoordinateSystem.xR
Gamma = mesh.Gamma

# Null space: constant v_theta in x,y coordinates
v_theta_fn_xy = r * mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

# --- Stokes solver ---

stokes = Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.saddle_preconditioner = 1.0

# Smooth density anomaly
rho = ((r / r_outer) ** k) * sympy.cos(n * th)
gravity_fn = -1.0 * unit_rvec
stokes.bodyforce = rho * gravity_fn

# Free-slip on both boundaries (penalty on normal velocity)
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Upper")
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Lower")

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

uw.pprint(0, "Solving Stokes...")
stokes.solve(verbose=True)

# --- Null space removal (rigid body rotation) ---

I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()

dv = uw.function.evaluate(norm * v_theta_fn_xy, v.coords).reshape(-1, 2) / vnorm
v.data[...] -= dv

# --- Compute norms ---

v_l2_integral = uw.maths.Integral(mesh, v.sym.dot(v.sym))
v_l2 = np.sqrt(v_l2_integral.evaluate())

p_l2_integral = uw.maths.Integral(mesh, p.sym.dot(p.sym))
p_l2 = np.sqrt(p_l2_integral.evaluate())

# Velocity magnitude stats
v_mag = np.sqrt(v.data[:, 0] ** 2 + v.data[:, 1] ** 2)
v_max = v_mag.max()

uw.pprint(0, "=" * 60)
uw.pprint(0, "Reference solution norms (rock-only annulus)")
uw.pprint(0, f"  r_inner={r_inner}, r_outer={r_outer}")
uw.pprint(0, f"  n={n}, k={k}, cellsize={cellsize}")
uw.pprint(0, f"  Velocity L2 norm:  {v_l2:.10e}")
uw.pprint(0, f"  Pressure L2 norm:  {p_l2:.10e}")
uw.pprint(0, f"  Max |v|:           {v_max:.10e}")
uw.pprint(0, "=" * 60)

# --- Save checkpoint ---
mesh.write_timestep("reference", meshVars=[v, p], outputPath=output_dir, index=0)
uw.pprint(0, f"Checkpoint saved to {output_dir}")
