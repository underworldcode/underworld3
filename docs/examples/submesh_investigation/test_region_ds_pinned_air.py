"""
Pinned-air approach: Dirichlet-constrain all air DOFs to zero.

Uses the existing solver infrastructure but adds essential BCs on the
"Outer" region label. This requires temporarily adding "Outer" to the
mesh boundaries enum so the solver's BC registration can find it.

With air velocity pinned to zero, the penalty on the internal boundary
becomes effectively one-sided — air-side closure data contributes zero.

Usage:
    pixi run -e default python tests/test_region_ds_pinned_air.py
"""

import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
import os
from enum import Enum

# --- Parameters ---

r_outer_full = 1.5
r_internal = 1.0
r_inner = 0.5
cellsize = 1/16
n = 2
k = 1
stokes_tol = 1.0e-4
vel_penalty = 1.0e4

output_dir = "./output/region_ds_pinned/"
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# --- Mesh ---

uw.pprint(0, "Creating full mesh...")

mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer_full,
    radiusInternal=r_internal,
    radiusInner=r_inner,
    cellSize=cellsize,
)

# Add region labels to the boundaries enum so the solver can find them
# for essential BC registration. This is a workaround until proper
# Region DS support is added to the solver.
from underworld3.discretisation.discretisation_mesh import extend_enum

@extend_enum([mesh.boundaries])
class extended_boundaries(Enum):
    Outer = mesh.regions.Outer.value  # 102

mesh.boundaries = extended_boundaries

uw.pprint(0, f"Boundaries: {[b.name for b in mesh.boundaries]}")
uw.pprint(0, f"Outer label value: {mesh.boundaries.Outer.value}")

# Verify the "Outer" DM label exists
outer_label = mesh.dm.getLabel("Outer")
uw.pprint(0, f"Outer DM label exists: {outer_label is not None}")

# --- Variables ---

v = uw.discretisation.MeshVariable("V", mesh, mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", mesh, 1, degree=1, continuous=True)

# --- Coordinate system ---

unit_rvec = mesh.CoordinateSystem.unit_e_0
r, th = mesh.CoordinateSystem.xR
Gamma = mesh.Gamma
v_theta_fn_xy = r * mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

# --- Stokes solver ---

stokes = Stokes(mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.saddle_preconditioner = 1.0

# Body force: same smooth density, applied everywhere (air DOFs are pinned anyway)
rho = ((r / r_internal) ** k) * sympy.cos(n * th)
stokes.bodyforce = rho * (-1.0 * unit_rvec)

# Free-slip on outer and inner boundaries
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Upper")
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Lower")

# Penalty on internal boundary (radial direction)
stokes.add_natural_bc(vel_penalty * v.sym.dot(unit_rvec) * unit_rvec, "Internal")

# Pin all air DOFs to zero (v=0, p=0 in outer region)
stokes.add_dirichlet_bc([0.0, 0.0], "Outer")

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

uw.pprint(0, "Solving with pinned air DOFs...")
stokes.solve(verbose=True)

# --- Null space removal ---

I0 = uw.maths.Integral(mesh, v_theta_fn_xy.dot(v.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()
dv = uw.function.evaluate(norm * v_theta_fn_xy, v.coords).reshape(-1, 2) / vnorm
v.data[...] -= dv

# --- Norms ---

rock_mask = sympy.Piecewise((1.0, r < r_internal), (0.0, True))
v_l2_rock = np.sqrt(uw.maths.Integral(mesh, rock_mask * v.sym.dot(v.sym)).evaluate())
p_l2_rock = np.sqrt(uw.maths.Integral(mesh, rock_mask * p.sym.dot(p.sym)).evaluate())

r_vals = uw.function.evaluate(r, v.coords)
inner_mask = r_vals.flatten() < r_internal
v_mag = np.sqrt(v.data[:, 0]**2 + v.data[:, 1]**2)
v_max_rock = v_mag[inner_mask].max()
v_max_air = v_mag[~inner_mask].max()

ref_v_l2 = 1.8061681957e-03
ref_p_l2 = 1.1796447277e-01

uw.pprint(0, "=" * 60)
uw.pprint(0, "Pinned-air approach (Dirichlet v=0 on Outer region)")
uw.pprint(0, f"  Rock-region norms:")
uw.pprint(0, f"    Velocity L2:  {v_l2_rock:.10e}  (ref: {ref_v_l2:.10e})")
uw.pprint(0, f"    Pressure L2:  {p_l2_rock:.10e}  (ref: {ref_p_l2:.10e})")
uw.pprint(0, f"  Relative errors:")
uw.pprint(0, f"    Velocity L2:  {abs(v_l2_rock - ref_v_l2) / ref_v_l2:.4e}")
uw.pprint(0, f"    Pressure L2:  {abs(p_l2_rock - ref_p_l2) / ref_p_l2:.4e}")
uw.pprint(0, f"  Max |v| rock:   {v_max_rock:.10e}")
uw.pprint(0, f"  Max |v| air:    {v_max_air:.10e}  (should be ~0)")
uw.pprint(0, "=" * 60)

# --- Checkpoint ---
mesh.write_timestep("pinned", meshVars=[v, p], outputPath=output_dir, index=0)
uw.pprint(0, f"Checkpoint saved to {output_dir}")
