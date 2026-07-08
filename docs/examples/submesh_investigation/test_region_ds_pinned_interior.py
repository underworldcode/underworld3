"""
Pinned air-interior approach: Dirichlet only on air DOFs that are NOT
on the internal boundary.

The previous pinned-air test applied Dirichlet v=0 on ALL points in the
"Outer" label, including vertices shared with the internal boundary.
Those interface vertices should be free to participate in the rock solve.

This test creates an "AirInterior" label excluding interface points,
and applies Dirichlet only there.

Usage:
    pixi run -e default python tests/test_region_ds_pinned_interior.py
"""

import underworld3 as uw
from underworld3.systems import Stokes
import numpy as np
import sympy
import os
from enum import Enum
from petsc4py import PETSc

# --- Parameters ---

r_outer_full = 1.5
r_internal = 1.0
r_inner = 0.5
cellsize = 1/16
n = 2
k = 1
stokes_tol = 1.0e-4
vel_penalty = 1.0e4

output_dir = "./output/region_ds_pinned_interior/"
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

# --- Create AirInterior label: Outer points minus Internal points ---

dm = mesh.dm
outer_label = dm.getLabel("Outer")
internal_label = dm.getLabel("Internal")

# Get point sets
outer_is = outer_label.getStratumIS(mesh.regions.Outer.value)
internal_is = internal_label.getStratumIS(mesh.boundaries.Internal.value)

outer_points = set(outer_is.getIndices()) if outer_is else set()
internal_points = set(internal_is.getIndices()) if internal_is else set()

# Air interior = outer minus internal boundary
air_interior_points = outer_points - internal_points

uw.pprint(0, f"Outer points: {len(outer_points)}")
uw.pprint(0, f"Internal points: {len(internal_points)}")
uw.pprint(0, f"Air interior points: {len(air_interior_points)}")
uw.pprint(0, f"Interface points removed: {len(outer_points) - len(air_interior_points)}")

# Create DM label
AIR_INTERIOR_VAL = 200
dm.createLabel("AirInterior")
air_label = dm.getLabel("AirInterior")
for pt in sorted(air_interior_points):
    air_label.setValue(pt, AIR_INTERIOR_VAL)

# Add to mesh boundaries so solver can find it
from underworld3.discretisation.discretisation_mesh import extend_enum

@extend_enum([mesh.boundaries])
class extended_boundaries(Enum):
    AirInterior = AIR_INTERIOR_VAL

mesh.boundaries = extended_boundaries

# Also stack into UW_Boundaries
uw_bc_label = dm.getLabel("UW_Boundaries")
air_is = air_label.getStratumIS(AIR_INTERIOR_VAL)
if air_is:
    uw_bc_label.setStratumIS(AIR_INTERIOR_VAL, air_is)

uw.pprint(0, f"AirInterior label created with {len(air_interior_points)} points, value={AIR_INTERIOR_VAL}")

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

# Body force everywhere (air DOFs are pinned anyway)
rho = ((r / r_internal) ** k) * sympy.cos(n * th)
stokes.bodyforce = rho * (-1.0 * unit_rvec)

# Free-slip on outer and inner boundaries
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Upper")
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Lower")

# Penalty on internal boundary
stokes.add_natural_bc(vel_penalty * v.sym.dot(unit_rvec) * unit_rvec, "Internal")

# Pin air-interior DOFs to zero (NOT interface DOFs)
stokes.add_dirichlet_bc([0.0, 0.0], "AirInterior")

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

uw.pprint(0, "Solving with pinned air-interior DOFs...")
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
uw.pprint(0, "Pinned air-interior (Dirichlet on Outer minus Internal)")
uw.pprint(0, f"  Rock-region norms:")
uw.pprint(0, f"    Velocity L2:  {v_l2_rock:.10e}  (ref: {ref_v_l2:.10e})")
uw.pprint(0, f"    Pressure L2:  {p_l2_rock:.10e}  (ref: {ref_p_l2:.10e})")
uw.pprint(0, f"  Relative errors:")
uw.pprint(0, f"    Velocity L2:  {abs(v_l2_rock - ref_v_l2) / ref_v_l2:.4e}")
uw.pprint(0, f"    Pressure L2:  {abs(p_l2_rock - ref_p_l2) / ref_p_l2:.4e}")
uw.pprint(0, f"  Max |v| rock:   {v_max_rock:.10e}")
uw.pprint(0, f"  Max |v| air:    {v_max_air:.10e}")
uw.pprint(0, "=" * 60)

# --- Checkpoint ---
mesh.write_timestep("pinned_interior", meshVars=[v, p], outputPath=output_dir, index=0)
uw.pprint(0, f"Checkpoint saved to {output_dir}")
