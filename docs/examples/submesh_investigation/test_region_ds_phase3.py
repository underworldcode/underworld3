"""
Phase 3: Region DS — restrict Stokes assembly to rock cells only.

Uses DMSetRegionDS to register a trivial (empty) DS on air cells.
Air DOFs are pinned to zero via Dirichlet on the "Outer" label.
The internal boundary penalty acts one-sided because air cells
contribute nothing to the residual/Jacobian.

Usage:
    pixi run -e default python tests/test_region_ds_phase3.py
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

output_dir = "./output/region_ds_phase3/"
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

# Build a complete "AirDOFs" label that includes ALL points (vertices, edges, cells)
# in the outer region, so P2 Dirichlet BCs cover every DOF.
from underworld3.discretisation.discretisation_mesh import extend_enum
from petsc4py import PETSc as _PETSc

dm = mesh.dm
outer_label = dm.getLabel("Outer")
outer_is = outer_label.getStratumIS(mesh.regions.Outer.value)
outer_cells = set(outer_is.getIndices()) if outer_is else set()

# Get all cells (depth == mesh.dim)
depth_label = dm.getLabel("depth")
cell_is = depth_label.getStratumIS(mesh.dim)
all_cells = set(cell_is.getIndices())
outer_cells_only = outer_cells & all_cells

# For each outer cell, get its closure (vertices + edges) and label them
AIR_DOFS_VAL = 200
dm.createLabel("AirDOFs")
air_label = dm.getLabel("AirDOFs")

air_points = set()
for cell in outer_cells_only:
    closure = dm.getTransitiveClosure(cell)[0]
    air_points.update(closure)

for pt in sorted(air_points):
    air_label.setValue(pt, AIR_DOFS_VAL)

uw.pprint(0, f"AirDOFs label: {len(air_points)} points (cells+edges+vertices in outer region)")

# Add to boundaries enum
@extend_enum([mesh.boundaries])
class extended_boundaries(Enum):
    Outer = mesh.regions.Outer.value
    AirDOFs = AIR_DOFS_VAL

mesh.boundaries = extended_boundaries

# Stack into UW_Boundaries
uw_bc_label = dm.getLabel("UW_Boundaries")
air_is = air_label.getStratumIS(AIR_DOFS_VAL)
if air_is:
    uw_bc_label.setStratumIS(AIR_DOFS_VAL, air_is)

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

# Body force everywhere (air cells won't assemble it due to Region DS)
rho = ((r / r_internal) ** k) * sympy.cos(n * th)
stokes.bodyforce = rho * (-1.0 * unit_rvec)

# Free-slip on outer and inner boundaries
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Upper")
stokes.add_natural_bc(vel_penalty * Gamma.dot(v.sym) * Gamma, "Lower")

# Penalty on internal boundary
stokes.add_natural_bc(vel_penalty * v.sym.dot(unit_rvec) * unit_rvec, "Internal")

# Pin air DOFs to zero (using complete label with vertices+edges+cells)
# Velocity (field 0) pinned to zero in air region
stokes.add_dirichlet_bc([0.0, 0.0], "AirDOFs")

# Configure Region DS: "Outer" cells get trivial DS (no assembly)
stokes.set_active_region("Outer", mesh.regions.Outer.value)

# --- Solver options ---

stokes.tolerance = stokes_tol
stokes.petsc_options["ksp_monitor"] = None
stokes.petsc_options["snes_monitor"] = None
stokes.petsc_options["snes_converged_reason"] = None
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

uw.pprint(0, "Solving with Region DS (air cells: trivial DS)...")
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
uw.pprint(0, "Region DS approach (trivial DS on air cells)")
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
mesh.write_timestep("phase3", meshVars=[v, p], outputPath=output_dir, index=0)
uw.pprint(0, f"Checkpoint saved to {output_dir}")
