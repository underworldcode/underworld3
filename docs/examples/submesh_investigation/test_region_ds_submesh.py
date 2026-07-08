"""
Submesh approach: extract the inner region from AnnulusInternalBoundary
and solve Stokes on it directly.

The submesh shares exact node positions with the full mesh, so solutions
can be mapped back by coordinate matching without interpolation.

Usage:
    pixi run -e default python tests/test_region_ds_submesh.py
"""

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3.cython.petsc_discretisation import petsc_dm_filter_by_label
from underworld3.discretisation import Mesh
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
stokes_tol = 1.0e-6
vel_penalty = 1.0e6

output_dir = "./output/region_ds_submesh/"
if uw.mpi.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

# --- Full mesh ---

uw.pprint(0, "Creating full mesh...")
full_mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer_full,
    radiusInternal=r_internal,
    radiusInner=r_inner,
    cellSize=cellsize,
)
uw.pprint(0, f"Full mesh: {full_mesh.dm.getChart()}")

# --- Extract inner region submesh ---

uw.pprint(0, "Extracting inner region submesh via DMPlexFilter...")
subdm = petsc_dm_filter_by_label(full_mesh.dm, "Inner", 101)

# Mark boundary faces on the submesh
subdm.markBoundaryFaces("All_Boundaries", 1001)

# The submesh needs boundary labels. The internal boundary (r=r_internal)
# becomes the outer boundary of the submesh. We'll set up boundaries
# by radius.

# Wrap in a UW3 Mesh
class submesh_boundaries(Enum):
    Lower = 1    # r = r_inner
    Upper = 2    # r = r_internal (was Internal on full mesh)

from underworld3.coordinates import CoordinateSystemType

rock_mesh = Mesh(
    subdm,
    degree=1,
    qdegree=2,
    boundaries=submesh_boundaries,
    coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D,
    verbose=False,
)

uw.pprint(0, f"Rock submesh: {rock_mesh.dm.getChart()}")

# Check coordinates
coords = rock_mesh.X.coords
r_coords = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
uw.pprint(0, f"Rock mesh r range: [{r_coords.min():.6f}, {r_coords.max():.6f}]")

# --- Label boundaries by radius ---
# The submesh lost the original boundary labels. Re-label by radius.

dm = rock_mesh.dm
dm.createLabel("UW_Boundaries")
uw_label = dm.getLabel("UW_Boundaries")
all_bd_label = dm.getLabel("All_Boundaries")

if all_bd_label:
    bd_is = all_bd_label.getStratumIS(1001)
    if bd_is:
        bd_points = bd_is.getIndices()
        uw.pprint(0, f"Boundary points: {len(bd_points)}")

        # Get vertex coordinates for boundary points
        # Only process vertices (depth 0)
        depth_label = dm.getLabel("depth")
        vert_is = depth_label.getStratumIS(0)
        verts = set(vert_is.getIndices()) if vert_is else set()

        coord_sec = dm.getCoordinateSection()
        coord_vec = dm.getCoordinatesLocal()

        n_lower = 0
        n_upper = 0
        for pt in bd_points:
            if pt in verts:
                off = coord_sec.getOffset(pt)
                x = coord_vec.getArray()[off]
                y = coord_vec.getArray()[off + 1]
                radius = np.sqrt(x**2 + y**2)

                if abs(radius - r_inner) < cellsize * 0.5:
                    uw_label.setValue(pt, submesh_boundaries.Lower.value)
                    n_lower += 1
                elif abs(radius - r_internal) < cellsize * 0.5:
                    uw_label.setValue(pt, submesh_boundaries.Upper.value)
                    n_upper += 1
            else:
                # Edges/faces: classify by checking if they're on inner or outer boundary
                # Use closure to find vertices and determine which boundary
                closure = dm.getTransitiveClosure(pt)[0]
                radii = []
                for cpt in closure:
                    if cpt in verts:
                        off = coord_sec.getOffset(cpt)
                        x = coord_vec.getArray()[off]
                        y = coord_vec.getArray()[off + 1]
                        radii.append(np.sqrt(x**2 + y**2))
                if radii:
                    mean_r = np.mean(radii)
                    if abs(mean_r - r_inner) < cellsize * 0.5:
                        uw_label.setValue(pt, submesh_boundaries.Lower.value)
                    elif abs(mean_r - r_internal) < cellsize * 0.5:
                        uw_label.setValue(pt, submesh_boundaries.Upper.value)

        uw.pprint(0, f"Labeled: {n_lower} lower vertices, {n_upper} upper vertices")

# --- Variables ---

v = uw.discretisation.MeshVariable("V", rock_mesh, rock_mesh.dim, degree=2)
p = uw.discretisation.MeshVariable("P", rock_mesh, 1, degree=1, continuous=True)

# --- Coordinate system ---

unit_rvec = rock_mesh.CoordinateSystem.unit_e_0
r, th = rock_mesh.CoordinateSystem.xR
Gamma = rock_mesh.Gamma
v_theta_fn_xy = r * rock_mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

# --- Stokes solver ---

stokes = Stokes(rock_mesh, velocityField=v, pressureField=p)
stokes.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes.saddle_preconditioner = 1.0

rho = ((r / r_internal) ** k) * sympy.cos(n * th)
stokes.bodyforce = rho * (-1.0 * unit_rvec)

# Free-slip on both boundaries
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

uw.pprint(0, "Solving Stokes on rock submesh...")
stokes.solve(verbose=True)

# --- Null space removal ---

I0 = uw.maths.Integral(rock_mesh, v_theta_fn_xy.dot(v.sym))
norm = I0.evaluate()
I0.fn = v_theta_fn_xy.dot(v_theta_fn_xy)
vnorm = I0.evaluate()
dv = uw.function.evaluate(norm * v_theta_fn_xy, v.coords).reshape(-1, 2) / vnorm
v.data[...] -= dv

# --- Norms ---

v_l2 = np.sqrt(uw.maths.Integral(rock_mesh, v.sym.dot(v.sym)).evaluate())
p_l2 = np.sqrt(uw.maths.Integral(rock_mesh, p.sym.dot(p.sym)).evaluate())
v_mag = np.sqrt(v.data[:, 0]**2 + v.data[:, 1]**2)

ref_v_l2 = 1.8061681957e-03
ref_p_l2 = 1.1796447277e-01
ref_v_max = 2.1782171120e-03

uw.pprint(0, "=" * 60)
uw.pprint(0, "Submesh approach (DMPlexFilter inner region)")
uw.pprint(0, f"  Velocity L2:  {v_l2:.10e}  (ref: {ref_v_l2:.10e})")
uw.pprint(0, f"  Pressure L2:  {p_l2:.10e}  (ref: {ref_p_l2:.10e})")
uw.pprint(0, f"  Max |v|:      {v_mag.max():.10e}  (ref: {ref_v_max:.10e})")
uw.pprint(0, f"  Relative errors:")
uw.pprint(0, f"    Velocity L2:  {abs(v_l2 - ref_v_l2) / ref_v_l2:.4e}")
uw.pprint(0, f"    Pressure L2:  {abs(p_l2 - ref_p_l2) / ref_p_l2:.4e}")
uw.pprint(0, f"    Max |v|:      {abs(v_mag.max() - ref_v_max) / ref_v_max:.4e}")
uw.pprint(0, "=" * 60)

# --- Checkpoint ---
rock_mesh.write_timestep("submesh", meshVars=[v, p], outputPath=output_dir, index=0)
uw.pprint(0, f"Checkpoint saved to {output_dir}")
