"""Investigation: why penalty solution differs between submesh and full mesh."""

import underworld3 as uw
from underworld3.systems import Stokes
from underworld3.cython.petsc_discretisation import petsc_dm_filter_by_label
from underworld3.discretisation import Mesh
from underworld3.coordinates import CoordinateSystemType
import numpy as np
import sympy
from enum import Enum
from scipy.spatial import cKDTree

r_internal = 1.0; r_inner = 0.5; r_outer_full = 1.5; cellsize = 1/16
n = 2; k = 1; vel_penalty = 1e4; stokes_tol = 1e-4

print("=" * 70, flush=True)
print("INVESTIGATION: Penalty comparison submesh vs full mesh", flush=True)
print("=" * 70, flush=True)

# Create both meshes
full_mesh = uw.meshing.AnnulusInternalBoundary(
    radiusOuter=r_outer_full, radiusInternal=r_internal,
    radiusInner=r_inner, cellSize=cellsize)

subdm = petsc_dm_filter_by_label(full_mesh.dm, "Inner", 101)
subdm.markBoundaryFaces("All_Boundaries", 1001)

class sub_bd(Enum):
    Lower = 1; Upper = 2

rock_mesh = Mesh(subdm, degree=1, qdegree=2, boundaries=sub_bd,
                 coordinate_system_type=CoordinateSystemType.CYLINDRICAL2D)

r_s, th_s = rock_mesh.CoordinateSystem.xR
r_f, th_f = full_mesh.CoordinateSystem.xR
unit_r_s = rock_mesh.CoordinateSystem.unit_e_0
unit_r_f = full_mesh.CoordinateSystem.unit_e_0
Gamma_s = rock_mesh.Gamma
Gamma_f = full_mesh.Gamma
v_theta_s = r_s * rock_mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))
v_theta_f = r_f * full_mesh.CoordinateSystem.rRotN.T * sympy.Matrix((0, 1))

# =====================================================================
# Solve submesh with penalty=1e4
# =====================================================================
v_ref = uw.discretisation.MeshVariable("V_ref", rock_mesh, rock_mesh.dim, degree=2)
p_ref = uw.discretisation.MeshVariable("P_ref", rock_mesh, 1, degree=1, continuous=True)

stokes_s = Stokes(rock_mesh, velocityField=v_ref, pressureField=p_ref)
stokes_s.constitutive_model = uw.constitutive_models.ViscousFlowModel
stokes_s.constitutive_model.Parameters.shear_viscosity_0 = 1.0
stokes_s.saddle_preconditioner = 1.0
rho_s = ((r_s / r_internal) ** k) * sympy.cos(n * th_s)
stokes_s.bodyforce = rho_s * (-1.0 * unit_r_s)
stokes_s.add_natural_bc(vel_penalty * Gamma_s.dot(v_ref.sym) * Gamma_s, "Upper")
stokes_s.add_natural_bc(vel_penalty * Gamma_s.dot(v_ref.sym) * Gamma_s, "Lower")
stokes_s.tolerance = stokes_tol
stokes_s.petsc_options["snes_type"] = "newtonls"
stokes_s.petsc_options["ksp_type"] = "fgmres"
stokes_s.petsc_options.setValue("fieldsplit_velocity_pc_mg_type", "kaskade")
stokes_s.petsc_options["fieldsplit_velocity_mg_coarse_pc_type"] = "svd"

print("\nSolving submesh (penalty=1e4)...", flush=True)
stokes_s.solve(verbose=False)

# =====================================================================
# POINT 3: Null space
# =====================================================================
print("\n--- POINT 3: Null space ---", flush=True)

# Submesh null space BEFORE removal
I0 = uw.maths.Integral(rock_mesh, v_theta_s.dot(v_ref.sym))
ns_sub = I0.evaluate()
I0.fn = v_theta_s.dot(v_theta_s)
ns_norm = I0.evaluate()
print(f"Submesh NS component (before removal): {ns_sub:.6e}", flush=True)
print(f"  NS norm: {ns_norm:.6e}, relative: {abs(ns_sub/ns_norm):.6e}", flush=True)

# Remove
dv = uw.function.evaluate(ns_sub * v_theta_s, v_ref.coords).reshape(-1, 2) / ns_norm
v_ref.data[...] -= dv

v_mag_ref = np.sqrt(v_ref.data[:, 0]**2 + v_ref.data[:, 1]**2)
print(f"Submesh |v| after NS removal: mean={v_mag_ref.mean():.4e}, max={v_mag_ref.max():.4e}", flush=True)

# Full mesh null space (from checkpoint)
v_pen = uw.discretisation.MeshVariable("V_pen", full_mesh, full_mesh.dim, degree=2)
v_pen.read_timestep("air_layer", "V", 0, outputPath="output/region_ds_air_layer/")

I0_f = uw.maths.Integral(full_mesh, v_theta_f.dot(v_pen.sym))
ns_full = I0_f.evaluate()
I0_f.fn = v_theta_f.dot(v_theta_f)
ns_norm_f = I0_f.evaluate()
print(f"\nFull mesh NS component (checkpoint, already removed): {ns_full:.6e}", flush=True)
print(f"  NS norm: {ns_norm_f:.6e}, relative: {abs(ns_full/ns_norm_f):.6e}", flush=True)

# =====================================================================
# POINT 1: Air incompressibility — radial velocity at interface
# =====================================================================
print("\n--- POINT 1: Radial velocity at r=1.0 ---", flush=True)

# Full mesh
r_at_v = np.sqrt(v_pen.coords[:, 0]**2 + v_pen.coords[:, 1]**2)
int_mask = np.abs(r_at_v - r_internal) < cellsize * 0.3
v_int = v_pen.data[int_mask]
c_int = v_pen.coords[int_mask]
r_hat = c_int / np.linalg.norm(c_int, axis=1, keepdims=True)
vr_full = np.sum(v_int * r_hat, axis=1)
print(f"Full mesh v_r at r=1.0: mean={vr_full.mean():.4e}, rms={np.sqrt((vr_full**2).mean()):.4e}, max|vr|={np.abs(vr_full).max():.4e}", flush=True)

# Submesh
r_at_vs = np.sqrt(v_ref.coords[:, 0]**2 + v_ref.coords[:, 1]**2)
int_mask_s = np.abs(r_at_vs - r_internal) < cellsize * 0.3
v_int_s = v_ref.data[int_mask_s]
c_int_s = v_ref.coords[int_mask_s]
r_hat_s = c_int_s / np.linalg.norm(c_int_s, axis=1, keepdims=True)
vr_sub = np.sum(v_int_s * r_hat_s, axis=1)
print(f"Submesh   v_r at r=1.0: mean={vr_sub.mean():.4e}, rms={np.sqrt((vr_sub**2).mean()):.4e}, max|vr|={np.abs(vr_sub).max():.4e}", flush=True)

print(f"\nRatio rms(vr) submesh/full: {np.sqrt((vr_sub**2).mean()) / np.sqrt((vr_full**2).mean()):.2f}", flush=True)
print("  >1 means submesh leaks MORE radially (no air resistance)", flush=True)

# =====================================================================
# POINT 2: Effective penalty — compare Gamma vs unit_rvec
# =====================================================================
print("\n--- POINT 2: Penalty form ---", flush=True)
print(f"Submesh BC: vel_penalty * Gamma.dot(v) * Gamma  (PETSc face normal)", flush=True)
print(f"Full mesh BC on Internal: vel_penalty * v.dot(unit_rvec) * unit_rvec  (analytical radial)", flush=True)
print(f"These are DIFFERENT penalty forms. Gamma may not align with radial on the submesh.", flush=True)

# =====================================================================
# Match and compare
# =====================================================================
print("\n--- MATCHED NODE COMPARISON ---", flush=True)

tree = cKDTree(v_ref.coords)
dists, idx = tree.query(v_pen.coords)
matched = dists < 1e-10

v_ref_m = v_ref.data[idx[matched]]
v_pen_m = v_pen.data[matched]
coords_m = v_ref.coords[idx[matched]]

def l2(a, b):
    return np.sqrt(np.sum((a - b)**2)) / np.sqrt(np.sum(b**2))

vmag_r = np.sqrt(v_ref_m[:, 0]**2 + v_ref_m[:, 1]**2)
vmag_p = np.sqrt(v_pen_m[:, 0]**2 + v_pen_m[:, 1]**2)

print(f"L2 rel error: {l2(v_pen_m, v_ref_m):.4e}", flush=True)
print(f"|v_ref| mean: {vmag_r.mean():.4e}", flush=True)
print(f"|v_pen| mean: {vmag_p.mean():.4e}", flush=True)
print(f"Ratio pen/ref: {vmag_p.mean()/vmag_r.mean():.4f}", flush=True)

print("\n" + "=" * 70, flush=True)
print("SUMMARY", flush=True)
print("=" * 70, flush=True)
print(f"1. Radial velocity at interface: submesh leaks {np.sqrt((vr_sub**2).mean()) / np.sqrt((vr_full**2).mean()):.1f}x more than full mesh", flush=True)
print(f"   -> Air incompressibility constrains radial flow even with low penalty", flush=True)
print(f"2. Different penalty forms: Gamma.dot(v)*Gamma vs v.dot(r_hat)*r_hat", flush=True)
print(f"3. Null space: submesh component = {abs(ns_sub/ns_norm):.2e}", flush=True)
print("=" * 70, flush=True)
