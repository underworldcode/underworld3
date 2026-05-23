"""Trace DOF 2107's SL trace-back path step-by-step.

DOF 2107 is at antipode (coord ≈ [-0.74, -0.67, +0.07]). After 6 SLCN
steps it has T_after = 0.727, but the analytic value is 0. The
diffusion-only solve works fine, so the bug must be in the
trace-back evaluation.

Manually replicate what SLCN does:
  1. mid_pt = coord - V(coord) * dt/2
  2. mid_pt_projected = return_coords_to_bounds(mid_pt)
  3. v_mid = global_evaluate(V_sym, mid_pt_projected)
  4. end_pt = coord - v_mid * dt
  5. end_pt_projected = return_coords_to_bounds(end_pt)
  6. T* = global_evaluate(T.sym, end_pt_projected)

Print everything at each step.
"""
import numpy as np
import sympy
import underworld3 as uw

mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.075)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
coords = np.asarray(T.coords)

sigma = 0.3
T.data[:, 0] = np.exp(
    -((coords[:, 0] - 1.0)**2 + coords[:, 1]**2 + coords[:, 2]**2) / (2 * sigma**2)
)

# DOF 2107 — the worst antipode anomaly.
idx = 2107
c = coords[idx]
print(f"DOF {idx} starting coord = {c},  r = {np.linalg.norm(c):.6f}")
print(f"T.data[{idx}] = {T.data[idx, 0]:+.6f}  (correctly ~0)")

# Velocity at the DOF coord, analytically: V = (-y, x, 0).
V_at_c = np.array([-c[1], c[0], 0.0])
print(f"\nV at DOF coord (analytic) = {V_at_c}")

dt = 2 * np.pi / 72

# Step 1: midpoint coord.
mid_pt = c - V_at_c * (0.5 * dt)
print(f"\nmid_pt = c - V*dt/2 = {mid_pt}  r = {np.linalg.norm(mid_pt):.6f}")

# Apply return_coords_to_bounds (radial → cell-plane).
mid_pt_proj = mesh.return_coords_to_bounds(mid_pt.reshape(1, 3).copy())[0]
print(f"return_coords_to_bounds(mid_pt) = {mid_pt_proj}  r = {np.linalg.norm(mid_pt_proj):.6f}")
print(f"  arc-distance from original DOF coord: "
      f"{np.degrees(np.arccos(np.clip(np.dot(c, mid_pt_proj) / (np.linalg.norm(c) * np.linalg.norm(mid_pt_proj)), -1, 1))):.2f}°")

# Evaluate V at mid_pt_proj.
x_sym, y_sym, z_sym = mesh.X
V_sym = sympy.Matrix([[-y_sym, x_sym, sympy.sympify(0)]])
v_mid_result = uw.function.global_evaluate(V_sym, mid_pt_proj.reshape(1, 3))
v_mid = np.asarray(v_mid_result).reshape(-1)
print(f"\nglobal_evaluate(V_sym, mid_pt_proj) = {v_mid}")
print(f"  analytic V at mid_pt_proj = "
      f"[{-mid_pt_proj[1]:.4f}, {mid_pt_proj[0]:.4f}, 0]")

# Step 2: endpoint coord.
end_pt = c - v_mid * dt
print(f"\nend_pt = c - v_mid*dt = {end_pt}  r = {np.linalg.norm(end_pt):.6f}")

end_pt_proj = mesh.return_coords_to_bounds(end_pt.reshape(1, 3).copy())[0]
print(f"return_coords_to_bounds(end_pt) = {end_pt_proj}  r = {np.linalg.norm(end_pt_proj):.6f}")
arc_to_orig = np.degrees(np.arccos(np.clip(
    np.dot(c, end_pt_proj) / (np.linalg.norm(c) * np.linalg.norm(end_pt_proj)),
    -1, 1
)))
arc_to_blob = np.degrees(np.arccos(np.clip(
    np.dot(np.array([1.0, 0.0, 0.0]), end_pt_proj),
    -1, 1
)))
print(f"  arc-distance from original DOF coord: {arc_to_orig:.2f}°")
print(f"  arc-distance from blob centre (1,0,0): {arc_to_blob:.2f}°")
print(f"  expected: arc to blob ≈ {np.degrees(np.arccos(np.dot(c, np.array([1.0,0,0])))):.2f}° "
      f"(antipodal)")

# Step 3: T* — evaluate T at the projected endpoint.
T_star = uw.function.global_evaluate(T.sym, end_pt_proj.reshape(1, 3))
T_star_value = float(np.asarray(T_star).reshape(-1)[0])
print(f"\nglobal_evaluate(T.sym, end_pt_proj) = {T_star_value:+.6f}")

# Direct (non-global) evaluate at the same coord for cross-check.
T_local = uw.function.evaluate(T.sym, end_pt_proj.reshape(1, 3))
print(f"uw.function.evaluate(T.sym, end_pt_proj) = "
      f"{float(np.asarray(T_local).reshape(-1)[0]):+.6f}  (rank-local)")

# Analytic expectation: end_pt is still at antipode, so Gaussian ≈ 0.
analytic_T_at_endpoint = float(np.exp(
    -((end_pt_proj[0] - 1.0)**2 + end_pt_proj[1]**2 + end_pt_proj[2]**2) / (2 * sigma**2)
))
print(f"analytic Gaussian at end_pt_proj = {analytic_T_at_endpoint:+.6f}")

# Now: what if we SKIP the cell-plane projection and only use the
# radial step?
print(f"\n=== Skip cell-plane projection ===")
radial_only = end_pt.copy()
r = np.linalg.norm(radial_only)
radial_only = radial_only / r  # to r=1
T_radial = uw.function.evaluate(T.sym, radial_only.reshape(1, 3))
print(f"end_pt radially-projected to r=1: {radial_only}")
print(f"evaluate(T.sym, radial_only) = "
      f"{float(np.asarray(T_radial).reshape(-1)[0]):+.6f}")

# And what if we don't project at all (use raw end_pt)?
print(f"\n=== Skip ALL projection ===")
T_raw = uw.function.evaluate(T.sym, end_pt.reshape(1, 3))
print(f"end_pt (unprojected): {end_pt}, r = {np.linalg.norm(end_pt):.6f}")
print(f"evaluate(T.sym, end_pt) = "
      f"{float(np.asarray(T_raw).reshape(-1)[0]):+.6f}")

# Which cell did my _project_to_nearest_cell_plane pick?
mesh._build_kd_tree_index()
_, closest_idx = mesh._centroid_index.query(end_pt.reshape(1, 3), k=1, sqr_dists=False)
cell_idx = int(np.asarray(closest_idx).reshape(-1)[0])
nav_dm = mesh._nav_dm if mesh._nav_dm is not None else mesh.dm
cStart, _ = nav_dm.getHeightStratum(0)
pStart, _ = nav_dm.getDepthStratum(0)
cone_pts = nav_dm.getTransitiveClosure(cStart + cell_idx)[0][-mesh.element.entities[mesh.dim]:]
nav_coords = mesh._nav_coords
vtx = nav_coords[cone_pts - pStart]
print(f"\n=== Cell my projection picked ===")
print(f"  cell idx = {cell_idx}, centroid = {vtx.mean(axis=0)}")
print(f"  vertex 0: {vtx[0]}")
print(f"  vertex 1: {vtx[1]}")
print(f"  vertex 2: {vtx[2]}")

# Find T values at these cell vertices (look up DOF indices via coord match).
print(f"  T at cell vertices:")
for i in range(3):
    d = np.linalg.norm(coords - vtx[i], axis=1)
    dof_match = int(np.argmin(d))
    print(f"    vertex {i} ≈ DOF {dof_match}, dist={d[dof_match]:.4e}, "
          f"T.data[{dof_match}] = {T.data[dof_match, 0]:+.6f}")
