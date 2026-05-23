"""Verify _project_to_nearest_cell_plane lands the coord EXACTLY
in the picked cell's plane.
"""
import numpy as np
import underworld3 as uw

mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.075)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
coords = np.asarray(T.coords)
T.data[:, 0] = coords[:, 2] + 0.3 * coords[:, 0]

# Use DOF 2107's end_pt from the earlier trace.
end_pt = np.array([-0.79784511, -0.59911955, 0.06707759])
print(f"Input coord: {end_pt}, r = {np.linalg.norm(end_pt):.6f}")

projected = mesh._project_to_nearest_cell_plane(end_pt.reshape(1, 3).copy())[0]
print(f"Projected:   {projected}, r = {np.linalg.norm(projected):.6f}")
print(f"Shift:       {np.linalg.norm(projected - end_pt):.4e}")

# Which cell did the projection pick?
mesh._build_kd_tree_index()
_, cell_idx = mesh._centroid_index.query(end_pt.reshape(1, 3), k=1, sqr_dists=False)
cell_idx = int(np.asarray(cell_idx).reshape(-1)[0])
nav_dm = mesh._nav_dm if mesh._nav_dm is not None else mesh.dm
cStart, _ = nav_dm.getHeightStratum(0)
pStart, _ = nav_dm.getDepthStratum(0)
cone_pts = nav_dm.getTransitiveClosure(cStart + cell_idx)[0][-mesh.element.entities[mesh.dim]:]
vtx = mesh._nav_coords[cone_pts - pStart]
a, b, c = vtx[0], vtx[1], vtx[2]
print(f"\nPicked cell {cell_idx}:")
print(f"  vertex a: {a}")
print(f"  vertex b: {b}")
print(f"  vertex c: {c}")

# Cell plane: passes through a, normal = (b-a) x (c-a) (un-normalised).
n_unnorm = np.cross(b - a, c - a)
n_hat = n_unnorm / np.linalg.norm(n_unnorm)
print(f"  cell normal (unit): {n_hat}")

# Signed distance from projected coord to the cell's plane.
signed_dist_in_plane = np.dot(projected - a, n_hat)
print(f"\n  signed distance from PROJECTED point to picked cell's plane = "
      f"{signed_dist_in_plane:+.4e}")

# Same for the input (un-projected) coord.
signed_dist_input = np.dot(end_pt - a, n_hat)
print(f"  signed distance from INPUT     point to picked cell's plane = "
      f"{signed_dist_input:+.4e}")

# Sanity: barycentric of projected coord in (a, b, c).
ab = b - a
ac = c - a
ap = projected - a
d1 = ab @ ap
d2 = ac @ ap
d3 = ab @ (projected - b)
d4 = ac @ (projected - b)
va = (b - projected) @ np.cross(c - projected, a - projected)  # simplified
# Use the algorithm in distance_pointcloud_triangle but for one point.
# Just compute the barycentric directly assuming we're in-plane.
# Project ap onto basis (ab, ac):
m = np.array([[ab @ ab, ab @ ac], [ab @ ac, ac @ ac]])
rhs = np.array([ab @ ap, ac @ ap])
sol = np.linalg.solve(m, rhs)
beta_, gamma_ = sol
alpha_ = 1.0 - beta_ - gamma_
print(f"\n  Barycentric of projected coord w.r.t. picked cell: "
      f"(α, β, γ) = ({alpha_:+.4f}, {beta_:+.4f}, {gamma_:+.4f})")
print(f"  (inside triangle iff all three are in [0, 1])")

# Manual FE evaluate of T (which is z + 0.3*x) at the projected coord
# using the picked cell's DOFs.
T_at_a = a[2] + 0.3 * a[0]
T_at_b = b[2] + 0.3 * b[0]
T_at_c = c[2] + 0.3 * c[0]
T_manual = alpha_ * T_at_a + beta_ * T_at_b + gamma_ * T_at_c
T_analytic = projected[2] + 0.3 * projected[0]
T_evaluate = float(np.asarray(uw.function.evaluate(T.sym, projected.reshape(1, 3))).reshape(-1)[0])

print(f"\n  At projected coord [{projected[0]:+.4f}, {projected[1]:+.4f}, {projected[2]:+.4f}]:")
print(f"  T_analytic         = {T_analytic:+.6f}")
print(f"  T_manual_bary      = {T_manual:+.6f}  (using picked cell {cell_idx})")
print(f"  T_evaluate         = {T_evaluate:+.6f}  (via DMInterpolation)")
