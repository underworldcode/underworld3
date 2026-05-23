"""Diagnose DOF 863: trace-back path from an antipode edge-midpoint."""
import numpy as np
import underworld3 as uw

mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.15)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
coords = np.asarray(T.coords)

sigma = 0.3
def gauss(c):
    return np.exp(-((c[..., 0] - 1.0)**2 + c[..., 1]**2 + c[..., 2]**2) / (2 * sigma**2))

T.data[:, 0] = gauss(coords)

# Target DOF: an antipode edge-midpoint, T_initial should be ~0.
c = coords[863]
print(f"DOF 863 coord = {c},  r = {np.linalg.norm(c):.4f}")
print(f"T.data[863] (initial Gaussian value) = {T.data[863, 0]:+.6f}")

# Evaluate T at the DOF coord (NOT projected). Should also be ~0.
val_at_dof = uw.function.evaluate(T.sym, c.reshape(1, 3))
print(f"evaluate(T.sym, [DOF coord])         = {np.asarray(val_at_dof).flatten()[0]:+.6f}")

# Apply return_coords_to_bounds (radial → cell-plane projection).
projected = mesh.return_coords_to_bounds(c.reshape(1, 3).copy())
print(f"return_coords_to_bounds(coord)       = {projected[0]},  r = {np.linalg.norm(projected[0]):.4f}")
val_at_proj = uw.function.evaluate(T.sym, projected)
print(f"evaluate(T.sym, [projected coord])   = {np.asarray(val_at_proj).flatten()[0]:+.6f}")

# Apply just the cell-plane projection (no radial step).
proj_cell_only = mesh._project_to_nearest_cell_plane(c.reshape(1, 3).copy())
print(f"_project_to_nearest_cell_plane()     = {proj_cell_only[0]},  r = {np.linalg.norm(proj_cell_only[0]):.4f}")
val_at_proj_cell = uw.function.evaluate(T.sym, proj_cell_only)
print(f"evaluate(T.sym, [cell-plane only])   = {np.asarray(val_at_proj_cell).flatten()[0]:+.6f}")

# What cell did we pick?
mesh._build_kd_tree_index()
_, closest_idx = mesh._centroid_index.query(c.reshape(1, 3), k=1, sqr_dists=False)
closest_cell_idx = int(np.asarray(closest_idx).flatten()[0])
nav_dm = mesh._nav_dm if mesh._nav_dm is not None else mesh.dm
cStart, _ = nav_dm.getHeightStratum(0)
pStart, _ = nav_dm.getDepthStratum(0)
cell_id = cStart + closest_cell_idx
cone_pts = nav_dm.getTransitiveClosure(cell_id)[0][-mesh.element.entities[mesh.dim]:]
nav_coords = mesh._nav_coords
vtx = nav_coords[cone_pts - pStart]
print()
print(f"Closest cell (centroid kdtree) idx = {closest_cell_idx}")
print(f"  vertex 0: {vtx[0]}")
print(f"  vertex 1: {vtx[1]}")
print(f"  vertex 2: {vtx[2]}")
print(f"  cell centroid: {vtx.mean(axis=0)}")
# Sanity: where is this cell on the sphere? Same hemisphere as DOF 863?
print(f"  cell centroid distance to DOF coord = {np.linalg.norm(vtx.mean(axis=0) - c):.4f}")
print(f"  cell centroid distance to (1,0,0) (blob) = {np.linalg.norm(vtx.mean(axis=0) - np.array([1.0, 0.0, 0.0])):.4f}")
