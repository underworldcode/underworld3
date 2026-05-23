"""Probe: does _project_to_nearest_cell_plane actually move the point
into the cell's plane? And does evaluate at the projected point match
the analytic field?
"""
import numpy as np
import underworld3 as uw

mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.3)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=2)
coords = np.asarray(T.coords)
# Analytic: T = z (simple, easy to track)
T.data[:, 0] = coords[:, 2]

# Probe points: a few on the sphere (r=1).
probes = np.array([
    [1.0, 0.0, 0.0],
    [0.5, 0.5, 0.7071067811865476],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [-0.7071, 0.0, 0.7071],
])
# Make sure they're exactly on the sphere.
probes = probes / np.linalg.norm(probes, axis=1, keepdims=True)

# Apply the cell-plane projection.
projected = mesh._project_to_nearest_cell_plane(probes)

# Check r before / after.
print("=== cell-plane projection inspection ===")
for i in range(probes.shape[0]):
    r_before = np.linalg.norm(probes[i])
    r_after = np.linalg.norm(projected[i])
    shift = np.linalg.norm(projected[i] - probes[i])
    print(f"probe {i}: input r={r_before:.6f}, proj r={r_after:.6f}, "
          f"shift={shift:.4e}")

print()
print("=== evaluate T at probe vs at projected ===")
# Evaluate T at the on-sphere probes (causes extrapolation).
T_on_sphere = uw.function.evaluate(T.sym, probes)
T_on_sphere = np.asarray(T_on_sphere).reshape(-1)
# Evaluate T at the in-plane projected coords.
T_in_plane = uw.function.evaluate(T.sym, projected)
T_in_plane = np.asarray(T_in_plane).reshape(-1)
# Analytic value: T = z.
T_analytic = probes[:, 2]

for i in range(probes.shape[0]):
    print(
        f"probe {i}: analytic z={T_analytic[i]:.4f}, "
        f"eval(on-sphere)={T_on_sphere[i]:.6f}, "
        f"eval(in-plane)={T_in_plane[i]:.6f}"
    )

# Also: explicitly call return_coords_to_bounds (the composed fn).
print()
print("=== return_coords_to_bounds(probes) ===")
proj2 = mesh.return_coords_to_bounds(probes.copy())
for i in range(probes.shape[0]):
    r_after = np.linalg.norm(proj2[i])
    shift = np.linalg.norm(proj2[i] - probes[i])
    print(f"probe {i}: r_after={r_after:.6f}, shift={shift:.4e}")
