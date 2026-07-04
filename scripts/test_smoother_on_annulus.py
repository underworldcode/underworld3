"""Minimal reproducer: smooth_mesh_interior on an Annulus, with
exception output."""
import sys
import traceback
import numpy as np
import underworld3 as uw

mesh = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                          cellSize=1.0 / 16, qdegree=3)
print("Annulus built. boundaries:",
      [b.name for b in mesh.boundaries], flush=True)
print(f"X.coords.shape = {mesh.X.coords.shape}", flush=True)

try:
    uw.meshing.smooth_mesh_interior(mesh, n_iters=5, alpha=0.5,
                                    verbose=True)
    print("smooth OK (no deformation case)", flush=True)
except Exception:
    print("smooth FAILED (no deformation case):", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)

# Now perturb a bit and try again.
coords = np.asarray(mesh.X.coords).copy()
r = np.sqrt(coords[:, 0] ** 2 + coords[:, 1] ** 2)
th = np.arctan2(coords[:, 1], coords[:, 0])
is_bnd = (np.abs(r - 0.5) < 1e-3) | (np.abs(r - 1.0) < 1e-3)
coords[~is_bnd] += 0.01 * np.column_stack([
    np.sin(7 * th[~is_bnd]),
    np.cos(5 * th[~is_bnd]),
])
mesh._deform_mesh(coords)
print("Mesh perturbed.", flush=True)

try:
    uw.meshing.smooth_mesh_interior(mesh, n_iters=5, alpha=0.5,
                                    verbose=True)
    print("smooth OK (deformed case)", flush=True)
except Exception:
    print("smooth FAILED (deformed case):", flush=True)
    traceback.print_exc(file=sys.stdout)
    sys.exit(1)

print("ALL OK", flush=True)
