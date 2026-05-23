"""Is DMInterpolation broken for points exactly in a cell's plane?

Construct test points by direct barycentric combination of known cell
vertices — so each test point is GUARANTEED to be in that specific
cell's plane and in its barycentric interior. Then call evaluate
and check whether the answer matches the manual barycentric
interpolation of the cell's DOF values.

If evaluate gives the right answer: DMInterpolation works fine when
the coord is exactly in a cell's plane, and our `_project_to_nearest_cell_plane`
must be producing coords NOT exactly in the picked cell's plane.

If evaluate gives wrong answers: DMInterpolation is broken for
in-plane points on a closed 2-manifold.
"""
import numpy as np
import underworld3 as uw

mesh = uw.meshing.SphericalManifold(radius=1.0, cellSize=0.15)
T = uw.discretisation.MeshVariable("T", mesh, 1, degree=1)
coords = np.asarray(T.coords)

# Set T to a known linear function in the embedded coords:
#    T(x, y, z) = z + 0.3 * x
# This is exact-in-cell for P1 (linear) elements: the FE
# interpolation of a linear function is EXACT to machine
# precision on every cell.
T.data[:, 0] = coords[:, 2] + 0.3 * coords[:, 0]

# Pick several cells, construct an in-cell point by barycentric
# combination of vertices, and check evaluate.
nav_dm = mesh._nav_dm if mesh._nav_dm is not None else mesh.dm
cStart, cEnd = nav_dm.getHeightStratum(0)
pStart, _ = nav_dm.getDepthStratum(0)
cell_num_points = mesh.element.entities[mesh.dim]
nav_coords = mesh._nav_coords

# Choose cells from different regions of the sphere.
test_cells = [0, 100, 200, cEnd - cStart - 1, cEnd - cStart - 100]

print("=== DMInterpolation test: point EXACTLY in a known cell's plane ===\n")
print(f"  T(x,y,z) = z + 0.3*x  (linear; FE-exact on P1 cells)\n")

for cidx in test_cells:
    cell_id = cStart + cidx
    cone_pts = nav_dm.getTransitiveClosure(cell_id)[0][-cell_num_points:]
    vtx = nav_coords[cone_pts - pStart]
    a, b, c = vtx[0], vtx[1], vtx[2]

    # Construct a point at barycentric (0.5, 0.3, 0.2):
    # exactly in the cell's plane, exactly in its triangle interior.
    alpha, beta, gamma = 0.5, 0.3, 0.2
    p_in_plane = alpha * a + beta * b + gamma * c
    r = np.linalg.norm(p_in_plane)
    arc_to_blob = float(np.degrees(np.arccos(np.clip(
        p_in_plane[0] / r, -1, 1
    ))))

    # Analytic value at this exact 3D point.
    T_analytic = p_in_plane[2] + 0.3 * p_in_plane[0]

    # FE-exact value from manual barycentric interpolation of the
    # cell's DOFs. For a P1 element with linear T, this equals
    # T_analytic up to floating-point error.
    T_a = a[2] + 0.3 * a[0]  # = T at vertex a (since T is linear)
    T_b = b[2] + 0.3 * b[0]
    T_c = c[2] + 0.3 * c[0]
    T_bary = alpha * T_a + beta * T_b + gamma * T_c

    # What does evaluate return?
    T_eval = float(np.asarray(uw.function.evaluate(T.sym, p_in_plane.reshape(1, 3))).reshape(-1)[0])

    err_eval = abs(T_eval - T_analytic)
    err_bary = abs(T_bary - T_analytic)

    print(f"Cell {cidx:5d}: centroid arc-to-blob={arc_to_blob:5.1f}°, point r={r:.6f}")
    print(f"  test point in-plane           : [{p_in_plane[0]:+.4f}, {p_in_plane[1]:+.4f}, {p_in_plane[2]:+.4f}]")
    print(f"  T_analytic = z + 0.3·x        : {T_analytic:+.6f}")
    print(f"  T (manual barycentric of cell): {T_bary:+.6f}  err={err_bary:.2e}")
    print(f"  T (DMInterpolation evaluate)  : {T_eval:+.6f}  err={err_eval:.2e}")
    if err_eval > 1e-3:
        print(f"  >>> evaluate gives wrong value (out by {err_eval:.4f})")
    print()
