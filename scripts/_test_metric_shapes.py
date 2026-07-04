"""Synthetic-shapes test for the follow_metric adapter.

Builds a square mesh and a T field with three sharp-boundary
shapes (rotated square, doughnut, triangle), each described as
a smooth level set. Adapts the mesh with follow_metric and
renders the result, so we can see whether the existing meshing
infrastructure resolves each shape's boundary correctly without
any convection coupling.

Decouples the meshing from any specific physics — pure
mesh-vs-metric exercise.
"""
import os
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes')
os.makedirs(OUT, exist_ok=True)


# ---------- Shape level-set indicators (numpy) ---------------------
# Each returns: distance INTO the shape (positive inside, negative
# outside). We smooth with tanh at the boundary so |∇T| is large
# in a narrow band ~ eps wide.
EPS = 0.02  # boundary smoothing scale


def square_signed_distance(p, centre, side, angle_deg):
    """Signed distance into an axis-aligned square at `centre`,
    side length `side`, rotated by `angle_deg`."""
    th = np.deg2rad(angle_deg)
    ct, st = np.cos(th), np.sin(th)
    dx = p[:, 0] - centre[0]
    dy = p[:, 1] - centre[1]
    xp = ct * dx + st * dy
    yp = -st * dx + ct * dy
    # SDF for axis-aligned square: outside is L_inf − side/2
    d = (side / 2.0) - np.maximum(np.abs(xp), np.abs(yp))
    return d


def doughnut_signed_distance(p, centre, r_in, r_out):
    """Signed distance into the annular region [r_in, r_out]
    around `centre`."""
    r = np.linalg.norm(p - centre, axis=1)
    # Inside if r_in <= r <= r_out
    return np.minimum(r - r_in, r_out - r)


def triangle_signed_distance(p, vertices):
    """Signed distance into a triangle (CCW vertices)."""
    v0, v1, v2 = [np.asarray(v) for v in vertices]
    # Distance to each edge (positive if INSIDE the half-plane
    # defined by the edge); take the MIN of those three to get
    # "negative if outside any half-plane, positive if inside all".
    def half_plane(a, b):
        # Outward normal to edge a→b for CCW polygon: n = (dy,-dx)
        # No wait — for CCW, interior is on the LEFT of a→b.
        # So inward normal is (-(b-a)_y, (b-a)_x).
        ed = b - a
        nrm = np.array([-ed[1], ed[0]])
        nrm /= np.linalg.norm(nrm)
        return (p - a) @ nrm
    d0 = half_plane(v0, v1)
    d1 = half_plane(v1, v2)
    d2 = half_plane(v2, v0)
    return np.minimum(np.minimum(d0, d1), d2)


def shape_field(p):
    """Composite T field: 1 inside any of the three shapes, 0
    outside, with smooth tanh transitions at boundaries."""
    d_sq = square_signed_distance(
        p, centre=(0.55, 0.35), side=0.4, angle_deg=30.0)
    d_dh = doughnut_signed_distance(
        p, centre=(-0.55, 0.45), r_in=0.15, r_out=0.30)
    d_tr = triangle_signed_distance(
        p, vertices=[(0.05, -0.65), (0.55, -0.35),
                     (-0.30, -0.30)])
    # Convert each to a smooth indicator [0,1] via tanh
    f_sq = 0.5 * (1.0 + np.tanh(d_sq / EPS))
    f_dh = 0.5 * (1.0 + np.tanh(d_dh / EPS))
    f_tr = 0.5 * (1.0 + np.tanh(d_tr / EPS))
    # Composite (max → union of shapes, soft-OR via tanh
    # composition gives the cleanest level-set)
    return np.maximum(np.maximum(f_sq, f_dh), f_tr)


def build_mesh_with_field():
    m = uw.meshing.UnstructuredSimplexBox(
        minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
        cellSize=0.04, qdegree=3)
    T = uw.discretisation.MeshVariable(
        "T_shapes", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    coords = np.asarray(T.coords)
    T.data[:, 0] = shape_field(coords)
    return m, T


# ------- Sweep follow_metric across refinement values --------------
CASES = [
    ("uniform (no adapt)", None),
    ("refinement = 2", 2.0),
    ("refinement = 3", 3.0),
    ("refinement = 5", 5.0),
]


for label, ref in CASES:
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("=", "")
              .replace("(", "").replace(")", ""))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "adapted.mesh.00000.h5")):
        print(f"{label}: already adapted")
        continue
    print(f"{label}: building & adapting")
    m, T = build_mesh_with_field()
    if ref is not None:
        old_X = np.asarray(m.X.coords).copy()
        old_T = np.asarray(T.data).copy()
        moved = uw.meshing.follow_metric(
            m, T, refinement=ref, skip_threshold=None,
            verbose=True)
        if moved:
            new_X = np.asarray(m.X.coords).copy()
            new_Tx = np.asarray(T.coords).copy()
            m._deform_mesh(old_X)
            T.data[...] = old_T
            rT = np.asarray(uw.function.evaluate(
                T.sym[0], new_Tx)).reshape(-1)
            m._deform_mesh(new_X)
            T.data[:, 0] = rT
    m.write_timestep(filename="adapted", index=0,
                     outputPath=out_dir, meshVars=[T],
                     meshUpdates=True, create_xdmf=True)


# ------- Render 2×2 grid --------------------------------------------
ncols, nrows = 2, 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1500 * ncols, 1500 * nrows),
                border=False)
pl.set_background("white")

for i, (label, ref) in enumerate(CASES):
    row, col = i // ncols, i % ncols
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("=", "")
              .replace("(", "").replace(")", ""))
    m = uw.discretisation.Mesh(
        os.path.join(out_dir, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_view_{i}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_shapes", 0, outputPath=out_dir)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(label, font_size=26, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="Blues",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=1.0,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_shapes_sweep.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
