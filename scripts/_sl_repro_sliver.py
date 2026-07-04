"""Reproduce the compounding-refinement sliver bug in a single
controlled script (no convection — just repeated follow_metric
calls). Saves the worst sliver mesh, then renders a zoomed view
so the bug is visible.

This is needed because the actual stuck-run snapshot was deleted
on restart. The mechanism is the same: repeated adapts each
shrink h0 and compound refinement, eventually producing a
near-degenerate triangle that crashes dt via the inscribed
radius.
"""
import os
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.meshing import smoothing as _sm
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser('~/+Simulations/StagnantLid/sliver_repro')
os.makedirs(OUT, exist_ok=True)


def report(m, T, label):
    tris = _sm._tri_cells(m.dm)
    p = np.asarray(m.X.coords)[tris]
    e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
    e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
    e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
    A = np.abs(_sm._signed_areas(np.asarray(m.X.coords), tris))
    q = 4 * np.sqrt(3) * A / (e0**2 + e1**2 + e2**2 + 1e-30)
    r_in = 2.0 * A / (e0 + e1 + e2)
    print(f"  {label}: A_min={A.min():.2e}  q_min={q.min():.3f}  "
          f"r_in_min={r_in.min():.2e}")


# Fresh annulus + tanh field (like the test fixture)
m = uw.meshing.Annulus(radiusOuter=1.0, radiusInner=0.5,
                       cellSize=1/16, qdegree=3)
T = uw.discretisation.MeshVariable(
    "T", m, vtype=uw.VarType.SCALAR, degree=3, continuous=True)
x, y = m.X
r = sympy.sqrt(x*x + y*y)
T_expr = 0.5 * (1.0 + sympy.tanh(40.0 * (0.7 - r)))
proj = uw.systems.Projection(m, T)
proj.smoothing = 0.0
proj.uw_function = T_expr
proj.solve()

# Force the mover to RE-READ its h0 from the deformed mesh by
# repeated follow_metric calls. The bug compounds at each call.
print("Repeated follow_metric calls (the compounding bug):")
report(m, T, "init      ")
for i in range(1, 9):
    uw.meshing.follow_metric(
        m, T, refinement=3.0, skip_threshold=None)
    report(m, T, f"after adapt #{i:2d}")

# Save final adapted state for rendering
m.write_timestep(filename="final", index=0, outputPath=OUT,
                 meshVars=[T], meshUpdates=True,
                 create_xdmf=True)


# Find worst-quality cell + render zoom
tris = _sm._tri_cells(m.dm)
coords = np.asarray(m.X.coords)
p = coords[tris]
e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
A = np.abs(_sm._signed_areas(coords, tris))
q = 4 * np.sqrt(3) * A / (e0**2 + e1**2 + e2**2 + 1e-30)
r_in = 2 * A / (e0 + e1 + e2)
i_worst = int(np.argmin(q))
worst_pts = p[i_worst]
centroid = worst_pts.mean(axis=0)
print(f"\nWorst cell: q={q[i_worst]:.4f}  A={A[i_worst]:.3e}  "
      f"r_in={r_in[i_worst]:.3e}")
print(f"  vertices: {worst_pts.tolist()}")
print(f"  edges: ({e0[i_worst]:.4f}, {e1[i_worst]:.4f}, {e2[i_worst]:.4f})")

# Render full + zoom
pv_T = vis.meshVariable_to_pv_mesh_object(T)
pv_T.point_data["T"] = np.asarray(T.data[:, 0])
edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
worst_poly = pv.PolyData(
    np.column_stack([worst_pts, np.zeros(3)]),
    faces=np.array([3, 0, 1, 2], dtype=np.int64),
)

pl = pv.Plotter(shape=(1, 2), off_screen=True,
                window_size=(3000, 1500), border=False)
pl.set_background("white")

pl.subplot(0, 0)
pl.add_text(f"Full annulus after 8 repeated follow_metric calls\n"
            f"worst q={q[i_worst]:.3f}, A={A[i_worst]:.2e}, "
            f"r_in={r_in[i_worst]:.2e}\n"
            f"dt_diff ≈ r_in² = {r_in[i_worst]**2:.2e}",
            font_size=18, color='black')
pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
            clim=(0.0, 1.0), show_edges=False,
            lighting=False, show_scalar_bar=False)
pl.add_mesh(edges, color="black", line_width=0.7,
            lighting=False, opacity=0.5)
pl.add_mesh(worst_poly, color="lime", show_edges=True,
            edge_color="lime", line_width=4)
pl.view_xy()
pl.camera.zoom(1.25)

pl.subplot(0, 1)
pl.add_text(f"Zoom at centroid ({centroid[0]:.4f}, "
            f"{centroid[1]:.4f})\nwindow ~0.10 × 0.10",
            font_size=18, color='black')
pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
            clim=(0.0, 1.0), show_edges=False,
            lighting=False, show_scalar_bar=False)
pl.add_mesh(edges, color="black", line_width=1.5,
            lighting=False, opacity=0.85)
pl.add_mesh(worst_poly, color="lime", show_edges=True,
            edge_color="lime", line_width=5, opacity=0.7)
# Use camera_position with a tight bounding box around the centroid
half = 0.05
pl.camera_position = [
    (centroid[0], centroid[1], 1.0),   # eye
    (centroid[0], centroid[1], 0.0),   # focal
    (0.0, 1.0, 0.0),                   # up
]
pl.camera.parallel_projection = True
pl.camera.parallel_scale = half
pl.reset_camera_clipping_range()

out_png = os.path.join(OUT, "plot_sliver_zoom.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
