"""Find the worst-quality cell in the stuck-run's last snapshot
and render a zoomed view of it (and a wider context view) so the
sliver becomes visible."""
import os
import glob
import re
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
from underworld3.meshing import smoothing as _sm
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/adapt_loop_movie_ref3')

# Find the last snapshot from the stuck v3 run (before we
# restarted with the h0 fix).
fs = sorted(glob.glob(os.path.join(OUT, "step*.mesh.00000.h5")))
if not fs:
    raise SystemExit("no snapshots — was the directory wiped?")
last_step = max(int(re.search(r"step(\d+)", f).group(1)) for f in fs)
stem = f"step{last_step:04d}"
print(f"Inspecting {stem}")

m = uw.discretisation.Mesh(os.path.join(OUT, f"{stem}.mesh.00000.h5"))
T = uw.discretisation.MeshVariable(
    "T_view", m, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T.read_timestep(stem, "T_v2p1", 0, outputPath=OUT)

tris = _sm._tri_cells(m.dm)
coords = np.asarray(m.X.coords)
p = coords[tris]
e0 = np.linalg.norm(p[:, 1] - p[:, 0], axis=1)
e1 = np.linalg.norm(p[:, 2] - p[:, 1], axis=1)
e2 = np.linalg.norm(p[:, 0] - p[:, 2], axis=1)
A = np.abs(_sm._signed_areas(coords, tris))
# Cell shape quality
q = 4.0 * np.sqrt(3.0) * A / (e0**2 + e1**2 + e2**2 + 1e-30)
# Inscribed radius — what estimate_dt uses
r_inscribed = 2.0 * A / (e0 + e1 + e2)

# Worst-quality cell, smallest-area cell, smallest-r_inscribed cell
i_q = int(np.argmin(q))
i_A = int(np.argmin(A))
i_r = int(np.argmin(r_inscribed))
print(f"\nWorst cells:")
print(f"  by quality q:        cell {i_q}, q={q[i_q]:.4f},  A={A[i_q]:.3e},  r_in={r_inscribed[i_q]:.4e}")
print(f"  by area A:           cell {i_A}, q={q[i_A]:.4f},  A={A[i_A]:.3e},  r_in={r_inscribed[i_A]:.4e}")
print(f"  by inscribed radius: cell {i_r}, q={q[i_r]:.4f},  A={A[i_r]:.3e},  r_in={r_inscribed[i_r]:.4e}")

# Render two panels: full annulus with a highlighted sliver, and a zoom.
pv_T = vis.meshVariable_to_pv_mesh_object(T)
pv_T.point_data["T"] = np.asarray(T.data[:, 0])
edges = vis.mesh_to_pv_mesh(m).extract_all_edges()

# Mark the worst cell by quality
worst_cell = i_q
centroid = p[worst_cell].mean(axis=0)
worst_pts = p[worst_cell]
# A pv polygon for the worst cell
worst_poly = pv.PolyData(
    np.column_stack([worst_pts, np.zeros(3)]),
    faces=np.array([3, 0, 1, 2], dtype=np.int64),
)

pl = pv.Plotter(shape=(1, 2), off_screen=True,
                window_size=(3000, 1500), border=False)
pl.set_background("white")

# Panel 1: full annulus, mark the sliver in lime green
pl.subplot(0, 0)
pl.add_text(f"{stem}: full annulus, sliver marked\n"
            f"worst q={q[worst_cell]:.3f}, A={A[worst_cell]:.2e}, "
            f"r_in={r_inscribed[worst_cell]:.2e}",
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

# Panel 2: zoom to centroid +/- 0.05
pl.subplot(0, 1)
pl.add_text(f"Zoomed view at centroid "
            f"({centroid[0]:.3f}, {centroid[1]:.3f})\n"
            f"window = 0.10 × 0.10",
            font_size=18, color='black')
pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
            clim=(0.0, 1.0), show_edges=False,
            lighting=False, show_scalar_bar=False)
pl.add_mesh(edges, color="black", line_width=1.5,
            lighting=False, opacity=0.85)
pl.add_mesh(worst_poly, color="lime", show_edges=True,
            edge_color="lime", line_width=5, opacity=0.7)
pl.view_xy()
# Center on the sliver and zoom in
pl.camera.SetFocalPoint(centroid[0], centroid[1], 0.0)
pl.camera.SetPosition(centroid[0], centroid[1], 1.0)
pl.camera.SetParallelScale(0.05)   # half-window 0.05

out_png = os.path.join(OUT, "plot_sliver_zoom.png")
pl.screenshot(out_png)
pl.close()
print(f"\nwrote {out_png}")
