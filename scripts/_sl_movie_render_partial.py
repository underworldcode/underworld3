"""Render the frames available so far (skips most recent step
in case it's still being written). T field + mesh + streamlines
overlay. Just renders PNGs — doesn't build the mp4."""
import os
import glob
import re
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True


def _sample_velocity_grid(V, n=200, r_inner=0.5, r_outer=1.0):
    """Sample the velocity field on a regular n×n grid covering
    the annulus bounding box. Returns a pv.ImageData with vector
    field 'velocity' (z component zero), with samples outside
    the annulus masked to zero so streamlines die in the holes."""
    # Build grid
    lo, hi = -r_outer * 1.05, r_outer * 1.05
    spacing = (hi - lo) / (n - 1)
    grid = pv.ImageData(
        dimensions=(n, n, 1),
        spacing=(spacing, spacing, 1.0),
        origin=(lo, lo, 0.0),
    )
    xs = np.linspace(lo, hi, n)
    XX, YY = np.meshgrid(xs, xs, indexing='xy')
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    # Sample velocity at grid points (uw.function.evaluate)
    v_pts = np.zeros((pts.shape[0], 3))
    try:
        v_eval = np.asarray(uw.function.evaluate(
            V.sym, pts)).reshape(pts.shape[0], 2)
        v_pts[:, 0] = v_eval[:, 0]
        v_pts[:, 1] = v_eval[:, 1]
    except Exception:
        # Some points outside the mesh — fall back to per-point try
        for i, p in enumerate(pts):
            try:
                v = np.asarray(uw.function.evaluate(
                    V.sym, p.reshape(1, 2))).reshape(2)
                v_pts[i, 0] = v[0]; v_pts[i, 1] = v[1]
            except Exception:
                pass
    # Mask outside annulus (set v=0)
    r2 = pts[:, 0]**2 + pts[:, 1]**2
    mask = (r2 < r_inner**2) | (r2 > r_outer**2)
    v_pts[mask, :] = 0.0
    grid.point_data["velocity"] = v_pts
    return grid

OUT = os.path.expanduser(
    os.environ.get('SL_MOVIE_OUT',
                   '~/+Simulations/StagnantLid/adapt_loop_movie_ref5'))
FRAMES = os.path.join(OUT, 'frames')
os.makedirs(FRAMES, exist_ok=True)

# Discover completed checkpoints (drop the most recent in case
# it's still being written)
files = sorted(glob.glob(os.path.join(OUT, "step*.mesh.00000.h5")))
steps = [int(re.search(r"step(\d+)", f).group(1)) for f in files]
# Drop the last one to avoid races
if len(steps) > 1:
    steps = steps[:-1]
    files = files[:-1]
if os.path.exists(os.path.join(OUT, "init.mesh.00000.h5")):
    steps = [0] + steps
    files = [os.path.join(OUT, "init.mesh.00000.h5")] + files
print(f"Rendering {len(steps)} frames (steps 0 to {max(steps)})")

hist = np.load(os.path.join(OUT, "history.npz"))
hist_step = np.asarray(hist['step']).astype(int)
hist_t = np.asarray(hist['t'])
step_to_t = {0: 0.0}
for i, s in enumerate(hist_step):
    step_to_t[int(s)] = float(hist_t[i])

for i, (s, mesh_file) in enumerate(zip(steps, files)):
    out_png = os.path.join(FRAMES, f"frame_{i:04d}.png")
    if os.path.exists(out_png):
        continue
    label = "init" if s == 0 else f"step{s:04d}"
    t_val = step_to_t.get(s, 0.0)
    m = uw.discretisation.Mesh(mesh_file)
    T = uw.discretisation.MeshVariable(
        f"T_view_{i}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    V = uw.discretisation.MeshVariable(
        f"V_view_{i}", m, vtype=uw.VarType.VECTOR,
        degree=2, continuous=True)
    T.read_timestep(label, "T_v2p1", 0, outputPath=OUT)
    V.read_timestep(label, "V_v2p1", 0, outputPath=OUT)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()

    # Streamlines on a regular sampling grid (pv 2D streamlines
    # don't work directly on the mesh PolyData when it's all in
    # the z=0 plane; the integrator can't find seeds inside the
    # cells reliably). Sampling onto a regular grid first works.
    v_data = np.asarray(V.data)
    grid = _sample_velocity_grid(V, n=200)
    streamlines = grid.streamlines_evenly_spaced_2D(
        vectors="velocity",
        start_position=(0.0, 0.75, 0.0),
        separating_distance=8.0,
        separating_distance_ratio=0.4,
        step_length=0.5,
        max_steps=4000,
    )

    pl = pv.Plotter(off_screen=True, window_size=(1500, 1500),
                    border=False)
    pl.set_background("white")
    vmax = float(np.linalg.norm(v_data, axis=1).max())
    pl.add_text(f"step {s:>4d}     t = {t_val:.5f}     "
                f"|v|max = {vmax:.2f}",
                font_size=22, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=0.7,
                lighting=False, opacity=0.5)
    if streamlines.n_points > 0:
        pl.add_mesh(streamlines, color="black", line_width=1.8,
                    opacity=0.5, lighting=False)
    pl.view_xy()
    pl.camera.zoom(1.25)
    pl.screenshot(out_png)
    pl.close()
    if i % 5 == 0:
        print(f"  frame {i}/{len(steps)}: step {s}, t={t_val:.5f}")

print(f"\n{len(steps)} frames rendered to {FRAMES}")
