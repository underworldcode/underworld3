"""Two runs side by side at matching checkpoint steps: T with streamlines.

  python render_compare.py -uw_runs box_supg_Ra1e4_C1_h0.03125_np1,box_slcn_Ra1e4_C1_h0.03125_np1 -uw_steps 0,100,300,800,1400 -uw_out figures/01_box_Ra1e4_supg_vs_slcn.png
"""
import os, glob
import numpy as np, pyvista as pv
import underworld3 as uw, underworld3.visualisation as vis
pv.OFF_SCREEN = True

params = uw.Params(uw_runs="", uw_steps="0,100,300,800,1400", uw_out="figures/compare.png", uw_degree=2)
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def model_time(run, step):
    """Model time at a checkpoint step, from the run's step log (interpolated
    between logged rows when the step itself was not logged)."""
    import re
    rows = []
    with open(os.path.join(D, "runs", run, "steps.log")) as f:
        for line in f:
            m = re.match(r"step\s+(\d+) t ([0-9.eE+-]+)", line)
            if m:
                rows.append((int(m.group(1)), float(m.group(2))))
    rows = [(0, 0.0)] + rows
    s, t = zip(*rows)
    return float(np.interp(step, s, t))
runs = str(params.uw_runs).split(","); steps = [int(s) for s in str(params.uw_steps).split(",")]

def load(run):
    cd = os.path.join(D, "runs", run, "checkpoints")
    mesh = uw.discretisation.Mesh(os.path.join(cd, "checkpoint.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(f"T_{run}", mesh, 1, degree=int(params.uw_degree))
    V = uw.discretisation.MeshVariable(f"U_{run}", mesh, mesh.dim, degree=2)
    return cd, mesh, T, V

def panel(pl, cd, mesh, T, V, run, step):
    T.read_timestep("checkpoint", "T", step, outputPath=cd)
    V.read_timestep("checkpoint", "U", step, outputPath=cd)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    pvmesh = vis.mesh_to_pv_mesh(mesh)
    pvmesh.point_data["V"] = vis.vector_fn_to_pv_points(pvmesh, V.sym)
    speed = np.linalg.norm(pvmesh.point_data["V"], axis=1).max()
    lo, hi = mesh.X.coords.min(axis=0), mesh.X.coords.max(axis=0)
    if "annulus" in run:
        rr, tt = np.meshgrid(np.linspace(0.58, 0.92, 4), np.linspace(0, 2 * np.pi, 25)[:-1])
        seed_xy = np.c_[(rr * np.cos(tt)).ravel(), (rr * np.sin(tt)).ravel()]
    else:
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], 8)[1:-1], np.linspace(lo[1], hi[1], 8)[1:-1])
        seed_xy = np.c_[gx.ravel(), gy.ravel()]
    seeds = pv.PolyData(np.c_[seed_xy, np.zeros(len(seed_xy))])
    L = float((hi - lo).max())
    lines = pvmesh.streamlines_from_source(
        seeds, vectors="V", integration_direction="both", max_time=40.0 * L / max(speed, 1e-12),
        initial_step_length=0.2, max_step_length=0.5, max_steps=20000, compute_vorticity=False)
    pl.set_background("white")
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r", clim=(0, 1), show_edges=False, lighting=False, show_scalar_bar=False)
    if lines.n_points:
        pl.add_mesh(lines, color="black", line_width=1.2, lighting=False)
    pl.add_text(f"{run.split('_')[1].upper()}  t = {model_time(run, step):.3f}  step {step}  max|v| {speed:.1f}", font_size=10, color="black")
    pl.view_xy(); pl.camera.zoom(1.35)

loaded = [load(r) for r in runs]
pl = pv.Plotter(off_screen=True, shape=(len(runs), len(steps)), window_size=(600 * len(steps), 600 * len(runs)), border=False)
for i, (run, (cd, mesh, T, V)) in enumerate(zip(runs, loaded)):
    for j, step in enumerate(steps):
        pl.subplot(i, j)
        panel(pl, cd, mesh, T, V, run, step)
out = os.path.join(D, str(params.uw_out)); os.makedirs(os.path.dirname(out), exist_ok=True)
pl.screenshot(out); pl.close(); print("wrote", out)
