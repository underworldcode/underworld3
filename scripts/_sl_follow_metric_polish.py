"""Render the effect of a Jacobi polish pass after follow_metric.

Compares no-polish vs 1-step, 2-step, 5-step Jacobi (alpha=0.2)
on the ref=3.0 stagnant-lid adapted mesh. The polish smooths
node positions toward neighbor-centroid average — a gentle local
cleanup that eliminates slivers and improves aspect without
undoing the metric distribution.
"""
import os
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
SRC = os.path.expanduser(
    '~/+Simulations/StagnantLid/aniso_dt_validate/R1.0_aniso')
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/follow_metric_polish')
os.makedirs(OUT, exist_ok=True)

CASES = [
    ("aniso only (no polish)", 0, 0.0),
    ("aniso + 1×Jacobi α=0.2", 1, 0.2),
    ("aniso + 2×Jacobi α=0.2", 2, 0.2),
    ("aniso + 5×Jacobi α=0.2", 5, 0.2),
]


def load_fresh():
    m = uw.discretisation.Mesh(
        os.path.join(SRC, "step0080.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("step0080", "T_v2p1", 0, outputPath=SRC)
    return m, T


for label, n_iters, alpha in CASES:
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("=", "")
            .replace("α", "a").replace("×", "x").replace(",", "")
            .replace("(", "").replace(")", "").replace(".", "p"))
    if os.path.exists(
            os.path.join(out_dir, "adapted.mesh.00000.h5")):
        print(f"{label}: already adapted")
        continue
    os.makedirs(out_dir, exist_ok=True)
    print(f"{label}: adapting + polishing")
    m, T = load_fresh()
    old_X = np.asarray(m.X.coords).copy()
    old_T = np.asarray(T.data).copy()
    moved = uw.meshing.follow_metric(
        m, T, refinement=3.0, skip_threshold=None)
    if n_iters > 0:
        uw.meshing.smooth_mesh_interior(
            m, n_iters=n_iters, alpha=alpha)
    new_X = np.asarray(m.X.coords).copy()
    if not np.allclose(new_X, old_X):
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


# Render 4-panel
ncols = 2
nrows = 2
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1500 * ncols, 1500 * nrows),
                border=False)
pl.set_background("white")
for i, (label, n_iters, alpha) in enumerate(CASES):
    row, col = i // ncols, i % ncols
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("=", "")
            .replace("α", "a").replace("×", "x").replace(",", "")
            .replace("(", "").replace(")", "").replace(".", "p"))
    m = uw.discretisation.Mesh(
        os.path.join(out_dir, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_view_{i}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_v2p1", 0, outputPath=out_dir)
    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()
    pl.subplot(row, col)
    pl.add_text(label, font_size=26, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    pl.add_mesh(edges, color="black", line_width=1.2,
                lighting=False, opacity=0.85)
    pl.view_xy()
    pl.camera.zoom(1.25)

out_png = os.path.join(OUT, "plot_polish_compare.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
