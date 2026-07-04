"""Compare uw.meshing.follow_metric() settings on a saturated
stagnant-lid T field.

Loads the step0080 snapshot from the validated R=1 uniform run,
then runs follow_metric() with several (refinement, coarsening,
metric) combinations. Each panel shows the T field + adapted
mesh overlay so the user can see how the new two-knob API trades
off refinement intensity vs grading transition.
"""
import os
import numpy as np
import sympy
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True

SRC = os.path.expanduser(
    '~/+Simulations/StagnantLid/aniso_dt_validate/R1.0_aniso')
SRC_STEM = "step0080"
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/follow_metric_compare')
os.makedirs(OUT, exist_ok=True)


CASES = [
    # (label, refinement, coarsening, metric)
    ("uniform (no adapt)",       None, None,   None),
    ("ref=1.5, coar=auto, FF",   1.5,  "auto", "front-following"),
    ("ref=2.0, coar=auto, FF",   2.0,  "auto", "front-following"),
    ("ref=3.0, coar=auto, FF",   3.0,  "auto", "front-following"),
    ("ref=2.0, coar=2.0, FF",    2.0,  2.0,    "front-following"),
    ("ref=2.0, coar=auto, GU",   2.0,  "auto", "gradient-uniform"),
]


def load_fresh():
    m = uw.discretisation.Mesh(
        os.path.join(SRC, f"{SRC_STEM}.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        "T_v2p1", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep(SRC_STEM, "T_v2p1", 0, outputPath=SRC)
    return m, T


# Pre-adapt the meshes and save (so the render step is decoupled)
for label, ref, coar, mc in CASES:
    out_dir = os.path.join(OUT, label.replace(" ", "_")
                            .replace(",", "")
                            .replace("=", "")
                            .replace("(", "")
                            .replace(")", ""))
    if os.path.exists(
            os.path.join(out_dir, "adapted.mesh.00000.h5")):
        print(f"{label}: already adapted")
        continue
    os.makedirs(out_dir, exist_ok=True)
    print(f"{label}: adapting")
    m, T = load_fresh()
    if ref is None:
        # No adaptation — just write back as-is
        pass
    else:
        # FE-remap T as in the production adapt step
        old_X = np.asarray(m.X.coords).copy()
        old_T = np.asarray(T.data).copy()
        moved = uw.meshing.follow_metric(
            m, T, refinement=ref, coarsening=coar,
            metric=mc, skip_threshold=None)
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
    print("  done")


# Render
ncols = 3
nrows = (len(CASES) + ncols - 1) // ncols
# Bump per-panel resolution so individual cell edges are legible
# at higher refinement (ref=3 has ~3× finer cells along the BL).
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1100 * ncols, 1100 * nrows),
                border=False)
pl.set_background("white")

for idx, (label, ref, coar, mc) in enumerate(CASES):
    row, col = idx // ncols, idx % ncols
    out_dir = os.path.join(OUT, label.replace(" ", "_")
                            .replace(",", "")
                            .replace("=", "")
                            .replace("(", "")
                            .replace(")", ""))
    m = uw.discretisation.Mesh(
        os.path.join(out_dir, "adapted.mesh.00000.h5"))
    T = uw.discretisation.MeshVariable(
        f"T_view_{idx}", m, vtype=uw.VarType.SCALAR,
        degree=3, continuous=True)
    T.read_timestep("adapted", "T_v2p1", 0, outputPath=out_dir)

    pv_T = vis.meshVariable_to_pv_mesh_object(T)
    pv_T.point_data["T"] = np.asarray(T.data[:, 0])
    edges = vis.mesh_to_pv_mesh(m).extract_all_edges()

    pl.subplot(row, col)
    pl.add_text(label, font_size=22, color='black')
    pl.add_mesh(pv_T, scalars="T", cmap="RdBu_r",
                clim=(0.0, 1.0), show_edges=False,
                lighting=False, show_scalar_bar=False)
    # Slightly thicker, more opaque edges so the cell pattern is
    # readable at the BL where cells are ~3× smaller.
    pl.add_mesh(edges, color="black", line_width=1.2,
                lighting=False, opacity=0.8)
    pl.view_xy()
    pl.camera.zoom(1.25)

out_png = os.path.join(OUT, "plot_follow_metric_compare.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
