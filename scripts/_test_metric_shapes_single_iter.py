"""Simplest possible case: ONE outer iteration of the mover.
No polish, no spring, no damping multiplier. Just the bare
Winslow elliptic solve applied once with a full step.

Sweep relax ∈ {0.1, 0.25, 0.5, 1.0} to see how the mover's
single-step displacement compares to the target."""
import os
import sys
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_single_iter')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes import build_mesh_with_field, shape_field


# Build a reference fine mesh + target ρ for the background
mesh_ref = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
    cellSize=0.012, qdegree=3)
T_ref = uw.discretisation.MeshVariable(
    "T_target", mesh_ref, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_ref.data[:, 0] = shape_field(np.asarray(T_ref.coords))
rho_expr = uw.meshing.metric_density_from_gradient(
    mesh_ref, T_ref, refinement=3.0, name="single_iter_rho")
rho_view = uw.discretisation.MeshVariable(
    "rho_view", mesh_ref, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
import underworld3.function as fn
rho_view.data[:, 0] = np.asarray(
    fn.evaluate(rho_expr, rho_view.coords)).reshape(-1)
pv_rho = vis.meshVariable_to_pv_mesh_object(rho_view)
pv_rho.point_data["rho"] = np.asarray(rho_view.data[:, 0])
rho_clim = (0.3, 9.0)


# Sweep: single iteration with different relax values.
CASES = [
    ("undeformed", None),
    ("n_outer=1, relax=0.1", dict(relax=0.1)),
    ("n_outer=1, relax=0.25", dict(relax=0.25)),
    ("n_outer=1, relax=0.5", dict(relax=0.5)),
    ("n_outer=1, relax=1.0", dict(relax=1.0)),
    # For reference: same case at the default
    ("default (n_outer=12, relax=0.2)", "default"),
]


def adapt_one(label, kw):
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("=", "")
              .replace(",", "").replace("(", "")
              .replace(")", "").replace(".", "p"))
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(
            os.path.join(out_dir, "adapted.mesh.00000.h5")):
        print(f"{label}: cached")
        return out_dir
    print(f"{label}: adapting")
    m, T = build_mesh_with_field()
    if kw is None:
        # No adaptation — just write out
        pass
    else:
        rho = uw.meshing.metric_density_from_gradient(
            m, T, refinement=3.0, name=f"sing_{label}")
        if kw == "default":
            method_kwargs = dict(relax=0.2, n_outer=12)
        else:
            method_kwargs = dict(n_outer=1, **kw)
        uw.meshing.smooth_mesh_interior(
            m, metric=rho, method="anisotropic",
            strategy="med",
            method_kwargs=method_kwargs, verbose=False)
    m.write_timestep(filename="adapted", index=0,
                     outputPath=out_dir, meshVars=[T],
                     meshUpdates=True, create_xdmf=True)
    return out_dir


for label, kw in CASES:
    adapt_one(label, kw)


# Render 3x2 grid: target ρ background + adapted mesh overlay
ncols, nrows = 2, 3
pl = pv.Plotter(shape=(nrows, ncols), off_screen=True,
                window_size=(1500 * ncols, 1500 * nrows),
                border=False)
pl.set_background("white")
for i, (label, kw) in enumerate(CASES):
    row, col = i // ncols, i % ncols
    out_dir = os.path.join(
        OUT, label.replace(" ", "_").replace("=", "")
              .replace(",", "").replace("(", "")
              .replace(")", "").replace(".", "p"))
    pl.subplot(row, col)
    pl.add_text(label, font_size=24, color='black')
    # Background: target ρ
    pl.add_mesh(pv_rho, scalars="rho", cmap="viridis",
                clim=rho_clim, show_edges=False,
                lighting=False, show_scalar_bar=False,
                opacity=0.55)
    # Overlay adapted mesh
    if os.path.exists(out_dir):
        try:
            m_ad = uw.discretisation.Mesh(
                os.path.join(out_dir, "adapted.mesh.00000.h5"))
            edges_ad = vis.mesh_to_pv_mesh(m_ad).extract_all_edges()
            pts = np.asarray(edges_ad.points)
            pts[:, 2] = 0.1
            edges_ad.points = pts
            pl.add_mesh(edges_ad, color="black", line_width=2.0,
                        lighting=False, opacity=0.95)
        except Exception as e:
            print(f"  skip {label}: {e}")
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_shapes_single_iter.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
