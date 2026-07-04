"""Render the TARGET T field and the metric ρ on a reference
fine mesh, so we can see what the mover is aiming to resolve."""
import os
import sys
import numpy as np
import underworld3 as uw
import underworld3.visualisation as vis
import pyvista as pv

pv.OFF_SCREEN = True
OUT = os.path.expanduser(
    '~/+Simulations/StagnantLid/synthetic_shapes_target')
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.dirname(__file__))
from _test_metric_shapes import shape_field


# Build a FINE reference mesh purely for visualisation
mesh_ref = uw.meshing.UnstructuredSimplexBox(
    minCoords=(-1.0, -1.0), maxCoords=(1.0, 1.0),
    cellSize=0.012, qdegree=3)
T_ref = uw.discretisation.MeshVariable(
    "T_target", mesh_ref, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
T_ref.data[:, 0] = shape_field(np.asarray(T_ref.coords))

# Compute the metric ρ that follow_metric would use,
# refinement=3, default settings
rho_expr = uw.meshing.metric_density_from_gradient(
    mesh_ref, T_ref, refinement=3.0, name="target_rho")

# Sample ρ at the reference vertices for rendering
import underworld3.function as fn
rho_view = uw.discretisation.MeshVariable(
    "rho_view", mesh_ref, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
rho_view.data[:, 0] = np.asarray(
    fn.evaluate(rho_expr, rho_view.coords)).reshape(-1)

# Compute |∇T| for the gradient-magnitude view
import sympy
X = mesh_ref.X
grad_T = T_ref.sym[0].diff(X[0])**2 + T_ref.sym[0].diff(X[1])**2
grad_view = uw.discretisation.MeshVariable(
    "grad_view", mesh_ref, vtype=uw.VarType.SCALAR,
    degree=3, continuous=True)
grad_view.data[:, 0] = np.sqrt(np.asarray(
    fn.evaluate(grad_T, grad_view.coords)).reshape(-1))


# Render three panels: T (binary-ish), |∇T| (boundary indicator),
# ρ (the metric the adapter sees)
ncols = 3
pl = pv.Plotter(shape=(1, ncols), off_screen=True,
                window_size=(1500 * ncols, 1500), border=False)
pl.set_background("white")

views = [
    ("Target T field", T_ref, "Blues", (0.0, 1.0)),
    ("|grad T|  (where edges are)", grad_view, "magma", None),
    ("Metric rho = (1 + amp*t)^p  ref=3", rho_view, "viridis",
     None),
]
for i, (label, var, cmap, clim) in enumerate(views):
    pv_v = vis.meshVariable_to_pv_mesh_object(var)
    pv_v.point_data["val"] = np.asarray(var.data[:, 0])
    pl.subplot(0, i)
    pl.add_text(label, font_size=26, color='black')
    kwargs = dict(scalars="val", cmap=cmap, show_edges=False,
                  lighting=False, show_scalar_bar=True,
                  scalar_bar_args=dict(color="black"))
    if clim is not None:
        kwargs["clim"] = clim
    pl.add_mesh(pv_v, **kwargs)
    pl.view_xy()
    pl.camera.zoom(1.15)

out_png = os.path.join(OUT, "plot_shapes_target.png")
pl.screenshot(out_png)
pl.close()
print(f"wrote {out_png}")
print(f"T range:    [{T_ref.data.min():.3f}, "
      f"{T_ref.data.max():.3f}]")
print(f"|grad T|:   [{grad_view.data.min():.3f}, "
      f"{grad_view.data.max():.3f}]")
print(f"rho:        [{rho_view.data.min():.3f}, "
      f"{rho_view.data.max():.3f}]")
